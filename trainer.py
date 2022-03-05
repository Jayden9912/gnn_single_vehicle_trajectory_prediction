import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import time
from datetime import datetime
import stp_ego
from dataset import *
from utils import *
import json
import pprint

class trainer:
    def __init__(self, options):
        self.opt = options
        opt_dict = self.opt.__dict__
        self.log_dir = "/home/jayden99/Desktop/URECA/"

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.model = stp_ego.stp(options)
        self.model.to(self.device)

        self.model_optimizer = optim.Adam(self.model.parameters(), self.opt.learning_rates)
        # self.model_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer,self.opt.scheduler_step_size, 0.1)
        # self.model_scheduler = optim.lr_scheduler.MultiStepLR(self.model_optimizer,[10,20,30,40,50],gamma = 0.1)
        self.model_scheduler = optim.lr_scheduler.MultiStepLR(self.model_optimizer, milestones = [5,20,30,40],gamma = 0.1)

        starting_time = datetime.now().strftime("%m%d%Y %H:%M:%S")
        self.save_filename = starting_time
        
        #print necessary info
        print("Model used:")
        print(self.model)
        pp = pprint.PrettyPrinter()
        pp.pprint(opt_dict)
        # print(self.opt)
        print("Training starts on {}".format(starting_time))
        print("Model and tensorboard event file are saved to %s" %self.log_dir)
        print("Training using %s" %self.device)
        print("Current lr:{}".format(self.model_scheduler.get_last_lr()))

        fpath = os.path.join(self.log_dir,"dataset", "{}.txt")
        train_path = fpath.format("train_xy")
        valid_path = fpath.format("valid_xy")

        train_filenames = read_file(train_path)
        valid_filenames = read_file(valid_path)

        train_dataset = trajectory_data(train_filenames)
        self.training_generator = DataLoader(train_dataset, self.opt.batch_size, shuffle = True, pin_memory = True, num_workers = 4)
        valid_dataset = trajectory_data(valid_filenames)
        self.valid_generator = DataLoader(valid_dataset, self.opt.batch_size, shuffle = True,pin_memory = True, num_workers = 4)
        self.val_iter = iter(self.valid_generator)

        if not os.path.exists(os.path.join(self.log_dir,"logfile")):
            os.makedirs(os.path.join(self.log_dir,"logfile"))
        self.writer = SummaryWriter(os.path.join(self.log_dir,"logfile"))

        # print("There are {} training data and {} validation data".format(len(train_filenames), len(valid_filenames)))
        print("There are {} training data".format(len(train_filenames)))
        
        #save the options used in this run
        self.save_opts()
        #set the sum of losses to 0
        self.losses_sum = 0


    def set_train(self):
        """convert all models to training mode
        """
        self.model.train()

    def set_eval(self):
        """convert all models to evaluation mode
        """
        self.model.eval()

    def train(self):
        """run the entire pipeline
        """
        print("Training")
        self.step = 0
        for ep in range(self.opt.num_epoch):
            self.epoch = ep
            self.run_epoch()
            if ep == (self.opt.num_epoch -1):
                self.save_model()
                ending_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                print("Training ends on {}".format(ending_time))

    def run_epoch(self):
        """run a single epoch of training and validation
        """
        self.set_train()
        for batch_idx, data in enumerate(self.training_generator):
            before_op_time = time.time()
            data = data.to(self.device)
            #downsample the input data
            # data.x =  data.x[:, ::2, :] 
            # data.y =  data.y[:,4::5,:] 
            #model training
            self.model_optimizer.zero_grad()
            pred = self.model(data,self.device)
            # pred = self.model(data)
            
            #calculating losses
            gt = data.y
            losses = self.compute_losses(pred,gt)
            losses.backward()
            self.losses_sum += losses.item()   
            self.model_optimizer.step()
            a = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            duration = time.time() - before_op_time
            self.step += 1

            if self.step%1000 == 0:
                average_losses = round((self.losses_sum/1000),4) #pls check
                self.log_time(batch_idx, duration, average_losses)
                self.losses_sum = 0
                self.log("Loss/Train",average_losses)
                self.val()
            # break
        self.model_scheduler.step()

    def val(self):
        """validate model on batches
        """
        self.set_eval()

        try:
            data = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.valid_generator)
            data = self.val_iter.next()

        with torch.no_grad():
            pred = self.model(data,self.device)
            gt = data.y.to(self.device)
            valid_losses = self.compute_losses(pred,gt)
            self.log("Loss/Validation", valid_losses)

        self.set_train()

    def compute_losses(self, pred, gt):
        pred_x = pred[:,:,0]
        pred_y = pred[:,:,1]
        gt_x = gt[:,:,0]
        gt_y = gt[:,:,1]
        batch_size = gt_y.shape[0]
        RMSE = torch.sqrt(torch.mean(20*torch.pow((pred_x-gt_x),2) + 0.5*torch.pow((pred_y-gt_y),2)))
        return RMSE

    def log_time(self, batch_idx,duration, losses):
        """print a logging statement to the terminal
        """
        current_lr = self.model_scheduler.get_last_lr()[0]
        samples_per_sec = self.opt.batch_size / duration
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:7.1f} | loss: {:.5f} | lr: {:.5f}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, losses, current_lr))

    def log(self,tags,losses):
        """write an event to tensorboard event file
        """
        self.writer.add_scalar("{}".format(tags),losses,self.step)

    def save_model(self):
        """save model weights
        """
        save_folder = os.path.join(self.log_dir,"models")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_path = os.path.join(save_folder,"{}_no_downsampling.pth".format(self.save_filename))
        torch.save(self.model.state_dict(),save_path)
  
    def save_opts(self):
        """save all the options for this training
        """
        saved_dir = os.path.join(self.log_dir,"models")
        saved_path = os.path.join(saved_dir,self.save_filename+"_no_downsampling.json")
        saved_options = self.opt.__dict__
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)
        
        with open(saved_path, 'w') as tmp:
            json.dump(saved_options, tmp, indent = 2)