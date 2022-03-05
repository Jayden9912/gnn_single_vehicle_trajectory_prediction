import torch
from torch_geometric.loader import DataLoader
from stp_ego import stp
from options_modified import NN_Options
from utils import *
from dataset import *

options = NN_Options()
opts = options.parse()

def compute_losses(pred, gt):
    """loss function
    """
    pred_x = pred[:,:,0]
    pred_y = pred[:,:,1]
    gt_x = gt[:,:,0]
    gt_y = gt[:,:,1]
    batch_size = gt_y.shape[0]
    RMSE = torch.sqrt(torch.mean(20*torch.pow((pred_x-gt_x),2) + 0.5*torch.pow((pred_y-gt_y),2)))
    return RMSE

"""evaluation
"""
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#load model
model_path = "/home/jayden99/Desktop/URECA/models/03052022_no_downsampling.pth"
print("Loading model from",model_path)
print("Loading pretrained model")
model = stp(opts)
model.load_state_dict(torch.load(model_path))
model.eval()
model.to(device)

#loading data
test_docpath = "/home/jayden99/Desktop/URECA/dataset/test_xy.txt"
test_fpath_list = read_file(test_docpath)
print("-- Predicting on %s test samples"%len(test_fpath_list))
test_dataset = trajectory_data(test_fpath_list)
test_generator = DataLoader(test_dataset, 1, shuffle = True,pin_memory = True, num_workers = 4)

model.eval()
with torch.no_grad():
    print("evaluating")
    losses = 0
    for i, data in enumerate(test_generator):
        pred = model(data, device)
        gt = data.y.to(device)
        losses += compute_losses(pred,gt)
avg_losses = losses/len(test_fpath_list)
print(" ".join(["average_losses:",avg_losses]))
