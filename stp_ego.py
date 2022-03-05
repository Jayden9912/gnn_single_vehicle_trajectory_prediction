import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class stp(nn.Module):
    def __init__(self,options):
        super(stp, self).__init__()
        self.opt = options

        ##embedding for the coordinates
        self.linear_embedding = nn.Linear(2,self.opt.emb_output_features) #input_features, output_features
        
        ##feature encoder
        # self.GRU = nn.GRU(input_size= self.opt.emb_output_features, hidden_size= self.opt.enc_output_features, num_layers = 1, batch_first = True)
        # self.ego_GRU = nn.GRU(input_size= self.opt.emb_output_features, hidden_size= self.opt.ego_enc_output_features, num_layers = 1, batch_first = True)
        self.LSTM = nn.LSTM(input_size= self.opt.emb_output_features, hidden_size= self.opt.enc_output_features, num_layers = 1, batch_first = True)
        # self.ego_LSTM = nn.LSTM(input_size= self.opt.emb_output_features, hidden_size= self.opt.ego_enc_output_features, num_layers = 1, batch_first = True)
        
        self.encoder_embedding = nn.Linear(self.opt.enc_output_features, self.opt.enc_emb_output_features)
        # self.ego_encoder_embedding = nn.Linear(self.opt.ego_enc_output_features,128)
        ##graph NN convolution
        self.conv1 = GATConv(in_channels= self.opt.enc_emb_output_features,out_channels= self.opt.conv_output_features ,heads= self.opt.conv_head,add_self_loops = False)
        self.conv2 = GATConv(in_channels= self.opt.conv_head*self.opt.conv_output_features,out_channels= self.opt.conv_output_features,heads= self.opt.conv_head,add_self_loops = False)
        
        ##fully connected layer
        self.conv_embedding = nn.Linear(self.opt.conv_head*self.opt.conv_output_features,self.opt.enc_output_features)
        self.fcl1 = nn.Linear(64,2)
        # self.fcl2 = nn.Linear(32,2)
        
        ##decoder
        self.decoder = nn.LSTM(input_size= self.opt.dec_input_features, hidden_size= 64 , num_layers= self.opt.dec_num_layers, batch_first = True)

        #activation
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, input, device):
        """pass a batch to the network and compute losses
        """
        self.device = device
        input.to(self.device)
        x = input.x #shape:(N,30,2)
        edge_index = input.edge_index #(2,X)
        target_index = torch.tensor([(input.batch==i).nonzero()[0] for i in range(input.num_graphs)]) #input.num_graphs = batch_size
        # final_x = torch.empty([input.num_graphs,10,128], dtype = torch.float32).to(self.device)
        #emb
        encoder_x = self.linear_embedding(x) #from (N,30,2) to (N,30,16)
        encoder_x = self.relu(encoder_x)
        # _,hn = self.GRU(x)
        # x = self.relu(hn)
        _,(hn,cn) = self.LSTM(encoder_x)
        encoder_x = self.relu(cn) #(1,N,hidden_size), hidden_size = 32
        encoder_x = encoder_x.squeeze(0)#(N,32)
        encoder_x = self.encoder_embedding(encoder_x) #(N,32)
        encoder_x = self.relu(encoder_x)
        conv_x = self.conv1(encoder_x,edge_index) #(N,128)
        conv_x = self.relu(conv_x)
        conv_x = self.conv2(conv_x,edge_index) #(N,128)
        conv_x = self.relu(conv_x)
        conv_x = self.conv_embedding(conv_x) #(N,32)
        conv_x = self.relu(conv_x)

        #ego_emb
        # ego_x = input.x[target_index] #(N,30,2)
        # ego_x = self.linear_embedding(ego_x) #(N,30,16)
        # ego_x = self.relu(ego_x)
        # _, ego_x = self.ego_GRU(ego_x)
        # _, (h_ego,c_ego) = self.ego_LSTM(ego_x)
        # ego_x = self.relu(c_ego).squeeze(0) #(N,64)
        # ego_x = self.ego_encoder_embedding(ego_x) #(N,128)
        #interchange input.num_graphs to 1
        # for i in range(input.num_graphs):
        #     index = [torch.flatten((input.batch==i).nonzero())][0].tolist()
        #     tmp_x = x[index[0]:index[-1]+1]
        #     tmp_ego_x = ego_x[i].unsqueeze(0)
        #     #combined_x with N = 50
        #     tmp_x = torch.cat((tmp_ego_x,tmp_x), axis = 0)
        #     #calculating the number of repetition
        #     num_of_repetition = 50//tmp_x.shape[0]
        #     #calculating the number of appending needed to reach 50
        #     num_of_append = 50%tmp_x.shape[0]
        #     tmp_x = tmp_x.repeat(num_of_repetition,1)
        #     for j in range(0,num_of_append):
        #         tmp_x = torch.cat((tmp_ego_x,tmp_x), axis=0) #(50,64)
        #     tmp_x = tmp_x.unsqueeze(0) #(1,50,64)
        #     final_x[i,:,:] = tmp_x
        #extract the ego vehicle from the input 
        fwd_ego_x = encoder_x[target_index] #(N,32)

        fwd_conv_x = conv_x[target_index]

        decoder_x = torch.cat((fwd_ego_x, fwd_conv_x), 1)

        decoder_x = decoder_x.unsqueeze(1) #(N,1,64)

        final_x = decoder_x.repeat(1,50,1)

        #decoder
        pred, (hn, cn) = self.decoder(final_x)
        pred = self.fcl1(pred)
        pred = self.relu(pred)
        # pred = self.fcl2(pred)
        # pred = self.relu(pred)
        return pred
