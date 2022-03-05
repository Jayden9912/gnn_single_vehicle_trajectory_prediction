import torch
import pickle
from torch_geometric.data import Dataset

def pickle_loader(path):
  with open(path,'rb') as tmp_file:
    data = pickle.load(tmp_file)
  return data

def torch_loader(path):
  data = torch.load(path)
  return data

class trajectory_data(Dataset):
  def __init__(self, file_path):
    self.file_path =file_path

  def __len__(self):
    return len(self.file_path)
  
  def __getitem__(self, idx):
    load_path = self.file_path[idx]
    # pyg_data = pickle_loader(load_path)
    ##################################
    pyg_data = torch_loader(load_path)
    pyg_data.x = pyg_data.node_feature.type(torch.FloatTensor)
    pyg_data.y = pyg_data.y.type(torch.FloatTensor)
    ##################################
    # x = pyg_data.x
    # y = pyg_data.y
    # edge_index = pyg_data.edge_index
    return pyg_data