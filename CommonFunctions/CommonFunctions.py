import os
from CommonFunctions.DatasetFunctions import BraninDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPM(nn.Module):
    pass

def TrainBranin(config):
    os.mkdir(config.save_dir)
    dataset = BraninDataset(config)
    dataloader = DataLoader(dataset, batch_size=config.batchSize, shuffle=True)
    
    ddpm = DDPM(config)

if __name__ == "__main__":
    import IPython
    IPython.embed()