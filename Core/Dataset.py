import numpy as np
import torch
from torch.utils.data import Dataset


class BraninDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.input_dim=config.inp_dim
        self.output_dim=config.out_dim
        
        self.a = 1
        self.b = 5.1/(4*(np.pi**2))
        self.c = 5/np.pi
        self.r = 6
        self.s = 10
        self.t = 1/(8*np.pi)
        
        self.range = [[-5, 10], [0, 15]]
        
    def func(self, x):
        y = (self.a * (x[..., 1] - self.b*(x[..., 0]**2) + self.c*x[..., 0] - self.r)**2 + self.s*(1 - self.t)*torch.cos(x[..., 0]) + self.s)
        return y.unsqueeze(dim=0)
    
    def __len__(self):
        return 100000
    
    def __getitem__(self, index):
        x = torch.rand(2)
        x[0] = self.range[0][0] + x[0]*(self.range[0][1] - self.range[0][0])
        x[1] = self.range[1][0] + x[1]*(self.range[1][1] - self.range[1][0])
        return {'x':x, 'y':self.func(x)}
    
    def GetEvalData(self, n_eval_pts):
        x1=torch.linspace(self.range[0][0], self.range[0][1], n_eval_pts)
        x2=torch.linspace(self.range[1][0], self.range[1][1], n_eval_pts)
        x1_grid,x2_grid=torch.meshgrid(x1,x2,indexing='ij')
        x=torch.stack([x1_grid.flatten(), x2_grid.flatten()]).T
        return {'x':x, 'y':self.func(x)}
        
if __name__ == "__main__":
    import IPython
    IPython.embed()