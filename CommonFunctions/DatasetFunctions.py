import numpy as np
import torch
from torch.utils.data import Dataset


class BraninDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.a = 1
        self.b = 5.1/(4*(np.pi**2))
        self.c = 5/np.pi
        self.r = 6
        self.s = 10
        self.t = 1/(8*np.pi)
        
        self.range = [[-5, 10], [0, 15]]
        
    def func(self, x):
        assert np.all(self.range[0][0] <= x[..., 0]) and np.all(x[..., 0] <= self.range[0][1])
        assert np.all(self.range[0][1] <= x[..., 1]) and np.all(x[..., 1] <= self.range[1][1])
        
        y = self.a * (x[..., 1] - self.b*(x[..., 0]**2) + self.c*x[..., 0] - self.r)**2 + self.s*(1 - self.t)*torch.cos(x[..., 0]) + self.s
        return y
    
    def __len__(self):
        return 100000
    
    def __getitem__(self, index):
        x = torch.rand(2)
        x[0] = self.range[0][0] + x[0]*(self.range[0][1] - self.range[0][0])
        x[1] = self.range[1][0] + x[1]*(self.range[1][1] - self.range[1][0])
        y = self.func(x)
        return {'x':x, 'y':y}
    
    
        
        
if __name__ == "__main__":
    import IPython
    IPython.embed()