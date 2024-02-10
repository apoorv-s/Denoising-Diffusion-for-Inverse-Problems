import numpy as np
import torch
from torch.utils.data import Dataset


class BraninDataset(Dataset):
    def __init__(self, filename):
        super().__init__()
        data = torch.load(filename)
        self.xData = data['xData']
        self.yData = data['yData']

    def __len__(self):
        return len(self.xData)
    
    def __getitem__(self, index):
        return {'x':self.xData[index], 'y':self.yData[index]}
    
class GenerateBraninDataset():
    def __init__(self) -> None:
        self.inputDim = 2
        self.outputDim = 1
        
        self.a = 1
        self.b = 5.1/(4*(np.pi**2))
        self.c = 5/np.pi
        self.r = 6
        self.s = 10
        self.t = 1/(8*np.pi)
        
        self.range = torch.tensor([[-5, 10], [0, 15]])
        
    def BraninFunction(self, x):        
        y = self.a * (x[..., 1] - self.b*(x[..., 0]**2) + self.c*x[..., 0] - self.r)**2 + self.s*(1 - self.t)*torch.cos(x[..., 0]) + self.s
        return y
    
    def GenerateData(self, nData, filename):
        xData = torch.rand((nData, self.inputDim))
        xData[:, 0] = self.range[0, 0] + (self.range[0, 1] - self.range[0, 0])*xData[:, 0]
        xData[:, 1] = self.range[1, 0] + (self.range[1, 1] - self.range[1, 0])*xData[:, 1]
        
        yData = torch.zeros((nData, self.outputDim))
        
        for iData in range(nData):
            yData[iData, :] = self.BraninFunction(xData[iData, :])
            
            
        dataFilename =  './Datasets/Branin/'+filename+'_data.pth'
        torch.save({'xData':xData, 'yData':yData}, dataFilename)
        return dataFilename
    
if __name__ == "__main__":
    import IPython
    IPython.embed()