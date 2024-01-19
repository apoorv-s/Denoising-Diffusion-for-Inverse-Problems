import os
from Core.Dataset import BraninDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.weight_norm import weight_norm

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CommonNNFunctions(nn.Module):
    def __init__(self) -> None:
        super.__init__(CommonNNFunctions, self)
        self.actDict = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid,
                        "tanh": nn.Tanh, "prelu": nn.PReLU,
                        "selu": nn.SELU, "gelu": nn.GELU}
        self.actLayerDict = {"none": nn.Identity, "batch": nn.BatchNorm1d,
                                       "layer": nn.LayerNorm, "instance": nn.InstanceNorm1d,}

    def GetActFn(self, actName):
        return self.actDict[actName]

    def GetActLayer(self, actLayerName):
        return self.actLayerDict[actLayerName]
    
    def LinearLayer(self, inputDim, outputDim, weightNormIndicator):
        linLayer = nn.Linear(inputDim, outputDim)
        if weightNormIndicator:
            linLayer = weight_norm(linLayer)
        return linLayer
    
    def ResidualBlock(self):
        pass
        #TBD
        

class DDPM(nn.Module):
    def __init__(self, config):
        super.__init__(DDPM, self)
        self.config = config
        self.nnf = CommonNNFunctions()
        
        tempSequence = [self.nnf.LinearLayer(config.inputDim, config.outputDim,
                                          config.weightNormIndicator),
                     self.nnf.GetActLayer(config.activationLayerName),
                     self.nnf.GetActFn(config.actFnName)()]
        self.inputBlock = nn.Sequential(*tempSequence)
        
        tempSequence # MyResBlockImplementation
        
        
    
    def Sample(self, yInput, condWeight):
        pass

class BraninDDPM():
    def __init__(self, config) -> None:
        self.config = config
    
    def TrainBranin(self):
        print("Training Branin DDPM")
        os.makedirs(self.config.trainDir, exist_ok=False)
        dataset = BraninDataset(self.config)
        dataloader = DataLoader(dataset, batch_size=self.config.batchSize, shuffle=True,
                                num_workers=self.config.numWorkers)
        
        ddpm = DDPM(self.config).to(device)
        optimizer = torch.optim.Adam(ddpm.parameters(), lr=self.config.learningRate,
                                    weight_decay= self.config.weightDecay)
        lrScheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0,
                                                        end_factor=0.01, total_iters=self.config.nEpochs)
        
        for ep in range(self.config.nEpochs):
            # tempLr = optimizer.param_groups[0]["lr"]
            # print(f"epoch: {ep}, learning rate: {tempLr}")
            
            ddpm.train() # Activating training mode
            progressBar = tqdm(dataloader, leave=False)
            lossEma = None
            for trainData in progressBar:
                optimizer.zero_grad()
                loss = ddpm(trainData["x"], trainData["y"])
                loss.backward()
                optimizer.step()
                
                if lossEma is None:
                    lossEma = loss.item()
                else:
                    lossEma = 0.95*lossEma + 0.05*loss.item()
                progressBar.set_description(f"EMA loss:{lossEma:.4f}")
            lrScheduler.step()
            
            ddpm.eval() # Activating evaluation mode
            with torch.no_grad():
                self.EvaluateBranin(ddpm, dataset)
                
            if self.config.saveIntermediateModels and (ep + 1)%10 == 0:
                torch.save(ddpm.state_dict(), self.config.trainDir + f"/model_{ep}.pt")        
        
    def EvaluateBranin(self, model, dataset):
        evalData = dataset.GetEvalData()
        for weight in self.config.conditioningWeights:
            xInverted = model.Sample()
            #TBD
                
    def ValidateBranin(self, config):
        assert config.validate == True and config.checkPoint is not None
        print("Validating Braning model from", config.checkPoint)
        # TBD
        

if __name__ == "__main__":
    import IPython
    IPython.embed()