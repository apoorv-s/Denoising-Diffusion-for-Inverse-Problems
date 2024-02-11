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
        super().__init__()
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
     
class DiffusionModel():
    def __init__(self) -> None:
        super().__init__()
    
    def GetLinearNoiseSchedule(self, beta1, beta2, nT):
        assert 0 < beta1 < beta2 < 1
        beta = torch.linspace(beta1, beta2, nT + 1)
        alpha = 1 - beta
        alphaBar = torch.cumsum(alpha.log(), dim=0).exp()
        
        sqrtAlpha = alpha.sqrt()
        # sqrtInvAlphaBar = 1/(alphaBar.sqrt())
        sqrtOneMinusAlphaBar = (1 - alphaBar).sqrt()
        # omAlphaOverSqrtOmAb = (1 - alpha)/sqrtOneMinusAlphaBar
        
        return dict(sqrtAlpha=sqrtAlpha, sqrtOneMinusAlphaBar=sqrtOneMinusAlphaBar)
    
    def GetCosineNoiseSchedule(self):
        pass

class ResidualBlockWithTimeScaling(nn.Module):
    def __init__(self, nnf, inpDim, nHidDims, actFnName, outDim, timeDim,
                 actLyrName, wtNormInd) -> None:
        super().__init__()
        
        self.block1 = nnf.LinearLayer(inpDim, nHidDims, wtNormInd)
        self.actLyr1 = nnf.GetActLayer(actLyrName)(nHidDims)
        self.actFn1 = nnf.GetActFn(actFnName)()
        
        self.block2 = nnf.LinearLayer(nHidDims, outDim, wtNormInd)
        self.actLyr2 = nnf.GetActLayer(actLyrName)(outDim)
        self.actFn2 = nnf.GetActFn(actFnName)()
        
        self.tBlock = nn.Sequential(nnf.GetActFn(actFnName)(),
                                    nn.Linear(timeDim, 2*nHidDims))
            
    def forward(self, xInp, tInp=None): # x: (B, 2*hidDim), t: (B, hidDim)
        x = self.block1(xInp)
        x = self.actLyr1(x)
        if tInp is not None:
            tScaleNShift = self.tBlock(tInp)
            tScale, tShift = tScaleNShift.chunk(2, dim=1)
            x = x*(1 + tScale) + tShift
            
        x = self.actFn1(x)
        x = self.block2(x)
        x = self.actLyr2(x)
        x = self.actFn2(x + xInp)
        return x

class NNModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.nnf = CommonNNFunctions()
        self.nTyHiddenLayers = 2
        self.nNpHiddenLayers = config.nHiddenLayers
        
        self.tEmbed = nn.Linear(1, config.nHiddenDims)
        self.yEmbed = nn.Linear(config.outputDim, config.nHiddenDims)
        
        # ty Embedding Residual Block
        tyInpLayers = [self.nnf.LinearLayer(2*config.nHiddenDims, config.nHiddenDims,
                                          config.weightNormIndicator),
                       self.nnf.GetActLayer(config.actLayerName)(config.nHiddenDims),
                       self.nnf.GetActFn(config.actFnName)()]
        self.tyInputBlock = nn.Sequential(*tyInpLayers)
        self.tyResidualBlocks = [ResidualBlockWithTimeScaling(nnf = self.nnf, inpDim = config.nHiddenDims,
                                                              nHidDims = config.nHiddenDims, actFnName = config.actFnName,
                                                              outDim = config.nHiddenDims,
                                                              timeDim = config.nHiddenDims,
                                                              actLyrName = config.actLayerName,
                                                              wtNormInd = config.weightNormIndicator) 
                                 for _ in range(self.nTyHiddenLayers)]
        self.tyOutLayers = self.nnf.LinearLayer(config.nHiddenDims,
                                                config.nHiddenDims,
                                                config.weightNormIndicator)
        
        
        # self.noisePrediction 
        npInpLayers = [self.nnf.LinearLayer(config.inputDim, config.nHiddenDims, 
                                                  config.weightNormIndicator),
                            self.nnf.GetActLayer(config.actLayerName)(config.nHiddenDims),
                            self.nnf.GetActFn(config.actFnName)()]
        self.npInputBlock = nn.Sequential(*npInpLayers)
        self.npResidualBlocks = [ResidualBlockWithTimeScaling(nnf = self.nnf, inpDim = config.nHiddenDims,
                                                              nHidDims = config.nHiddenDims, actFnName = config.actFnName,
                                                              outDim = config.nHiddenDims,
                                                              timeDim = config.nHiddenDims,
                                                              actLyrName = config.actLayerName,
                                                              wtNormInd = config.weightNormIndicator) 
                                 for _ in range(config.nHiddenLayers)]
        self.npOutLayers = self.nnf.LinearLayer(config.nHiddenDims,
                                                config.inputDim,
                                                config.weightNormIndicator)
    
    def forward(self, xInput, yInput, tSteps, yBlock):
        t = self.tEmbed(tSteps) # (B, hidDim)
        y = self.yEmbed(yInput) # (B, hidDim)
        y = y*(1 - yBlock) ## Classifier free guidance - classification embedding set to zero
        
        ty = torch.cat((t, y), dim=-1) # (B, 2*hidDim)
        ty = self.tyInputBlock(ty) # (B, hidDim)
        for iResblock in range(self.nTyHiddenLayers):
            ty = self.tyResidualBlocks[iResblock](ty, t) # (B, hidDim)
        ty = self.tyOutLayers(ty) # (B, hidDim)
        
        x = self.npInputBlock(xInput) # (B, hidDim)
        for iResblock in range(self.nNpHiddenLayers):
            x = self.npResidualBlocks[iResblock](x, ty) # (B, hidDim)
        x = self.npOutLayers(x) # (B, inputDim)
        return x

class DDPM(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.config = config
        self.nT = config.nT
        self.dropProb = config.dropProb
        self.noiseSched = DiffusionModel().GetLinearNoiseSchedule(*config.beta, config.nT)
        self.nnModel = NNModel(config)
        
    def forward(self, x, y):
        batchSize = x.shape[0]
        sampledTimes = torch.randint(1, self.nT + 1, (batchSize,)) # (B,)
        stdNormalNoise = torch.randn_like(x) # (B, xDim)
        
        # Forward Diffusion Model: x_t
        xAtSampledTimes = (self.noiseSched["sqrtAlpha"][sampledTimes][:, None]*x +
                           self.noiseSched["sqrtOneMinusAlphaBar"][sampledTimes][:, None]*stdNormalNoise)
        
        # For classifier free guidance
        yBlock = torch.bernoulli(torch.zeros(batchSize) + self.dropProb)[:, None]
        
        tSteps = (sampledTimes/self.nT)[:, None]
        loss = (stdNormalNoise - self.nnModel(xAtSampledTimes, y, tSteps, yBlock)).square().mean()
        return loss
    
    @torch.no_grad()
    def Sample(self, yInput, condWeight):
        batchSize = yInput.shape

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