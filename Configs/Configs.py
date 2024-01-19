class BraninConfig:
    # data
    inputDim = 2
    outputDim = 1
    
    saveDir = "./SavedResults/Branin"
    
    
    weightNormIndicator = False
    actLayerName = "batch"
    actFnName = "relu"
    
    batchSize = 64
    numWorkers = 4
    learningRate = 1e-3
    weightDecay = 1e-3
    
    saveIntermediateModels = True
    nEpochs = 200
    nHiddenLayers = 4
    nHiddenDims = 64
    
    conditioningWeights = [0, 0.5, 2]
    
    
    