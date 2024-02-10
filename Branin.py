import argparse
from Configs.Configs import BraninConfig
import Core.Utils as cf

if __name__ == "__main__":
    taskName = "Branin"
    
    parser = argparse.ArgumentParser(description="DDPM parser")
    parser.add_argument("--validate", default=False, action="store_true")
    parser.add_argument("--runNumber", required=True)
    args = parser.parse_args()
    
    config = BraninConfig()
    config.runNumber = args.runNumber
    config.trainDir = config.saveDir + f"/run{config.runNumber}/train/"
    
    cf.TrainBranin(config)
    
    

