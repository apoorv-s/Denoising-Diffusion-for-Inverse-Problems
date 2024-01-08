import argparse
from Configs.BraninConfig import BraninConfig
import CommonFunctions.CommonFunctions as cf

if __name__ == "__main__":
    taskName = "Branin"
    
    parser = argparse.ArgumentParser(description="DDPM Branin Function")
    parser.add_argument("--dir", default="./SavedResults/taskName/", type=str)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--num_layers", default=4, type=int)
    parser.add_argument("--n_epochs", default=100, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--validate", default=False, action="store_true")
    args = parser.parse_args()
    
    config = BraninConfig()
    
    cf.TrainBranin(config)
    
    

