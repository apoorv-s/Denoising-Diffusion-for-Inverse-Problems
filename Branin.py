import argparse
from Configs.Configs import BraninConfig
from Core.Branin import BraninDDPM

if __name__ == "__main__":
    task_name = "Branin"
    
    parser = argparse.ArgumentParser(description="DDPM parser")
    parser.add_argument("--validate", default=False, action="store_true")
    parser.add_argument("--runNumber", required=True)
    args = parser.parse_args()
    
    config = BraninConfig()
    config.run_number = args.run_number
    
    branin_obj=BraninDDPM(config)
    branin_obj.train()
    
    print("Training Complete")
    
    

