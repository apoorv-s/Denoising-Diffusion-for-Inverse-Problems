import argparse
from Configs.Configs import BraninConfig
from Core.Branin import BraninDDPM

if __name__ == "__main__":
    task_name = "Branin"
    
    parser = argparse.ArgumentParser(description="DDPM parser")
    parser.add_argument("--train", default=False, action="store_true")
    parser.add_argument("--run_number", required=True)
    args = parser.parse_args()
    
    if args.train:
        config = BraninConfig()
        branin_obj=BraninDDPM()
        branin_obj.train(config, args.run_number)
        
        
    
    

