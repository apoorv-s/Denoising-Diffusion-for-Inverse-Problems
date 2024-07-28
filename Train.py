import argparse
from Configs.Configs import BraninMLPConfig, PoseMLPConfig, PoseTransformersConfig, BraninTransformerConfig
from Core.DDPM import DDPM

if __name__ == "__main__":
    print("Initiating Training")
    
    parser = argparse.ArgumentParser(description="DDPM parser")
    # parser.add_argument("--train", default=False, action="store_true")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--run_number", required=True)
    args = parser.parse_args()
    
    if args.model_name == 'branin_mlp':
        config = BraninMLPConfig()
    elif args.model_name == 'branin_transformer':
        config = BraninTransformerConfig()
    elif args.model_name == 'pose_mlp':
        config = PoseMLPConfig()
    elif args.model_name == 'pose_transformer':
        config = PoseTransformersConfig()
    else:
        raise NameError("Unknown model:", args.model_name)
        
    DDPM_obj = DDPM(args.model_name)
    DDPM_obj.train(config, args.run_number)
        
        
    
    

