import argparse
from Configs.Configs import BraninMLPConfig, PoseMLPConfig, PoseTransformersConfig, BraninTransformerConfig
from Core.DDPM import DDPM
import scipy.io

if __name__ == "__main__":
    print("Initiating Evaluation")
    
    parser = argparse.ArgumentParser(description="DDPM parser")
    # parser.add_argument("--train", default=False, action="store_true")
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--model_number", required=True, type=int)
    parser.add_argument("--target_y", required=True, type=float)
    parser.add_argument("--n_test_points", required=True, type=int)
    parser.add_argument("--cond_weight", required=True, type=float)
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
        
    model_dir = config.save_dir+"/run"+str(args.model_number)+"/train" + f"/final_model_{499}.pt"
    
    DDPM_obj = DDPM(args.model_name)
    DDPM_obj.load_pretrained_model(model_dir)
    result = DDPM_obj.sample(args.target_y, args.n_test_points, args.cond_weight)
    
    x_T = result[0, :, :]
    x_T = x_T.cpu().detach().numpy()
    
    scipy.io.savemat(f"Data/BraninSamples/branin_sample_{args.n_test_points}_target_{args.target_y}.mat",
                     dict(x_T = x_T, model_name=args.model_name, model_dir = model_dir,
                          target_y = args.target_y, n_test_points = args.n_test_points,
                          cond_weight = args.cond_weight))
    
    print("Sampling complete")
    
    