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
    parser.add_argument("--target_y", required=False, type=float)
    parser.add_argument("--n_test_points", required=False, type=int)
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
    
    
    if args.model_name == 'branin_mlp' or args.model_name == 'branin_transformer':
        ## Save the matrices to plot later (BraninTests.ipynb)
        result = DDPM_obj.sample(args.target_y, args.n_test_points, args.cond_weight)
        
        x_T = result[0, :, :]
        x_T = x_T.cpu().detach().numpy()
        
        scipy.io.savemat(f"Data/BraninSamples/branin_sample_{args.n_test_points}_target_{args.target_y}.mat",
                        dict(x_T = x_T, model_name=args.model_name, model_dir = model_dir,
                            target_y = args.target_y, n_test_points = args.n_test_points,
                            cond_weight = args.cond_weight))
    else:
        print("Make sure to update sequence in dataset initialization.")
        from Core.Dataset import PoseModelDataset
        import matplotlib.pyplot as plt
        dataset = PoseModelDataset(config)
        test_sample = dataset[0]
        img = dataset.viz(test_sample['x'], test_sample)

        target_y = test_sample['y'][0]
        result = DDPM_obj.sample(target_y, 8, 0.01) # n_sample 8 to ensure that the dataset shapes are matched
        
        x_T = result[0, :, :]
        x_T = x_T.cpu().detach()
        
        res_img = dataset.viz(x_T, test_sample)

        fig, axs = plt.subplots(1, 4, figsize=(24, 6))  # 1 row, 4 columns, larger figure size

        # Display the reference image
        axs[0].imshow(img[0])
        axs[0].axis('off')  # Hide axes
        axs[0].set_title("Reference Image")

        # Display the sampled images through DDPM
        axs[1].imshow(res_img[0])
        axs[1].axis('off')  # Hide axes
        axs[1].set_title("Sampled through DDPM")

        axs[2].imshow(res_img[1])
        axs[2].axis('off')  # Hide axes
        axs[2].set_title("Sampled through DDPM")

        axs[3].imshow(res_img[2])
        axs[3].axis('off')  # Hide axes
        axs[3].set_title("Sampled through DDPM")

        # Adjust the layout to be more compact
        plt.subplots_adjust(wspace=0.1, hspace=0)  # Adjust horizontal and vertical space
        plt.savefig("PoseModelSampling.png")
        plt.show()
    
    print("Sampling complete")
    
    