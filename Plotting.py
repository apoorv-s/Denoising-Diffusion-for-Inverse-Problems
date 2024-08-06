from Configs.Configs import BraninMLPConfig, PoseTransformersConfig
from Core.Dataset import BraninDataset, PoseModelDataset
from Core.DDPM import DDPM

import matplotlib.pyplot as plt
import scipy.io


## Branin

config=BraninMLPConfig()
dataset=BraninDataset(config)

n_eval_pts=500
eval_data = dataset.get_eval_data(n_eval_pts)

x=eval_data['x']
y=eval_data['y']

x1_grid_n=x[:, 0].reshape(n_eval_pts, n_eval_pts)
x2_grid_n=x[:, 1].reshape(n_eval_pts, n_eval_pts)
y_grid=y.reshape(n_eval_pts, n_eval_pts)

y10_data = scipy.io.loadmat("Data/BraninSamples/branin_sample_10000_target_10.0.mat")["x_T"]
y50_data = scipy.io.loadmat("Data/BraninSamples/branin_sample_5000_target_50.0.mat")["x_T"]
y100_data = scipy.io.loadmat("Data/BraninSamples/branin_sample_5000_target_100.0.mat")["x_T"]
y200_data = scipy.io.loadmat("Data/BraninSamples/branin_sample_5000_target_200.0.mat")["x_T"]

contour = plt.contourf(x1_grid_n, x2_grid_n, y_grid, 100)
plt.scatter(y10_data[:, 0], y10_data[:, 1], label = "y = 10")
plt.scatter(y50_data[:, 0], y50_data[:, 1], label = "y = 50")
plt.scatter(y100_data[:, 0], y100_data[:, 1], label = "y = 100")
plt.scatter(y200_data[:, 0], y200_data[:, 1], label = "y = 200")
plt.colorbar(contour)
plt.legend(fontsize = 10)
plt.title("Branin Function and points sampled through DDPM", fontsize=14)
plt.savefig("BraninResults.png", dpi=600)
plt.show()

## Pose Model

config = PoseTransformersConfig()
dataset = PoseModelDataset(config)

run_number = 1
model_name = 'pose_transformer'

model_dir = config.save_dir+"/run"+str(run_number)+"/train" + f"/final_model_{499}.pt"
DDPM_obj = DDPM(model_name)
DDPM_obj.load_pretrained_model(model_dir)

test_sample = dataset[0]
img = dataset.viz(test_sample['x'], test_sample)

target_y = test_sample['y'][0]
result = DDPM_obj.sample(target_y, 8, 0.01) # n_sample 8 to ensure that the dataset shapes are matched

x_T = result[0, :, :]
x_T = x_T.cpu().detach()

res_img = dataset.viz(x_T, test_sample)

fig, axs = plt.subplots(1, 4, figsize=(24, 6))

# Display the reference image
axs[0].imshow(img[0])
axs[0].axis('off')  # Hide axes
axs[0].set_title("Reference Image")

# Display the sampled images through DDPM
axs[1].imshow(res_img[-1])
axs[1].axis('off')  # Hide axes
axs[1].set_title("Sampled through DDPM")

axs[2].imshow(res_img[-2])
axs[2].axis('off')  # Hide axes
axs[2].set_title("Sampled through DDPM")

axs[3].imshow(res_img[-3])
axs[3].axis('off')  # Hide axes
axs[3].set_title("Sampled through DDPM")

# Adjust the layout to be more compact
plt.subplots_adjust(wspace=0.1, hspace=0)  # Adjust horizontal and vertical space
plt.show()