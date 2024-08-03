import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
from tensorboardX import SummaryWriter
from tqdm import trange

from Core.Dataset import BraninDataset, PoseModelDataset
from Core.Branin import BraninMLPDiffusion, BraninTranformerDiffusion
from Core.PoseModel import PoseMLPDiffusion, PoseTranformerDiffusion

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from Core.Utils import generate_discription, diffusion_noise_schedule

class DDPM():
    def __init__(self, model_name:str) -> None:
        if model_name == 'branin_mlp':
            self.problem = 'branin'
            self.model_class = BraninMLPDiffusion
            self.dataset_class = BraninDataset
        elif model_name == 'branin_transformer':
            self.problem = 'branin'
            self.model_class = BraninTranformerDiffusion
            self.dataset_class = BraninDataset
        elif model_name == "pose_mlp":
            self.problem = 'pose'
            self.model_class = PoseMLPDiffusion
            self.dataset_class = PoseModelDataset
        elif model_name == "pose_transformer":
            self.problem = 'pose'
            self.model_class = PoseTranformerDiffusion
            self.dataset_class = PoseModelDataset
        else:
            raise NameError("Unknown model class:", model_name)
        
        self.pretrained_model = None
        
    def train(self, config, run_number):
        description = generate_discription(config, run_number)
        print("****** Training Initiated ******")
        print(description)
        print("training on", device)
        train_dir=config.save_dir+"/run"+str(run_number)+"/train"
        os.makedirs(train_dir, exist_ok=False)
        
        model = self.model_class(config).to(device)
        dataset=self.dataset_class(config)
        dataloader=DataLoader(dataset, batch_size=config.batch_size, num_workers=config.n_workers, pin_memory=True)
        optimizer=torch.optim.Adam(model.parameters(), lr=config.lr)
        loss_function = nn.MSELoss()
        
        [alpha, beta, sqrt_alpha_bar,
         sqrt_om_alpha_bar]=diffusion_noise_schedule(config.n_time, config.beta_range)
        
        writer=SummaryWriter(train_dir+'/tboard')
        writer.add_text('description', description)
        
        counter=0
        
        mode="train"
        model.train()
        pbar = trange(config.n_epochs)
        for ep in pbar:
            running_loss=0.0
            epoch_loss = 0.0
            for ind, batch_data in enumerate(dataloader):
                optimizer.zero_grad()
                data_x = batch_data['x'].view(-1, batch_data['x'].shape[-1]) # 1024x2
                batch_size=data_x.shape[0] # 1024
                inp_noise=torch.randn_like(data_x)
                inp_t_ind=torch.randint(0, config.n_time, (batch_size, 1))
                inp_x = (data_x*sqrt_alpha_bar[inp_t_ind] + inp_noise*sqrt_om_alpha_bar[inp_t_ind]).to(device)
                inp_y = batch_data['y'].view(-1, batch_data['y'].shape[-1]).to(device) #1024x1
                inp_t = (inp_t_ind/config.n_time).to(device) #1024x1
                pred_noise = model(inp_x, inp_y, inp_t, mode).to(device)
                loss = loss_function(pred_noise, inp_noise.to(device)).to(device)
                loss.backward()
                optimizer.step()
                running_loss=running_loss+loss.item()
                # print(ind, running_loss)
                if (ind + 1)%10==0:
                    writer.add_scalar('running_train_loss', running_loss, counter)
                    counter=counter+1
                    epoch_loss += running_loss
                    running_loss=0.0
                    writer.flush()
                
            pbar.set_description(f"epoch stats: epoch = {ep}, epoch_loss = {epoch_loss:.4f}")
            writer.add_scalar('epoch_loss', epoch_loss, global_step=ep)
                    
            if (ep + 1)%config.n_epochs_bw_saves==0:
                torch.save({"state_dict":model.state_dict(),
                            "description":description, 
                            "config":config}, train_dir + f"/int_model_{ep}.pt")
        
        torch.save({"state_dict":model.state_dict(),
                    "description":description,
                    "config":config}, train_dir + f"/final_model_{ep}.pt")
        
        print("Training complete")
    
    def sample(self, target_y, n_test_points, cond_weight):
        assert self.pretrained_model, "Load pretrained model"
        mode = "eval"
        self.pretrained_model.to(device)
        self.pretrained_model.eval()
        
        [alpha, beta, sqrt_alpha_bar,
         sqrt_om_alpha_bar]=diffusion_noise_schedule(self.pretrained_config.n_time,
                                                     self.pretrained_config.beta_range)
        om_alpha = 1 - alpha
        
        # Move tensors to device
        alpha = alpha.to(device)
        beta = beta.to(device)
        sqrt_alpha_bar = sqrt_alpha_bar.to(device)
        sqrt_om_alpha_bar = sqrt_om_alpha_bar.to(device)
        om_alpha = om_alpha.to(device)
        
        x_mat = torch.zeros((self.pretrained_config.n_time, n_test_points, self.pretrained_config.inp_dim), device=device)
        x_mat[-1, :, :] = torch.randn((n_test_points, self.pretrained_config.inp_dim), device=device)
        y_inp = torch.ones((n_test_points, self.pretrained_config.out_dim), device=device) * target_y

        with torch.no_grad():
            for it in trange(self.pretrained_config.n_time - 1, 0, -1):
                t_inp = torch.ones((n_test_points, 1), device=device) * it / self.pretrained_config.n_time
                cond_pred_noise = self.pretrained_model(x_mat[it, :, :], y_inp, t_inp, mode)
                uncond_pred_noise = self.pretrained_model(x_mat[it, :, :], None, t_inp, mode)
                pred_noise = (1 + cond_weight) * cond_pred_noise + cond_weight * uncond_pred_noise
                z = torch.randn_like(x_mat[it, :, :], device=device) if it > 1 else torch.zeros_like(x_mat[it, :, :], device=device)
                x_mat[it - 1, :, :] = (x_mat[it, :, :] - (om_alpha[it] * pred_noise / sqrt_om_alpha_bar[it])) / (alpha[it].sqrt()) + beta[it] * z
                del cond_pred_noise, uncond_pred_noise, pred_noise, z, t_inp  # Clear memory of unnecessary variables
                torch.cuda.empty_cache()
        
        return x_mat
        
    def load_pretrained_model(self, model_path):
        saved_obj = torch.load(model_path, map_location=device)
        self.pretrained_description = saved_obj["description"]
        self.pretrained_config = saved_obj["config"]
        self.pretrained_model = self.model_class(saved_obj["config"])
        print(self.pretrained_model.load_state_dict(saved_obj["state_dict"]))
        self.pretrained_model.eval()


