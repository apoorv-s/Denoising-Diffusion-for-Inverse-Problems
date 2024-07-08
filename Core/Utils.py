import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
from tensorboardX import SummaryWriter
from tqdm import trange

from Core.Dataset import BraninDataset, PoseModelDataset
from Configs.Configs import BraninConfig, PoseMLPConfig, PoseTransformersConfig
from Core.Transformers import AttentionBlock, CrossAttentionBlock

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def diffusion_noise_schedule(n_steps, beta_range):
    beta=torch.linspace(beta_range[0], beta_range[1], n_steps)
    alpha=1-beta
    alpha_bar=torch.cumprod(alpha, dim=0)
    om_alpha_bar=1-alpha_bar
    return [alpha, beta, alpha_bar.sqrt(), om_alpha_bar.sqrt()]

def get_act_fn(act_fn):
    return {'relu':nn.ReLU, 'tanh':nn.Tanh, 'gelu':nn.GELU}[act_fn]

class ResNet(nn.Module):
    def __init__(self, inp_dim, hid_dim, act_fn) -> None:
        super().__init__()
        self.lin1=nn.Linear(inp_dim, hid_dim)
        self.bn1=nn.BatchNorm1d(hid_dim)
        self.act1=get_act_fn(act_fn)()
        
        self.lin2=nn.Linear(hid_dim, inp_dim)
        self.bn2=nn.BatchNorm1d(inp_dim)
        self.act2=get_act_fn(act_fn)()
        
    def forward(self, inp):
        temp_inp=self.lin1(inp)
        temp_inp=self.bn1(temp_inp)
        temp_inp=self.act1(temp_inp)
        
        temp_inp=self.lin2(temp_inp)
        temp_inp=self.bn2(temp_inp)
        out=self.act2(temp_inp+inp)
        return out

def generate_discription(config, run_number = None):
        config_vars = vars(config)
        config_str = f"Run number: {run_number}\nConfig:\n"
        config_str += "\n".join(f"    {key} = {repr(value)}" for key, value in config_vars.items())
        return config_str
    
class DDPM():
    def __init__(self, model_name:str) -> None:
        if model_name == 'branin':
            self.model_class = BraninDiffusionModel
            self.dataset_class = BraninDataset
        elif model_name == "pose_mlp":
            self.model_class = PoseMLPDiffusion
            self.dataset_class = PoseModelDataset
        elif model_name == "pose_transformer":
            self.model_class = PoseTranformerDiffusion
            self.dataset_class = PoseModelDataset
        else:
            raise NameError("Unknown model class:", model_name)
        
        self.pretrained_model = None
        
    def train(self, config, run_number):
        description = generate_discription(config, run_number)
        print("****** Training Initiated ******")
        print(description)
        
        train_dir=config.save_dir+"/run"+str(run_number)+"/train"
        os.makedirs(train_dir, exist_ok=False)
        
        model = self.model_class(config).to(device)
        dataset=self.dataset_class(config)
        dataloader=DataLoader(dataset, batch_size=config.batch_size)
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
                data_x = batch_data['x']
                batch_size=data_x.shape[0]
                inp_noise=torch.randn_like(data_x)
                inp_t_ind=torch.randint(0, config.n_time, (batch_size, 1))
                inp_x = (data_x*sqrt_alpha_bar[inp_t_ind] + inp_noise*sqrt_om_alpha_bar[inp_t_ind]).to(device)
                inp_y = batch_data['y'].to(device)
                inp_t = (inp_t_ind/config.n_time).to(device)
                pred_noise = model(inp_x, inp_y, inp_t, mode).to(device)
                loss = loss_function(pred_noise, inp_noise.to(device)).to(device)
                loss.backward()
                optimizer.step()
                running_loss=running_loss+loss.item()
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
        self.pretrained_model.eval()
        
        [alpha, beta, sqrt_alpha_bar,
         sqrt_om_alpha_bar]=diffusion_noise_schedule(self.pretrained_config.n_time,
                                                     self.pretrained_config.beta_range)
        om_alpha = 1 - alpha
        
        x_mat = torch.zeros((self.pretrained_config.n_time, n_test_points, self.pretrained_config.inp_dim))
        x_mat[-1, :, :] = torch.randn((n_test_points, self.pretrained_config.inp_dim))
        y_inp = torch.ones((n_test_points, self.pretrained_config.out_dim))*target_y
        
        for it in range(self.pretrained_config.n_time-1, 0, -1):
            t_inp = torch.ones((n_test_points, 1))*it/self.pretrained_config.n_time
            cond_pred_noise = self.pretrained_model(x_mat[it, :, :], y_inp, t_inp, mode)
            uncond_pred_noise = self.pretrained_model(x_mat[it, :, :], None, t_inp, mode)
            pred_noise = (1 + cond_weight)*cond_pred_noise + cond_weight*uncond_pred_noise
            z = torch.randn_like(x_mat[it, :, :]) if it > 1 else 0            
            x_mat[it - 1, :, :] = (x_mat[it, :, :] - (om_alpha[it]*pred_noise/sqrt_om_alpha_bar[it]))/(alpha[it].sqrt()) + beta[it]*z
            
        return x_mat
        
    def load_pretrained_model(self, model_path):
        saved_obj = torch.load(model_path, map_location=device)
        self.pretrained_description = saved_obj["description"]
        self.pretrained_config = saved_obj["config"]
        self.pretrained_model = self.model_class(saved_obj["config"])
        print(self.pretrained_model.load_state_dict(saved_obj["state_dict"]))
        self.pretrained_model.eval()

class BraninDiffusionModel(nn.Module):
    def __init__(self, config:BraninConfig) -> None:
        super().__init__()
        self.drop_prob=config.drop_prob
        self.n_hid_dim = config.n_hid_dim
        self.t_embed_nn=nn.Sequential(*[nn.Linear(1, config.n_hid_dim),
                                       nn.BatchNorm1d(config.n_hid_dim),
                                       get_act_fn(config.act_fn)(),
                                       nn.Linear(config.n_hid_dim, config.n_hid_dim)])
        self.y_embed_nn=nn.Sequential(*[nn.Linear(config.out_dim, config.n_hid_dim),
                                       nn.BatchNorm1d(config.n_hid_dim),
                                       get_act_fn(config.act_fn)(),
                                       nn.Linear(config.n_hid_dim, config.n_hid_dim)])
        self.ty_embed_nn=nn.Sequential(*[nn.Linear(2*config.n_hid_dim, config.n_hid_dim),
                                       nn.BatchNorm1d(config.n_hid_dim),
                                       get_act_fn(config.act_fn)(),
                                       nn.Linear(config.n_hid_dim, config.n_hid_dim)])
        self.x_embed_nn=nn.Sequential(*[nn.Linear(config.inp_dim, config.n_hid_dim),
                                       nn.BatchNorm1d(config.n_hid_dim),
                                       get_act_fn(config.act_fn)(),
                                       nn.Linear(config.n_hid_dim, config.n_hid_dim)])
        self.xty_embed_nn=nn.Sequential(*[nn.Linear(2*config.n_hid_dim, config.n_hid_dim),
                                       nn.BatchNorm1d(config.n_hid_dim),
                                       get_act_fn(config.act_fn)(),
                                       nn.Linear(config.n_hid_dim, config.n_hid_dim)])
        
        self.resnets=nn.ModuleList()
        for _ in range(config.n_resnets):
            self.resnets.append(ResNet(config.n_hid_dim, config.n_hid_dim, config.act_fn))
            
        self.final_layer=nn.Sequential(*[nn.Linear(config.n_hid_dim, config.n_hid_dim),
                                       nn.BatchNorm1d(config.n_hid_dim),
                                       get_act_fn(config.act_fn)(),
                                       nn.Linear(config.n_hid_dim, config.inp_dim)])
        
    def forward(self, noisy_x, y, t, mode):
        t_embed=self.t_embed_nn(t)
        if mode=="train":
            y_embed=self.y_embed_nn(y)
            y_embed=y_embed*(torch.bernoulli(torch.ones(y_embed.shape[0], 1)-self.drop_prob).to(device))
        elif mode=="eval":
            if y is None:
                y_embed=torch.zeros((noisy_x.shape[0], self.n_hid_dim))
            else:
                y_embed=self.y_embed_nn(y)
        else:
            raise ValueError("unknown mode: not train or eval")
        
        ty_embed=self.ty_embed_nn(torch.cat([t_embed, y_embed], dim=1))
        x_embed=self.x_embed_nn(noisy_x)
        
        xty_embed=self.xty_embed_nn(torch.cat([ty_embed, x_embed], dim=1))
        for temp_resnet in self.resnets:
            xty_embed=temp_resnet(xty_embed)
        
        out=self.final_layer(xty_embed)
        return out        

class PoseMLPDiffusion(nn.Module):
    def __init__(self, config:PoseMLPConfig) -> None:
        super().__init__()
        self.drop_prob=config.drop_prob
        self.n_hid_dim = config.n_hid_dim
        self.t_embed_nn=nn.Sequential(*[nn.Linear(1, config.n_hid_dim),
                                       nn.BatchNorm1d(config.n_hid_dim),
                                       get_act_fn(config.act_fn)(),
                                       nn.Linear(config.n_hid_dim, config.n_hid_dim)])
        self.y_embed_nn=nn.Sequential(*[nn.Linear(config.out_dim, config.n_hid_dim),
                                       nn.BatchNorm1d(config.n_hid_dim),
                                       get_act_fn(config.act_fn)(),
                                       nn.Linear(config.n_hid_dim, config.n_hid_dim)])
        self.ty_embed_nn=nn.Sequential(*[nn.Linear(2*config.n_hid_dim, config.n_hid_dim),
                                       nn.BatchNorm1d(config.n_hid_dim),
                                       get_act_fn(config.act_fn)(),
                                       nn.Linear(config.n_hid_dim, config.n_hid_dim)])
        self.x_embed_nn=nn.Sequential(*[nn.Linear(config.inp_dim, config.n_hid_dim),
                                       nn.BatchNorm1d(config.n_hid_dim),
                                       get_act_fn(config.act_fn)(),
                                       nn.Linear(config.n_hid_dim, config.n_hid_dim)])
        self.xty_embed_nn=nn.Sequential(*[nn.Linear(2*config.n_hid_dim, config.n_hid_dim),
                                       nn.BatchNorm1d(config.n_hid_dim),
                                       get_act_fn(config.act_fn)(),
                                       nn.Linear(config.n_hid_dim, config.n_hid_dim)])
        
        self.resnets=nn.ModuleList()
        for _ in range(config.n_resnets):
            self.resnets.append(ResNet(config.n_hid_dim, config.n_hid_dim, config.act_fn))
            
        self.final_layer=nn.Sequential(*[nn.Linear(config.n_hid_dim, config.n_hid_dim),
                                       nn.BatchNorm1d(config.n_hid_dim),
                                       get_act_fn(config.act_fn)(),
                                       nn.Linear(config.n_hid_dim, config.inp_dim)])
        
    def forward(self, noisy_x, y, t, mode):
        t_embed=self.t_embed_nn(t)
        if mode=="train":
            y_embed=self.y_embed_nn(y)
            y_embed=y_embed*(torch.bernoulli(torch.ones(y_embed.shape[0], 1)-self.drop_prob).to(device))
        elif mode=="eval":
            if y is None:
                y_embed=torch.zeros((noisy_x.shape[0], self.n_hid_dim))
            else:
                y_embed=self.y_embed_nn(y)
        else:
            raise ValueError("unknown mode:", mode)
        
        ty_embed=self.ty_embed_nn(torch.cat([t_embed, y_embed], dim=1))
        x_embed=self.x_embed_nn(noisy_x)
        
        xty_embed=self.xty_embed_nn(torch.cat([ty_embed, x_embed], dim=1))
        for temp_resnet in self.resnets:
            xty_embed=temp_resnet(xty_embed)
        
        out=self.final_layer(xty_embed)
        return out
    
class PoseTranformerDiffusion(nn.Module):
    def __init__(self, config:PoseTransformersConfig) -> None:
        super().__init__()
        
        self.x_dim_1 = config.x_dim_1
        self.x_dim_2 = config.x_dim_2
        self.y_dim_1 = config.y_dim_1
        self.y_dim_2 = config.y_dim_2
        
        self.t_dim = 1
        
        self.emb_dim = config.emb_dim
        
        self.y_drop_prob = config.diff_drop_prob
        
        # x_shape: B x  x_dim_1 x x_dim_2
        self.x_emb = nn.Linear(self.x_dim_2, self.emb_dim)
        self.x_pos_emb = nn.Parameter(torch.randn(self.x_dim_1,
                                                  self.emb_dim))    
        
        # y_shape: B x y_dim_1 x y_dim_2
        self.y_emb = nn.Linear(self.y_dim_2, self.emb_dim)
        self.y_pos_emb = nn.Parameter(torch.randn(self.y_dim_1,
                                                  self.emb_dim))
        
        # t: B x 1 
        self.t_emb = nn.Linear(self.t_dim, self.emb_dim)
        
        temp_list = []
        for _ in range(config.n_x_attn):
            temp_list.append(AttentionBlock(config.n_heads, config.emb_dim,
                                            config.kq_dim, config.mlp_hid_dim,
                                            config.mlp_act_fn))
        self.x_attn = nn.ModuleList(temp_list)
        
        temp_list = []
        for _ in range(config.n_y_attn):
            temp_list.append(AttentionBlock(config.n_heads, config.emb_dim,
                                            config.kq_dim, config.mlp_hid_dim,
                                            config.mlp_act_fn))
        self.y_attn = nn.ModuleList(temp_list)
        
        temp_list = []
        for _ in range(config.n_xyt_attn):
            temp_list.append(CrossAttentionBlock(config.n_heads, config.emb_dim,
                                                 config.kq_dim, config.emb_dim, config.mlp_hid_dim,
                                                 config.mlp_act_fn))
        self.xy_cross_attn = nn.ModuleList(temp_list)
        
        temp_list = []
        for _ in range(config.n_xyt_attn):
            temp_list.append(CrossAttentionBlock(config.n_heads, config.emb_dim,
                                                 config.kq_dim, config.emb_dim, config.mlp_hid_dim,
                                                 config.mlp_act_fn))
        self.xt_cross_attn = nn.ModuleList(temp_list)
        
        self.out_layer = nn.Linear(config.emb_dim, config.x_dim_2)
    
    def forward(self, x, y, t, mode):
        batch_size = x.shape[0]
        
        x = x.reshape(batch_size, self.x_dim_1, self.x_dim_2)
        x = self.x_emb(x) + self.x_pos_emb
        
        if y is None:
            y = torch.zeros((batch_size, self.y_dim_1, self.emb_dim))
        else:
            y = y.reshape(batch_size, self.y_dim_1, self.y_dim_2)
            y = self.y_emb(y)

        if mode=="train":
            y =y*(torch.bernoulli(torch.ones(batch_size, 1, 1)-self.y_drop_prob).to(device))
        y = y + self.y_pos_emb
        
        t = self.t_emb(t[:, None, None])
        
        for x_attn in self.x_attn:
            x = x_attn(x)

        for y_attn in self.y_attn:
            y = y_attn(y)
        
        for i in range(len(self.xy_cross_attn)):
            x = self.xt_cross_attn[i](x, t)
            x = self.xy_cross_attn[i](x, y)
            
        x = self.out_layer(x)
        x = x.reshape(batch_size, self.x_dim_1*self.x_dim_2)
        return x
    


    
