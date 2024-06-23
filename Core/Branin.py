import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter

from Core.Dataset import BraninDataset
from tqdm import trange
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def diffusion_noise_schedule(n_steps, beta_range):
    beta=torch.linspace(beta_range[0], beta_range[1], n_steps)
    alpha=1-beta
    alpha_bar=torch.cumprod(alpha, dim=0)
    om_alpha_bar=1-alpha_bar
    return [alpha, beta, alpha_bar.sqrt(), om_alpha_bar.sqrt()]

def get_act_fn(act_fn):
    return {'relu':nn.ReLU, 'tanh':nn.Tanh}[act_fn]

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

class DiffusionModel(nn.Module):
    def __init__(self, config) -> None:
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
            y_embed=y_embed*torch.bernoulli(torch.ones(y_embed.shape[0], 1)-self.drop_prob)
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

class BraninDDPM():
    def __init__(self) -> None:
        self.pretrained_model = None
            
    def generate_discription(self, config, run_number = None):
        config_vars = vars(config)
        config_str = f"Run number: {run_number}\nConfig:\n"
        config_str += "\n".join(f"    {key} = {repr(value)}" for key, value in config_vars.items())
        return config_str
        
    def train(self, config, run_number):
        description = self.generate_discription(config, run_number)
        
        train_dir=config.save_dir+"/run"+str(run_number)+"/train"
        os.makedirs(train_dir, exist_ok=False)
        
        model = DiffusionModel(config).to(device)
        dataset=BraninDataset(config)
        dataloader=DataLoader(dataset, batch_size=512)
        optimizer=torch.optim.Adam(model.parameters(), lr=config.lr)
        loss_function = nn.MSELoss(reduction='sum')
        
        [alpha, beta, sqrt_alpha_bar,
         sqrt_om_alpha_bar]=diffusion_noise_schedule(config.n_time, config.beta_range)
        
        writer=SummaryWriter(train_dir+'/tboard')
        counter=0
        
        mode="train"
        model.train()
        for ep in trange(config.n_epochs):
            running_loss=0.0
            for ind, batch_data in enumerate(dataloader):
                optimizer.zero_grad()
                data_x = batch_data['x']    
                batch_size=data_x.shape[0]
                inp_noise=torch.randn_like(data_x)
                inp_t_ind=torch.randint(0, config.n_time, (batch_size, 1))
                inp_x = (data_x*sqrt_alpha_bar[inp_t_ind] + inp_noise*sqrt_om_alpha_bar[inp_t_ind]).to(device)
                inp_y = batch_data['y'].to(device)
                inp_t = (inp_t_ind/config.n_time).to(device)
                pred_noise = model(inp_x, inp_y, inp_t, mode)
                loss = loss_function(pred_noise, inp_noise)
                loss.backward()
                optimizer.step()
                running_loss=running_loss+loss.item()
                if ind%100==99:
                    writer.add_scalar('running_train_loss', running_loss/100, counter)
                    counter=counter+1
                    print('[epoch=%d--batch_id=%d] loss=%.3f'%(ep, ind, running_loss/100))
                    running_loss=0.0
                    
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
        
        for it in range(self.pretrained_config.n_time-1, -1, -1):
            t_inp = torch.ones((n_test_points, 1))*it/self.pretrained_config.n_time
            cond_pred_noise = self.pretrained_model(x_mat[it, :, :], y_inp, t_inp, mode)
            uncond_pred_noise = self.pretrained_model(x_mat[it, :, :], None, t_inp, mode)
            pred_noise = (1 + cond_weight)*cond_pred_noise + cond_weight*uncond_pred_noise
            z = torch.randn_like(x_mat[it, :, :])            
            x_mat[it - 1, :, :] = (x_mat[it, :, :] - (om_alpha[it]*pred_noise/sqrt_om_alpha_bar[it]))/(alpha[it].sqrt()) + beta[it]*z
            
        return x_mat
        
    def load_pretrained_model(self, model_path):
        saved_obj = torch.load(model_path, map_location=device)
        self.pretrained_description = saved_obj["description"]
        self.pretrained_config = saved_obj["config"]
        self.pretrained_model = DiffusionModel(saved_obj["config"])
        self.pretrained_model.eval()