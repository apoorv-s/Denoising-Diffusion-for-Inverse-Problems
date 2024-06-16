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
    return [alpha_bar.sqrt(), om_alpha_bar.sqrt()]

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

class NNModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.drop_prob=config.drop_prob
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
        
        self.resnets=[]
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
                y_embed=torch.zeros((y.shape[0], self.config.n_hid_dim))
            else:
                y_embed=self.y_embed_nn(y)
        else:
            raise ValueError("unknown mode")
        
        ty_embed=self.ty_embed_nn(torch.cat([t_embed, y_embed], dim=1))
        x_embed=self.x_embed_nn(noisy_x)
        
        xty_embed=self.xty_embed_nn(torch.cat([ty_embed, x_embed], dim=1))
        for temp_resnet in self.resnets:
            xty_embed=temp_resnet(xty_embed)
        
        out=self.final_layer(xty_embed)
        return out        

class DiffusionModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config=config
        self.nn_model=NNModel(config)
        [self.sqrt_alpha_bar,
         self.sqrt_om_alpha_bar]=diffusion_noise_schedule(self.config.n_time,
                                                                   self.config.beta_range)
        
    def forward(self, inp, out, mode):
        batch_size=inp.shape[0]
        
        inp_noise=torch.randn_like(inp)
        t_indices=torch.randint(0, self.config.n_time, (batch_size, 1))
        noisy_x = inp*self.sqrt_alpha_bar[t_indices] + inp_noise*self.sqrt_om_alpha_bar[t_indices]
        
        predicted_noise=self.nn_model(noisy_x, out, t_indices/self.config.n_time, mode)
        return (inp_noise-predicted_noise).square().mean()
        

class BraninDDPM():
    def __init__(self, config, model_path=None) -> None:
        self.config=config
        if model_path is None:
            self.pretrained=False
            self.model=DiffusionModel(config).to(device)
            print("untrained model")
        else:
            self.pretrained=True
            self.model=torch.load(model_path)
            print("using trained model")
        
    def train(self):
        dataset=BraninDataset(self.config)
        dataloader=DataLoader(dataset, batch_size=64, shuffle=True)
        optimizer=torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        
        train_dir=self.config.save_dir+"/run"+str(self.config.run_number)+"/train"
        os.makedirs(train_dir, exist_ok=False)
        
        writer=SummaryWriter(train_dir+'/tboard')
        counter=0
        mode="train"
        for ep in trange(self.config.n_epochs):
            
            running_loss=0.0
            for ind, batch_data in enumerate(dataloader):
                optimizer.zero_grad()
                loss=self.model(batch_data['x'].to(device), batch_data['y'].to(device), mode)
                loss.backward()
                optimizer.step()
                running_loss=running_loss+loss.item()
                if ind%100==99:
                    writer.add_scalar('running_train_loss', running_loss/100, counter)
                    counter=counter+1
                    print('[epoch=%d--batch_id=%d] loss=%.3f'%(ep, ind, running_loss/100))
                    running_loss=0.0
            
            if ep%20==19:
                torch.save(self.model, train_dir+"/int_model_"+str(ep)+".pt")
        
        torch.save(self.model, train_dir+"/final_model_"+str(ep)+".pt")
    
    def sample(self):
        pass
        
    def validate(self, n_points, y_test):
        pass