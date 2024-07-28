import torch
import torch.nn as nn

## Functions
def diffusion_noise_schedule(n_steps, beta_range):
    beta=torch.linspace(beta_range[0], beta_range[1], n_steps)
    alpha=1-beta
    alpha_bar=torch.cumprod(alpha, dim=0)
    om_alpha_bar=1-alpha_bar
    return [alpha, beta, alpha_bar.sqrt(), om_alpha_bar.sqrt()]


def get_act_fn(act_fn):
    return {'relu':nn.ReLU, 'tanh':nn.Tanh, 'gelu':nn.GELU}[act_fn]


def generate_discription(config, run_number = None):
        config_vars = vars(config)
        config_str = f"Run number: {run_number}\nConfig:\n"
        config_str += "\n".join(f"    {key} = {repr(value)}" for key, value in config_vars.items())
        return config_str

# Classes
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
    
