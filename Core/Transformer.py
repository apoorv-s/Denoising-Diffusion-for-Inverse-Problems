import torch
import torch.nn as nn
from Configs.Configs import PoseTransformersConfig


class Transformer(nn.Module):
    def __init__(self, config:PoseTransformersConfig) -> None:
        super().__init__()
        
        self.x_dim_1 = config.x_dim_1
        self.x_dim_2 = config.x_dim_2
        self.y_dim_1 = config.y_dim_1
        self.y_dim_2 = config.y_dim_2
        
        self.t_dim = 1
        
        self.hid_dim = config.hid_dim
        
        # x_shape: B x  x_dim_1 x x_dim_2
        self.x_emb = nn.Linear(self.x_dim_1, self.hid_dim)
        self.x_pos_emb = nn.parameter(torch.randn(self.x_dim_1,
                                                  self.hid_dim))    
        
        # y_shape: B x y_dim_1 x y_dim_2
        self.y_emb = nn.Linear(self.y_dim_2, self.hid_dim)
        self.y_pos_emb = nn.parameter(torch.randn(self.y_dim_1,
                                                  self.hid_dim))
        
        # t: B x 1
        self.t_emb = nn.Linear(self.t_dim, self.hid_dim)
        
        
        
    
    def forward(self, x, y, t):
        batch_size = x.shape[0]
        
        x = x.view(batch_size, self.x_dim_1, self.x_dim_2)
        y = y.view(batch_size, self.y_dim_1, self.y_dim_2)
        
        
        x_emb = self.x_emb(x) + self.x_pos_emb
        
        # Include deop probability for classifier free guidance
        y_emb = self.y_emb(y)
        # Drop prob
        y_emb = y_emb + self.y_pos_emb
        
        t_emb = self.t_emb(t)
        

        
        