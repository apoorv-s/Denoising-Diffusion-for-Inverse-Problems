import torch
import torch.nn as nn

from Configs.Configs import BraninMLPConfig, BraninTransformerConfig
from Core.Transformers import AttentionBlock, CrossAttentionBlock

from Core.Utils import get_act_fn, ResNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BraninMLPDiffusion(nn.Module):
    def __init__(self, config:BraninMLPConfig) -> None:
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

class BraninTranformerDiffusion(nn.Module):
    def __init__(self, config:BraninTransformerConfig) -> None:
        super().__init__()
        
        self.x_dim_1 = 1
        self.x_dim_2 = config.inp_dim
        self.y_dim_1 = 1
        self.y_dim_2 = config.out_dim
        
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
        
        self.out_layer = nn.Linear(config.emb_dim, self.x_dim_2)
    
    def forward(self, x, y, t, mode):
        batch_size = x.shape[0]
        
        x = x.reshape(batch_size, self.x_dim_1, self.x_dim_2)
        x = self.x_emb(x) + self.x_pos_emb
        
        if y is None:
            y = torch.zeros((batch_size, self.y_dim_1, self.emb_dim)).to(device)
        else:
            y = y.reshape(batch_size, self.y_dim_1, self.y_dim_2)
            y = self.y_emb(y)

        if mode=="train":
            y =y*(torch.bernoulli(torch.ones(batch_size, 1, 1)-self.y_drop_prob).to(device))
        y = y + self.y_pos_emb
        
        t = self.t_emb(t[:, None])
        
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
   
