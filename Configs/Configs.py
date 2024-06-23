class BraninConfig():
    def __init__(self) -> None:
        # data
        self.inp_dim = 2
        self.out_dim = 1
        
        self.save_dir = "./Models/Branin"
        
        self.wt_norm_ind = False
        self.act_lyr = "batch"
        self.act_fn = "gelu"
        
        self.batch_size = 1024
        self.n_workers = 4
        self.lr = 1e-4
        self.wt_decay = 1e-3
        
        self.save_bw_ep = True
        self.n_epochs = 500
        self.n_resnets = 6
        self.n_hid_dim = 64
        
        self.drop_prob = 0.1
        self.n_time = 1000
        
        # Noise Schedule:
        self.beta_range = (1e-4, 0.02)
        
        self.n_epochs_bw_saves = 50
    
class PoseConfig():
    def __init__(self) -> None:
        # data
        self.inp_dim = None
        self.out_dim = None
        
        self.save_dir = "./Models/PoseModel"
        
        self.wt_norm_ind = False
        self.act_lyr = "batch"
        self.act_fn = "relu"
        
        self.batch_size = 1024
        self.n_workers = 4
        self.lr = 1e-4
        self.wt_decay = 1e-3
        
        self.save_bw_ep = True
        self.n_epochs = 500
        self.n_resnets = 3
        self.n_hid_dim = 128
        
        self.drop_prob = 0.1
        self.n_time = 500
        
        # Noise Schedule:
        self.beta_range = (1e-4, 0.02)
        
        self.n_epochs_bw_saves = 50
    
    