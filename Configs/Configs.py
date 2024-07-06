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
        self.lr = 1e-6
        self.wt_decay = 1e-4
        
        self.save_bw_ep = True
        self.n_epochs = 500
        self.n_resnets = 6
        self.n_hid_dim = 64
        
        self.drop_prob = 0.5
        self.n_time = 500
        
        # Noise Schedule:
        self.beta_range = (1e-4, 0.02)
        
        self.n_epochs_bw_saves = 50
    
class PoseMLPConfig():
    def __init__(self) -> None:
        # data
        self.inp_dim = 24*6
        self.out_dim = 25*2
        
        self.data_dir = "./Data/PoseData"
        self.smpl_model_dir = "./SupportingFiles/SMPLModel/smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl"
        self.save_dir = "./Models/PoseMLP"
        
        self.subset = "full"
        
        self.wt_norm_ind = False
        self.act_lyr = "batch"
        self.act_fn = "gelu"
        
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
       
class PoseTransformersConfig():
    def __init__(self) -> None:
        # data
        self.x_dim_1 = 24
        self.x_dim_2 = 6
        self.y_dim_1 = 25
        self.y_dim_2 = 2
        
        self.data_dir = "./Data/PoseData"
        self.smpl_model_dir = "./SupportingFiles/SMPLModel/smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl"
        self.save_dir = "./Models/PoseMLP"
        
        self.wt_norm_ind = False
        self.act_lyr = "batch"
        self.act_fn = "gelu"
        
        self.batch_size = 1024
        self.n_workers = 4
        self.lr = 1e-4
        self.wt_decay = 1e-3
        
        self.save_bw_ep = True
        self.n_epochs = 500
        self.n_resnets = 3
        self.hid_dim = 128
        
        self.drop_prob = 0.1
        self.n_time = 500
        
        # Noise Schedule:
        self.beta_range = (1e-4, 0.02)
        
        self.n_epochs_bw_saves = 50
        