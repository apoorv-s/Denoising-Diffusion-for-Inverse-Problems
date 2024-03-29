class BraninConfig:
    # data
    inp_dim = 2
    out_dim = 1
    
    save_dir = "./SavedResults/Branin"
    
    
    wt_norm_ind = False
    act_lyr = "batch"
    act_fn = "relu"
    
    batch_size = 64
    n_workers = 4
    lr = 1e-3
    wt_decay = 1e-3
    
    save_bw_ep = True
    n_epochs = 200
    n_resnets = 2
    n_hid_dim = 128
    
    conditioning_weights = [0, 0.5, 2]
    
    drop_prob = 0.1
    n_time = 500
    
    # Noise Schedule:
    beta_range = (1e-4, 0.02)
    
    