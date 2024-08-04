import torch
from torch.utils.data import Dataset

import numpy as np
import copy
import glob
from Core.Rotations import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from torch.utils.data.dataloader import default_collate

from Core.VisualizationUtils import (SMPL, random_camera,
                                     projection, j2d_to_y,
                                     viz_smpl, show_points,
                                     dcn)

class BraninDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.input_dim=config.inp_dim
        self.output_dim=config.out_dim
        
        self.a = 1
        self.b = 5.1/(4*(np.pi**2))
        self.c = 5/np.pi
        self.r = 6
        self.s = 10
        self.t = 1/(8*np.pi)
        
        self.range = torch.tensor([[-5, 10], [0, 15]])
        
    def func(self, x):
        y = (self.a * (x[..., 1] - self.b*(x[..., 0]**2) + self.c*x[..., 0] - self.r)**2 + self.s*(1 - self.t)*torch.cos(x[..., 0]) + self.s)
        return y.unsqueeze(dim=0)
    
    def __len__(self):
        return 100000
    
    def __getitem__(self, index):
        x = torch.rand(2)
        x[0] = self.range[0][0] + x[0]*(self.range[0][1] - self.range[0][0])
        x[1] = self.range[1][0] + x[1]*(self.range[1][1] - self.range[1][0])
        return {'x':x, 'y':self.func(x)}
    
    def get_eval_data(self, n_eval_pts):
        x1=torch.linspace(self.range[0][0], self.range[0][1], n_eval_pts)
        x2=torch.linspace(self.range[1][0], self.range[1][1], n_eval_pts)
        x1_grid,x2_grid=torch.meshgrid(x1,x2,indexing='ij')
        x=torch.stack([x1_grid.flatten(), x2_grid.flatten()]).T
        return {'x':x, 'y':self.func(x)}

class PoseModelDataset(Dataset):
    def __init__(self, config) -> None:
        super().__init__()
        self.data_dir = config.data_dir
        # subset = config.subset
        # assert subset in ["oneseq", "cmu", "full"]
        # if subset == "oneseq":
        #     self.sequences = sorted(glob.glob(self.data_dir + "/CMU/*/*_poses.npz"))
        #     self.sequences = self.sequences[-99:-98]
        #     self.sequences = self.sequences*11000
        # elif subset == "cmu":
        #     self.sequences = sorted(glob.glob(self.data_dir + "/CMU/*/*_poses.npz"))
        #     self.sequences = self.sequences * 6
        # elif subset == "full":
        #     self.sequences = sorted(glob.glob(self.data_dir + "/*/*/*_poses.npz"))
        
        self.sequences = sorted(glob.glob(self.data_dir + "/*.npz"))

        print("Number of sequences", len(self.sequences))
        self.smpl = SMPL(config.smpl_model_dir)
        self.N = 8

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx, frame_idx=None):
        sequence = self.sequences[idx]
        bdata = np.load(sequence)
        try:
            num_frames = bdata["trans"].shape[0]
        except:
            print(sequence)
            exit(0)
        is_sequence = False
        if frame_idx is None:
            frame_idx = np.random.randint(0, num_frames, self.N)
        elif frame_idx == -1:
            fps = int(bdata["mocap_framerate"])
            frame_idx = slice(0, 5 * fps, fps // 30)
            is_sequence = True

        poses = torch.tensor(bdata["poses"][frame_idx]).float()
        global_orient = poses[..., :3].float()
        body_pose = poses[..., 3 : 3 + 23 * 3].float()
        transl = torch.tensor(bdata["trans"][frame_idx]).float()

        global_orient = axis_angle_to_matrix(global_orient)
        body_pose = axis_angle_to_matrix(body_pose.unflatten(-1, (-1, 3)))

        betas = torch.from_numpy(bdata["betas"][:10]).float()
        betas = betas[None].expand(body_pose.shape[0], -1)
        betas = torch.zeros_like(betas)

        data = {
            "global_orient": global_orient,
            "body_pose": body_pose,
            "transl": transl,
            "betas": betas,
        }

        bmout = self.smpl(**{k: v for k, v in data.items()})
        joints = bmout.joints
        B, J = joints.shape[:2]
        if is_sequence:
            cam = random_camera(joints.mean((0, 1)))
            cam = default_collate([cam] * B)
        else:
            cam = random_camera(joints.mean(1))

        j2d = projection(joints, cam)
        data.update({f"cam_{k}": v for k, v in cam.items()})

        data["j2d"] = j2d
        data["x"], data["y"] = self.data_to_xy(data)

        if B == 1:
            data = {k: v[0] for k, v in data.items()}
        return data

    def data_to_xy(self, data):
        x = torch.cat([data["global_orient"][:, None], data["body_pose"]], 1)  # BJ33
        x = matrix_to_rotation_6d(x).view(x.shape[0], -1)

        y = j2d_to_y(data["j2d"], data["cam_height"], data["cam_width"])
        return x, y

    def get_sequence_data(self, idx):
        return self.__getitem__(idx, frame_idx=-1)

    def viz(self, x, data, is_gt=False, is_inp=False):
        B = x.shape[0]
        data = copy.deepcopy(data)

        data["global_orient"] = rotation_6d_to_matrix(x[:, :6])
        data["body_pose"] = rotation_6d_to_matrix(x[:, 6:].unflatten(-1, (-1, 6)))

        bmout = self.smpl(**data)
        cam = {
            k.replace("cam_", "", 1): v for k, v in data.items() if k.startswith("cam_")
        }
        if is_inp:
            bmout.vertices += 10000
            img = viz_smpl(bmout, self.smpl.faces, cam)
            img = show_points(dcn(data["j2d"]), img, color="green")
            return img

        img = viz_smpl(bmout, self.smpl.faces, cam)
        img = show_points(dcn(data["j2d"]), img, color="green")
        if not is_gt:
            pred_j2d = projection(bmout.joints, cam)
            img = show_points(pred_j2d, img, color="red")
        return img
    
if __name__ == "__main__":
    import IPython
    IPython.embed()