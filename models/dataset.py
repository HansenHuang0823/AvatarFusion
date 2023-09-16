import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import json
import imageio
import logging
from scipy import ndimage
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

class SMPL_Dataset:
    def __init__(self, conf):
        super(SMPL_Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf
        self.poses = []
        with open(r'./transforms_train.json', 'r') as fp:
            meta = json.load(fp)
        for frame in meta['frames']:
            self.poses.append(np.array(frame['transform_matrix']))
        self.poses = np.array(self.poses).astype(np.float32)
        self.poses = torch.from_numpy(self.poses).to(self.device)

        R = self.poses[58, :3, :3]
        T = self.poses[58, :3, 3]
        
        self.H, self.W = (256, 256)
        camera_angle_x = 1.0471975511965976
        self.focal = .5 * self.W / np.tan(.5 * camera_angle_x)
        self.render_poses = torch.stack([pose_spherical(90, angle, 2.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
        self.image_pixels = self.H * self.W

        self.object_bbox_min = np.array([-1.01, -1.01, -1.01])
        self.object_bbox_max = np.array([ 1.01,  1.01,  1.01])

        self.K = np.array([
            [self.focal, 0,          0.5*self.W],
            [0,          self.focal, 0.5*self.H],
            [0,          0,          1         ]
        ])
        self.K = torch.from_numpy(self.K).cpu()

        print('Load data: End')
    
    def gen_rays_silhouettes(self, pose, max_ray_num, mask):
        if mask.sum() == 0:
            return self.gen_rays_pose(pose, resolution_level=4)
        struct = ndimage.generate_binary_structure(2, 2)
        dilated_mask = ndimage.binary_dilation(mask, structure=struct, iterations=10).astype(np.int32)
        current_ratio = dilated_mask.sum() / float(mask.shape[0] * mask.shape[1])
        W = H = min(self.H, int(np.sqrt(max_ray_num / current_ratio)))
        tx = torch.linspace(0, self.W - 1, W)
        ty = torch.linspace(0, self.H - 1, H)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        pixels_x = pixels_x.t()
        pixels_y = pixels_y.t()
        p = torch.stack([(pixels_x - self.K[0][2]) / self.K[0][0],
                         -(pixels_y - self.K[1][2]) / self.K[1][1],
                         -torch.ones_like(pixels_x)], -1).float()
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.sum(rays_v[..., None, :] * pose[:3, :3], -1)
        rays_o = pose[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        resized_dilated_mask = torch.nn.functional.interpolate(
            torch.from_numpy(dilated_mask).reshape(256, 256, 1).permute(2, 0, 1).unsqueeze(0).float(), size=(H, W)).squeeze()
        masked_rays_v = rays_v[resized_dilated_mask > 0]
        masked_rays_o = rays_o[resized_dilated_mask > 0]

        return masked_rays_o, masked_rays_v, W, resized_dilated_mask > 0

    def gen_rays_pose(self, pose, resolution_level=1):
        """
        Generate rays at world space given pose.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, int(self.W // l))
        ty = torch.linspace(0, self.H - 1, int(self.H // l))
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        pixels_x = pixels_x.t()
        pixels_y = pixels_y.t()
        p = torch.stack([(pixels_x - self.K[0][2]) / self.K[0][0],
                         -(pixels_y - self.K[1][2]) / self.K[1][1],
                         -torch.ones_like(pixels_x)], -1).float()
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.sum(rays_v[..., None, :] * pose[:3, :3], -1)
        rays_o = pose[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o, rays_v

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, int(self.W // l))
        ty = torch.linspace(0, self.H - 1, int(self.H // l))
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        pixels_x = pixels_x.t()
        pixels_y = pixels_y.t()
        p = torch.stack([(pixels_x - self.K[0][2]) / self.K[0][0],
                         -(pixels_y - self.K[1][2]) / self.K[1][1],
                         -torch.ones_like(pixels_x)], -1).float()
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        # rays_v = torch.matmul(self.poses[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = torch.sum(rays_v[..., None, :] * self.poses[img_idx, :3, :3], -1)
        rays_o = self.poses[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o, rays_v

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        p = torch.stack([(pixels_x - self.K[0][2]) / self.K[0][0],
                         -(pixels_y - self.K[1][2]) / self.K[1][1],
                         -torch.ones_like(pixels_x)], -1).float()
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        # rays_v = torch.matmul(self.poses[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_v = torch.sum(rays_v[..., None, :] * self.poses[img_idx, :3, :3], -1)
        rays_o = self.poses[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()    # batch_size, 10

    def near_far_from_sphere(self, rays_o, rays_d, is_sphere=False):
        # if not is_sphere:
        #     return 0.5, 3
        # else:
        #     return 0.5, 1
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1
        near[near < 0] = 0
        far = mid + 1
        return near, far
    
    def near_far_from_box(self, rays_o, rays_d, bound_min, bound_max):
        # r.dir is unit direction vector of ray
        if (rays_d == 0).any():
            pass
        dirfrac = 1 / rays_d
        # lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
        # r.org is origin of ray
        t_1 = (bound_min - rays_o) * dirfrac
        t_2 = (bound_max - rays_o) * dirfrac
        tmin = torch.max(torch.minimum(t_1, t_2), dim=1).values
        tmax = torch.min(torch.maximum(t_1, t_2), dim=1).values

        mask = torch.ones(rays_o.shape[0])
        # print(mask.sum())
        mask[tmax < 0] = 0
        # print(mask.sum())
        mask[tmin > tmax] = 0
        # print(mask.sum())
        return tmin.unsqueeze(-1), tmax.unsqueeze(-1), mask

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        img = img[:, ::-1, :]
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)
    
    def get_item(self, index):
        result = {}
        result["pose"] = torch.Tensor(np.load("./dataset/{}/pose.npy".format(index)))
        # result["gradients_grid"] = torch.Tensor(np.load("./dataset/{}/gradients_grid.npy".format(index)).reshape(1, 256, 256, 256, 3).transpose(0, 4, 3, 2, 1))
        # result["sdf_grid"] = torch.Tensor(np.load("./dataset/{}/sdf_grid.npy".format(index)).reshape(1, 256, 256, 256, 1).transpose(0, 4, 3, 2, 1))
        bound_info = np.load("./dataset/{}/bound_stand.npz".format(index))
        # result["bound_min"] = torch.Tensor(bound_info["bound_min"])
        # result["bound_max"] = torch.Tensor(bound_info["bound_max"])
        result["vertices"] = np.load("./dataset/{}/vertices.npy".format(index))
        result["bound_miner"] = torch.Tensor(bound_info["bound_miner"])
        result["bound_maxer"] = torch.Tensor(bound_info["bound_maxer"])
        bound_maxer = result["bound_maxer"]
        bound_miner = result["bound_miner"]
        bound_center = (result["bound_miner"] + result["bound_maxer"]) * 0.5
        cube_length = (bound_maxer - bound_miner).max()
        bound_cube_min = bound_center - cube_length / 2
        bound_cube_max = bound_center + cube_length / 2
        result["bound_cube_min"] = bound_cube_min
        result["bound_cube_max"] = bound_cube_max
        result["scale"] = bound_info["scale"]
        result["A"] = torch.Tensor(np.load("./dataset/{}/A.npy".format(index)))
        return result
