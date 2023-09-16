import os
from models.jointrenderer import JointRenderer
from models.clothrenderer import ClothRenderer
import logging
import argparse
import random
import numpy as np
import cv2 as cv
import torch
import torch.nn.functional as F
from torchvision import transforms
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import SMPL_Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF, SDFSMPL, ClothSDFNetwork
from models.renderer import NeuSRenderer
from models.utils import sphere_coord, random_eye_normal, rgb2hsv, differentiable_histogram, random_eye_body, random_eye_front, my_lbs, readOBJ, get_view_direction, four_eye, lookat, random_eye, random_at, render_one_batch, batch_rodrigues, eye_front
from smplx import build_layer, SMPL
from smplx.lbs import batch_rigid_transform, batch_rodrigues, blend_shapes, vertices2joints
from torchvision import transforms
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
import torch
from smpl_models.smpl.smpl_verts import vertex_ids
try:
    from torch.linalg import inv as inv
except:
    from torch import inverse as inv
from scheduler.schedule import Schedule
from scheduler.stage import Stage

class Runner:
    def __init__(self, conf_path, mode='train', is_continue=False, is_colab=False, conf=None):
        self.device = torch.device('cuda')
        self.conf_path = conf_path

        if is_colab:
            self.conf = conf
        else:
            # Configuration
            f = open(self.conf_path)
            conf_text = f.read()
            f.close()
            self.conf = ConfigFactory.parse_string(conf_text)

        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = SMPL_Dataset(self.conf['dataset'])
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.diffusion_learning_rate = self.conf.get_float('train.diffusion_learning_rate')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.max_ray_num = self.conf.get_int('train.max_ray_num', default=512 * 512)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.clip_mask_weight = self.conf.get_float('train.clip_mask_weight')
        self.diffusion_mask_weight = self.conf.get_float('train.diffusion_mask_weight')
        self.diffusion_weight = self.conf.get_float('train.diffusion_weight')
        try:
            self.clip_weight = self.conf.get_float('train.clip_weight')
        except:
            self.clip_weight = None
        try:
            self.extra_color = self.conf.get_bool('model.rendering_network.extra_color')
        except:
            self.extra_color = False
        try:
            self.add_no_texture = self.conf.get_bool('train.add_no_texture')
        except:
            self.add_no_texture = False
        try:
            self.texture_cast_light = self.conf.get_bool('train.texture_cast_light')
        except:
            self.texture_cast_light = False
        try:
            self.use_face_prompt = self.conf.get_bool('train.use_face_prompt')
        except:
            self.use_face_prompt = False
        try:
            self.use_back_prompt = self.conf.get_bool('train.use_back_prompt')
        except:
            self.use_back_prompt = False
        try:
            self.use_silhouettes = self.conf.get_bool('train.use_silhouettes')
        except:
            self.use_silhouettes = False
        try:
            self.head_height = self.conf.get_float('train.head_height')
            print("Use head height: {}".format(self.head_height))
        except:
            self.head_height = 0.65
        try:
            self.use_bg_aug = self.conf.get_bool('train.use_bg_aug')
        except:
            self.use_bg_aug = True
        try:
            self.seed = self.conf.get_int('train.seed')
            # Constrain all sources of randomness
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            print("Fix seed to: {}".format(self.seed))
        except:
            pass

        try:
            self.smpl_model_path = self.conf.get_string('general.smpl_model_path')
        except:
            self.smpl_model_path = '../../smpl_models'

        try:
            self.pose_type = self.conf.get_string('general.pose_type')
            assert self.pose_type in ['stand_pose', 't_pose']
        except:
            self.pose_type = 'stand_pose'

        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.nerf_outside = None #NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_smpl = SDFSMPL()
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        self.body_color_network = RenderingNetwork(**self.conf['model.body_rendering_network']).to(self.device)
        self.cloth_sdf_network = ClothSDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.cloth_color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)

        params_to_train += list(self.sdf_smpl.parameters())
        params_to_train += list(self.sdf_network.parameters())
        # params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())
        params_to_train += list(self.body_color_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)
        self.diffusion_optimizer = torch.optim.Adam(params_to_train, lr=self.diffusion_learning_rate * 0.4)
        self.body_optimizer = torch.optim.Adam(params_to_train, lr=self.diffusion_learning_rate * 0.4)
        Stage.optimizer_map["diffusion"] = self.diffusion_optimizer
        Stage.optimizer_map["body"] = self.body_optimizer

        params_to_train = []
        params_to_train += list(self.cloth_sdf_network.parameters())
        self.cloth_sdf_optimizer = torch.optim.Adam(params_to_train, lr=self.diffusion_learning_rate)
        params_to_train += list(self.cloth_color_network.parameters())
        self.cloth_optimizer = torch.optim.Adam(params_to_train, lr=self.diffusion_learning_rate * 2)
        
        params_to_train = []
        params_to_train += list(self.cloth_color_network.parameters())
        self.cloth_color_optimizer = torch.optim.Adam(params_to_train, lr=self.diffusion_learning_rate)

        Stage.optimizer_map["cloth"] = self.cloth_optimizer
        Stage.optimizer_map["cloth_sdf"] = self.cloth_sdf_optimizer
        Stage.optimizer_map["cloth_color"] = self.cloth_color_optimizer

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_smpl,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     self.body_color_network,
                                     **self.conf['model.neus_renderer'])
        
        self.joint_renderer = JointRenderer(self.nerf_outside,
                                     self.sdf_smpl,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     self.body_color_network,
                                     self.cloth_sdf_network,
                                     self.cloth_color_network,
                                     **self.conf['model.neus_renderer'])
        
        self.cloth_renderer = ClothRenderer(self.nerf_outside,
                                     self.sdf_smpl,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     self.cloth_sdf_network,
                                     self.cloth_color_network,
                                     **self.conf['model.neus_renderer'])

        try:
            pretrain_pth = self.conf.get_string('train.pretrain')
        except:
            pretrain_pth = None
        if pretrain_pth is not None:
            logging.info('Load pretrain: {}'.format(pretrain_pth))
            self.load_pretrain(pretrain_pth)

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def init_diffusion(self):
        device = torch.device('cuda')
        from nerf.sd import StableDiffusion
        self.guidance = StableDiffusion(device)
        Stage.diffusion_model = self.guidance
        
        Stage.text_embeddings_map["face"] = {}
        Stage.text_embeddings_map["front"] = {}
        Stage.text_embeddings_map["body"] = {}


        face_prompt = self.conf.get_string('diffusion.face_prompt')
        self.encoded_face_text_diffusion = self.guidance.get_text_embeds([face_prompt], [''])

        front_prompt = self.conf.get_string('diffusion.front_prompt')
        self.encoded_front_text_diffusion = self.guidance.get_text_embeds([front_prompt], [''])

        body_prompt = self.conf.get_string('diffusion.body_prompt')
        self.encoded_body_text_diffusion = self.guidance.get_text_embeds([body_prompt], [''])

        self.skin_color = self.conf.get_list('diffusion.skin_color')

        for view in ['front', 'side', 'back', 'top', 'bottom']:
            Stage.text_embeddings_map["face"][view] = self.guidance.get_text_embeds([face_prompt + f', {view} view'], [''])
            Stage.text_embeddings_map["front"][view] = self.guidance.get_text_embeds([front_prompt + f', {view} view'], [''])
            Stage.text_embeddings_map["body"][view] = self.guidance.get_text_embeds([body_prompt + f', {view} view'], [''])

    def init_smpl(self):
        try:
            template_obj_fname = self.conf['dataset.template_obj']
        except:
            template_obj_fname = None

        model_folder = './smpl_models'
        model_type = 'smpl'
        gender = 'neutral'
        num_betas = 10
        self.smpl_model = build_layer(
            model_folder, model_type = model_type,
            gender = gender, num_betas = num_betas, dtype=torch.float32).cuda()
        
        self.pose_type = "t_pose"
        if self.pose_type == 'stand_pose':
            with open('../ShapeGen/output/stand_pose.npy', 'rb') as f:
                new_pose = np.load(f)
        elif self.pose_type == 't_pose':
            new_pose = np.zeros([1, 24, 3])
            new_pose[:, 0, 0] = np.pi / 2
        else:
            raise NotImplementedError

        new_pose = torch.from_numpy(new_pose.astype(np.float32)).cuda()
        pose_rot = batch_rodrigues(new_pose.reshape(-1, 3)).reshape(1, 24, 3, 3)

        # if template_obj_fname is not None:
        if False:
            # v_dict = torch.load(template_obj_fname)
            # v_shaped = v_dict['v'].reshape(1, -1, 3).cuda()
            v_shaped, _, _, _ = readOBJ(template_obj_fname)
            v_shaped = torch.from_numpy(v_shaped.astype(np.float32)).reshape(1, -1, 3).cuda()
            full_pose = pose_rot.reshape(1, -1, 3, 3)
            vertices, joints = my_lbs(
                v_shaped, full_pose, self.smpl_model.v_template,
                self.smpl_model.shapedirs, self.smpl_model.posedirs,
                self.smpl_model.J_regressor, self.smpl_model.parents,
                self.smpl_model.lbs_weights, pose2rot=False,
            )
            self.v = vertices.clone()
        else:
            beta = torch.zeros([1, 10]).cuda()
            so = self.smpl_model(betas = beta, body_pose = pose_rot[:, 1:], global_orient = pose_rot[:, 0, :, :].view(1, 1, 3, 3))
            self.v = so['vertices'].clone()
            del so

        self.f = self.smpl_model.faces.copy()

    def from_smplparams_to_vertices_and_transforms(self, theta, betas):
        pose = theta

        batch_size = 1
        theta = theta.reshape(batch_size, 24, 3)
        betas = betas.reshape(batch_size, 10)

        # vertices, joints = lbs(betas, full_pose, self.v_template, self.shapedirs, self.posedirs, self.J_regressor, self.parents, self.lbs_weights, pose2rot=pose2rot)
        # a copy of lbs
        device, dtype = betas.device, betas.dtype

        # Add shape contribution
        v_shaped = self.smpl_model.v_template + blend_shapes(betas, self.smpl_model.shapedirs)

        # Get the joints
        # NxJx3 array
        J = vertices2joints(self.smpl_model.J_regressor, v_shaped)

        # 3. Add pose blend shapes
        # N x J x 3 x 3
        ident = torch.eye(3, dtype=dtype, device=device)
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view([batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 36

        pose_offsets = torch.matmul(
            pose_feature, self.smpl_model.posedirs).view(batch_size, -1, 3)

        v_posed = pose_offsets + v_shaped
        # 4. Get the global joint location
        J_transformed, A = batch_rigid_transform(rot_mats, J, self.smpl_model.parents, dtype=dtype)

        # 5. Do skinning:
        # W is N x V x (J + 1)
        W = self.smpl_model.lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
        # (N x V x (J + 1)) x (N x (J + 1) x 16)
        num_joints = self.smpl_model.J_regressor.shape[0]
        T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
            .view(batch_size, -1, 4, 4)

        homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                                dtype=dtype, device=device)
        v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

        vertices = v_homo[:, :, :3, 0]
        joints = J_transformed

        return vertices, A, joints

    def refresh_smpl(self, pose, t_pose):
        '''
            input: Tensors: pose
            return: Tensors: Vertices in camera coordinates [6890, 3],
            Relative A from pose to cano [24, 4, 4],
            Joints in camera coordinates [24, 3];
        '''
        vertices, A, joints = self.from_smplparams_to_vertices_and_transforms(pose, torch.zeros(10))
        rot_mat = torch.from_numpy(np.array(
                [[ 1.,  0.,  0.],
                [ 0.,  0., -1.],
                [ 0.,  1.,  0.]], dtype=np.float32)).cuda()
        vertices = torch.matmul(vertices, rot_mat)
        joints = torch.matmul(joints, rot_mat)
        A = A.squeeze()

        # t_pose = torch.zeros_like(pose, device=pose.device)
        # t_pose[:3] = torch.Tensor([np.pi/2, 0, 0])
        # angle = 30
        # t_pose[5] = np.deg2rad(angle)
        # t_pose[8] = np.deg2rad(-angle)
        t_pose = t_pose.clone()
        t_pose_vertices, A_t, joints_t = self.from_smplparams_to_vertices_and_transforms(torch.Tensor(t_pose), torch.zeros(10).cuda())
        A_t = A_t.squeeze()
        A_c = torch.from_numpy(np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)).cuda()
        A_r = torch.matmul(torch.matmul(A_c, A), inv(torch.matmul(A_c, A_t)))
        return vertices[0], A_r, joints[0]

    def train_diffusion(self, schedule):
        self.update_learning_rate()

        motion = np.load("./stand_pose.npy")
        scaler = torch.cuda.amp.GradScaler()
        for iter_i in tqdm(range(self.iter_step, self.end_iter)):
            stage = schedule.get_stage(iter_i)

            data = {}

            theta = torch.from_numpy(motion[iter_i % motion.shape[0]]).cuda()
            theta[:3] = torch.Tensor([np.pi/2, 0, 0])
            big_poses = np.zeros((72))
            angle = 30
            big_poses[5] = np.deg2rad(angle)
            big_poses[8] = np.deg2rad(-angle)
            big_poses = torch.Tensor(big_poses).cuda()
            big_poses[:3] = torch.Tensor([np.pi/2, 0, 0])
            vertices, A, joints = self.refresh_smpl(theta, big_poses)
            data["pose"] = theta
            data["vertices"] = vertices
            data["A"] = A

            eye, at, theta, phi = stage.sample_camera_pose()
            if stage.use_last_camera_pose:
                eye = self.eye
                at = self.at
                theta = self.theta
                phi = self.phi
            else:
                self.eye = eye
                self.at = at
                self.theta = theta
                self.phi = phi
            
            view_prompt = get_view_direction(theta, phi, np.pi/6, np.pi/3)

            pose = lookat(eye, at, np.array([0, 1, 0]))

            true_rgb = torch.from_numpy(render_one_batch(torch.Tensor(data["vertices"]).unsqueeze(0), self.f, torch.from_numpy(eye).cuda(), torch.from_numpy(at).cuda()))
            
            ori_mask = torch.zeros_like(true_rgb)
            ori_mask[true_rgb != 0] = 1
            ori_mask = ori_mask[..., 0]

            rays_o, rays_d, W, dilated_mask = self.dataset.gen_rays_silhouettes(torch.from_numpy(pose).cuda(), self.max_ray_num, ori_mask)
            H = W
            rays_o = rays_o.float()
            rays_d = rays_d.float()

            true_rgb = torch.nn.functional.interpolate(true_rgb.reshape(256, 256, 3).permute(2, 0, 1).unsqueeze(0), \
                                                       size=(H, W)).squeeze(0).permute(1, 2, 0).cuda().reshape(-1, 3)

            if True:
                toimg = transforms.ToPILImage()
                true_rgb_img = true_rgb.clone().reshape(H,W,3).cpu().transpose(1, 2).transpose(1, 0)
                img = toimg(true_rgb_img)
                os.makedirs(os.path.join(self.base_exp_dir, 'images'), exist_ok=True)
                img_path = os.path.join(self.base_exp_dir, 'images', '{:0>8d}_truergb.jpg'.format(iter_i % 100))
                img.save(img_path)


            mask = torch.zeros_like(true_rgb)
            mask[true_rgb != 0] = 1
            mask = mask[..., :1]
            
            bound_min = torch.min(vertices, dim=0).values
            bound_max = torch.max(vertices, dim=0).values
            bound_center = (bound_min + bound_max) / 2
            bound_miner = bound_center - 1.2 * (bound_center - bound_min)
            bound_maxer = bound_center + 1.2 * (bound_max - bound_center)
            data["bound_miner"] = bound_miner
            data["bound_maxer"] = bound_maxer
            near, far, box_mask = self.dataset.near_far_from_box(rays_o, rays_d, data["bound_miner"], data["bound_maxer"])

            valid_rays = dilated_mask[dilated_mask == 1]
            valid_rays[box_mask == 0] = 0
            dilated_mask[dilated_mask == 1] = valid_rays
            rays_o = rays_o[box_mask == 1]
            rays_d = rays_d[box_mask == 1]
            near = near[box_mask == 1]
            far = far[box_mask == 1]
            
            choice_i = stage.get_background_choice()
            background_rgb = None
            if choice_i == -1:
                background_rgb = torch.ones([1, 3])
            if choice_i == 0:
                amb = np.random.normal(0.75, 0.1)
                if amb > 1:
                    amb = 1
                elif amb < 0:
                    amb = 0
                background_rgb = torch.ones([1, 3]) * amb
            elif choice_i == 1:
                gaussian = torch.normal(torch.zeros([H, W, 1]) + 0.5, torch.zeros([H, W, 1]) + 0.2)
                background_rgb = torch.clamp(gaussian, min=0, max=1).reshape(-1, 1)
            elif choice_i == 2:
                chess_board = torch.zeros([H, W, 1]) + 0.2
                chess_length = H // np.random.choice(np.arange(10,20))
                i, j = np.meshgrid(np.arange(H, dtype=np.int32), np.arange(W, dtype=np.int32), indexing='xy')
                div_i, div_j = i // chess_length, j // chess_length
                white_i, white_j = i[(div_i + div_j) % 2 == 0], j[(div_i + div_j) % 2 == 0]
                chess_board[white_i, white_j] = 0.8
                blur_fn = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))
                background_rgb = blur_fn(chess_board.unsqueeze(0).permute(0, 3, 1, 2)).squeeze(0).permute(1, 2, 0).reshape(-1, 1)
                background_rgb = background_rgb.reshape(-1, 1)
            
            if choice_i == 1 or choice_i == 2:
                masked_background_rgb = background_rgb.reshape(H, W, 1)[dilated_mask].reshape(-1, 1)
            else:
                masked_background_rgb = background_rgb
            
            with torch.cuda.amp.autocast(enabled=False):
                if stage.render_type == "inner":
                    render_out = self.renderer.render(rays_o, rays_d, near, far, data, 0.1,
                                                background_rgb=masked_background_rgb,
                                                cos_anneal_ratio=self.get_cos_anneal_ratio(), **stage.render_kwargs)
                elif stage.render_type == "outer":
                    render_out = self.cloth_renderer.render(rays_o, rays_d, near, far, data, 0.1,
                                                background_rgb=masked_background_rgb,
                                                cos_anneal_ratio=self.get_cos_anneal_ratio(), is_refine=iter_i>=iter_4)
                elif stage.render_type == "joint":
                    render_out = self.joint_renderer.render(rays_o, rays_d, near, far, data, 0.1,
                                                background_rgb=masked_background_rgb,
                                                cos_anneal_ratio=self.get_cos_anneal_ratio(), **stage.render_kwargs)
                else:
                    assert 1 == 0, "Invalid renderer type."
            
                color_fine = render_out['color_fine']
                if stage.diffusion_loss_type == "train_step_pixel" or stage.diffusion_loss_type == "train_step_latent":
                    color_fine = render_out['fake_color']

                ## cast light
                if self.add_no_texture or stage.texture_cast_light:
                    n_samples = self.renderer.n_samples + self.renderer.n_importance
                    normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                    normals = normals.sum(dim=1)
                    normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-7)

                    light_dir = sphere_coord(theta + np.random.uniform(-np.pi/4, np.pi/4), phi + np.random.uniform(-np.pi/4, np.pi/4))
                    light_dir = torch.from_numpy(light_dir).float()
                    rand_light_d = torch.zeros_like(normals).float().to(normals.device) + light_dir.to(normals.device)
                    rand_light_d = rand_light_d / (torch.norm(rand_light_d, dim=-1, keepdim=True) + 1e-7)
                    
                    rand_diffuse_shading = (normals * rand_light_d).sum(-1, keepdim=True).clamp(min=0, max=1)
                    rand_diffuse_shading[torch.isnan(rand_diffuse_shading)] = 1.0
                    ambience = np.random.uniform(0, 0.2)
                    diffuse = 1 - ambience
                    rand_shading = ambience + diffuse * rand_diffuse_shading

                    rand_shading_rgb = rand_shading.clone()
                    rand_shading_rgb = rand_shading_rgb.reshape(-1, 1).repeat(1, 3).float()
                    weight_sum = render_out['weight_sum'].reshape(-1)
                    # rand_shading_rgb[weight_sum < 0.5] = 0.0
                    rand_shading_rgb[weight_sum < 0.5] = color_fine[weight_sum < 0.5]

                    l_ratio = 1
                    rand_shading = l_ratio * rand_shading + 1 - l_ratio
                    rand_shading[weight_sum < 0.5] = 1.0
                    texture_shading = (color_fine * rand_shading).clamp(min=0, max=1)

                s_val = render_out['s_val']
                cdf_fine = render_out['cdf_fine']
                gradient_error = render_out['gradient_error']
                weight_max = render_out['weight_max']
                weight_sum = render_out['weight_sum']
                sdf_residual = render_out['sdf_residual']

                background = torch.zeros([H, W, 3]).cuda()
                if choice_i == -1:
                    background[:] = 1
                elif choice_i == 0:
                    background[:] = amb
                elif choice_i == 1 or choice_i == 2:
                    background[~dilated_mask] = background_rgb.reshape(H, W, 1).repeat(1, 1, 3)[~dilated_mask]
                    
                if self.add_no_texture or stage.texture_cast_light:
                    full_texture_shading = background.clone()
                    full_texture_shading[dilated_mask] = texture_shading
                    texture_shading = full_texture_shading.reshape(-1, 3)

                    full_rand_shading_rgb = background.clone()
                    full_rand_shading_rgb[dilated_mask] = rand_shading_rgb
                    rand_shading_rgb = full_rand_shading_rgb.reshape(-1, 3)

                full_color_fine = background.clone()
                full_color_fine[dilated_mask] = color_fine
                color_fine = full_color_fine.reshape(-1, 3)

                full_weight_sum = torch.zeros([H, W, 1]).cuda()
                full_weight_sum[dilated_mask] = weight_sum
                weight_sum = full_weight_sum.reshape(-1, 1)


                mask_loss = F.binary_cross_entropy(weight_sum.clamp(1e-3, 1.0 - 1e-3), mask)
            # with torch.cuda.amp.autocast():
                if stage.texture_cast_light:
                    pred_rgb = texture_shading.reshape(H, W, 3).unsqueeze(0).permute(0, 3, 1, 2)
                    skin = texture_shading[weight_sum.squeeze(1)>0.8]
                else:
                    pred_rgb = color_fine.reshape(H, W, 3).unsqueeze(0).permute(0, 3, 1, 2)
                    skin = color_fine[weight_sum.squeeze(1)>0.8]

                augmented_extra_rgb_hack = pred_rgb.clone()

                if True:
                    toimg = transforms.ToPILImage()
                    augmented_extra_rgb_img = augmented_extra_rgb_hack[0].cpu()
                    img = toimg(augmented_extra_rgb_img)
                    os.makedirs(os.path.join(self.base_exp_dir, 'images'), exist_ok=True)
                    img_path = os.path.join(self.base_exp_dir, 'images', '{:0>8d}.jpg'.format(iter_i % 100))
                    img.save(img_path)
                
                if stage.backward == False:
                    self.body_pred_rgb = pred_rgb.detach()
                    pred_rgb.mean().backward()
                    continue

                loss_kwargs = {}
                loss_kwargs["sdf_resi_value"] = sdf_residual
                loss_kwargs["mask_loss"] = mask_loss
                loss_kwargs["entropy_loss"] = render_out.get('weight_entropy')
                loss_kwargs["diffusion_loss_kwargs"] = {}
                loss_kwargs["diffusion_loss_kwargs"]["pred_rgb"] = pred_rgb
                loss_kwargs["diffusion_loss_kwargs"]["view_prompt"] = view_prompt
                loss_kwargs["diffusion_loss_kwargs"]["skin"] = skin
                loss_kwargs["diffusion_loss_kwargs"]["skin_color"] = torch.Tensor(self.skin_color).to("cuda")
                if hasattr(self, "body_pred_rgb"):
                    loss_kwargs["diffusion_loss_kwargs"]["neg_pred_rgb"] = self.body_pred_rgb

                loss = stage.get_loss(loss_kwargs)
            
            if stage.scale_grad:
                # This somehow stablize the training procedure
                stage.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(stage.optimizer)
                scaler.update()
            else:
                stage.optimizer.zero_grad()
                loss.backward()
                stage.optimizer.step()


            self.iter_step += 1

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                # self.validate_image(idx = 4, coe=0.1, renderer_type="body")
                # self.validate_image(idx = 33, coe=0.1, renderer_type="body")
                self.validate_image(idx = 58, coe=0.1, renderer_type="body")
                
                if stage.clothing_decoupling:
                    # self.validate_image(idx = 4, coe=0.1, renderer_type="joint")
                    self.validate_image(idx = 58, coe=0.1, renderer_type="joint")
                    # self.validate_image(idx = 4, coe=0.1, renderer_type="cloth")
                    self.validate_image(idx = 58, coe=0.1, renderer_type="cloth")


            self.update_learning_rate()

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        iter_i = self.iter_step
        iter_1 = -1
        iter_2 = 25000
        iter_3 = 50000
        iter_4 = 55000
        if iter_i < iter_2:
            self.warm_up_end = 5000
        elif iter_i < iter_3:
            self.warm_up_end = 30000
        elif iter_i < iter_4:
            self.warm_up_end = 51000
        else:
            self.warm_up_end = 60000

        if self.iter_step < self.warm_up_end:
            learning_factor = 1
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
        if iter_i < iter_2:
            for g in self.diffusion_optimizer.param_groups:
                g['lr'] = self.diffusion_learning_rate * learning_factor
            for g in self.body_optimizer.param_groups:
                g['lr'] = self.diffusion_learning_rate * learning_factor * 0.2
        elif iter_i < iter_4:
            for g in self.cloth_optimizer.param_groups:
                g['lr'] = self.diffusion_learning_rate * learning_factor
        else:
            for g in self.cloth_color_optimizer.param_groups:
                g['lr'] = self.diffusion_learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        # self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        # self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        if 'body_color_network' in checkpoint.keys():
            self.body_color_network.load_state_dict(checkpoint['body_color_network'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']
        self.cloth_sdf_network.load_state_dict(checkpoint['cloth_sdf_network'])
        self.cloth_color_network.load_state_dict(checkpoint['cloth_color_network'])

        logging.info('End')

    def load_pretrain(self, checkpoint_name):
        # checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        checkpoint = torch.load(checkpoint_name, map_location=self.device)
        self.sdf_smpl.load_state_dict(checkpoint['sdf_smpl_fine'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'], strict=False)

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'sdf_smpl_fine': self.sdf_smpl.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'body_color_network': self.body_color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
            'cloth_sdf_network': self.cloth_sdf_network.state_dict(),
            'cloth_color_network': self.cloth_color_network.state_dict(),
            'cloth_sdf_optimizer': self.cloth_sdf_optimizer.state_dict(),
            'cloth_color_optimizer': self.cloth_color_optimizer.state_dict(),
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1, coe = 0.1, renderer_type="joint"):
        
        human_pose = torch.Tensor(np.load("./stand_pose.npy")[0]).cuda()
        human_pose[:3] = torch.Tensor([np.pi/2, 0, 0])
        big_poses = np.zeros((72))
        angle = 30
        big_poses[5] = np.deg2rad(angle)
        big_poses[8] = np.deg2rad(-angle)
        big_poses = torch.Tensor(big_poses).cuda()
        big_poses[:3] = torch.Tensor([np.pi/2, 0, 0])
        vertices_camera, A, joints = self.refresh_smpl(human_pose, human_pose)

        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_extra_rgb_fine = []
        out_normal_fine = []

        bound_min = torch.min(vertices_camera, dim=0).values
        bound_max = torch.max(vertices_camera, dim=0).values
        bound_center = (bound_min + bound_max) / 2
        bound_miner = bound_center - 1.2 * (bound_center - bound_min)
        bound_maxer = bound_center + 1.2 * (bound_max - bound_center)
        # print(bound_miner, bound_maxer)

        cube_length = (bound_maxer - bound_miner).max()
        bound_cube_min = bound_center - cube_length / 2
        bound_cube_max = bound_center + cube_length / 2

        data = {}
        data["pose"] = human_pose
        data["vertices"] = vertices_camera
        data["A"] = A

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):

            near, far, box_mask = self.dataset.near_far_from_box(rays_o_batch, rays_d_batch, bound_miner, bound_maxer)
            near = torch.minimum(near, far)
            far = torch.maximum(near, far)

            background_rgb = torch.ones([1, 3])
            if renderer_type == "body":
                render_out = self.renderer.render(rays_o_batch, rays_d_batch, near, far, data, coe,
                                              background_rgb=background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio())
            elif renderer_type == "joint":
                render_out = self.joint_renderer.render(rays_o_batch, rays_d_batch, near, far, data, coe,
                                              background_rgb=background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(), is_refine=False, is_all=True)
            elif renderer_type == "cloth":
                render_out = self.cloth_renderer.render(rays_o_batch, rays_d_batch, near, far, data, coe,
                                              background_rgb=background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(), is_refine=self.iter_step>50000)

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 255).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.poses[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}_{}.png'.format(self.iter_step, i, idx, renderer_type)), cv.cvtColor(img_fine[..., i], cv.COLOR_RGB2BGR))
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}_{}.png'.format(self.iter_step, i, idx, renderer_type)),
                           normal_img[..., i])

    def demo(self):
        pass
                    


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--schedule', type=int, default=0)
    parser.add_argument('--t_pose', default=False, action="store_true")

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    runner = Runner(args.conf, args.mode, args.is_continue)

    if args.mode == 'demo':
        runner.init_smpl()
        runner.demo()
    elif args.mode == 'train_diffusion':
        runner.init_diffusion()
        runner.init_smpl()
        if args.schedule == 0:
            stages = [[Stage.optimize_face_stage(), Stage.optimize_body_stage()], [Stage.optimize_joint_stage_PSDS()], [Stage.optimize_joint_stage_SDS_refine()]]
        elif args.schedule == 1:
            stages = [[Stage.optimize_face_stage(), Stage.optimize_body_stage()], [Stage.render_body_stage(), Stage.optimize_joint_PSDS_latent_stage()], [Stage.optimize_joint_stage_SDS_refine()]]
        else:
            stages = [[Stage.optimize_face_stage(), Stage.optimize_body_stage()], [Stage.render_body_stage(), Stage.optimize_joint_PSDS_pixel_stage()], [Stage.optimize_joint_stage_SDS_refine()]]

        schedule = Schedule([25000, 50000, 75000], stages)
        runner.train_diffusion(schedule)
