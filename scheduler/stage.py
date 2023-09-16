from models.utils import four_eye, lookat, random_eye, random_at, render_one_batch, batch_rodrigues, eye_front, at_face
from models.utils import sphere_coord, random_eye_normal, rgb2hsv, differentiable_histogram, random_eye_body, random_eye_front, get_view_direction
import torch
import numpy as np
import random

class Stage():
    diffusion_model = None
    text_embeddings_map = {}
    optimizer_map = {}
    def __init__(self, camera_posi_func, camera_posi_kwargs, camera_at_func, camera_at_kwargs, 
            render_type, render_kwargs, diffusion_loss_type, loss_kwargs, text_type, neg_text_type=None, 
            optimizer_type="diffusion", clothing_decoupling=True, texture_cast_light=False, use_last_camera_pose=False, backward=True, bg_choices=[0], scale_grad=True):
        '''
            1. camera_posi_func: the function to get camera position.
            2. camera_at_func: the function to get view point position.

        '''
        self.camera_posi_func = camera_posi_func
        self.camera_posi_kwargs = camera_posi_kwargs
        self.camera_at_func = camera_at_func
        self.camera_at_kwargs = camera_at_kwargs
        self.render_type = render_type
        self.render_kwargs = render_kwargs
        self.diffusion_loss_type = diffusion_loss_type
        self.loss_kwargs = loss_kwargs
        if self.loss_kwargs["sdf_loss_type"] == "MSE":
            self.sdf_loss_criterion = torch.nn.MSELoss()
        self.text_type = text_type
        # self.text_embeddings = self.text_embeddings_map[text_type]
        if neg_text_type:
            self.neg_text_type = neg_text_type
            # self.neg_text_embeddings = self.text_embeddings_map[neg_text_type]
        self.optimizer = self.optimizer_map[optimizer_type]
        self.clothing_decoupling = clothing_decoupling
        self.texture_cast_light = texture_cast_light
        self.use_last_camera_pose = use_last_camera_pose
        self.backward = backward
        self.bg_choices = bg_choices
        self.scale_grad = scale_grad
    
    def get_background_choice(self):
        return random.choice(self.bg_choices)

    def sample_camera_at(self):
        return self.camera_at_func(**self.camera_at_kwargs)

    def sample_camera_pose(self):
        at = self.sample_camera_at()
        eye, theta, phi, is_front = self.camera_posi_func(**self.camera_posi_kwargs)
        return eye + at, at, theta, phi

    def get_loss(self, data):
        diffusion_loss = self.get_diffusion_loss(data["diffusion_loss_kwargs"])
        if not self.loss_kwargs["sdf_loss_weight"] == 0:
            sdf_loss = self.sdf_loss_criterion(data["sdf_resi_value"] - self.loss_kwargs["cloth_thickness"],  torch.zeros_like(data["sdf_resi_value"])) * self.loss_kwargs["sdf_loss_weight"]
        else:
            sdf_loss = 0
        
        if not self.loss_kwargs["mask_loss_weight"] == 0:
            mask_loss = data["mask_loss"] * self.loss_kwargs["mask_loss_weight"]
        else:
            mask_loss = 0
        
        if not self.loss_kwargs["entropy_loss_weight"] == 0:
            entropy_loss = self.loss_kwargs["entropy_loss_weight"] * data["entropy_loss"]
        else:
            entropy_loss = 0

        loss = diffusion_loss + sdf_loss + entropy_loss + mask_loss
        return loss
    
    # def set_diffusion(self, diffusion_model):
    #     self.diffusion_model = diffusion_model

    def get_diffusion_loss(self, diffusion_kwargs):
        if self.diffusion_loss_type == "train_step":
            diffusion_loss = self.diffusion_model.train_step(self.text_embeddings_map[self.text_type][diffusion_kwargs["view_prompt"]], diffusion_kwargs["pred_rgb"])
        elif self.diffusion_loss_type == "train_step_latent":
            diffusion_loss = self.diffusion_model.train_step_latent(self.text_embeddings_map[self.text_type][diffusion_kwargs["view_prompt"]],
                                                             self.text_embeddings_map[self.neg_text_type][diffusion_kwargs["view_prompt"]], 
                                                             diffusion_kwargs["pred_rgb"],
                                                             diffusion_kwargs["neg_pred_rgb"])
        elif self.diffusion_loss_type == "train_step_pixel":
            diffusion_loss = self.diffusion_model.train_step_pixel(self.text_embeddings_map[self.text_type][diffusion_kwargs["view_prompt"]],
                                                             self.text_embeddings_map[self.neg_text_type][diffusion_kwargs["view_prompt"]], 
                                                             diffusion_kwargs["pred_rgb"],
                                                             diffusion_kwargs["neg_pred_rgb"])
        elif self.diffusion_loss_type == "train_step_joint_joint":
            diffusion_loss = self.diffusion_model.train_step_joint_joint(self.text_embeddings_map[self.text_type][diffusion_kwargs["view_prompt"]],
                                                             self.text_embeddings_map[self.neg_text_type][diffusion_kwargs["view_prompt"]], 
                                                             diffusion_kwargs["pred_rgb"])
        elif self.diffusion_loss_type == "train_step_perp_neg":
            diffusion_loss = self.diffusion_model.train_step_perp_neg(self.text_embeddings_map[self.text_type][diffusion_kwargs["view_prompt"]],
                                                             self.text_embeddings_map[self.neg_text_type][diffusion_kwargs["view_prompt"]], 
                                                             diffusion_kwargs["pred_rgb"])
        elif self.diffusion_loss_type == "skin_color":
            diffusion_loss = self.sdf_loss_criterion(diffusion_kwargs["skin"] - diffusion_kwargs["skin_color"], torch.zeros_like(diffusion_kwargs["skin"]))
        return diffusion_loss

    @staticmethod
    def optimize_body_stage():
        return Stage(
            camera_posi_func = random_eye_normal,
            camera_posi_kwargs = {},
            camera_at_func = random_at,
            camera_at_kwargs = {},
            render_type = "inner",
            render_kwargs = {"is_body": True},
            diffusion_loss_type = "skin_color",
            loss_kwargs = {
                "sdf_loss_weight" : 0,
                "entropy_loss_weight" : 0,
                "mask_loss_weight" : 1500,
                "sdf_loss_type" : "MSE",
                "cloth_thickness": 0.2
            },
            text_type = "body",
            neg_text_type = None,
            optimizer_type = "body",
            clothing_decoupling=False,
            texture_cast_light=True
        )

    
    @staticmethod
    def optimize_joint_stage():
        return Stage(
            camera_posi_func = random_eye_normal,
            camera_posi_kwargs = {},
            camera_at_func = random_at,
            camera_at_kwargs = {},
            render_type = "joint",
            render_kwargs = {
                "is_refine": True,
                "is_all": True
                },
            diffusion_loss_type = "train_step_joint_joint",
            loss_kwargs = {
                "sdf_loss_weight" : 1,
                "entropy_loss_weight" : 0.3,
                "mask_loss_weight" : 0,
                "sdf_loss_type" : "MSE",
                "cloth_thickness": 0.2
            },
            text_type = "front",
            neg_text_type = "body",
            optimizer_type = "cloth",
            clothing_decoupling=True
        )
    
    @staticmethod
    def optimize_joint_stage_SDS_refine():
        return Stage(
            camera_posi_func = random_eye_normal,
            camera_posi_kwargs = {},
            camera_at_func = random_at,
            camera_at_kwargs = {},
            render_type = "joint",
            render_kwargs = {
                "is_refine": True,
                "is_all": True
                },
            diffusion_loss_type = "train_step",
            loss_kwargs = {
                "sdf_loss_weight" : 1,
                "entropy_loss_weight" : 0.3,
                "mask_loss_weight" : 0,
                "sdf_loss_type" : "MSE",
                "cloth_thickness": 0.2
            },
            text_type = "front",
            neg_text_type = "body",
            optimizer_type = "cloth",
            clothing_decoupling=True
        )

    @staticmethod
    def optimize_joint_refine_stage():
        return Stage(
            camera_posi_func = random_eye_normal,
            camera_posi_kwargs = {},
            camera_at_func = random_at,
            camera_at_kwargs = {},
            render_type = "joint",
            render_kwargs = {
                "is_refine": True,
                "is_all": True
                },
            diffusion_loss_type = "train_step",
            loss_kwargs = {
                "sdf_loss_weight" : 15000,
                "entropy_loss_weight" : 1000,
                "mask_loss_weight" : 0,
                "sdf_loss_type" : "MSE",
                "cloth_thickness": 0
            },
            text_type = "front",
            neg_text_type = "body",
            optimizer_type = "cloth",
            clothing_decoupling=True
        )
    
    @staticmethod
    def optimize_face_stage():
        return Stage(
            camera_posi_func = random_eye,
            camera_posi_kwargs = {"distance":0.4, "theta_std":np.pi/4, "phi_std": np.pi/8},
            camera_at_func = at_face,
            camera_at_kwargs = {},
            render_type = "inner",
            render_kwargs = {"is_face": True},
            diffusion_loss_type = "train_step",
            loss_kwargs = {
                "sdf_loss_weight" : 0,
                "entropy_loss_weight" : 0,
                "mask_loss_weight" : 1000,
                "sdf_loss_type" : "MSE",
                "cloth_thickness": 0.2
            },
            text_type = "face",
            neg_text_type = None,
            optimizer_type = "diffusion",
            clothing_decoupling=False
        )
    
    
    @staticmethod
    def render_body_stage():
        return Stage(
            camera_posi_func = random_eye_normal,
            camera_posi_kwargs = {},
            camera_at_func = random_at,
            camera_at_kwargs = {},
            render_type = "inner",
            render_kwargs = {},
            diffusion_loss_type = "train_step",
            loss_kwargs = {
                "sdf_loss_weight" : 0,
                "entropy_loss_weight" : 0,
                "mask_loss_weight" : 1000,
                "sdf_loss_type" : "MSE",
                "cloth_thickness": 0.2
            },
            text_type = "face",
            neg_text_type = None,
            optimizer_type = "diffusion",
            clothing_decoupling=False,
            backward=False,
            bg_choices=[3]
        )
    
    def optimize_joint_PSDS_latent_stage():
        return Stage(
            camera_posi_func = random_eye_normal,
            camera_posi_kwargs = {},
            camera_at_func = random_at,
            camera_at_kwargs = {},
            render_type = "joint",
            render_kwargs = {
                "is_refine": False,
                "is_all": True
                },
            diffusion_loss_type = "train_step_latent",
            loss_kwargs = {
                "sdf_loss_weight" : 20000,
                "entropy_loss_weight" : 0,
                "mask_loss_weight" : 0,
                "sdf_loss_type" : "MSE",
                "cloth_thickness": 0.02
            },
            text_type = "front",
            neg_text_type = "body",
            optimizer_type = "cloth",
            clothing_decoupling=True,
            use_last_camera_pose=True,
            bg_choices=[0,1,2],
            scale_grad=False
        )
    
    def optimize_joint_PSDS_pixel_stage():
        return Stage(
            camera_posi_func = random_eye_normal,
            camera_posi_kwargs = {},
            camera_at_func = random_at,
            camera_at_kwargs = {},
            render_type = "joint",
            render_kwargs = {
                "is_refine": False,
                "is_all": True
                },
            diffusion_loss_type = "train_step_pixel",
            loss_kwargs = {
                "sdf_loss_weight" : 20000,
                "entropy_loss_weight" : 0,
                "mask_loss_weight" : 0,
                "sdf_loss_type" : "MSE",
                "cloth_thickness": 0.02
            },
            text_type = "front",
            neg_text_type = "body",
            optimizer_type = "cloth",
            clothing_decoupling=True,
            use_last_camera_pose=True,
            bg_choices=[0,1,2],
            scale_grad=False
        )
    
    
    @staticmethod
    def optimize_joint_stage_PSDS():
        return Stage(
            camera_posi_func = random_eye_normal,
            camera_posi_kwargs = {},
            camera_at_func = random_at,
            camera_at_kwargs = {},
            render_type = "joint",
            render_kwargs = {
                "is_refine": False,
                "is_all": True
                },
            diffusion_loss_type = "train_step_joint_joint",
            loss_kwargs = {
                "sdf_loss_weight" : 5,
                "entropy_loss_weight" : 0,
                "mask_loss_weight" : 0,
                "sdf_loss_type" : "MSE",
                "cloth_thickness": 0.02
            },
            text_type = "front",
            neg_text_type = "body",
            optimizer_type = "cloth",
            clothing_decoupling=True
        )

    
    
    

class EndStage(Stage):
    def __init__(self):
        super().__init__()