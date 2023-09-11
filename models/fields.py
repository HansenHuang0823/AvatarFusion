import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models.embedder import get_embedder
from smplx.body_models import SMPL
from smplx.lbs import batch_rigid_transform, batch_rodrigues, blend_shapes, vertices2joints
import trimesh
from scipy import ndimage
from psbody.mesh import Mesh

# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init and l != self.num_layers - 2:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            else:
                torch.nn.init.normal_(lin.weight, 0.0, 0.0001)
                torch.nn.init.constant_(lin.bias, 0.0)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

class ClothSDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(ClothSDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init and l != self.num_layers - 2:
            # if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            else:
                torch.nn.init.normal_(lin.weight, 0.0, 0.0001)
                torch.nn.init.constant_(lin.bias, 0.0)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class RenderingNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True,
                 extra_color=False):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        self.extra_color = extra_color
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        if self.extra_color:
            self.extra_lin = nn.Linear(dims[self.num_layers - 2], d_out)
            if weight_norm:
                self.extra_lin = nn.utils.weight_norm(self.extra_lin)

        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

            if self.extra_color and l == self.num_layers - 3:
                extra_x = self.extra_lin(x)

        if self.extra_color:
            x = torch.cat([x, extra_x], -1)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x


# This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
class NeRF(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
                 multires=0,
                 multires_view=0,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.variance = init_val
        # self.register_parameter('variance', torch.tensor(init_val))

    def forward(self, x):
        return torch.ones([len(x), 1]) * math.exp(self.variance * 10.0)

# class SDFSMPL(nn.Module):
#     def __init__(self, n_layers=2, latent_dim=64):
#         super(SDFSMPL, self).__init__()
#         self.smpl_model = SMPL("./smpl_models/smpl/SMPL_NEUTRAL.pkl")

#         embed_fn, input_ch = get_embedder(1, input_dims=3 + 72 + 10)
#         self.embed_fn_fine = embed_fn

#         self.deform_linears = nn.ModuleList(
#             [nn.Linear(input_ch, latent_dim)] +
#             [nn.Linear(latent_dim, latent_dim) for i in range(n_layers - 1)])
#         self.delta_pts_cano_linears = nn.Linear(latent_dim, 3)
        

#         nn.init.normal_(self.delta_pts_cano_linears.weight, 0.0, 0.0001)
#         nn.init.constant_(self.delta_pts_cano_linears.bias, 0.0)


#         self.A = None
#         with open("./new_data/sdf_grid_6144_100.npy", "rb") as f:
#             self.sdf_grid = torch.Tensor(np.load(f).reshape(1, 1, 256, 256, 256)).transpose(2, 4)
#         with open("./new_data/gradients_grid_6144_100.npy", "rb") as f:
#             self.gradients_grid = torch.Tensor(np.load(f).reshape(1, 256, 256, 256, 3).transpose(0, 4, 3, 2, 1))
#         with open("./new_data/bound.npz", "rb") as f:
#             bound = np.load(f)
#             self.bound_min = torch.Tensor(bound["bound_miner"])
#             self.bound_max = torch.Tensor(bound["bound_maxer"])

#         with open("./pose1.npy", "rb") as f:
#             self.bweights_grid = torch.Tensor(np.load(f).reshape(1, 128, 128, 128, 24).transpose(0, 4, 3, 2, 1))  
#         with open("./bound.npz", "rb") as f:
#             pose_bound = np.load(f)
#             self.pose_bound_min = torch.Tensor(pose_bound["bound_miner"])
#             self.pose_bound_max = torch.Tensor(pose_bound["bound_maxer"])
#         # self.bweights_grid = np.load("pose1.npy")
    
#     def transform_to_cano(self, points, theta):
#         nearest_smpl_weights = self.calculate_weights(points)
#         pts_smpl, pts_cano = self.calculate_pts_cano(points, nearest_smpl_weights, self.A, theta)
#         return pts_smpl, pts_cano

#     def forward(self, pts, theta):
#         # sdf = self.calculate_sdf(pts)
#         # return sdf, pts

#         # transform points from pose space to canonical space
#         nearest_smpl_weights = self.calculate_weights(pts)
#         self.nearest_smpl_weights = nearest_smpl_weights
#         pts_smpl, pts_cano = self.calculate_pts_cano(pts, nearest_smpl_weights, self.A, theta)
#         # calculate sdf of SMPL mesh in canonical space
#         sdf = self.calculate_sdf(pts_smpl)
#         return sdf, pts_cano
    
#     def calculate_sdf(self, points):
#         pts_grid = points.reshape(1, 1, 1, -1, 3)
#         pts_grid = 2 * (pts_grid - self.bound_min) / (self.bound_max - self.bound_min) - 1
#         sdf = F.grid_sample(self.sdf_grid, pts_grid, align_corners=True, padding_mode="border").detach()
#         sdf = sdf.squeeze().unsqueeze(-1)
#         return sdf
    
#     def calculate_weights(self, points):
#         pts_grid = points.reshape(1, 1, 1, -1, 3)
#         pts_grid = 2 * (pts_grid - self.pose_bound_min) / (self.pose_bound_max - self.pose_bound_min) - 1
#         bweights = F.grid_sample(self.bweights_grid, pts_grid, align_corners=True, padding_mode="border").detach()
#         bweights = bweights.squeeze().transpose(0, 1)
#         return bweights

    
#     def from_smplparams_to_vertices_and_transforms(self, theta, betas):
#         # deformed_vertices = self.smpl_model(betas=self.betas, global_orient=theta[0], body_pose=theta[1:]).vertices
#         # a copy of self.smpl_model.forward()
#         pose = theta

#         batch_size = 1
#         theta = theta.reshape(batch_size, 24, 3)
#         betas = betas.reshape(batch_size, 10)

#         # vertices, joints = lbs(betas, full_pose, self.v_template, self.shapedirs, self.posedirs, self.J_regressor, self.parents, self.lbs_weights, pose2rot=pose2rot)
#         # a copy of lbs
#         device, dtype = betas.device, betas.dtype

#         # Add shape contribution
#         v_shaped = self.smpl_model.v_template + blend_shapes(betas, self.smpl_model.shapedirs)

#         # Get the joints
#         # NxJx3 array
#         J = vertices2joints(self.smpl_model.J_regressor, v_shaped)

#         # 3. Add pose blend shapes
#         # N x J x 3 x 3
#         ident = torch.eye(3, dtype=dtype, device=device)
#         rot_mats = batch_rodrigues(pose.view(-1, 3)).view([batch_size, -1, 3, 3])

#         pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
#         # (N x P) x (P, V * 3) -> N x V x 3
#         pose_offsets = torch.matmul(
#             pose_feature, self.smpl_model.posedirs).view(batch_size, -1, 3)

#         v_posed = pose_offsets + v_shaped
#         # 4. Get the global joint location
#         J_transformed, A = batch_rigid_transform(rot_mats, J, self.smpl_model.parents, dtype=dtype)

#         # 5. Do skinning:
#         # W is N x V x (J + 1)
#         W = self.smpl_model.lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
#         # (N x V x (J + 1)) x (N x (J + 1) x 16)
#         num_joints = self.smpl_model.J_regressor.shape[0]
#         T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
#             .view(batch_size, -1, 4, 4)

#         homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
#                                 dtype=dtype, device=device)
#         v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
#         v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

#         vertices = v_homo[:, :, :3, 0]
#         joints = J_transformed

#         return vertices, A

#     # def calculate_sdf(self, points, bound_min, bound_max):
#     #     # sdf = mesh_to_sdf(mesh, points.cpu().numpy(), surface_point_method="sample")
#     #     pts_grid = points.reshape(1, 1, 1, -1, 3)
#     #     pts_grid = 2 * (pts_grid - bound_min) / (bound_max - bound_min) - 1
#     #     # with open("/home/huangshuo07/Models/AvatarCLIP_co2fine/AvatarGen/AppearanceGen/new_data/sdf.npy", "rb") as f:
#     #     #     self.sdf_grid = torch.Tensor(np.load(f).reshape(1, 1, 256, 256, 256)).transpose(2, 4)
#     #     sdf = F.grid_sample(self.sdf_grid, pts_grid, align_corners=True, padding_mode="border").detach()
#     #     sdf = sdf.squeeze().unsqueeze(-1)
#     #     return sdf

#     def barycentric_interpolation(self, val, coords):
#         """
#         :param val: verts x 3 x d input matrix
#         :param coords: verts x 3 barycentric weights array
#         :return: verts x d weighted matrix
#         """
#         t = val * coords[..., np.newaxis]
#         ret = t.sum(axis=1)
#         return ret

#     # def calculate_weights(self, points):
#     #     points = points.detach().cpu()
#     #     mesh = Mesh(self.vertices.cpu(), self.smpl_model.faces)
#     #     closest_face, closest_points = mesh.closest_faces_and_points(points)
#     #     vert_ids, bary_coords = mesh.barycentric_coordinates_for_points(closest_points, closest_face.astype('int32'))
#     #     bweights = self.barycentric_interpolation(self.smpl_model.lbs_weights[vert_ids.astype('int32')].cpu(),
#     #                                             bary_coords).reshape(-1, 24)
#     #     bweights = bweights.cuda()
#     #     return bweights
    
#     def calculate_pts_cano(self, points, nearest_smpl_weights, A, pose):
#         return points, points
#         # 1: SMPL inverse skinning
#         batch_size = points.shape[0]
#         num_joints = 24
#         T = torch.matmul(nearest_smpl_weights.float(), A.view(1, num_joints, 16)) \
#             .view(batch_size, 4, 4)
#         homogen_coord = torch.ones([batch_size, 1]).cuda()
#         points_homo = torch.cat([points, homogen_coord], dim=1)
#         T = T.cpu().numpy()
#         T_inv = np.linalg.inv(T)
#         pts_cano = torch.matmul(torch.Tensor(T_inv).cuda(), torch.unsqueeze(points_homo, dim=-1)).squeeze()[:, :-1]
#         # 2: delta_x_cano received from MLP.
#         # input_embedding = self.embed_fn_fine(torch.cat([pts_cano, pose.reshape(-1).unsqueeze(0).expand(batch_size, 72), torch.zeros(10).reshape(-1).unsqueeze(0).expand(batch_size, 10)], dim=1))

#         # h = input_embedding
#         # for i, l in enumerate(self.deform_linears):
#         #     h = self.deform_linears[i](h)
#         #     h = F.relu(h)
#         # delta_pts_cano = self.delta_pts_cano_linears(h)
#         return pts_cano, pts_cano


#     def jacobian(self, pts, theta):
#         # nearest_smpl_weights = self.calculate_weights(pts)
#         # return torch.ones(list(pts.shape)+[3])
#         pts.requires_grad = True
#         # self.nearest_smpl_weights = None
#         pts_smpl, pts_cano = self.calculate_pts_cano(pts, self.nearest_smpl_weights, self.A, theta)
#         jacobian = torch.stack([torch.autograd.grad([pts_cano[:, i].sum()], [pts], retain_graph=True, create_graph=True)[0] for i in range(3)], dim=-1)
#         return jacobian

    
#     def cano_gradient(self, pts_cano):
#         pts_grid = pts_cano.reshape(1, 1, 1, -1, 3)
#         pts_grid = 2 * (pts_grid - self.bound_min) / (self.bound_max - self.bound_min) - 1
#         gradient = F.grid_sample(self.gradients_grid, pts_grid, align_corners=True, padding_mode="border").detach()
#         gradient = gradient.squeeze().transpose(0, 1)
#         # print(torch.sqrt(torch.sum(gradient * gradient, -1)).mean())
#         gradient = gradient
#         return gradient

class SDFSMPL(nn.Module):
    def __init__(self, n_layers=2, latent_dim=64):
        super(SDFSMPL, self).__init__()
        self.smpl_model = SMPL("./smpl_models/smpl/SMPL_NEUTRAL.pkl")

        embed_fn, input_ch = get_embedder(1, input_dims=3 + 72 + 10)
        self.embed_fn_fine = embed_fn

        self.deform_linears = nn.ModuleList(
            [nn.Linear(input_ch, latent_dim)] +
            [nn.Linear(latent_dim, latent_dim) for i in range(n_layers - 1)])
        self.delta_pts_cano_linears = nn.Linear(latent_dim, 3)
        

        nn.init.normal_(self.delta_pts_cano_linears.weight, 0.0, 0.0001)
        nn.init.constant_(self.delta_pts_cano_linears.bias, 0.0)


        self.A = None
        with open("./new_data/sdf_grid_stand.npy", "rb") as f:
            self.sdf_grid = torch.Tensor(np.load(f).reshape(1, 1, 256, 256, 256)).transpose(2, 4)
        with open("./new_data/gradients_grid_stand.npy", "rb") as f:
            self.gradients_grid = torch.Tensor(np.load(f).reshape(1, 256, 256, 256, 3).transpose(0, 4, 3, 2, 1))
        with open("./new_data/bound_stand.npz", "rb") as f:
            bound = np.load(f)
            self.bound_min = torch.Tensor(bound["bound_miner"])
            self.bound_max = torch.Tensor(bound["bound_maxer"])
    
    def transform_to_cano(self, points, theta):
        nearest_smpl_weights = self.calculate_weights(points)
        pts_smpl, pts_cano = self.calculate_pts_cano(points, nearest_smpl_weights, self.A, theta)
        return pts_smpl, pts_cano

    def forward(self, pts, theta):
        sdf = self.calculate_sdf(pts)
        return sdf, pts

        # transform points from pose space to canonical space
        nearest_smpl_weights = self.calculate_weights(pts)
        self.nearest_smpl_weights = nearest_smpl_weights
        pts_smpl, pts_cano = self.calculate_pts_cano(pts, nearest_smpl_weights, self.A, theta)
        # calculate sdf of SMPL mesh in canonical space
        sdf = self.calculate_sdf(pts_smpl)
        return sdf, pts_cano
    
    def calculate_sdf(self, points):
        pts_grid = points.reshape(1, 1, 1, -1, 3)
        pts_grid = 2 * (pts_grid - self.bound_min) / (self.bound_max - self.bound_min) - 1
        sdf = F.grid_sample(self.sdf_grid, pts_grid, align_corners=True, padding_mode="border").detach()
        sdf = sdf.squeeze().unsqueeze(-1)
        return sdf

    
    def from_smplparams_to_vertices_and_transforms(self, theta, betas):
        # deformed_vertices = self.smpl_model(betas=self.betas, global_orient=theta[0], body_pose=theta[1:]).vertices
        # a copy of self.smpl_model.forward()
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
        # (N x P) x (P, V * 3) -> N x V x 3
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

        return vertices, A

    # def calculate_sdf(self, points, bound_min, bound_max):
    #     # sdf = mesh_to_sdf(mesh, points.cpu().numpy(), surface_point_method="sample")
    #     pts_grid = points.reshape(1, 1, 1, -1, 3)
    #     pts_grid = 2 * (pts_grid - bound_min) / (bound_max - bound_min) - 1
    #     # with open("/home/huangshuo07/Models/AvatarCLIP_co2fine/AvatarGen/AppearanceGen/new_data/sdf.npy", "rb") as f:
    #     #     self.sdf_grid = torch.Tensor(np.load(f).reshape(1, 1, 256, 256, 256)).transpose(2, 4)
    #     sdf = F.grid_sample(self.sdf_grid, pts_grid, align_corners=True, padding_mode="border").detach()
    #     sdf = sdf.squeeze().unsqueeze(-1)
    #     return sdf

    def barycentric_interpolation(self, val, coords):
        """
        :param val: verts x 3 x d input matrix
        :param coords: verts x 3 barycentric weights array
        :return: verts x d weighted matrix
        """
        t = val * coords[..., np.newaxis]
        ret = t.sum(axis=1)
        return ret

    def calculate_weights(self, points):
        points = points.detach().cpu()
        mesh = Mesh(self.vertices.cpu(), self.smpl_model.faces)
        closest_face, closest_points = mesh.closest_faces_and_points(points)
        vert_ids, bary_coords = mesh.barycentric_coordinates_for_points(closest_points, closest_face.astype('int32'))
        bweights = self.barycentric_interpolation(self.smpl_model.lbs_weights[vert_ids.astype('int32')].cpu(),
                                                bary_coords).reshape(-1, 24)
        bweights = bweights.cuda()
        return bweights
    
    def calculate_pts_cano(self, points, nearest_smpl_weights, A, pose):
        return points, points
        # 1: SMPL inverse skinning
        batch_size = points.shape[0]
        num_joints = 24
        T = torch.matmul(nearest_smpl_weights.float(), A.view(1, num_joints, 16)) \
            .view(batch_size, 4, 4)
        homogen_coord = torch.ones([batch_size, 1]).cuda()
        points_homo = torch.cat([points, homogen_coord], dim=1)
        T = T.cpu().numpy()
        T_inv = np.linalg.inv(T)
        pts_cano = torch.matmul(torch.Tensor(T_inv).cuda(), torch.unsqueeze(points_homo, dim=-1)).squeeze()[:, :-1]
        # 2: delta_x_cano received from MLP.
        # input_embedding = self.embed_fn_fine(torch.cat([pts_cano, pose.reshape(-1).unsqueeze(0).expand(batch_size, 72), torch.zeros(10).reshape(-1).unsqueeze(0).expand(batch_size, 10)], dim=1))

        # h = input_embedding
        # for i, l in enumerate(self.deform_linears):
        #     h = self.deform_linears[i](h)
        #     h = F.relu(h)
        # delta_pts_cano = self.delta_pts_cano_linears(h)
        return pts_cano, pts_cano


    def jacobian(self, pts, theta):
        # nearest_smpl_weights = self.calculate_weights(pts)
        # pts.requires_grad = True
        # self.nearest_smpl_weights = None
        return torch.eye(3)
        # pts_smpl, pts_cano = self.calculate_pts_cano(pts, self.nearest_smpl_weights, self.A, theta)
        # jacobian = torch.stack([torch.autograd.grad([pts_cano[:, i].sum()], [pts], retain_graph=True, create_graph=True)[0] for i in range(3)], dim=-1)
        # return jacobian

    
    def cano_gradient(self, pts_cano):
        pts_grid = pts_cano.reshape(1, 1, 1, -1, 3)
        pts_grid = 2 * (pts_grid - self.bound_min) / (self.bound_max - self.bound_min) - 1
        gradient = F.grid_sample(self.gradients_grid, pts_grid, align_corners=True, padding_mode="border").detach()
        gradient = gradient.squeeze().transpose(0, 1)
        # print(torch.sqrt(torch.sum(gradient * gradient, -1)).mean())
        gradient = gradient
        return gradient
