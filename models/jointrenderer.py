from operator import is_
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import mcubes
from icecream import ic


def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class JointRenderer:
    def __init__(self,
                 nerf,
                 sdf_smpl,
                 sdf_network,
                 deviation_network,
                 color_network,
                 body_color_network,
                 cloth_sdf_network,
                 cloth_color_network,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb,
                 extra_color=False):
        self.nerf = nerf
        self.sdf_smpl = sdf_smpl
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.body_color_network = body_color_network
        self.cloth_sdf_network = cloth_sdf_network
        self.cloth_color_network = cloth_color_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb
        self.extra_color = extra_color

    def render_core_outside(self, rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None):
        """
        Render background
        """
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

        dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
        pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)       # batch_size, n_samples, 4

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        dirs = dirs.reshape(-1, 3)

        density, sampled_color = nerf(pts, dirs)
        alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)
        alpha = alpha.reshape(batch_size, n_samples)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights,
        }

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, theta, coe, last=False, is_refine=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_smpl_sdf, pts_cano = self.sdf_smpl(pts.reshape(-1, 3), theta)
            new_smpl_sdf = new_smpl_sdf.reshape(batch_size, n_importance)

            new_body_sdf = new_smpl_sdf + coe * self.sdf_network.sdf(pts_cano.reshape(-1, 3)).reshape(batch_size, n_importance)
            new_cloth_sdf = new_body_sdf + coe * self.cloth_sdf_network.sdf(pts_cano.reshape(-1, 3)).reshape(batch_size, n_importance)

            new_sdf = torch.zeros_like(new_smpl_sdf)
            is_cloth = new_body_sdf >=  (0.00001 if not is_refine else 0.002)
            is_body = torch.logical_not(is_cloth)
            new_sdf[is_body] = new_body_sdf[is_body]
            new_sdf[is_cloth] = torch.min(new_cloth_sdf[is_cloth], new_body_sdf[is_cloth])
            
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)
        return z_vals, sdf

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    sdf_smpl,
                    sdf_network,
                    deviation_network,
                    color_network,
                    body_color_network,
                    cloth_sdf_network,
                    cloth_color_network,
                    theta,
                    coe,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0,
                    is_refine=False,
                    is_all=False):
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        smpl_sdf, pts_cano = sdf_smpl(pts, theta)

        sdf_nn_output = sdf_network(pts_cano)
        sdf_residual = sdf_nn_output[:, :1]
        body_sdf = smpl_sdf + coe * sdf_residual
        feature_vector = sdf_nn_output[:, 1:]

        # seperate cloth and body points to reduce cuda mem.
        is_cloth = (body_sdf >=  (0.00001 if not is_refine else 0.002)).squeeze(1)
        is_body = torch.logical_not(is_cloth)
        pts_cano_cloth = pts_cano[is_cloth]

        cloth_sdf_nn_output = cloth_sdf_network(pts_cano)
        cloth_sdf_residual = cloth_sdf_nn_output[:, :1]

        cloth_sdf = body_sdf + coe * cloth_sdf_residual
        
        if is_all:
            cloth_feature_vector = cloth_sdf_nn_output[:, 1:]
        else:
            cloth_feature_vector = torch.zeros_like(cloth_sdf_nn_output[:, 1:])

        gradients_cano = sdf_smpl.cano_gradient(pts_cano)
        gradients_cano_body = gradients_cano + coe * sdf_network.gradient(pts_cano).squeeze()
        gradients_cano_cloth = gradients_cano_body + coe * cloth_sdf_network.gradient(pts_cano).squeeze()


        gradients_cano_body = gradients_cano_body / (torch.norm(gradients_cano_body, dim=-1, keepdim=True) + 1e-7)
        gradients_cano_cloth = gradients_cano_cloth / (torch.norm(gradients_cano_cloth, dim=-1, keepdim=True) + 1e-7)

        jacobian = sdf_smpl.jacobian(pts.clone(), theta)
        gradients_body = torch.matmul(jacobian, gradients_cano_body.unsqueeze(-1)).squeeze()
        gradients_cloth = torch.matmul(jacobian, gradients_cano_cloth.unsqueeze(-1)).squeeze()

        raw_color_all = torch.zeros((batch_size * n_samples, 3), device=pts_cano.device)

        real_head_color = color_network(pts_cano, gradients_cano_body, dirs, feature_vector)
        real_body_color = body_color_network(pts_cano, gradients_cano_body, dirs, torch.zeros_like(feature_vector))

        head_center = torch.Tensor([-0.0055,  0.6681,  0.3625]).cuda()
        dis_to_head = torch.linalg.norm(pts - head_center, ord = 2, dim=-1)
        ratio_of_head = torch.ones_like(dis_to_head)
        ratio_of_head[dis_to_head > 0.15] = 0
        ratio_of_head = ratio_of_head.unsqueeze(-1)

        # inner model blended color
        raw_color_body = real_head_color * ratio_of_head + real_body_color * (1 - ratio_of_head)

        raw_color_cloth = cloth_color_network(pts_cano_cloth, gradients_cano_cloth[is_cloth], dirs[is_cloth], cloth_feature_vector[is_cloth])
        raw_color_all[is_body] = raw_color_body[is_body]

        true_cos_body = (dirs * gradients_body).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        # iter_cos_body = -(F.relu(-true_cos_body * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
        #              F.relu(-true_cos_body) * cos_anneal_ratio)  # always non-positive
        iter_cos_body = -F.relu(-true_cos_body)
        # Estimate signed distances at section points
        estimated_next_sdf_body = body_sdf + iter_cos_body * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf_body = body_sdf - iter_cos_body * dists.reshape(-1, 1) * 0.5

        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        prev_cdf_body = torch.sigmoid(estimated_prev_sdf_body * inv_s)
        next_cdf_body = torch.sigmoid(estimated_next_sdf_body * inv_s)

        p_body = prev_cdf_body - next_cdf_body
        c_body = prev_cdf_body

        alpha_body = ((p_body + 1e-5) / (c_body + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        true_cos_cloth = (dirs * gradients_cloth).sum(-1, keepdim=True)
        iter_cos_cloth = -F.relu(-true_cos_cloth)

        # Estimate signed distances at section points
        estimated_next_sdf_cloth = cloth_sdf + iter_cos_cloth * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf_cloth = cloth_sdf - iter_cos_cloth * dists.reshape(-1, 1) * 0.5

        prev_cdf_cloth = torch.sigmoid(estimated_prev_sdf_cloth * inv_s)
        next_cdf_cloth = torch.sigmoid(estimated_next_sdf_cloth * inv_s)

        p_cloth = prev_cdf_cloth - next_cdf_cloth
        c_cloth = prev_cdf_cloth

        alpha_cloth = ((p_cloth + 1e-5) / (c_cloth + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)
        
        alpha = torch.zeros_like(alpha_body)
        is_body_ = is_body.reshape(batch_size, n_samples)
        is_cloth_ = is_cloth.reshape(batch_size, n_samples)
        alpha[is_body_] = alpha_body[is_body_]
        alpha[is_cloth_] = alpha_body[is_cloth_] + alpha_cloth[is_cloth_] - alpha_body[is_cloth_] * alpha_cloth[is_cloth_]

        alpha_body_ = alpha_body[is_cloth_].unsqueeze(1)
        alpha_cloth_ = alpha_cloth[is_cloth_].unsqueeze(1)
        
        body_color_weight = alpha_body_ / (alpha_body_ + alpha_cloth_)
        cloth_color_weight = alpha_cloth_ / (alpha_body_ + alpha_cloth_)
        raw_color_all[is_cloth] = body_color_weight * raw_color_body[is_cloth] + cloth_color_weight * raw_color_cloth
        sampled_color = torch.zeros_like(raw_color_all)
        sampled_color[is_body] = raw_color_body[is_body]
        sampled_color[is_cloth] = body_color_weight * raw_color_body[is_cloth] + cloth_color_weight * raw_color_cloth
        raw_color_all = raw_color_all.reshape(batch_size, n_samples, 3)
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)

        gradients = torch.zeros_like(gradients_cano, dtype=gradients_body.dtype)
        gradients[is_body] = gradients_body[is_body]
        gradients[is_cloth] = (body_color_weight * gradients_body[is_cloth] + cloth_color_weight * gradients_cloth[is_cloth]).type_as(gradients)

        gradients = gradients / (torch.norm(gradients, dim=-1, keepdim=True) + 1e-7)

        pts_norm = torch.linalg.norm(pts_cano, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.2).float().detach()
        relax_inside_sphere = (pts_norm < 1.4).float().detach()

        # Render with background
        if background_alpha is not None:
            alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
            sampled_color = sampled_color * inside_sphere[:, :, None] +\
                            background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
            sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        body_weights = torch.zeros_like(weights)
        body_weights[is_body_] = 1 * weights[is_body_]
        body_weights[is_cloth_] = body_color_weight.squeeze(1) * weights[is_cloth_]
        
        cloth_weights = torch.zeros_like(weights)
        cloth_weights[is_body_] = 0
        cloth_weights[is_cloth_] = cloth_color_weight.squeeze(1) * weights[is_cloth_]

        weights_sum = weights.sum(dim=-1, keepdim=True)
        cloth_weights_sum = cloth_weights.sum(dim=-1, keepdim=True)
        body_weights_sum = body_weights.sum(dim=-1, keepdim=True)
        cloth_weights_sum = cloth_weights_sum[weights_sum > 0.5] / weights_sum[weights_sum > 0.5]
        body_weights_sum = body_weights_sum[weights_sum > 0.5] / weights_sum[weights_sum > 0.5]
        cloth_weights_sum = cloth_weights_sum.clip(5e-2, 1-5e-2)
        body_weights_sum = body_weights_sum.clip(5e-2, 1-5e-2)
        ray_entropy = (-torch.log2(cloth_weights_sum) * cloth_weights_sum - torch.log2(body_weights_sum) * body_weights_sum) / cloth_weights_sum.shape[0]
        color = (sampled_color * weights[:, :, None]).sum(dim=1)

        if background_rgb is not None:    # Fixed background, usually black
            color = color + background_rgb * (1.0 - weights_sum)

        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients_cano.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        return {
            'weight_entropy': ray_entropy.sum(),
            'sdf_residual': cloth_sdf_residual[is_cloth],
            'color': color,
            'sdf': body_sdf,
            'dists': dists,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'cdf': c_body.reshape(batch_size, n_samples),
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere
        }

    def render(self, rays_o, rays_d, near, far, data, coe, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0, is_refine=False, is_all=False):
        self.sdf_smpl.eval()
        self.sdf_network.eval()
        self.color_network.eval()
        self.cloth_sdf_network.train()
        self.cloth_color_network.train()
        self.sdf_smpl.A = data["A"]
        self.sdf_smpl.vertices = data["vertices"]
        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples   # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]

        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside)

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

            if self.n_outside > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]])
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        if self.n_outside > 0:
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

        background_alpha = None
        background_sampled_color = None

        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                smpl_sdf, pts_cano = self.sdf_smpl(pts.reshape(-1, 3), data["pose"])
                smpl_sdf = smpl_sdf.reshape(batch_size, self.n_samples)
                body_sdf = smpl_sdf + coe * self.sdf_network.sdf(pts_cano.reshape(-1, 3)).reshape(batch_size, self.n_samples)
                cloth_sdf = body_sdf + coe * self.cloth_sdf_network.sdf(pts_cano.reshape(-1, 3)).reshape(batch_size, self.n_samples)
                sdf = torch.zeros_like(smpl_sdf)
                is_cloth = body_sdf >=  (0.00001 if not is_refine else 0.002)
                is_body = torch.logical_not(is_cloth)
                sdf[is_body] = body_sdf[is_body]
                sdf[is_cloth] = torch.min(cloth_sdf[is_cloth], body_sdf[is_cloth])
                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2**i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  data["pose"],
                                                  coe,
                                                  last=(i + 1 == self.up_sample_steps),
                                                  is_refine=is_refine)

            n_samples = self.n_samples + self.n_importance

        # Background model
        if self.n_outside > 0:
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)

            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']

        # Render core
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    self.sdf_smpl,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    self.body_color_network,
                                    self.cloth_sdf_network,
                                    self.cloth_color_network,
                                    data["pose"],
                                    coe,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    cos_anneal_ratio=cos_anneal_ratio,
                                    is_refine=is_refine,
                                    is_all=is_all)

        color_fine = ret_fine['color']
        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)
        gradients = ret_fine['gradients']

        return {
            'weight_entropy': ret_fine['weight_entropy'],
            'sdf_residual': ret_fine['sdf_residual'],
            'color_fine': color_fine,
            's_val': s_val,
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'weights': weights,
            'gradients': gradients,
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere']
        }
