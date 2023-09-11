from smplx import SMPL
import numpy as np
import torch
import trimesh
import skimage.metrics
from psbody.mesh import Mesh
def barycentric_interpolation(val, coords):
    """
    :param val: verts x 3 x d input matrix
    :param coords: verts x 3 barycentric weights array
    :return: verts x d weighted matrix
    """
    t = val * coords[..., np.newaxis]
    ret = t.sum(axis=1)
    return ret
reso = 128
model = SMPL(r"./smpl_models/smpl/SMPL_NEUTRAL.pkl")
theta = np.load("stand_pose.npy")
# big_poses = np.zeros((69))
# angle = 30
# big_poses[2] = np.deg2rad(angle)
# big_poses[5] = np.deg2rad(-angle)
output = model(global_orient=torch.Tensor([[np.pi/2, 0, 0]]), body_pose=torch.Tensor(theta[0,3:]).unsqueeze(0))
# output = model(global_orient=torch.Tensor([[np.pi/2, 0, 0]]), body_pose=torch.Tensor(big_poses).unsqueeze(0))
vertices = output.vertices.cpu().detach().numpy()
rot_mat = np.array(
                [[ 1.,  0.,  0.],
                [ 0.,  0., -1.],
                [ 0.,  1.,  0.]], dtype=np.float32)
vertices = np.matmul(vertices, rot_mat)
# print(vertices)
# np.save("./data/vertices.npy", vertices)
bound_min = np.min(vertices, axis=1)
bound_max = np.max(vertices, axis=1)
bound_center = (bound_min + bound_max) / 2
bound_miner = bound_center - 1.2 * (bound_center - bound_min)
bound_maxer = bound_center + 1.2 * (bound_max - bound_center)
# print(bound_miner, bound_maxer)
np.savez("./bound.npz", bound_miner=bound_miner[0], bound_maxer=bound_maxer[0])
points = np.meshgrid(
        np.linspace(bound_miner[0][0], bound_maxer[0][0], reso),
        np.linspace(bound_miner[0][1], bound_maxer[0][1], reso),
        np.linspace(bound_miner[0][2], bound_maxer[0][2], reso)
    )
points = np.stack(points, axis=-1).transpose(1,0,2,3).reshape(-1, 3)

smpl_mesh = Mesh(vertices[0], model.faces)
closest_face, closest_points = smpl_mesh.closest_faces_and_points(points)
vert_ids, bary_coords = smpl_mesh.barycentric_coordinates_for_points(closest_points, closest_face.astype('int32'))
bweights = barycentric_interpolation(model.lbs_weights[vert_ids.astype('int32')],
                                        bary_coords).reshape(reso, reso, reso, 24)
np.save("pose1.npy", bweights)