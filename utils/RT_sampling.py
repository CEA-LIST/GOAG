import open3d as o3d
import torch
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

def sample_gripper_poses(robot_name, object_pcd, r=0.01, num_samples=200):
    """
    Samples poses uniformly on the convex hull of a given point cloud.
    Each pose's x or y-axis is directed towards the hull face normal (inward).

    Args:
        robot_name (str): The name of the robot (e.g., 'allegro', 'barrett', 'shadowhand').
        object_pcd (torch.Tensor): Nx3 array of point cloud coordinates.
        r (float): Radius for sampling around the convex hull.
        num_samples (int): Number of poses to sample.

    Returns:
        poses (torch.Tensor): (num_samples, 9) A tensor of sampled poses.
    """
    device = object_pcd.device

    # Compute convex hull using Open3D for its efficient implementation
    pcd_np = object_pcd.to(torch.float32).cpu().numpy()
    pcd_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_np))
    hull, _ = pcd_o3d.compute_convex_hull()
    hull_vertices = torch.tensor(np.asarray(hull.vertices), dtype=torch.float32)
    hull_triangles = torch.tensor(np.asarray(hull.triangles), dtype=torch.long)
    sampled_points = torch.from_numpy(np.asarray(hull.sample_points_poisson_disk(num_samples).points)).float()

    # Compute face normals using vectorized operations
    v0, v1, v2 = (hull_vertices[hull_triangles[:, i]] for i in range(3))
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    face_normals /= torch.norm(face_normals, dim=1, keepdim=True)

    # Assign each sampled point to the closest face using trimesh
    trimesh_hull = trimesh.Trimesh(vertices=hull_vertices.numpy(), faces=hull_triangles.numpy(), process=False)
    _, _, face_indices = trimesh_hull.nearest.on_surface(sampled_points.numpy())
    hull_normals = face_normals[torch.tensor(face_indices, dtype=torch.long)]

    def get_arbitrary_axis(arbitrary, primary, idx, alt):
        arbitrary_axis = torch.tensor(arbitrary, dtype=torch.float32).repeat(num_samples, 1)
        mask = torch.abs(primary[:, idx]) >= 0.99
        arbitrary_axis[mask] = torch.tensor(alt, dtype=torch.float32)
        return arbitrary_axis

    # Vectorized frame creation
    if robot_name == 'allegro':
        primary_axis = -hull_normals
        arbitrary = get_arbitrary_axis([0, 1, 0], primary_axis, 1, [1, 0, 0])
        y_axis = torch.cross(primary_axis, arbitrary, dim=1)
        y_axis /= torch.norm(y_axis, dim=1, keepdim=True)
        z_axis = torch.cross(primary_axis, y_axis, dim=1)
        z_axis /= torch.norm(z_axis, dim=1, keepdim=True)
        axes = (primary_axis, y_axis, z_axis)
        translation_dir = primary_axis
        offset = -r * translation_dir
        rot_axis = primary_axis
    elif robot_name == 'shadowhand':
        primary_axis = hull_normals
        arbitrary = get_arbitrary_axis([1, 0, 0], primary_axis, 0, [0, 1, 0])
        x_axis = torch.cross(primary_axis, arbitrary, dim=1)
        x_axis /= torch.norm(x_axis, dim=1, keepdim=True)
        z_axis = torch.cross(x_axis, primary_axis, dim=1)
        z_axis /= torch.norm(z_axis, dim=1, keepdim=True)
        axes = (x_axis, primary_axis, z_axis)
        translation_dir = -primary_axis
        offset = -r * translation_dir
        rot_axis = primary_axis
    elif robot_name == 'barrett':
        primary_axis = -hull_normals
        arbitrary = get_arbitrary_axis([0, 1, 0], primary_axis, 1, [1, 0, 0])
        x_axis = torch.cross(arbitrary, primary_axis, dim=1)
        x_axis /= torch.norm(x_axis, dim=1, keepdim=True)
        y_axis = torch.cross(primary_axis, x_axis, dim=1)
        y_axis /= torch.norm(y_axis, dim=1, keepdim=True)
        axes = (x_axis, y_axis, primary_axis)
        translation_dir = primary_axis
        offset = -(r + 0.06) * translation_dir
        rot_axis = primary_axis
    else:
        raise ValueError("Not Implemented for this Robot.")

    # Random rotation around primary axis
    angle = torch.rand(num_samples, 1) * 2 * np.pi
    sin_half, cos_half = torch.sin(angle / 2), torch.cos(angle / 2)
    quat = torch.stack([
        rot_axis[:, 0] * sin_half.squeeze(),
        rot_axis[:, 1] * sin_half.squeeze(),
        rot_axis[:, 2] * sin_half.squeeze(),
        cos_half.squeeze()
    ], dim=1)
    R_rand = torch.from_numpy(Rotation.from_quat(quat.numpy()).as_matrix()).float()

    R = torch.stack(axes, dim=2)
    R = torch.matmul(R_rand, R)

    frame_xyz = sampled_points + offset
    if robot_name == 'shadowhand':
        frame_xyz = frame_xyz - 0.08 * R[:, :, 2]

    frame_rot6d = torch.cat((R[:, :, 0], R[:, :, 1]), dim=1)
    poses_6d = torch.cat((frame_xyz, frame_rot6d), dim=1)

    return poses_6d.to(device)