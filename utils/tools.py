import numpy as np
import random
from scipy.spatial import KDTree, ConvexHull
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Tuple
import colorsys
from utils_data.custom_bps import bps_torch, compute_aligned_dist_v2
from utils.rot6d import * 
from utils.constants import SHADOWHAND_FOREARM2WRIST, SHADOWHAND_WRIST2PALM

# Consistent color map for integer labels between 0 and 21
label_range = np.arange(0, 21)
label_to_color = {
    label: colorsys.hsv_to_rgb((i) / (len(label_range)), torch.rand(1).item() * 0.5 + 0.5, torch.rand(1).item() * 0.5 + 0.5)
    for i, label in enumerate(label_range)
}
def mycustom_cmap(tensor):
    # tensor: 1D torch tensor of ints in [0, 21]
    arr = tensor.cpu().numpy()
    colors_arr = np.array([label_to_color[int(x)] for x in arr])
    return colors_arr


def set_global_seed(seed=42):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_transformation_matrix(translation: torch.Tensor, rotation: torch.Tensor, deg=False) -> torch.Tensor:
    """
    Create a transformation matrix from translation and rotation.

    Args:
        translation (torch.Tensor): Translation vector of shape (3,).
        rotation (torch.Tensor): Rotation angles in degrees of shape (3,).

    Returns:
        torch.Tensor: Transformation matrix of shape (4, 4).
    """

    batch_size = translation.shape[0] if translation.dim() > 1 else 1

    # Convert rotation angles from degrees to radians
    if deg:
        rotation = torch.deg2rad(rotation)

    # Handle batching
    if batch_size == 1 and translation.dim() == 1:
        # Single transformation case
        # Compute rotation matrices for each axis
        Rx = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(rotation[0]), -torch.sin(rotation[0])],
            [0, torch.sin(rotation[0]), torch.cos(rotation[0])]
        ], device=translation.device)

        Ry = torch.tensor([
            [torch.cos(rotation[1]), 0, torch.sin(rotation[1])],
            [0, 1, 0],
            [-torch.sin(rotation[1]), 0, torch.cos(rotation[1])]
        ], device=translation.device)

        Rz = torch.tensor([
            [torch.cos(rotation[2]), -torch.sin(rotation[2]), 0],
            [torch.sin(rotation[2]), torch.cos(rotation[2]), 0],
            [0, 0, 1]
        ], device=translation.device)

        # Combined rotation: Rz @ Ry @ Rx (ZYX convention)
        R = Rx @ Ry @ Rz

        # Create the transformation matrix
        T = torch.eye(4, device=translation.device)
        T[:3, :3] = R
        T[:3, 3] = translation

        return T
    else:
        # Batched transformation case
        # Create identity matrices for batch
        Rx = torch.eye(3, device=translation.device).unsqueeze(0).repeat(batch_size, 1, 1)
        Ry = torch.eye(3, device=translation.device).unsqueeze(0).repeat(batch_size, 1, 1)
        Rz = torch.eye(3, device=translation.device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Fill rotation matrices
        cos_x, sin_x = torch.cos(rotation[:, 0]), torch.sin(rotation[:, 0])
        Rx[:, 1, 1] = cos_x
        Rx[:, 1, 2] = -sin_x
        Rx[:, 2, 1] = sin_x
        Rx[:, 2, 2] = cos_x

        cos_y, sin_y = torch.cos(rotation[:, 1]), torch.sin(rotation[:, 1])
        Ry[:, 0, 0] = cos_y
        Ry[:, 0, 2] = sin_y
        Ry[:, 2, 0] = -sin_y
        Ry[:, 2, 2] = cos_y

        cos_z, sin_z = torch.cos(rotation[:, 2]), torch.sin(rotation[:, 2])
        Rz[:, 0, 0] = cos_z
        Rz[:, 0, 1] = -sin_z
        Rz[:, 1, 0] = sin_z
        Rz[:, 1, 1] = cos_z

        # Combined rotation: Rz @ Ry @ Rx (ZYX convention)
        R = torch.bmm(torch.bmm(Rx, Ry), Rz)

        # Create the transformation matrices
        T = torch.eye(4, device=translation.device).unsqueeze(0).repeat(batch_size, 1, 1)
        T[:, :3, :3] = R
        T[:, :3, 3] = translation

        return T


def get_transformation_matrix_from_rot6d(translation: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """
    Create a transformation matrix from translation and rotation.

    Args:
        translation (torch.Tensor): Translation vector of shape (3,).
        rotation (torch.Tensor): Rotation angles in degrees of shape (3,).

    Returns:
        torch.Tensor: Transformation matrix of shape (4, 4).
    """

    batch_size = translation.shape[0] if translation.dim() > 1 else 1

    # Handle single transformation case
    if batch_size == 1 and translation.dim() == 1:
        rotation = rotation.unsqueeze(0)
        translation = translation.unsqueeze(0)

    u = rotation[:, 0:3]
    v = rotation[:, 3:6]
    u = F.normalize(u, p=2, dim=1)
    w = torch.cross(u, v, dim=1)
    w = F.normalize(w, p=2, dim=1)
    v = torch.cross(w, u, dim=1)
    
    R = torch.stack([u, v, w], dim=-1)
    T = torch.eye(4, device=translation.device).unsqueeze(0).repeat(batch_size, 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = translation
    
    return T


def rotate_pcd(pcd, x, y, z, normals=None):
    """Rotate a point cloud around the x, y, z axes.
    Args:
        pcd (torch.tensor): (B, N, 3) or (B, N, 4), point cloud
        x (float): (B, 1) rotation angle around the x-axis in radians
        y (float): (B, 1) rotation angle around the y-axis in radians
        z (float): (B, 1) rotation angle around the z-axis in radians
    Returns:
        torch.tensor: Rotated point cloud of shape (B, N, 3) or (B, N, 4)
    """
    
    # Handle both single and batch cases
    if x.dim() == 0:
        # Single case - add batch dimension
        batch_size = 0
        rotation = torch.stack([x, y, z]).unsqueeze(0)  # (1, 3)
    else:
        # Batch case
        batch_size = x.shape[0]
        rotation = torch.stack([x, y, z], dim=1)  # (B, 3)

    T = get_transformation_matrix(
        translation=torch.zeros(batch_size, 3, device=pcd.device),  # No translation
        rotation=rotation,  # Rotation angles
        deg=False  # Angles are in radians
    )

    if batch_size == 0:
        rotation_matrix = T[:3, :3]  # (3, 3)
        # Translate pcd to origin, rotate, then translate back
        rotated_pcd = pcd.clone()
        rotated_pcd[:, :3] = pcd[:, :3] @ rotation_matrix.T

        if normals is not None:
            rotated_normals = normals @ rotation_matrix.T
            return rotated_pcd, rotated_normals
        
        return rotated_pcd
    else:
        rotation_matrix = T[:, :3, :3]  # (B, 3, 3)
        # Batched rotation
        rotated_pcd = pcd[:batch_size].clone()
        rotated_pcd[:, :, :3] = torch.bmm(pcd[:batch_size, :, :3], rotation_matrix.transpose(-2, -1))

        if normals is not None:
            rotated_normals = torch.bmm(normals[:batch_size], rotation_matrix.transpose(-2, -1))
            return rotated_pcd, rotated_normals
        
        return rotated_pcd


def rotate_pcd_6d(pcd, rot_6d, normals=None):
    """
    Rotate a point cloud using a 6D rotation representation.

    Args:
        pcd (torch.tensor): (B, N, 3) or (N, 3), point cloud
        rot_6d (torch.tensor): (B, 6) or (6,), 6D rotation vector
        normals (torch.tensor, optional): (B, N, 3) or (N, 3), point normals.

    Returns:
        tuple or torch.tensor: Rotated point cloud and optionally rotated normals.
    """
    
    # Handle both single and batch cases
    is_batched = pcd.dim() == 3

    if not is_batched:
        pcd = pcd.unsqueeze(0)
        rot_6d = rot_6d.unsqueeze(0)
        if normals is not None:
            normals = normals.unsqueeze(0)

    batch_size = rot_6d.shape[0]

    # Get the 4x4 transformation matrix from the 6D rotation
    T = get_transformation_matrix_from_rot6d(
        translation=torch.zeros(batch_size, 3, device=pcd.device),
        rotation=rot_6d
    )

    # Extract the 3x3 rotation matrix from the transformation matrix
    rotation_matrix = T[:, :3, :3]  # (B, 3, 3)

    # Rotate the point cloud using batched matrix multiplication
    rotated_pcd = pcd[:batch_size].clone()
    rotated_pcd[:, :, :3] = torch.bmm(pcd[:batch_size, :, :3], rotation_matrix.transpose(-2, -1))

    # Rotate normals if they are provided
    if normals is not None:
        rotated_normals = torch.bmm(normals[:batch_size], rotation_matrix.transpose(-2, -1))
        # Squeeze if original input was not batched
        if not is_batched:
            return rotated_pcd.squeeze(0), rotated_normals.squeeze(0)
        return rotated_pcd, rotated_normals

    # Squeeze if original input was not batched
    if not is_batched:
        return rotated_pcd.squeeze(0)
    
    return rotated_pcd


def move_pcd(xyz, rot, pcd, normals=None):
    if not isinstance(xyz, torch.Tensor):
        xyz = torch.tensor(xyz, device=pcd.device, dtype=pcd.dtype)
    if not isinstance(rot, torch.Tensor):
        rot = torch.tensor(rot, device=pcd.device, dtype=pcd.dtype)

    if normals is not None:
        pcd, normals = rotate_pcd(pcd, x=rot[:, 0], y=rot[:, 1], z=rot[:, 2], normals=normals)
        pcd[:, :, :3] += xyz.unsqueeze(1)
        return pcd, normals
    else:
        pcd = rotate_pcd(pcd, x=rot[:, 0], y=rot[:, 1], z=rot[:, 2])
        pcd[:, :, :3] += xyz.unsqueeze(1)
        return pcd


def move_pcd_6d(xyz, rot, pcd, normals=None):
    if not isinstance(xyz, torch.Tensor):
        xyz = torch.tensor(xyz, device=pcd.device, dtype=pcd.dtype)
    if not isinstance(rot, torch.Tensor):
        rot = torch.tensor(rot, device=pcd.device, dtype=pcd.dtype)

    if normals is not None:
        pcd, normals = rotate_pcd_6d(pcd, rot, normals=normals)
        pcd[:, :, :3] += xyz.unsqueeze(1)
        return pcd, normals
    else:
        pcd = rotate_pcd_6d(pcd, rot)
        pcd[:, :, :3] += xyz.unsqueeze(1)
        return pcd

    
def compute_6d_gripper_pose(object_xyz, object_rot):
    """
    Compute the gripper pose in the object frame.

    Args:
        object_xyz (torch.Tensor): (3,) translation of the object.
        object_rot (torch.Tensor): (3,) rotation angles (degrees) for x, y, z axes.

    Returns:
        torch.Tensor: (9,) pose vector [t_x, t_y, t_z, u_x, u_y, u_z, v_x, v_y, v_z]
    """
    T = get_transformation_matrix(
        translation=object_xyz,
        rotation=object_rot
    )

    # Inverse transformation matrix to express the gripper in the object frame
    T_hand_in_o = torch.inverse(T)

    t_inv = T_hand_in_o[:3, 3]
    R_inv = T_hand_in_o[:3, :3]

    # For u, v rotation, let's use the first two columns of R_inv as u, v
    u = R_inv[:, 0]
    v = R_inv[:, 1]

    # Concatenate to form the output tensor
    return torch.cat([t_inv, u, v], dim=0)


def compute_pose_in_world_frame(t, R):
    
    T = get_transformation_matrix(
        translation=t,
        rotation=R,
    )

    t_w = T[:, :3, 3]  # (B, 3)
    R_w = matrix_to_euler(T[:, :3, :3])  # (B, 3, 3)

    return t_w, R_w


def compute_object_pose(gripper_xyz, gripper_rot):
    """
    Compute the object pose in the gripper frame.
    Args:
        gripper_xyz (torch.Tensor): (B, 3) translation of the object.
        gripper_rot (torch.Tensor): (B, 3) rotation angles (degrees) for x, y, z axes.

    Returns:
        torch.Tensor: (B, 3) pose vector [t_x, t_y, t_z, u_x, u_y, u_z, v_x, v_y, v_z]
    """

    T_w2g = get_transformation_matrix(
        translation=gripper_xyz,
        rotation=gripper_rot
    ) # (B, 4, 4)

    # Inverse transformation matrix to express the gripper in the object frame
    T_g2o = torch.inverse(T_w2g)  # (B, 4, 4)

    t_inv = T_g2o[:, :3, 3]  # (B, 3)
    R_inv = T_g2o[:, :3, :3]  # (B, 3, 3)

    rot_euler = matrix_to_euler(R_inv)  # (B, 3)
    # Concatenate to form the output tensor
    pos_euler = torch.cat([t_inv, rot_euler], dim=1)  # (B, 6)
    return pos_euler


def compute_object_pose_6d(gripper_xyz, gripper_rot):
    """
    Compute the object pose in the gripper frame.
    Args:
        gripper_xyz (torch.Tensor): (B, 3) translation of the object.
        gripper_rot (torch.Tensor): (B, 6) 6D rotation vector.

    Returns:
        torch.Tensor: (B, 6) pose vector [t_x, t_y, t_z, u_x, u_y, u_z, v_x, v_y, v_z]
    """

    T_w2g = get_transformation_matrix_from_rot6d(
        translation=gripper_xyz,
        rotation=gripper_rot
    ) # (B, 4, 4)

    # Inverse transformation matrix to express the gripper in the object frame
    T_g2o = torch.inverse(T_w2g)  # (B, 4, 4)

    t_inv = T_g2o[:, :3, 3]  # (B, 3)
    R_inv = T_g2o[:, :3, :3]  # (B, 3, 3)

    rot6d = matrix_to_rot6d(R_inv)  # (B, 6)

    # Concatenate to form the output tensor
    pos_6d = torch.cat([t_inv, rot6d], dim=1)  # (B, 9)
    return pos_6d


def pad_batch_list_to_0(batch_list):
    # batch_contact_points = torch.cat([torch.stack(sample, dim=0) for sample in all_contact_points], dim=0)      # (n_s * B, N, 3)

    # Flatten the list to get all pcds, then find the max size
    all_data = [p for sublist in batch_list for p in sublist]
    max_data_size = max(p.shape[0] for p in all_data)
        
    padded_data = []
    for data_sample in batch_list:
        padded_sample = []
        for d in data_sample:
            if d.shape[0] < max_data_size:
                if d.dim() <= 1:
                    padding = torch.zeros((max_data_size - d.shape[0]), dtype=d.dtype, device=d.device)
                else:
                    padding = torch.zeros((max_data_size - d.shape[0], d.shape[1]), dtype=d.dtype, device=d.device)
                padded = torch.cat([d, padding], dim=0)
            else:
                padded = d
            padded_sample.append(padded)
        padded_data.append(torch.stack(padded_sample, dim=0))
    padded_data = torch.stack(padded_data, dim=0)

    return padded_data


def pad_batch_list_copy(batch_list, n=None):
    all_data = [p for sublist in batch_list for p in sublist if p.numel() > 0]
    if all_data == []:
        return None
    max_data_size = max(p.shape[0] for p in all_data)
    max_data_size = max(max_data_size, n) if n is not None else max_data_size  # Use n if provided, otherwise use max size
    
    padded_data = []
    for data_sample in batch_list:
        padded_sample = []
        for d in data_sample:
            if d.numel() == 0:  # Check if d is empty tensor
                # Create empty tensor with correct shape but no values (uninitialized)
                padded = torch.empty((max_data_size, d.shape[1] if d.dim() > 1 else 1), dtype=d.dtype, device=d.device)
            elif d.shape[0] < max_data_size:
                # Repeat data to reach max_data_size by cycling through existing data
                repeat_times = (max_data_size + d.shape[0] - 1) // d.shape[0]  # Ceiling division
                repeated_d = d.repeat(repeat_times, 1) if d.dim() > 1 else d.repeat(repeat_times)
                padded = repeated_d[:max_data_size]
            else:
                padded = d
            padded_sample.append(padded)
        padded_data.append(torch.stack(padded_sample, dim=0))
    padded_data = torch.stack(padded_data, dim=0)
    return padded_data


def convert_shadow_pose(q, rot_type='euler'):
    """For ShadowHand: Convert a 6+24 DOF pose to a 6+22 DOF pose by removing first two joints and change the base position."""
    B = q.shape[0]

    if rot_type == 'euler':
        t = q[:, :3]  # (B, 3)
        r_eul = q[:, 3:6]  # (B, 3)
        q_wrj1, q_wrj2 = q[:, 7], q[:, 6]  # (B, 1)
    else:
        t = q[:, :3]  # (B, 3)
        r_eul = rot6d_to_euler(q[:, 3:9])  # (B, 3)
        q_wrj1, q_wrj2 = q[:, 9], q[:, 10]  # (B, 1)

    t_forearm2wrist = SHADOWHAND_FOREARM2WRIST.unsqueeze(0).repeat(B, 1).to(q.device)
    t_wrist2palm = SHADOWHAND_WRIST2PALM.unsqueeze(0).repeat(B, 1).to(q.device)

    T_world2forearm = get_transformation_matrix(
        translation=t,  # (B, 3)
        rotation=r_eul,  # (B, 3)
    )  # (B, 4, 4)
    T_forearm2wrj2 = get_transformation_matrix(
        translation=t_forearm2wrist,  # (B, 3)
        rotation=torch.zeros_like(r_eul, device=q.device),  # (B, 3)
    )  # (B, 4, 4)

    # 2. T_WRJ2_rotation(q_WRJ2): Rotation for WRJ2 around its axis (Y-axis)
    #    <axis xyz="0 1 0"/>
    cos_q2 = torch.cos(q_wrj2) # (B, 1)
    sin_q2 = torch.sin(q_wrj2) # (B, 1)

    # Construct rotation matrix for Y-axis rotation (B, 3, 3)
    R_WRJ2_q2 = torch.zeros(B, 3, 3, dtype=torch.float32, device=q_wrj2.device)
    R_WRJ2_q2[:, 0, 0] = cos_q2.squeeze()
    R_WRJ2_q2[:, 0, 2] = sin_q2.squeeze()
    R_WRJ2_q2[:, 1, 1] = 1.0
    R_WRJ2_q2[:, 2, 0] = -sin_q2.squeeze()
    R_WRJ2_q2[:, 2, 2] = cos_q2.squeeze()

    T_WRJ2_rotation = get_transformation_matrix(
        translation=torch.zeros(B, 3, device=q_wrj2.device),  # No translation
        rotation=matrix_to_euler(R_WRJ2_q2),
    )  # (B, 4, 4)

    # 3. T_WRJ2_out_to_WRJ1_origin: Constant transform from WRJ2 child (wrist) to WRJ1 joint origin
    #    This is from WRJ1's <origin rpy="0 0 0" xyz="0 0 0.034"/> relative to its parent "wrist"
    T_WRJ2_out_to_WRJ1_origin = get_transformation_matrix(
        translation=t_wrist2palm,  # (B, 3)
        rotation=torch.zeros_like(r_eul, device=q.device),  # (B, 3)
    )  # (B, 4, 4)

    # 4. T_WRJ1_rotation(q_WRJ1): Rotation for WRJ1 around its axis (X-axis)
    #    <axis xyz="1 0 0"/>
    cos_q1 = torch.cos(q_wrj1) # (B, 1)
    sin_q1 = torch.sin(q_wrj1) # (B, 1)

    # Construct rotation matrix for X-axis rotation (B, 3, 3)
    R_WRJ1_q1 = torch.zeros(B, 3, 3, dtype=torch.float32, device=q_wrj1.device)
    R_WRJ1_q1[:, 0, 0] = 1.0
    R_WRJ1_q1[:, 1, 1] = cos_q1.squeeze()
    R_WRJ1_q1[:, 1, 2] = -sin_q1.squeeze()
    R_WRJ1_q1[:, 2, 1] = sin_q1.squeeze()
    R_WRJ1_q1[:, 2, 2] = cos_q1.squeeze()

    T_WRJ1_rotation = get_transformation_matrix(
        translation=torch.zeros(B, 3, device=q_wrj1.device),  # No translation
        rotation=matrix_to_euler(R_WRJ1_q1),  # (B, 3, 3)
    )  # (B, 4, 4)

    # Transformation from forearm to the 'wrist' link (which is the child of WRJ2)
    # This includes the fixed offset and the rotation of WRJ2.
    T_forearm2wrist = torch.bmm(T_forearm2wrj2, T_WRJ2_rotation)  # (B, 4, 4)

    # Transformation from the 'wrist' link to the 'palm' link (which is the child of WRJ1)
    # This includes the fixed offset and the rotation of WRJ1.
    T_wrist2palm = torch.bmm(T_WRJ2_out_to_WRJ1_origin, T_WRJ1_rotation)  # (B, 4, 4)

    # Final calculation for T_world2palm
    T_world2wrist = torch.bmm(T_world2forearm, T_forearm2wrist)  # (B, 4, 4)
    T_world2palm = torch.bmm(T_world2wrist, T_wrist2palm)  # (B, 4, 4)

    new_t = T_world2palm[:, :3, 3]  # (B, 3)
    new_r_eul = matrix_to_euler(T_world2palm[:, :3, :3])  # (B, 3)

    
    if rot_type == 'euler':
        q_22dof = torch.zeros((B, 6+22), device=q.device, dtype=q.dtype)
        q_22dof[:, :3] = new_t
        q_22dof[:, 3:6] = new_r_eul
        q_22dof[:, 6:] = q[:, 8:]  # Copy last 22 DOFs (skip joints 7 and 8)
    else:
        q_22dof = torch.zeros((B, 9+22), device=q.device, dtype=q.dtype)
        q_22dof[:, :3] = new_t
        q_22dof[:, 3:9] = euler_to_rot6d(new_r_eul)
        q_22dof[:, 9:] = q[:, 11:]  # Copy last 22 DOFs (skip joints 9 and 10)
    return q_22dof


def save_tensor_pcd_as_ply(tensor, filename):
    """Save a tensor as a ply file.

    Args:
        tensor (torch.tensor): (N, 3+)
        filename (string): path to save the ply file
    """
    if 'o3d' not in globals():
        import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(tensor[:, :3].cpu().numpy())
    o3d.io.write_point_cloud(filename, pcd)


def find_key_by_item(dictionary, item):
    """
    Find the key associated with a given item in the dictionary.

    Parameters:
    - dictionary: The dictionary to search.
    - item: The item to find the key for.

    Returns:
    - The key associated with the item, or None if the item is not found.
    """
    for key, items in dictionary.items():
        if item in items:
            return key
    return None


def count_occurences(list):
    """Count the occurrences of each value in the list

    Args:
        list (_list_)

    Returns:
        _dict_: Dict of each values occurences
    """
    occurences = {}
    for item in list:
        if item not in occurences:
            occurences[item] = 1
        else:
            occurences[item] += 1
    return occurences


def farthest_point_sampling_with_labels(point_cloud, num_points=4096):
    """
    :param point_cloud: (N, 4), point cloud with link index
    :param num_points: int, number of sampled points
    :return: ((N, 3) or (N, 4), list), sampled point cloud (numpy) & index
    """
    point_cloud_origin = point_cloud
    if point_cloud.shape[1] == 4:
        point_cloud = point_cloud[:, :3]

    # Perform farthest point sampling
    selected_indices = [0]
    distances = torch.norm(point_cloud - point_cloud[selected_indices[-1]], dim=1)
    for i in tqdm(range(num_points - 1), desc="Farthest Point Sampling"):
        farthest_point_idx = torch.argmax(distances)
        selected_indices.append(farthest_point_idx)
        new_distances = torch.norm(point_cloud - point_cloud[farthest_point_idx], dim=1)
        distances = torch.min(distances, new_distances)
    sampled_point_cloud = point_cloud_origin[selected_indices]

    # Convert tensors to numpy arrays for use with KDTree
    point_cloud_np = point_cloud.cpu().numpy()
    selected_points_np = point_cloud[selected_indices].cpu().numpy()
    
    print("Building KDTree...", end='\r')
    # Build a KDTree for the original point cloud
    kdtree = KDTree(selected_points_np)
    # Query the KDTree for the closest points
    _, indices = kdtree.query(point_cloud_np, k=1)
    print("Building KDTree... Done !")
    
    # Collect labels for each selected index
    labels_per_sampled_point = []
    for idx in tqdm(range(len(selected_indices)), desc="Collecting Labels"):
        # Find all points for which this selected point is the closest
        is_closest = indices == idx
        # Collect labels (fourth element) for these points
        labels = torch.unique(point_cloud_origin[is_closest, 3])
        labels_per_sampled_point.append(labels)

    return sampled_point_cloud, labels_per_sampled_point


def farthest_point_sampling(point_cloud, num_points=4096):
    """
    :param point_cloud: (N, 3) or (N, 3+), point cloud (with link index, normals, etc.)
    :param num_points: int, number of sampled points
    :return: ((N, 3) or (N, 3+), list), sampled point cloud (numpy) & index
    """
    point_cloud_origin = point_cloud
    if point_cloud.shape[1] > 3:
        point_cloud = point_cloud[:, :3]

    selected_indices = [0]
    distances = torch.norm(point_cloud - point_cloud[selected_indices[-1]], dim=1)
    for i in range(num_points - 1):
        farthest_point_idx = torch.argmax(distances)
        selected_indices.append(farthest_point_idx)
        new_distances = torch.norm(point_cloud - point_cloud[farthest_point_idx], dim=1)
        distances = torch.min(distances, new_distances)
    sampled_point_cloud = point_cloud_origin[selected_indices]

    return sampled_point_cloud, selected_indices


def farthest_point_sampling_batch(point_cloud, num_points=512):
    """
    :param point_cloud: (B, N, 3) or (B, N, 4), batched point cloud (with optional link index)
    :param num_points: int, number of sampled points
    :return: ((B, num_points, 3) or (B, num_points, 4), list of lists), sampled point clouds & indices
    """
    batch_size, num_original_points, dim = point_cloud.shape
    point_cloud_origin = point_cloud
    if dim == 4:
        point_cloud = point_cloud[:, :, :3]
    else:
        point_cloud = point_cloud

    selected_indices = torch.zeros(batch_size, num_points, dtype=torch.long, device=point_cloud.device)

    # Initialize the first point to be the first in each batch
    selected_indices[:, 0] = 0
    print("selected_indices: ", selected_indices.shape)
    distances = torch.norm(point_cloud - point_cloud[:, 0].unsqueeze(1), dim=2)
    print("distances: ", distances.shape)
    for i in range(1, num_points):
        farthest_point_idx = torch.argmax(distances, dim=1)
        print("farthest_point_idx: ", farthest_point_idx)
        selected_indices[:, i] = farthest_point_idx[:]
        print("selected_indices: ", selected_indices.shape)
        print(point_cloud[:, selected_indices[:, i]].unsqueeze(1).shape)
        new_distances = torch.norm(point_cloud - point_cloud[:, selected_indices[:, i]].unsqueeze(1), dim=2)
        print("new_distances: ", new_distances.shape)
        distances = torch.min(distances, new_distances)
    sampled_point_cloud = point_cloud_origin[torch.arange(batch_size).unsqueeze(1), selected_indices]

    return sampled_point_cloud


def get_batch_joint_samples(joint_ranges, batch_size, start_idx, device):
    """Generate a batch of joint configurations starting from index `start_idx`.

    Args:
        joint_ranges (_type_): _description_
        batch_size (_type_): _description_
        start_idx (_type_): _description_
        device (_type_): _description_

    Returns:
        _type_: _description_
    """
    indices = torch.arange(start_idx, start_idx + batch_size, device=device)
    
    # Compute the total number of possibilities per joint
    num_options = torch.tensor([len(j) for j in joint_ranges], device=device)

    # Convert linear index to multi-dimensional indices
    multi_indices = torch.zeros((batch_size, len(joint_ranges)), dtype=torch.long, device=device)
    for i in reversed(range(len(joint_ranges))):
        multi_indices[:, i] = indices % num_options[i]
        indices = torch.div(indices, num_options[i], rounding_mode='trunc')

    # Retrieve the actual joint values
    q_batch = torch.stack([joint_ranges[i][multi_indices[:, i]] for i in range(len(joint_ranges))], dim=1)
    
    # Convert the array to a PyTorch tensor
    pos_carte = torch.tensor([0, 0, 0, 1, 0, 0, 0, 1, 0], device=q_batch.device, dtype=q_batch.dtype) 

    # Repeat prefix for each row and concatenate
    prefix_batch = pos_carte.repeat(q_batch.shape[0], 1)  # Shape: [batch_size, 9]
    q_batch = torch.cat([prefix_batch, q_batch], dim=1)  # Concatenate along dim=1

    return q_batch


def center_grasp_on_hand(hand_pcd, object_pcd, hand_config):
    
    device = hand_pcd.device

    # Transformation matrix expressing hand into object frame
    R = robust_compute_rotation_matrix_from_ortho6d(hand_config[3:9].unsqueeze(0))

    T_hand_obj = torch.eye(4, device=device)
    T_hand_obj[:3, :3] = R
    T_hand_obj[:3, 3] = hand_config[:3]

    # Transformation matrix expressing object's points into hand frame
    T_obj_hand = torch.inverse(T_hand_obj)

    # Transform pcd
    hand_pcd_homogeneous = torch.cat((hand_pcd[:, :3], torch.ones(hand_pcd.size(0), 1, device=device)), dim=1)  # Convert to homogeneous coordinates
    hand_pcd_transformed_homogeneous = torch.mm(T_obj_hand, hand_pcd_homogeneous.t()).t()
    hand_pcd_transformed = hand_pcd.clone()
    hand_pcd_transformed[:, :3] = hand_pcd_transformed_homogeneous[:, :3]

    # hand_pcd_transformed = hand_pcd_transformed.repeat(B, 1, 1)             # (B, N, 4)
    # print(hand_pcd_transformed.shape)

    object_pcd_homogeneous = torch.cat((object_pcd[:, :3], torch.ones(object_pcd.size(0), 1, device=device)), dim=1)  # Convert to homogeneous coordinates
    object_pcd_transformed_homogeneous = torch.mm(T_obj_hand, object_pcd_homogeneous.t()).t()
    object_pcd_transformed = object_pcd.clone()
    object_pcd_transformed[:, :3] = object_pcd_transformed_homogeneous[:, :3]

    # object_pcd_transformed = object_pcd_transformed.repeat(B, 1, 1)         # (B, N, 4)
    # print(object_pcd_transformed.shape)

    # print(f"Hand: {hand_pcd_transformed.shape}, Object: {object_pcd_transformed.shape}")
    return hand_pcd_transformed, object_pcd_transformed


def compute_label_barycenters(contact_points, labels):
        """
        Compute barycenters for each unique label in each batch.
        
        Args:
            contact_points: (B, N, 3) tensor of contact points
            labels: (B, N) tensor of labels
        
        Returns:
            barycenters: (B, m, 3) tensor where m is max number of unique labels across batches
            out_labels: (B, m) tensor of unique labels
        """
        device = contact_points.device
        if contact_points.dim() >= 3:
            B, _ = contact_points.shape[0], contact_points.shape[1]
        
            # Find all unique labels across all batches
            all_unique_labels = torch.unique(labels)
            m = len(all_unique_labels)
            
            barycenters = []
            out_labels = []
            
            for b in range(B):
                batch_points = contact_points[b]  # (N, 3)
                batch_labels = labels[b]  # (N,)

                barycenters_b = []
                out_labels_b = []
                for label in all_unique_labels:
                    mask = batch_labels == label
                    if mask.sum() > 0:
                        barycenters_b.append(batch_points[mask].mean(dim=0))
                        out_labels_b.append(label)
                # If less than m found, repeat the first barycenter to fill up to m
                if len(barycenters_b) == 0:
                    # fallback: fill with zeros if no label found at all
                    barycenters_b = [torch.zeros(3, device=device)] * m
                    out_labels_b = [all_unique_labels[0]] * m
                elif len(barycenters_b) < m:
                    repeat_count = m - len(barycenters_b)
                    barycenters_b += [barycenters_b[0]] * repeat_count
                    out_labels_b += [out_labels_b[0]] * repeat_count
                barycenters.append(torch.stack(barycenters_b, dim=0))
                out_labels.append(torch.tensor(out_labels_b, device=device))
            barycenters = torch.stack(barycenters, dim=0)
            out_labels = torch.stack(out_labels, dim=0)
            return barycenters, out_labels
        else:
            all_unique_labels = torch.unique(labels)
            m = len(all_unique_labels)
            barycenters = []
            out_labels = []

            for i, label in enumerate(all_unique_labels):
                mask = labels == label
                if mask.sum() > 0:
                    barycenters.append(contact_points[mask].mean(dim=0))
                    out_labels.append(label)
            barycenters = torch.stack(barycenters, dim=0)
            out_labels = torch.tensor(out_labels, device=contact_points.device)
            return barycenters, out_labels


def check_force_closure(contact_points, object_pcd, object_normals, num_friction_vectors=8, mu=0.5, epsilon=1e-4, verbose=False):
    """
    Check if the contact_points are in force closure.

    Args:
        contact_points (torch.tensor): A tensor of shape (B, N, 3) representing the contact points.
        object_pcd (torch.tensor): A tensor of shape (B, M, 3) representing the object point cloud.
        object_normals (torch.tensor): A tensor of shape (B, M, 3) representing the object normals.
        num_friction_vectors (int): Number of vectors to approximate the friction cone.
        mu (float): Friction coefficient.
        epsilon (float): Tolerance for checking if the origin is inside the convex hull.
    Returns:
        bool: True if force closure, False otherwise.
        torch.tensor: The global wrench matrix.
    """

    if contact_points.dim() == 2:
        # Single case - add batch dimension
        contact_points = contact_points.unsqueeze(0)
        object_pcd = object_pcd.unsqueeze(0)
        object_normals = object_normals.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    B, N, _ = contact_points.shape
    _, M, _ = object_pcd.shape

    # Project contact points onto closest object points
    distances = torch.cdist(contact_points, object_pcd)  # (B, N, M)
    closest_distances, closest_indices = torch.min(distances, dim=2)  # (B, N)
    
    # Define threshold for valid projections
    threshold = 0.05 
    dist_mask = closest_distances <= threshold  # (B, N)

    batch_indices = torch.arange(B, device=contact_points.device).unsqueeze(1).expand(-1, N)  # (B, N)
    projected_points = object_pcd[batch_indices, closest_indices]  # (B, N, 3)
    projected_normals = object_normals[batch_indices, closest_indices]  # (B, N, 3)
    
    # # Apply mask to filter out points beyond threshold and zero coordinates
    # projected_points = projected_points * dist_mask.unsqueeze(-1)  # (B, N, 3)
    # projected_normals = projected_normals * dist_mask.unsqueeze(-1)  # (B, N, 3)

    # Create friction cone vectors for each contact point
    friction_vectors = []
    for i in range(num_friction_vectors):
        angle = torch.tensor(2 * torch.pi * i / num_friction_vectors)
        # Create tangential directions perpendicular to normal
        # First, find two orthogonal vectors in the tangent plane
        normals_batch = projected_normals  # (B, N, 3)
        
        # Create arbitrary vector not parallel to normal
        arbitrary = torch.tensor([1., 0., 0.], device=projected_normals.device).expand_as(normals_batch).clone()
        # If normal is too close to [1,0,0], use [0,1,0] instead
        parallel_mask = torch.abs(torch.sum(normals_batch * arbitrary, dim=2)) > 0.9
        arbitrary[parallel_mask] = torch.tensor([0., 1., 0.], device=projected_normals.device)
        
        # First tangent vector (cross product)
        tangent1 = torch.cross(normals_batch, arbitrary, dim=2)
        tangent1 = tangent1 / (torch.norm(tangent1, dim=2, keepdim=True) + 1e-8)
        
        # Second tangent vector (cross product of normal and first tangent)
        tangent2 = torch.cross(normals_batch, tangent1, dim=2)
        tangent2 = tangent2 / (torch.norm(tangent2, dim=2, keepdim=True) + 1e-8)
        
        # Create friction vector in the cone
        tangent_component = torch.cos(angle) * tangent1 + torch.sin(angle) * tangent2
        friction_force = -projected_normals + mu * tangent_component  # normal + friction
        friction_force = friction_force / (torch.norm(friction_force, dim=2, keepdim=True) + 1e-8)
        friction_vectors.append(friction_force)

    friction_vectors = torch.stack(friction_vectors, dim=0)  # (num_friction_vectors, B, N, 3)

    # Compute object center for moment calculation
    object_center = object_pcd.mean(dim=1, keepdim=True)  # (B, 1, 3)

    # Compute wrenches for all friction vectors
    all_wrenches = []
    for fv in friction_vectors:
        # Compute moment arms (from object center to projected contact points)
        moment_arms = projected_points - object_center  # (B, N, 3)
        
        # Compute torques (cross product of moment arms and forces)
        torques = torch.cross(moment_arms, fv, dim=2)  # (B, N, 3)
        
        # Construct wrench: [forces; torques] for each contact point
        wrench = torch.cat([fv, torques], dim=2)  # (B, N, 6)
        all_wrenches.append(wrench)

    all_wrenches = torch.stack(all_wrenches, dim=0)  # (num_friction_vectors, B, N, 6)

    # Combine all wrenches into a single set for force closure analysis
    batch = all_wrenches.shape[1]  # Number of batches
    S = all_wrenches.permute(1, 0, 2, 3).reshape(batch, -1, 6)  # (B, num_friction_vectors*N, 6)

    # # The dist_mask from earlier needs to be expanded to match the shape of the wrenches
    # dist_mask_expanded = dist_mask.unsqueeze(1).expand(-1, num_friction_vectors, -1).reshape(batch, -1) # (B, num_friction_vectors*N)


    # Compute convex hull 
    is_force_closure = torch.zeros(batch, dtype=torch.bool, device=contact_points.device)

    all_hull = []
    all_r = []
    for b in range(batch):
        S_batch = S[b]  # (num_friction_vectors*N, 6)
        
        # dist_mask_batch = dist_mask_expanded[b] # (num_friction_vectors*N)
        # # Filter out zero wrenches
        # S_batch = S_batch[dist_mask_batch]

        try:
            hull = ConvexHull(S_batch.cpu().numpy())
            all_hull.append(hull)
            # Check if origin is inside convex hull
            # A point is inside if all hyperplane equations are satisfied
            origin = np.zeros(6)
            is_inside = all(np.dot(hull.equations[i, :-1], origin) + hull.equations[i, -1] <= np.finfo(np.float32).eps 
                            for i in range(len(hull.equations)))

            if is_inside:
                # Compute the signed distances from the origin to each facet
                distances = torch.from_numpy(np.abs(hull.equations[:, -1]) / np.linalg.norm(hull.equations[:, :-1], axis=1)).to(contact_points.device)
                R = torch.min(distances)
                is_force_closure[b] = R > epsilon
                # print(f"Batch {b}: Inside - Force closure: {is_force_closure[b]}, R: {R}")
            else:
                R = 0.0
                is_force_closure[b] = False
                # print(f"Batch {b}: Outside - Force closure: {is_force_closure[b]}, R: {R}")
            all_r.append(R)
        except:
            is_force_closure[b] = False

    if squeeze_output:
        is_force_closure = is_force_closure.item()
        S = S.squeeze(0)  # (num_friction_vectors*N, 6)
        projected_points = projected_points.squeeze(0)  # (N, 3)

    if verbose:
        return is_force_closure, S, projected_points, all_hull, all_r
    return is_force_closure


def get_cp_from_grasp(pcd, label, list_grasp_labels, ratio=0.25):
    """
    Get contact points given the grasp labels.

    Args:
        pcd (torch.tensor): (N, 3), point cloud of the gripper in a specific configuration.
        label (torch.tensor): (N, 1), point cloud labels correspoding to the hand fingers.
        list_grasp_labels (list): List of grasp labels to sample from.
        ratio (float, optional): Ratio of points in contact relative to the total number of points per grasps labels. Defaults to 0.25 (25%).

    Returns:
        _type_: _description_
    """
    grasp = torch.zeros_like(label, device=label.device)

    for grasp_labels in list_grasp_labels:
        idx = torch.tensor([], dtype=torch.long, device=label.device)
        for l in grasp_labels:
            idx = torch.cat((idx, torch.where(label[:, 0] == l)[0]))

        if idx.numel() > 0:  # Check if idx has any elements
            K = int(idx.size(0) * ratio)
            selected_idx = idx[torch.randint(0, idx.size(0), (1,))]                         # Select one index randomly
            distances = torch.norm(pcd[idx] - pcd[selected_idx], dim=1)                     # Compute distances to the selected point
            nearest_neighbors = torch.topk(distances, k=K, largest=False).indices           # Get nearest neighbors
            grasp[idx[nearest_neighbors]] = 1                                               # Update grasp for the selected neighbors

    return grasp.squeeze(-1)


def get_cp_clusters_from_grasp(pcd, label, list_grasp_labels, ratio=0.25):
    """
    Get contact points given the grasp labels.

    Args:
        pcd (torch.tensor): (N, 3), point cloud of the gripper in a specific configuration.
        label (torch.tensor): (N, 1), point cloud labels correspoding to the hand fingers.
        list_grasp_labels (list): List of grasp labels to sample from.
        ratio (float, optional): Ratio of points in contact relative to the total number of points per grasps labels. Defaults to 0.25 (25%).

    Returns:
        torch.tensor: Contact points of shape (N_cp, 1).
    """
    grasp = torch.zeros_like(label, device=label.device)
    
    # Select a random number of clusters, at least 2
    n_clusters = random.randint(2, len(list_grasp_labels))
    sampled_labels = random.sample(list_grasp_labels, n_clusters)                           # Randomly sample labels

    for grasp_labels in sampled_labels:
        idx = torch.tensor([], dtype=torch.long, device=label.device)
        for l in grasp_labels:
            idx = torch.cat((idx, torch.where(label[:, 0] == l)[0]))

        if idx.numel() > 0:  # Check if idx has any elements
            K = int(idx.size(0) * ratio)
            selected_idx = idx[torch.randint(0, idx.size(0), (1,))]                         # Select one index randomly
            distances = torch.norm(pcd[idx] - pcd[selected_idx], dim=1)                     # Compute distances to the selected point
            nearest_neighbors = torch.topk(distances, k=K, largest=False).indices           # Get nearest neighbors
            grasp[idx[nearest_neighbors]] = 1                                               # Update grasp for the selected neighbors

    return grasp.squeeze(-1)


def get_contact_map(pcd, target, target_labels=None):
    """
    Get the contact map between the point cloud and the target points.

    Args:
        pcd (torch.Tensor): Point cloud data of shape (B, N, 3).
        target (torch.Tensor): Target points of shape (B, M, 6).
        target_labels (torch.Tensor, optional): Labels for the target points of shape (B, M, 1).

    Returns:
        torch.Tensor: Contact map of shape (B, N, 1).
    """
    if pcd.dim() == 2:
        # Single case - add batch dimension
        pcd = pcd.unsqueeze(0)
    if target.dim() == 2:
        # Single case - add batch dimension
        target = target.unsqueeze(0)
        target_labels = target_labels.unsqueeze(0) if target_labels is not None else None

    # If there are no target points, fill with zeros
    if target.shape[1] == 0:
        return torch.zeros((pcd.shape[0], pcd.shape[1]), device=pcd.device)

    aligned_dist, indices = compute_aligned_dist_v2(X=target, Y=pcd, gamma=2.0, delta=0.1, use_sqrt=True)
    # aligned_dist, indices = aligned_dist.squeeze(0), indices.squeeze(0)  # Remove batch dimension

    batch_indices = torch.arange(pcd.shape[0], device=pcd.device).unsqueeze(1)  # (B, 1)
    deltas = target[batch_indices, indices, 3:]  # (B, N, 3), Copy the normals from target points to the pcd points
    # deltas = target[indices, 3:]  # (N, 3), Copy the normals from target points to the pcd points

    # Compute the contact values based on GenDexGrasp's formula
    # contact_values = (1. - 2. * (torch.sigmoid(aligned_dist) - 0.5))
    contact_values = (2. * (torch.sigmoid(1 / (1 * aligned_dist)) - 0.5))
    # Normalize the contact values
    contact_values = (contact_values - contact_values.min(dim=1, keepdim=True).values) / (contact_values.max(dim=1, keepdim=True).values - contact_values.min(dim=1, keepdim=True).values + 1e-8)

    if target_labels is not None:
        projected_labels = target_labels[batch_indices, indices]
        projected_labels[projected_labels < 0] = 0
        return contact_values, deltas, projected_labels

    return contact_values, deltas


def get_bps(bps: bps_torch, points):
    """
    Get the BPS points from the point cloud.

    Args:
        points (torch.tensor): (N, 3) or (N, 4), point cloud (with link index)
        cp_values (torch.tensor): (N, 1), contact points

    Returns:
        bps_dists (torch.tensor): (M, 1), BPS distances
        bps_ids (torch.tensor): (M, 1), BPS indices
    """
    # bps_encoded = bps.encode(points, feature_type=['dists'], x_features=None, custom_basis=None)
    bps_encoded = bps.encode(points)
    bps_dists = bps_encoded['dists'].squeeze(0)                                     # (M)
    bps_ids =  bps_encoded['ids'].squeeze(0)                                        # (M)

    return bps_ids, bps_dists


def get_bps_cp(bps: bps_torch, points, cp_values, epsilon=0.01, noise=False) -> Tuple[torch.tensor, torch.tensor]:
    """
    Get the BPS and CP points from the point cloud.

    Args:
        points (torch.tensor): (N, 3) or (N, 4), point cloud (with link index)
        cp_values (torch.tensor): (N, 1), contact points
        epsilon (float): threshold for contact points

    Returns:
        bps_dists (torch.tensor): (4096, 1), BPS distances
        bps_cp (torch.tensor): (4096, 1), Contact Points (bool)
    """
    cp_values = cp_values.squeeze(-1)
    # bps_encoded = bps.encode(points, feature_type=['dists', 'deltas'], x_features=None, custom_basis=None)
    bps_encoded = bps.encode(points)
    bps_dists = bps_encoded['dists'].squeeze(0)                                     # (4096)
    bps_ids =  bps_encoded['ids'].squeeze(0)                                        # (4096)
    mask = bps_dists <= epsilon
    bps_cp = (cp_values[bps_ids] > 0.5).float() * mask.float()

    if noise:
        idx_cp_points = bps_cp > 0.5
        noisy_bps = torch.rand_like(bps_dists, device=bps_dists.device) * 0.2
        noisy_bps[idx_cp_points] = bps_dists[idx_cp_points].clone()
        return bps_ids, noisy_bps, bps_cp
    
    return bps_ids, bps_dists, bps_cp


def find_local_maxima(point_cloud, contact_values, radius=0.02):
    # Create a KDTree for efficient neighbor search
    tree = KDTree(point_cloud)

    # Query the tree to find the indices of the nearest neighbors for each point
    # _, neighbor_indices = tree.query(point_cloud, k=num_neighbors + 1)  # +1 to include the point itself
    neighbor_indices = tree.query_ball_point(point_cloud, r=radius)

    # Initialize a list to store the indices of local maxima
    local_maximum_indices = []
    # Initialize a set to store the indices of points that have been processed
    processed_indices = set()
    
    # Iterate over each point and its neighbors
    for i in range(len(point_cloud)):
        if i in processed_indices:
            continue

        neighbors = neighbor_indices[i]
        current_value = contact_values[i]

        if current_value < 0.5:
            continue

        # Check if the current point's contact value is greater than or equal to all its neighbors
        is_local_maximum = all(current_value >= contact_values[neighbors[j]] for j in range(1, len(neighbors)))
        if is_local_maximum:
            # Check if any neighbor has the same contact value
            same_value_neighbors = [neighbors[j] for j in range(1, len(neighbors)) if contact_values[neighbors[j]] == current_value]

            if same_value_neighbors:
                # Add the current point and its same-value neighbors to the processed set
                processed_indices.update(same_value_neighbors)

            local_maximum_indices.append(i)

    return local_maximum_indices


def get_barycenter_of_clusters(pcd, cp, cluster_ids):
    """
    Get the barycenter of each cluster.
    """
    # For each cluster, find its 10 closest neighbors in filtered_workspace with highest bps_cp_filtered value
    neighbors_indices = []
    tree = KDTree(pcd.cpu().numpy())
    for idx in cluster_ids:
        # Find all points within a reasonable radius
        indices = tree.query_ball_point(pcd[idx].cpu().numpy(), r=0.02)
        # Sort neighbors by bps_cp_filtered value (descending)
        sorted_neighbors = sorted(indices, key=lambda i: cp[i].item(), reverse=True)
        # Take top 10
        top_neighbors = sorted_neighbors[:10]
        neighbors_indices.append(top_neighbors)
    # print("Top 10 neighbors (by bps_cp_filtered) for each cluster:", neighbors_indices)

    # Retrieve neighbors coordinates and compute barycenter for each cluster
    barycenters = []
    for i, cluster_neighbors in enumerate(neighbors_indices):
        coords = pcd[cluster_neighbors]
        barycenter = coords.mean(dim=0)
        barycenters.append(barycenter)
    
    if len(barycenters) == 0:
        return torch.tensor([])
    barycenters = torch.stack(barycenters)
    return barycenters





