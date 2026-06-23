import torch
from scipy.spatial.transform import Rotation

def matrix_to_euler(matrix):
    device = matrix.device
    # forward_kinematics() requires intrinsic euler ('XYZ')
    euler = Rotation.from_matrix(matrix.cpu().numpy()).as_euler('XYZ')
    return torch.tensor(euler, dtype=torch.float32, device=device)

def euler_to_matrix(euler):
    """euler should be in radians"""
    device = euler.device
    matrix = Rotation.from_euler('XYZ', euler.cpu().numpy(), degrees=False).as_matrix()
    return torch.tensor(matrix, dtype=torch.float32, device=device)

# def matrix_to_rot6d(matrix):
#     return matrix.T.reshape(9)[:6]

def matrix_to_rot6d(matrix):
    # Ensure the matrix is in the correct shape for batched operations
    if matrix.ndimension() == 3:
        # Batched case: transpose the last two dimensions
        matrix = matrix.transpose(-1, -2)
    else:
        # Single matrix case: transpose the matrix
        matrix = matrix.T
    # Reshape to 6D representation
    return matrix.reshape(matrix.shape[:-2] + (-1,))[..., :6]

def rot6d_to_matrix(rot6d):
    x = normalize(rot6d[..., 0:3])
    y = normalize(rot6d[..., 3:6])
    a = normalize(x + y)
    b = normalize(x - y)
    x = normalize(a + b)
    y = normalize(a - b)
    z = normalize(torch.cross(x, y, dim=-1))
    matrix = torch.stack([x, y, z], dim=-2).mT
    return matrix

def euler_to_rot6d(euler):
    matrix = euler_to_matrix(euler)
    return matrix_to_rot6d(matrix)

def rot6d_to_euler(rot6d):
    matrix = rot6d_to_matrix(rot6d)
    return matrix_to_euler(matrix)

def axisangle_to_matrix(axis, angle):
    (x, y, z), c, s = axis, torch.cos(angle), torch.sin(angle)
    return torch.tensor([
        [(1 - c) * x * x + c, (1 - c) * x * y - s * z, (1 - c) * x * z + s * y],
        [(1 - c) * x * y + s * z, (1 - c) * y * y + c, (1 - c) * y * z - s * x],
        [(1 - c) * x * z - s * y, (1 - c) * y * z + s * x, (1 - c) * z * z + c]
    ])

def euler_to_quaternion(euler):
    device = euler.device
    quaternion = Rotation.from_euler('XYZ', euler.cpu().numpy()).as_quat()
    return torch.tensor(quaternion, dtype=torch.float32, device=device)

def normalize(v):
    return v / torch.norm(v, dim=-1, keepdim=True)

def q_euler_to_q_rot6d(q_euler):
    """ euler should be in radians """
    return torch.cat([q_euler[..., :3], euler_to_rot6d(q_euler[..., 3:6]), q_euler[..., 6:]], dim=-1)

def q_rot6d_to_q_euler(q_rot6d):
    """ Convert 6D representation back to euler angles in radians """
    return torch.cat([q_rot6d[..., :3], rot6d_to_euler(q_rot6d[..., 3:9]), q_rot6d[..., 9:]], dim=-1)

def robust_compute_rotation_matrix_from_ortho6d(poses):
    """
    Instead of making 2nd vector orthogonal to first
    create a base that takes into account the two predicted
    directions equally
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    # Create orthonormal vectors
    x = normalize_vector(x_raw)  # batch*3
    y = normalize_vector(y_raw)  # batch*3
    middle = normalize_vector(x + y)
    orthmid = normalize_vector(x - y)
    x = normalize_vector(middle + orthmid)
    y = normalize_vector(middle - orthmid)
    # Their scalar product should be small !
    z = normalize_vector(torch.cross(x, y, dim=1))

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    # Check for reflection in matrix ! If found, flip last vector TODO
    # assert (torch.stack([torch.det(mat) for mat in matrix ])< 0).sum() == 0
    return matrix

def normalize_vector(v):
    v_mag = torch.norm(v, dim=1, keepdim=True)
    v_mag = torch.clamp(v_mag, min=1e-8) # Avoid division by zeros
    v = v / v_mag
    return v

