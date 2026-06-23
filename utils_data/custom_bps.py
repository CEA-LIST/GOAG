import torch

def compute_aligned_dist_v2(X, Y, gamma=2.0, delta=0.1, use_sqrt=True):
    """
    Computes the aligned distance between two point sets X and Y.

    Args:
        X (torch.Tensor): Point set X, with normals, of shape (B, N, 6).
        Y (torch.Tensor): Point set Y of shape (B, M, 3).

    Returns:
        torch.Tensor: Aligned distances of shape (B, M).
        torch.Tensor: Indices of the closest points in X for each point in Y of shape (B, M).
    """
    # print(f"X shape: {X.shape}, Y shape: {Y.shape}")  # Debugging shape

    if X.dim() < 3:
        X = X.unsqueeze(0)  # Ensure X is at least 3D (B, N, 6)

    B, N, _ = X.shape
    if Y.dim() < 3:
        Y = Y.unsqueeze(0).repeat(B, 1, 1)  # Ensure Y is at least 3D (B, M, 3)
    _, M, _ = Y.shape
    
    # Extract points and normals from X
    X_points = X[:, :, :3]  # (B, N, 3)
    X_normals = X[:, :, 3:]  # (B, N, 3)
    
    # Expand dimensions for pairwise computation
    X_points_expanded = X_points.unsqueeze(2)  # (B, N, 1, 3)
    X_normals_expanded = X_normals.unsqueeze(2)  # (B, N, 1, 3)
    Y_expanded = Y.unsqueeze(1)  # (B, 1, M, 3)
    
    # Repeat to create pairwise combinations
    X_points_expanded = X_points_expanded.repeat(1, 1, M, 1)  # (B, N, M, 3)
    X_normals_expanded = X_normals_expanded.repeat(1, 1, M, 1)  # (B, N, M, 3)
    Y_expanded = Y_expanded.repeat(1, N, 1, 1)  # (B, N, M, 3)
    
    # print(f"X_points_expanded: {X_points_expanded.shape}, Y_expanded: {Y_expanded.shape}")  # Debugging shape

    # Compute distances
    deltas = Y_expanded - X_points_expanded  # (B, N, M, 3)
    dists = deltas.norm(dim=3)  # (B, N, M)
    
    # Compute alignment
    alignment = (deltas * X_normals_expanded).sum(dim=3)  # (B, N, M)
    alignment = alignment / (dists + 1e-5)  # Normalize by distance
    
    # Compute aligned distance
    aligned_dist = dists * torch.exp(gamma * (1.0 - alignment))  # (B, N, M)

    # Take minimum over N dimension to get final result
    result, indices = aligned_dist.min(dim=1)  # result: (B, M), indices: (B, M)
    result = result / delta

    if use_sqrt:
        result = torch.sqrt(result)

    return result, indices


def to_tensor(array, dtype=torch.float32):
    if not torch.is_tensor(array):
        array = torch.tensor(array)
    return array.to(dtype)


class bps_torch():
    def __init__(self, custom_basis, n_dims=3):

        basis_set = to_tensor(custom_basis)

        self.bps = basis_set.reshape(1,-1,n_dims)

        if self.bps.ndim > 2:
            self.bps = self.bps.squeeze(0)

    def encode(self, x):

        x = to_tensor(x)
        is_batch = True if x.ndim > 2 else False

        if not is_batch:
            x = x.unsqueeze(0)

        bps = to_tensor(self.bps)

        aligned_dist, indices = compute_aligned_dist_v2(X=x, Y=bps, gamma=2.0, delta=0.1, use_sqrt=True)

        x_bps = {}
        x_bps['dists'] = aligned_dist
        x_bps['ids'] = indices
        return x_bps