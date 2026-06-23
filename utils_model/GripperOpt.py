import torch
import torch.nn.functional as F
import time
from tqdm import tqdm

from utils.get_models import get_handmodel
from utils_model.HandModel import HandModel
from utils_data.custom_bps import compute_aligned_dist_v2

class GripperOpt:
    """
    Run the optimization for the gripper. Uses Adam optimizer on q based on an energy function.
    """
    def __init__(self, cfg, logger=None):

        self.device = cfg['device']
        self.learning_rate = float(cfg['learning_rate'])
        self.w_pen = float(cfg['W_PENETRATION'])
        self.w_spen = float(cfg['W_SELF_PENETRATION'])
        self.w_joints = float(cfg['W_JOINTS'])
        self.max_iter = cfg['max_iterations']
        self.dist_type = cfg['distance_type']
        assert self.dist_type in ['euclidean', 'aligned'], "Distance type in config file must be either 'euclidean' or 'aligned'."

        self.batch_size = None
        self.robot_name = None
        self.object_name = None
        self.object_pcd = None
        self.object_normals = None
        self.target_contact_points = None
        self.labels = None
        self.q_start = None

        self.has_reset = False

        self.q_current = None
        self.compute_energy = None
        self.optimizer = None
        self.scheduler = None
        self.energy = None
        self.global_step = None
        self.target_points = None
        self.time = 0

        self.logger = logger

    def reset(self, robot_name, object_name, object_pcd=None, object_normals=None, contact_points=None, labels=None, q_start=None):
        """Reset and prepare GripperOpt state.

        Args:
            q_start (B, n_joints): Initial joint values for the gripper.
            object_pcd (B, N, 3): Object point cloud.
            object_normals (B, N, 3): Object normals corresponding to the point cloud.
            contact_points (B, n, 3/6): Target contact points for the gripper to reach.
            labels (B, n): Labels indicating which contact points belong to which finger.
        """
        self.robot_name = robot_name
        self.object_name = object_name
        self.object_pcd = object_pcd                                                                        # (B, 2048, 3)
        self.object_normals = object_normals                                                                # (B, 2048, 3)
        self.target_contact_points = contact_points                                                         # (B, N, 3)
        self.labels = labels                                                                                # (B, N)

        self.batch_size = contact_points.shape[0]                                                           # Number of batches

        self.hand_model: HandModel = get_handmodel(self.robot_name, batch_size=self.batch_size, hand_scale=1.0, device=self.device, num_points=512)
        self.q_upper_bound = self.hand_model.revolute_joints_q_upper.detach()  # (B, n_joints)
        self.q_lower_bound = self.hand_model.revolute_joints_q_lower.detach()  # (B, n_joints)
        self.q_start = q_start if (q_start is not None) else self.hand_model.straight_pose.clone()[:, 9:]  # (B, n_joints)
        # self.q_start = torch.rand_like(self.hand_model.straight_pose.clone()[:, 9:]) * (self.q_upper_bound - self.q_lower_bound) + self.q_lower_bound  # Random initialization within bounds

        self.global_step = 0
        if self.dist_type == 'euclidean':
            self.compute_energy = self.compute_energy_euclidean_dist
        elif self.dist_type == 'aligned':
            self.compute_energy = self.compute_energy_aligned_dist

        self.q_current = self.q_start.clone().to(self.device)
        self.q_current = torch.nn.Parameter(self.q_current, requires_grad=True)  # Make it a parameter for optimization
        self.optimizer = torch.optim.Adam([self.q_current], lr=self.learning_rate)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=15, threshold=0.0005, verbose=True)

        self.hand_model.update_kinematics_no_base(jv=self.q_current)  # Update hand model with initial joint values

        self.has_reset = True

    def get_current_q(self):
        """
        Get the current joint values.
        """
        return self.q_current

    def compute_energy_euclidean_dist(self):
        """
        Compute the energy function based on the current gripper pose.
        """
        if self.target_contact_points.shape[-1] > 3:
            self.target_contact_points = self.target_contact_points[:, :, :3]

        all_keypoints = self.hand_model.get_keypoints_differentiable(with_base=True)                                    # (B, K, 3)
            # Exclude base
        keypoints_no_base = all_keypoints[:, 1:, :] 

        # Energy: EUCLIDEAN DISTANCE
            # Mask distances where self.labels == 0
        mask = (self.labels != 0).float()  
            # Get the keypoints corresponding to the labels
        labels_exp = self.labels.unsqueeze(-1).expand(-1, -1, 3)
        keypoints_per_labels = torch.gather(all_keypoints, 1, labels_exp)
            # Compute distances between keypoints and target contact points
        distances = torch.norm(keypoints_per_labels - self.target_contact_points, dim=-1)                               # (B, N)
        distances = distances * mask
        energy_dist = distances.mean(dim=1)                                                                             # (B,)

        # Energy: HAND-OBJECT PENETRATION
        hand_surface_points_ = self.hand_model.get_handprint_points_differentiable()                                    # (B, 2048, 3)
        npts_obj = self.object_pcd.shape[1]
        npts_hand = hand_surface_points_.shape[1]
            # Object pcd: (2048, 3) --> (B, 2048, 2048, 3)
        batch_object_pcd = self.object_pcd.reshape(self.batch_size, 1, npts_obj, 3)                                     # (B, 1, 2048, 3)
        batch_object_pcd = batch_object_pcd.repeat(1, npts_hand, 1, 1)                                                  # (B, 2048, 2048, 3)
            # Hand surface points: (B, 2048, 3) --> (B, 2048, 2048, 3)
        hand_surface_points = hand_surface_points_.reshape(self.batch_size, 1, npts_hand, 3)                            # (B, 1, 2048, 3)
        hand_surface_points = hand_surface_points.repeat(1, npts_obj, 1, 1).transpose(1, 2)                             # (B, 2048, 2048, 3)
            # Compute distances between hand surface points and object points
        hand_object_distances = torch.norm(hand_surface_points - batch_object_pcd, dim=-1)                              # (B, 2048, 2048)
        hand_object_distances, hand_object_indices = torch.min(hand_object_distances, dim=2)                            # (B, 2048), (B, 2048)
            # Get the closest object points and normals per hand point
        hand_object_points = torch.gather(self.object_pcd, 1, hand_object_indices.unsqueeze(-1).expand(-1, -1, 3))      # (B, 2048, 3)
        hand_object_normals = torch.gather(self.object_normals, 1, hand_object_indices.unsqueeze(-1).expand(-1, -1, 3)) # (B, 2048, 3)
        hand_object_signs = torch.sum((hand_object_points - hand_surface_points_) * hand_object_normals, dim=-1)        # (B, 2048)
        hand_object_signs = (hand_object_signs > 0.0).float()                                                           # (B, 2048)
            # Compute penetration energy
        energy_pen = torch.mean(hand_object_signs * hand_object_distances, dim=1)                                       # (B,)

        # Energy: SELF-PENETRATION
        distances_keypoints = torch.cdist(keypoints_no_base, keypoints_no_base, p=2)                                    # (B, K-1, K-1)
            # Exclude intra-finger distances from self-penetration energy
        if self.robot_name == 'allegro':
            # finger_ids = torch.arange(20, device=self.device) // 5              # Finger indices: 0-4 (LF), 5-9 (MF), 10-14 (IF), 15-19 (TH)
            finger_ids = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3], device=self.device)
        elif self.robot_name == 'shadowhand':
            finger_ids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4], device=self.device)
        elif self.robot_name == 'barrett':
            finger_ids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], device=self.device)
        else:
            raise ValueError(f"Unsupported robot name: {self.robot_name}")
            # Create mask: True for intra-finger pairs, False otherwise
        intra_finger_mask = (finger_ids.unsqueeze(0) == finger_ids.unsqueeze(1)).unsqueeze(0)                           # (K-1, K-1)
        distances_keypoints = torch.where(intra_finger_mask, torch.ones_like(distances_keypoints), distances_keypoints) # (B, K-1, K-1)
            # Compute self-penetration energy
        if self.robot_name == 'allegro':
            s_norm = F.relu(0.03 - distances_keypoints)
        elif self.robot_name == 'shadowhand':
            s_norm = F.relu(0.02 - distances_keypoints)
        elif self.robot_name == 'barrett':
            s_norm = F.relu(0.024 - distances_keypoints)
        else:
            raise ValueError(f"Unsupported robot name: {self.robot_name}")
        energy_spen = s_norm.sum(dim=(1, 2))                                                                            # (B,)

        # Energy: JOINTS => upper | lower
        z_norm = F.relu(self.q_current - self.q_upper_bound) + F.relu(self.q_lower_bound - self.q_current)
        energy_joints = z_norm.sum(dim=1)

        # Energy: TOTAL
        energy = energy_dist + self.w_pen * energy_pen + self.w_spen * energy_spen + self.w_joints * energy_joints  # (B,)

        self.energy = energy

        if self.logger is not None:
            self.logger.log_metrics({
                'energy': energy.mean().item(),
                'energy_dist': energy_dist.mean().item(),
                'energy_pen': self.w_pen * energy_pen.mean().item(),
                'energy_spen': self.w_spen * energy_spen.mean().item(),
                'energy_joints': self.w_joints * energy_joints.mean().item(),
            }, step=self.global_step)

        return energy

    def compute_energy_aligned_dist(self):
        """
        Compute the energy function based on the current gripper pose.
        """
        assert self.target_contact_points.shape[-1] == 6, "Target contact points must have 6 dimensions: normals included."

        all_keypoints = self.hand_model.get_keypoints_differentiable(with_base=True)                                    # (B, K, 3)
            # Exclude base
        keypoints_no_base = all_keypoints[:, 1:, :] 

        # Energy: ALIGN DISTANCE
            # Mask distances where self.labels == 0
        mask = (self.labels != 0).float()  
            # Get the keypoints corresponding to the labels
        labels_exp = self.labels.unsqueeze(-1).expand(-1, -1, 3)
        keypoints_per_labels = torch.gather(all_keypoints, 1, labels_exp)
            # Compute aligned distances between keypoints and target contact points
        aligned_dist, _ = compute_aligned_dist_v2(X=self.target_contact_points, Y=keypoints_per_labels, gamma=2.0, delta=0.1, use_sqrt=True)
        aligned_dist = aligned_dist * mask
        energy_dist = aligned_dist.mean(dim=1)                                                                             # (B,)

        # Energy: HAND-OBJECT PENETRATION
        hand_surface_points_ = self.hand_model.get_handprint_points_differentiable()                                    # (B, 2048, 3)
        npts_obj = self.object_pcd.shape[1]
        npts_hand = hand_surface_points_.shape[1]
            # Object pcd: (2048, 3) --> (B, 2048, 2048, 3)
        batch_object_pcd = self.object_pcd.reshape(self.batch_size, 1, npts_obj, 3)                                     # (B, 1, 2048, 3)
        batch_object_pcd = batch_object_pcd.repeat(1, npts_hand, 1, 1)                                                  # (B, 2048, 2048, 3)
            # Hand surface points: (B, 2048, 3) --> (B, 2048, 2048, 3)
        hand_surface_points = hand_surface_points_.reshape(self.batch_size, 1, npts_hand, 3)                            # (B, 1, 2048, 3)
        hand_surface_points = hand_surface_points.repeat(1, npts_obj, 1, 1).transpose(1, 2)                             # (B, 2048, 2048, 3)
            # Compute distances between hand surface points and object points
        hand_object_distances = torch.norm(hand_surface_points - batch_object_pcd, dim=-1)                              # (B, 2048, 2048)
        hand_object_distances, hand_object_indices = torch.min(hand_object_distances, dim=2)                            # (B, 2048), (B, 2048)
            # Get the closest object points and normals per hand point
        hand_object_points = torch.gather(self.object_pcd, 1, hand_object_indices.unsqueeze(-1).expand(-1, -1, 3))      # (B, 2048, 3)
        hand_object_normals = torch.gather(self.object_normals, 1, hand_object_indices.unsqueeze(-1).expand(-1, -1, 3)) # (B, 2048, 3)
        hand_object_signs = torch.sum((hand_object_points - hand_surface_points_) * hand_object_normals, dim=-1)        # (B, 2048)
        hand_object_signs = (hand_object_signs > 0.0).float()                                                           # (B, 2048)
            # Compute penetration energy
        energy_pen = torch.mean(hand_object_signs * hand_object_distances, dim=1)                                       # (B,)

        # Energy: SELF-PENETRATION
        distances_keypoints = torch.cdist(keypoints_no_base, keypoints_no_base, p=2)                                    # (B, K-1, K-1)
            # Exclude intra-finger distances from self-penetration energy
        if self.robot_name == 'allegro':
            # finger_ids = torch.arange(20, device=self.device) // 5              # Finger indices: 0-4 (LF), 5-9 (MF), 10-14 (IF), 15-19 (TH)
            finger_ids = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3], device=self.device)
        elif self.robot_name == 'shadowhand':
            finger_ids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4], device=self.device)
        elif self.robot_name == 'barrett':
            finger_ids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], device=self.device)
        else:
            raise ValueError(f"Unsupported robot name: {self.robot_name}")
            # Create mask: True for intra-finger pairs, False otherwise
        intra_finger_mask = (finger_ids.unsqueeze(0) == finger_ids.unsqueeze(1)).unsqueeze(0)                           # (K-1, K-1)
        distances_keypoints = torch.where(intra_finger_mask, torch.ones_like(distances_keypoints), distances_keypoints) # (B, K-1, K-1)
            # Compute self-penetration energy
        if self.robot_name == 'allegro':
            s_norm = F.relu(0.03 - distances_keypoints)
        elif self.robot_name == 'shadowhand':
            s_norm = F.relu(0.02 - distances_keypoints)
        elif self.robot_name == 'barrett':
            s_norm = F.relu(0.024 - distances_keypoints)
        else:
            raise ValueError(f"Unsupported robot name: {self.robot_name}")
        energy_spen = s_norm.sum(dim=(1, 2))                                                                            # (B,)

        # Energy: JOINTS => upper | lower
        z_norm = F.relu(self.q_current - self.q_upper_bound) + F.relu(self.q_lower_bound - self.q_current)
        energy_joints = z_norm.sum(dim=1)

        # Energy: TOTAL
        energy = energy_dist + self.w_pen * energy_pen + self.w_spen * energy_spen + self.w_joints * energy_joints  # (B,)

        self.energy = energy

        if self.logger is not None:
            self.logger.log_metrics({
                'energy': energy.mean().item(),
                'energy_dist': energy_dist.mean().item(),
                'energy_pen': self.w_pen * energy_pen.mean().item(),
                'energy_spen': self.w_spen * energy_spen.mean().item(),
                'energy_joints': self.w_joints * energy_joints.mean().item(),
            }, step=self.global_step)

        return energy

    def step(self):
        self.optimizer.zero_grad()
        self.hand_model.update_kinematics_no_base(jv=self.q_current)
        energy = self.compute_energy()
        energy_mean = energy.mean()
        energy_mean.backward()
        self.optimizer.step()
        self.global_step += 1

    def run(self, verbose=True):

        if not self.has_reset:
            raise RuntimeError("GripperOpt has not been reset. Call reset() before run().")

        q_traj = []
        energy_per_iter = []
        time_per_step = []
        if verbose:
            with tqdm(total=self.max_iter, desc=f"[{self.robot_name}/{self.object_name}] Optimization - Energy: n/a (avg)") as pbar:
                for i in range(self.max_iter):
                    t0 = time.time()
                    self.step()
                    t1 = time.time()
                    time_per_step.append(t1 - t0)
                    with torch.no_grad():
                        q = self.get_current_q()
                        q_traj.append(q.clone().detach())
                        energy = self.energy.detach().cpu()
                        energy_per_iter.append(energy)
                        pbar.set_description(f"[{self.robot_name}/{self.object_name}] Optimization - Energy: {energy.mean():.4f} (avg)")
                        pbar.update(1)
        else:
            for i in range(self.max_iter):
                t0 = time.time()
                self.step()
                t1 = time.time()
                time_per_step.append(t1 - t0)
                with torch.no_grad():
                    q = self.get_current_q()
                    q_traj.append(q.clone().detach())
                    energy = self.energy.detach().cpu()
                    energy_per_iter.append(energy)
        self.time = sum(time_per_step) / len(time_per_step)
        # print(f"Optimization finished. Mean time per iter: {self.time:.4f} Energy: {energy.min():.4f} (min)")

        q_traj = torch.stack(q_traj, dim=0).transpose(0, 1)  # (B, self.max_iter, 16)
        energy_per_iter = torch.stack(energy_per_iter, dim=0).transpose(0, 1)    # (B, self.max_iter)

        return q_traj, energy_per_iter