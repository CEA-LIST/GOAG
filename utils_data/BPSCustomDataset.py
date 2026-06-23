import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm
from utils_data.custom_bps import bps_torch

from utils.constants import DATA_PATH, ALLEGRO_GRASPS_LABELS, SHADOWHAND_GRASPS_LABELS, BARRETT_GRASPS_LABELS

from utils.get_models import get_handmodel
from utils_model.HandModel import HandModel
from utils.tools import get_cp_from_grasp, get_contact_map, get_bps, get_cp_clusters_from_grasp

class BPSCustomDataset(Dataset):
    def __init__(self,
                 robot_name='allegro',
                 normalize_jv=False,
                 split=True,
                 subset=False,
                 debug=False,
                 ):

        self.normalize_jv = normalize_jv
        self.robot_name = robot_name
        self.workspace = None
        self.handprints = None
        self.upper = None
        self.lower = None
        self.data_per_handprint = None
        self.size = 0
        self.debug = debug

        self.sets_per_grasp = {
            'c6': 50,
            'f27': 50,
            'f23': 50,
            'f34': 50,
            'f29': 50,
            'c1': 50,
        }

        ## Compute data per handprint
        robot_grasps_labels = {
            'allegro': ALLEGRO_GRASPS_LABELS,
            'shadowhand': SHADOWHAND_GRASPS_LABELS,
            'barrett': BARRETT_GRASPS_LABELS
        }
        self.grasps_labels = robot_grasps_labels[self.robot_name]


        if not subset:

            self.workspace = torch.load(os.path.join(DATA_PATH, f'workspaces/{self.robot_name}_workspace.pt'), map_location='cpu', weights_only=True)[:, :3]     # (8192, 3)
            self.handprints = torch.load(os.path.join(DATA_PATH, f'handprints/{self.robot_name}_10000_handprints_normals.pt'), map_location='cpu', weights_only=True)[:100]

            hand_model : HandModel = get_handmodel(robot_name, batch_size=1, hand_scale=1.0, device='cpu')
            self.upper = hand_model.revolute_joints_q_upper.squeeze(0)
            self.lower = hand_model.revolute_joints_q_lower.squeeze(0)

            self.data_per_handprint = self.precompute_data()

            self.size = len(self.handprints) * sum(self.sets_per_grasp.values())
        
        if split:
            split_index = round(0.8 * len(self.handprints) / 10) * 10 - 1
            self.train_handprints = self.handprints[:split_index]
            self.test_handprints = self.handprints[split_index:]
            
            train_data_per_handprint = {k: [] for k in self.data_per_handprint.keys()}
            test_data_per_handprint = {k: [] for k in self.data_per_handprint.keys()}
            for k, v in self.data_per_handprint.items():
                train_data_per_handprint[k] = v[:split_index]
                test_data_per_handprint[k] = v[split_index:]

            self.train_dataset = self._create_subset(self.train_handprints, train_data_per_handprint)
            self.test_dataset = self._create_subset(self.test_handprints, test_data_per_handprint)

        
        
    def precompute_data(self):
        device = 'cuda' #if torch.cuda.is_available() else 'cpu'
        data_per_handprint = {}
        
        # Move workspace and bps to GPU for faster computation
        workspace_gpu = self.workspace.to(device)
        bps_gpu = bps_torch(custom_basis=workspace_gpu.unsqueeze(0), n_dims=3)
        
        for data in tqdm(self.handprints, desc=f"[Dataset] Computing data per handprint", total=len(self.handprints)):
            # Handprint Data - move to GPU
            handprint_label = data[1].to(device)                     # (2048, 4)
            handprint, label = torch.split(handprint_label, [6, 1], dim=1)      # (2048, 6), (2048, 1)

            for grasp in self.sets_per_grasp.keys():
                if grasp not in data_per_handprint.keys():
                    data_per_handprint[grasp] = []

                base_grasp = get_cp_from_grasp(handprint[:, :3], label, list_grasp_labels=self.grasps_labels[grasp], ratio=1.0)
                _, base_bps_dists = get_bps(bps_gpu, handprint[base_grasp == 1.0])

                # Move results back to CPU
                data_per_handprint[grasp].append((
                    handprint[base_grasp == 1.0].cpu(), 
                    label[base_grasp == 1.0].cpu(), 
                    base_bps_dists.cpu()
                ))
        
        return data_per_handprint

    def _create_subset(self, handprints, data_per_handprint):
        subset = BPSCustomDataset(robot_name=self.robot_name, normalize_jv=self.normalize_jv, split=False, subset=True, debug=self.debug)
        subset.workspace = self.workspace
        subset.handprints = handprints
        subset.data_per_handprint = data_per_handprint
        subset.size = len(handprints) * sum(self.sets_per_grasp.values())
        subset.upper = self.upper
        subset.lower = self.lower
        return subset

    def __len__(self):
        return self.size
    
    def __collate_fn__(self, batch):
        if self.debug:
            handprint = [b[0] for b in batch]
            label = [b[1] for b in batch]
            grasp = [b[2] for b in batch]
            joint_values = torch.stack([b[3] for b in batch], dim=0)
            pcd = [b[4] for b in batch]
            pcd_labels = [b[5] for b in batch]
            bps_d = torch.stack([b[6] for b in batch], dim=0)
            bps_cp = torch.stack([b[7] for b in batch], dim=0)
            bps_labels = torch.stack([b[8] for b in batch], dim=0)

            max_pcd_size = max(p.shape[0] for p in pcd)
            
            padded_pcd = []
            padded_labels = []
            for i in range(len(pcd)):
                if pcd[i].shape[0] < max_pcd_size:
                    padding = torch.zeros(max_pcd_size - pcd[i].shape[0], pcd[i].shape[1], dtype=pcd[i].dtype, device=pcd[i].device)
                    padded_pcd.append(torch.cat([pcd[i], padding], dim=0))
                    padded_labels.append(torch.cat([pcd_labels[i], padding[:, 0]], dim=0))
                else:
                    padded_pcd.append(pcd[i])
                    padded_labels.append(pcd_labels[i])

            padded_pcd = torch.stack(padded_pcd, dim=0)
            padded_labels = torch.stack(padded_labels, dim=0).long()

            return handprint, label, grasp, joint_values, padded_pcd, padded_labels, bps_d, bps_cp, bps_labels
        else:
            joint_values = torch.stack([b[1] for b in batch], dim=0)
            pcd = [b[2] for b in batch]
            pcd_labels = [b[3] for b in batch]
            bps_d = torch.stack([b[4] for b in batch], dim=0)
            bps_cp = torch.stack([b[5] for b in batch], dim=0)
            bps_labels = torch.stack([b[6] for b in batch], dim=0)

            max_pcd_size = max(p.shape[0] for p in pcd)
            
            padded_pcd = []
            padded_labels = []
            for i in range(len(pcd)):
                if pcd[i].shape[0] < max_pcd_size:
                    padding = torch.zeros(max_pcd_size - pcd[i].shape[0], pcd[i].shape[1], dtype=pcd[i].dtype, device=pcd[i].device)
                    padded_pcd.append(torch.cat([pcd[i], padding], dim=0))
                    padded_labels.append(torch.cat([pcd_labels[i], padding[:, 0]], dim=0))
                else:
                    padded_pcd.append(pcd[i])
                    padded_labels.append(pcd_labels[i])

            padded_pcd = torch.stack(padded_pcd, dim=0)
            padded_labels = torch.stack(padded_labels, dim=0).long()

            return False, joint_values, padded_pcd, padded_labels, bps_d, bps_cp, bps_labels

    def __getitem__(self, index):
        
        # Calculate which handprint and which set within that handprint
        handprint_idx = index // sum(self.sets_per_grasp.values())
        set_idx = index % sum(self.sets_per_grasp.values())
        # Determine which grasp based on set_idx
        cumulative_sets = 0
        grasp_name = None
        for grasp, nb_set in self.sets_per_grasp.items():
            if set_idx < cumulative_sets + nb_set:
                grasp_name = grasp
                break
            cumulative_sets += nb_set

        # Joint Values
        joint_values = self.handprints[handprint_idx][0].squeeze(0)                        # (n_joints)
        if self.normalize_jv:
            joint_values = (joint_values - self.lower) / (self.upper - self.lower)
        
        # Get the handprint data for the specific grasp
        handprint, label, bps_d = self.data_per_handprint[grasp_name][handprint_idx]

        # Switch to GPU
        device = 'cuda'
        workspace = self.workspace.to(device)
        handprint, label, bps_d = handprint.to(device), label.to(device), bps_d.to(device)

        grasp = get_cp_clusters_from_grasp(
            handprint[:, :3], 
            label, 
            list_grasp_labels=self.grasps_labels[grasp_name], 
            ratio=0.25
        )

        target = handprint[grasp == 1.0]
        
        target_labels = label[grasp == 1.0].squeeze(-1)

        bps_cp, bps_deltas, bps_labels = get_contact_map(workspace, target, target_labels)  # (4096)
        bps_cp, bps_deltas, bps_labels = bps_cp.squeeze(0), bps_deltas.squeeze(0), bps_labels.squeeze(0)  # (N, 3), (N, 3), (N)

        # Filter points having too low contact values
        mask = (bps_cp > 0.75)
        # Labels for pointnet
        pcd_labeled = bps_labels[mask]       # (N)
        pcd_labeled[pcd_labeled < 0] = 0      # Palm = label 0
        # Clusters for pointnet
        pcd = workspace[mask]          # (N, 3)
        pcd_deltas = bps_deltas[mask]  # (N, 3)
        pcd_labels = torch.cat((pcd, pcd_deltas), dim=-1)  # (N, 6)

        # return False, joint_values.cpu(), pcd_labels.cpu(), pcd_labeled.cpu(), bps_d.cpu(), bps_cp.cpu(), bps_labels.cpu()
        if self.debug:
            return handprint.cpu(), label.cpu(), grasp.cpu(), joint_values, pcd_labels.cpu(), pcd_labeled.cpu(), bps_d.cpu(), bps_cp.cpu(), bps_labels.cpu()

        return False, joint_values, pcd_labels.cpu(), pcd_labeled.cpu(), bps_d.cpu(), bps_cp.cpu(), bps_labels.cpu()


