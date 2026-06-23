import sys
import time
import yaml
import os
import json
import argparse
from tqdm import tqdm
from termcolor import cprint
import torch
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

# from bps_torch.bps import bps_torch
from utils_data.custom_bps import bps_torch
from datetime import datetime

if not sys.stdout.isatty():
    pass

from utils.get_models import get_handmodel
from utils_model.HandModel import HandModel
from utils.constants import DATA_PATH, ROOT_PATH
from utils.tools import convert_shadow_pose, farthest_point_sampling, move_pcd_6d, compute_object_pose_6d, compute_label_barycenters, check_force_closure
from utils.RT_sampling import *
from utils.rot6d import rot6d_to_euler, q_rot6d_to_q_euler

from utils_model.CVAE import CVAE
from utils_model.PointNetLabels import PointNetLabels
from utils_model.GripperOpt import GripperOpt

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_name', default='allegro', type=str, help='Name of the gripper to use')
    parser.add_argument('--object_name', default=None, type=str, help='Name of the object to validate')
    parser.add_argument('--radius', default=0.01, type=float, help='Radius for sampling gripper poses')
    parser.add_argument('--dataset', default='multidex', type=str, help='Dataset name')
    args_ = parser.parse_args()
    return args_


if __name__=="__main__":
    args = get_parser()

    cvae_name = f'{args.robot_name}_cvae_0801'
    pointnet_name = f'{args.robot_name}_pointnet_0801'

    device = torch.device('cuda')
    robot_name = args.robot_name
    
    global_batch_size = 10
    n_grasps_per_obj = 2

    # Load CVAE config
    cvae_model_basedir = os.path.join(ROOT_PATH, f'logs/{cvae_name}')
    cvae_config_path = os.path.join(cvae_model_basedir, "config.yaml")
    with open(cvae_config_path, "r") as file:
        cvae_cfg = yaml.safe_load(file)
    cvae_cfg = cvae_cfg['cvae']
    cvae_cfg['batch_size'] = global_batch_size
    cvae_cfg['robot_name'] = robot_name

    # Load pointnet config
    pointnet_model_basedir = os.path.join(ROOT_PATH, f'logs/{pointnet_name}')
    pointnet_config_path = os.path.join(pointnet_model_basedir, "config.yaml")
    with open(pointnet_config_path, "r") as file:
        pointnet_cfg = yaml.safe_load(file)
    pointnet_cfg = pointnet_cfg['pointnet']
    pointnet_cfg['batch_size'] = global_batch_size
    pointnet_cfg['robot_name'] = robot_name

    # Load Optimization Config
    opti_config_path = os.path.join(ROOT_PATH, 'configs/optimization.yaml')
    with open(opti_config_path, "r") as file:
        opti_cfg = yaml.safe_load(file)

    if not os.path.exists(cvae_model_basedir) or not os.path.exists(pointnet_model_basedir):
        raise ValueError("Model name not valid.")

    # CVAE Model
    cvae_state_dict = torch.load(os.path.join(cvae_model_basedir, 'ckpts_dir', f'{cvae_name}_state_dict.pth'), weights_only=False, map_location=device)
    cvae_model = CVAE(cvae_cfg).to(device)
    cvae_model.load_state_dict(cvae_state_dict)
    cvae_model.eval()

    # PointNet Model
    pointnet_state_dict = torch.load(os.path.join(pointnet_model_basedir, 'ckpts_dir', f'{pointnet_name}_state_dict.pth'), weights_only=False, map_location=device)
    pointnet_model = PointNetLabels(pointnet_cfg).to(device)
    pointnet_model.load_state_dict(pointnet_state_dict)
    pointnet_model.eval()

    # Optimization Model
    opti_logs_name = None       #f'logs_opti_{args.robot_name}'
    if opti_logs_name:
        opti_tb_logger = TensorBoardLogger(save_dir=os.path.join(ROOT_PATH, 'logs_opti'), name=opti_logs_name, log_graph=False)
    else:
        opti_tb_logger = None
    opt_model = GripperOpt(cfg=opti_cfg, logger=opti_tb_logger)

    # Load hand model
    hand_model : HandModel = get_handmodel(robot=robot_name, batch_size=global_batch_size, device=device)
    upper = hand_model.revolute_joints_q_upper.squeeze(0)
    lower = hand_model.revolute_joints_q_lower.squeeze(0)

    # BPS: Load workspace and create BPS basis
    workspace = torch.load(os.path.join(DATA_PATH, f'workspaces/{robot_name}_workspace.pt'), map_location=device)[:, :3]       # (8192, 3)
    bps = bps_torch(custom_basis=workspace.unsqueeze(0), n_dims=3)

    # Objects list
    DATASET = args.dataset.lower()
    dataset_folder = os.path.join(DATA_PATH, 'pointclouds', DATASET)
    with open(os.path.join(DATA_PATH, 'pointclouds', f'split_{DATASET}.json'), 'r') as f:
        split_data = json.load(f)
    objects_list = split_data['test_split']

    all_time_cvae, all_time_pointnet, all_time_cvae_pointnet, all_time_opti = [], [], [], []
    predicted_data = []
    # For each object
    cprint(f"********************************************************** [{robot_name.upper()} - Prediction]", 'magenta', attrs=['bold'])
    for i, object_name in tqdm(enumerate(objects_list), desc=f"[{robot_name}] Validating on {DATASET} dataset: ", unit='objects', total=len(objects_list)):
        object_name = object_name[:-3] if object_name.endswith('.pt') else object_name
        if args.object_name is not None and (object_name != args.object_name):
            continue
        
        batch_size = global_batch_size

        # Load object point cloud
        object_path = os.path.join(DATA_PATH, 'pointclouds', DATASET, f'{object_name}.pt')
        object_pcd_normals = torch.load(object_path, map_location=device).to(torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)           # (B, 2048, 6)
        object_pcd = object_pcd_normals[:, :, :3]  # (B, 2048, 3)
        object_normals = object_pcd_normals[:, :, 3:]  # (B, 2048, 3)

        #   Keep tracks of the original object
        object_pcd_at_origin = object_pcd.clone()
        all_gripper_pos_6d = sample_gripper_poses(robot_name, object_pcd[0], r=args.radius, num_samples=n_grasps_per_obj)

        all_objects_pcd = []
        all_objects_normals = []
        all_cp_hat = []
        all_contact_points = []
        all_contact_points_normals = []
        all_labels_hat = []

        data_count = 0
        while data_count < n_grasps_per_obj:
            gripper_pose = all_gripper_pos_6d[data_count : data_count + batch_size]                           # (B, 16)
            batch_size = gripper_pose.shape[0]

            data_count += batch_size

            gripper_xyz = gripper_pose[:, :3]
            gripper_rot = gripper_pose[:, 3:9]

            object_pos = compute_object_pose_6d(gripper_xyz, gripper_rot)                  # (B, 6)
            #  Move the object to the desired R,t
            object_xyz = object_pos[:, :3]                                              # (B, 3)
            object_rot = object_pos[:, 3:]                                              # (B, 6)

            object_pcd_moved, object_normals_moved = move_pcd_6d(xyz=object_xyz, rot=object_rot, pcd=object_pcd, normals=object_normals)
            # Encode object point cloud
            object_pcd_moved_normals = torch.cat((object_pcd_moved, -object_normals_moved), dim=-1)  # (B, 2048, 6)
            bps_encoded = bps.encode(object_pcd_moved_normals)
            bps_d = bps_encoded['dists'].to(device)                                      # (B, 8192)

            time_cvae, time_pointnet, time_cvae_pointnet = [], [], []
            list_valid_cp_hat, list_valid_contact_points, list_valid_contact_points_normals, list_valid_labels_hat = [], [], [], []
            shape_check = True
            nb_try = 0
            start_time_cvae_pointnet = time.time()
            while shape_check:
                with torch.no_grad():
                    # Sample latent space                
                    z_latent_code = torch.randn((batch_size, cvae_model.latent_size), dtype=torch.float32, device=device)  # torch.rand or torch.randn
                    # Infer CVAE
                    start_time = time.time()
                    cp_hat = cvae_model.inference(bps_d, z_latent_code).squeeze(0)              # (B, 8192)
                    time_cvae.append(time.time() - start_time)

                if batch_size == 1:
                    cp_hat = cp_hat.unsqueeze(0)

                # Select top_k contact points based on the predicted scores                
                top_k = 100
                cp_idx = torch.zeros_like(cp_hat, dtype=torch.bool)                              # (B, 8192)       
                topk = torch.topk(cp_hat, top_k, dim=-1, sorted=False)                          # topk.indices: (B, top_k), topk.values: (B, top_k)
        
                batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, top_k)
                cp_idx[batch_indices, topk.indices] = True                                      

                workspace_pts = workspace.clone().unsqueeze(0).repeat(batch_size, 1, 1)                                                         # (B, 8192, 3)
                contact_points = workspace_pts[cp_idx]                                          # (B*top_k, 3)
                contact_points = contact_points.view(batch_size, top_k, 3)                      # (B, top_k, 3)

                cp_hat_filtered = cp_hat[cp_idx].view(batch_size, top_k)                              # (B, top_k)
                
                # Find closest points in object_pcd_moved for each contact point and get their normals (vectorized)
                # Compute distances between contact points and object point cloud for all batches
                distances = torch.cdist(contact_points, object_pcd_moved)  # (B, top_k, 2048)
                closest_indices = torch.argmin(distances, dim=2)  # (B, top_k)
                batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, top_k)  # (B, top_k)
                contact_points_normals = -object_normals_moved[batch_indices, closest_indices]  # (B, top_k, 3)

                # Predict labels
                with torch.no_grad():
                    start_time = time.time()
                    labels_hat = pointnet_model(torch.cat([contact_points, contact_points_normals], dim=-1))                                   # (B, top_k, n_labels)
                    time_pointnet.append(time.time() - start_time)
                labels_hat = torch.argmax(labels_hat, dim=-1)                                   # (B, top_k)

                # Check force closure condition
                barycenters, l = compute_label_barycenters(contact_points, labels_hat)  # Compute barycenters of the contact points for each label
                force_closure_condition = check_force_closure(barycenters, object_pcd_moved, object_normals_moved, mu=0.5, epsilon=1e-3, verbose=False)
                if not force_closure_condition.any():
                    continue
                
                list_valid_cp_hat.append(cp_hat_filtered[force_closure_condition].clone())
                list_valid_contact_points.append(contact_points[force_closure_condition].clone())
                list_valid_contact_points_normals.append(contact_points_normals[force_closure_condition].clone())
                list_valid_labels_hat.append(labels_hat[force_closure_condition].clone())
                # Concatenate valid results
                valid_cp_hat = torch.cat(list_valid_cp_hat, dim=0)                                       # (n_grasps_per_obj, top_k)
                valid_contact_points = torch.cat(list_valid_contact_points, dim=0)                       # (n_grasps_per_obj, top_k, 3)
                valid_contact_points_normals = torch.cat(list_valid_contact_points_normals, dim=0)       # (n_grasps_per_obj, top_k, 3)
                valid_labels_hat = torch.cat(list_valid_labels_hat, dim=0)                               # (n_grasps_per_obj, top_k)

                shape_check = valid_cp_hat.shape[0] < batch_size
                nb_try += 1
                if nb_try > 20:
                    break
            if nb_try > 20:
                cprint(f"Too many tries ({nb_try}), skipping object.", 'red', attrs=['bold'])
                continue
            
            time_cvae_pointnet.append(time.time() - start_time_cvae_pointnet)
            
            all_objects_pcd.append(object_pcd_moved.clone())
            all_objects_normals.append(object_normals_moved.clone())
            all_cp_hat.append(valid_cp_hat[:batch_size])
            all_contact_points.append(valid_contact_points[:batch_size])
            all_contact_points_normals.append(valid_contact_points_normals[:batch_size])
            all_labels_hat.append(valid_labels_hat[:batch_size])

        # Optimization for one object
            # Prepare batch
        batch_object_pcd = torch.cat(all_objects_pcd, dim=0)                                    # (n_grasps_per_obj, 2048, 3)
        batch_object_normals = torch.cat(all_objects_normals, dim=0)                            # (n_grasps_per_obj, 2048, 3)
        batch_cp_hat = torch.cat(all_cp_hat, dim=0)                                             # (n_grasps_per_obj, top_k)
        batch_contact_points = torch.cat(all_contact_points, dim=0)                             # (n_grasps_per_obj, top_k, 3)
        batch_contact_points_normals = torch.cat(all_contact_points_normals, dim=0)             # (n_grasps_per_obj, top_k, 3)
        batch_contact_points_with_normals = torch.cat([batch_contact_points, batch_contact_points_normals], dim=-1)  # (n_grasps_per_obj, top_k, 6)
        batch_labels_hat = torch.cat(all_labels_hat, dim=0)                                     # (n_grasps_per_obj, top_k)

            # Run optimization 
        # opt_model.reset(robot_name, object_name, batch_object_pcd, batch_object_normals, batch_contact_points, batch_labels_hat)             # If dist euclidean
        opt_model.reset(robot_name, object_name.replace("/", "+"), batch_object_pcd, batch_object_normals, batch_contact_points_with_normals, batch_labels_hat)  # If dist aligned
        # opt_model.reset(robot_name, object_name, batch_object_pcd, batch_object_normals, projected_barycenters, batch_bary_labels)           # For barycenters instead of contact points
        start_time = time.time()
        q_traj, energy = opt_model.run(verbose=False)                                                             # (n_grasps_per_obj, max_iter, 16)
        opti_time = time.time() - start_time

        # Select the configuration with the lowest energy for each grasp
        # min_energy_indices = torch.argmin(energy, dim=1)  # (n_grasps_per_obj,)
        # batch_indices = torch.arange(n_grasps_per_obj, device=device)
        # q_opti = q_traj[batch_indices, min_energy_indices, :]  # (n_grasps_per_obj, 16)
        q_opti = q_traj[:, -1, :]   # Get the optimized gripper positions (last iteration)      # (n_grasps_per_obj, 16)

        all_gripper_pos_euler = q_rot6d_to_q_euler(all_gripper_pos_6d)
        predicted_q_full = torch.cat([all_gripper_pos_euler, q_opti], dim=1)  # Concatenate along feature dimension
        predicted_data.append({
            'object_name': object_name.replace("/", "+"),
            'predicted_q': predicted_q_full.cpu(),
        })

        # Print time statistics
        mean_time_cvae_per_grasps = (sum(time_cvae) / len(time_cvae)) / batch_size
        mean_time_pointnet_per_grasps = (sum(time_pointnet) / len(time_pointnet)) / batch_size
        mean_time_cvae_pointnet_per_grasps = (sum(time_cvae_pointnet) / len(time_cvae_pointnet)) / batch_size
        opti_time_per_grasps = opti_time / n_grasps_per_obj

        all_time_cvae.append(mean_time_cvae_per_grasps)
        all_time_pointnet.append(mean_time_pointnet_per_grasps)
        all_time_cvae_pointnet.append(mean_time_cvae_pointnet_per_grasps)
        all_time_opti.append(opti_time_per_grasps)

        cprint(f"[{args.robot_name}/{object_name}] CVAE: ", end='')
        cprint(f"{mean_time_cvae_per_grasps:.4e}", 'light_green', end='')
        cprint(" sec/grasps, PointNet: ", end='')
        cprint(f"{mean_time_pointnet_per_grasps:.4f}", 'light_green', end='')
        cprint(" sec/grasps, Full Pipeline: ", end='')
        cprint(f"{mean_time_cvae_pointnet_per_grasps:.4f}", 'light_green', end='')
        cprint(" sec/grasps, Optimization: ", end='')
        cprint(f"{opti_time_per_grasps:.4f}", 'light_green', end='')
        cprint(" sec/grasps, Overall time: ", end='')
        cprint(f"{mean_time_cvae_pointnet_per_grasps + opti_time_per_grasps:.4f}", 'light_green', attrs=['bold'], end='')
        cprint(" sec/grasps")

    date_str = datetime.now().strftime('%m%d%Y')
    result_path = os.path.join(ROOT_PATH, 'logs_inference_grasps', f'{date_str}')
    if not os.path.exists(result_path):
        os.makedirs(result_path, exist_ok=False)
    file_name = f'{DATASET}_{robot_name}_predicted_q_r{str(args.radius)[2:]}.pt'
    torch.save(predicted_data, os.path.join(result_path, file_name))

    # Print overall statistics
    mean_time_cvae = sum(all_time_cvae) / len(all_time_cvae)
    mean_time_pointnet = sum(all_time_pointnet) / len(all_time_pointnet)
    mean_time_cvae_pointnet = sum(all_time_cvae_pointnet) / len(all_time_cvae_pointnet)
    mean_time_opti = sum(all_time_opti) / len(all_time_opti)
    
    cprint(f"********************************************************** [{args.robot_name.upper()}] OVERALL MEAN TIME", 'cyan')
    cprint(f"CVAE: ", 'cyan', end='')
    cprint(f"{mean_time_cvae:.4e}", 'cyan', attrs=['bold'], end='')
    cprint(" sec/grasps, PointNet: ", 'cyan', end='')
    cprint(f"{mean_time_pointnet:.4f}", 'cyan', attrs=['bold'], end='')
    cprint(" sec/grasps, Full Pipeline: ", 'cyan', end='')
    cprint(f"{mean_time_cvae_pointnet:.4f}", 'cyan', attrs=['bold'], end='')
    cprint(" sec/grasps, Optimization: ", 'cyan', end='')
    cprint(f"{mean_time_opti:.4f}", 'cyan', attrs=['bold'], end='')
    cprint(" sec/grasps, Overall time: ", 'cyan', end='')
    cprint(f"{mean_time_cvae_pointnet + mean_time_opti:.4f}", 'cyan', attrs=['bold'], end='')
    cprint(" sec/grasps", 'cyan')

