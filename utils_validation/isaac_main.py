from utils_validation.isaac_validator import IsaacValidator # Import isaacgym modules before importing torch to avoid segmentation fault

from termcolor import cprint
import os
from datetime import datetime
import argparse
from tqdm import tqdm
import json
from utils.constants import ROOT_PATH, DATA_PATH


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_name', default='allegro', type=str, help='Name of the gripper to use')
    parser.add_argument('--rt', default='given', type=str, help='RT Given or Sampled')
    parser.add_argument('--radius', default=0.01, type=float, help='Radius for sampling gripper poses')
    parser.add_argument('--dataset', default='none', type=str, help='Dataset name if specific object')
    args_ = parser.parse_args()
    return args_


if __name__ == "__main__":
    
    import torch

    args = get_parser()

    device = 'cuda'
    object_scale=1.0

    # Read q values from the grasps files and create a q_batch tensor
    date_str = datetime.now().strftime('%m%d%Y')
    log_path = os.path.join(ROOT_PATH, 'logs_inference_grasps', f'{date_str}')
    all_files = os.listdir(log_path)
    file_name = f'{args.dataset}_{args.robot_name}_predicted_q_r{str(args.radius)[2:]}.pt'
    metadata = torch.load(os.path.join(log_path, file_name), map_location=device, weights_only=True)

    cprint(f"********************************************************** [{args.robot_name.upper()} - Isaac Validation]", 'yellow', attrs=['bold'])    

    save_every_steps = 25   # Save every n steps and quit to avoid segmentation fault from isaacgym
    
    # Resume logic: check for existing partial results
    result_path = os.path.join(ROOT_PATH, 'logs_isaac', f'{date_str}')
    if not os.path.exists(result_path):
        os.makedirs(result_path, exist_ok=True)
    partial_file_name = f'partial_{args.robot_name}_validation_results.pt'
    partial_file_path = os.path.join(result_path, partial_file_name)

    start_idx = 0
    if os.path.exists(partial_file_path):
        checkpoint = torch.load(partial_file_path, weights_only=True)
        all_validation_results = checkpoint['all_validation_results']
        all_success_rates = checkpoint['all_success_rates']
        all_success_q = checkpoint['all_success_q']
        matches = checkpoint['matches']
        start_idx = checkpoint['last_idx'] + 1
        cprint(f"Resuming from iteration {start_idx}", 'yellow', attrs=['bold'])
    else:
        all_validation_results = []
        all_success_rates = []
        all_success_q = []
        matches = []
        start_idx = 0

    with open(os.path.join(DATA_PATH, 'pointclouds', f'split_{args.dataset}.json'), 'r') as f:
        split_data = json.load(f)
    objects_list = split_data['test_split']

    for idx, data in enumerate(tqdm(metadata, desc=f"[{args.robot_name}] Validating: ", unit='objects')):
        if idx < start_idx:
            continue
        i = idx + 1
        object_name = data['object_name'].replace("+", "/")

        if object_name not in objects_list:
            continue

        q_batch = data['predicted_q'].to(device)

        cprint(f"--- [{args.robot_name}] Validating object {i}/{len(metadata)}: {object_name} ---", 'magenta', attrs=['bold'])

        object_path = os.path.join(DATA_PATH, 'urdf/objects')

        if args.dataset == 'multidex':
            object_name_split = object_name.split('/')
            object_urdf_path = f'{args.dataset}/{object_name_split[0]}/{object_name_split[1]}/coacd_decomposed_object_one_link.urdf'
        else:
            object_urdf_path = f'{args.dataset}/{object_name}.urdf'

        try:
            simulator = IsaacValidator(
                robot_name=args.robot_name,
                batch_size=1,
                use_gui=False,
                use_controller=False,
                use_stiffness=True,
                gpu=0,
                debug_interval=0.01,
            )
            simulator.batch_size = q_batch.shape[0]
            simulator.set_asset(
                object_path=object_path,
                object_file=object_urdf_path,
                scale=object_scale,
            )
            simulator.create_envs()
            simulator.set_actor_pose_dof(q_batch.to(torch.device('cpu')))
            success, q_isaac = simulator.run_sim()
            simulator.destroy()
        except:
            simulator.destroy()
            continue
        
        

        all_validation_results.append({
            'object_name': object_name.replace("/", "+"),
            'predicted_q': q_batch.cpu(),
            'q_isaac': q_isaac.cpu(),
            'success': success.cpu(),
        })
        # Print Success Rate
        success_num = success.sum().item()
        success_rate = success_num / q_batch.shape[0] * 100
        all_success_rates.append(success_rate)
        all_success_q.append(q_isaac[success])

        color = 'light_red' if success_rate < 34 else 'light_yellow' if success_rate < 67 else 'light_green'
        cprint(f"[{args.robot_name}/{object_name}] Result: ", end='')
        cprint(f"{success_num}/{q_batch.shape[0]}", color, attrs=['bold'], end='')
        cprint(f" (", end='')
        cprint(f"{success_rate:.2f} %", color, attrs=['bold'], end='')
        cprint(f")")

        matches.append((object_name, success_num, q_batch.shape[0]))

        if i == len(metadata):
            result_path = os.path.join(ROOT_PATH, 'logs_isaac', f'{date_str}')
            if not os.path.exists(result_path):
                os.makedirs(result_path, exist_ok=False)
            file_name = f'{args.robot_name}_validation_results_{args.dataset}_r{str(args.radius)[2:]}.pt'
            torch.save(all_validation_results, os.path.join(result_path, file_name))
            break

        # Save every steps and quit
        elif (i % save_every_steps == 0):
            torch.save({
                'all_validation_results': all_validation_results,
                'all_success_rates': all_success_rates,
                'all_success_q': all_success_q,
                'matches': matches,
                'last_idx': idx
            }, partial_file_path)
            cprint(f"Saved checkpoint at iteration {i}. Exiting to avoid segmentation fault.", 'red', attrs=['bold'])
            break

