import subprocess
import os
from termcolor import cprint
import argparse
import re
from datetime import datetime

from utils.constants import ROOT_PATH

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_name', default='allegro', type=str, help='Name of the gripper to use')
    parser.add_argument('--rt', default='given', type=str, help='RT Given or Sampled')
    parser.add_argument('--radius', default=0.01, type=float, help='Radius for sampling gripper poses')
    parser.add_argument('--dataset', default='multidex', type=str, help='Dataset name if specific object')

    args_ = parser.parse_args()
    return args_

if __name__ == "__main__":
    args = get_parser()

    cprint(f"********************************************************** [{args.robot_name.upper()} - Isaac Validation]", 'yellow', attrs=['bold'])    

    # Ensure the path to the Isaac Validator script is correct
    script_path = os.path.join(ROOT_PATH, 'utils_validation/isaac_main.py')
    # Run the Isaac Validator script as a subprocess
    subprocess_args = [
        'python',
        script_path,
        '--robot_name', args.robot_name,
        '--rt', args.rt,
        '--radius', str(args.radius),
        '--dataset', args.dataset,
    ]

    ret = subprocess.run(subprocess_args, stdout=subprocess.PIPE)

    # Find all result lines and extract object names, successes and totals
    pattern = re.compile(r'\[(.*?)\] Result: (\d+)/(\d+)')
    matches = pattern.findall(ret.stdout.decode('utf-8'))

    # Find all diversity lines and extract diversity values
    diversity_pattern = re.compile(r'Diversity \(std\): ([\d\.]+)')
    diversity_matches = diversity_pattern.findall(ret.stdout.decode('utf-8'))
    diversity_std = float(diversity_matches[-1]) if diversity_matches else 0.0

    # Prepare output strings
    output_lines = ["\n"]
    output_lines.append(f"********************************************************** [{args.robot_name.upper()} - Results]")
    
    cprint(f"********************************************************** [{args.robot_name.upper()} - Results]", 'light_green', attrs=['bold'])    
    
    total_success = 0
    total_trials = 0
    for object_name, success, trials in matches:
        success = int(success)
        trials = int(trials)
        success_rate = 100.0 * success / trials if trials > 0 else 0.0
        total_success += success
        total_trials += trials

        object_name = object_name.split('/')[-1]  # Get the last part of the object name
        color = 'light_red' if success_rate < 50 else 'light_yellow' if success_rate < 80 else 'light_green'
        
        # Format the line for both display and logging
        if len(object_name) < 10:
            line = f"|-- {object_name}\t\t\t --  {success}/{trials}\t({success_rate:.2f} %)"
            cprint(f"|-- {object_name}\t\t\t -- ", end='')
        elif len(object_name) < 20:
            line = f"|-- {object_name}\t\t --  {success}/{trials}\t({success_rate:.2f} %)"
            cprint(f"|-- {object_name}\t\t -- ", end='')
        else:
            line = f"|-- {object_name}\t --  {success}/{trials}\t({success_rate:.2f} %)"
            cprint(f"|-- {object_name}\t -- ", end='')
            
        output_lines.append(line)
        
        cprint(f" {success}/{trials}\t", color, attrs=['bold'], end='')
        cprint(f"(", end='')
        cprint(f"{success_rate:.2f} %", color, attrs=['bold'], end='')
        cprint(f")")        

    if total_trials != 0:
        overall_success_rate = 100.0 * total_success / total_trials
    else:
        overall_success_rate = 0.0
    output_lines.append(f"********************************************************** [{args.robot_name.upper()}] Overall Success Rate: {overall_success_rate:.2f} % - Diversity (std): {diversity_std:.4f}")
    
    cprint(f"********************************************************** [{args.robot_name.upper()}] Overall Success Rate: ", 'cyan', end='')
    cprint(f"{overall_success_rate:.2f} %", 'cyan', attrs=['bold'], end=' ')
    cprint(f" - Diversity (std): ", 'cyan', end='')
    cprint(f"{diversity_std:.4f}", 'cyan', attrs=['bold'])
    
    # Write to log file
    date_str = datetime.now().strftime('%m%d%Y')
    log_file = os.path.join('logs_validation', f'{date_str}_results_{args.dataset}.txt')
    if not os.path.exists('logs_validation'):
        os.makedirs('logs_validation', exist_ok=False)
    with open(log_file, 'a') as f:
        f.write('\n'.join(output_lines) + '\n')