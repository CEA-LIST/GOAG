import os
import torch
import viser
import time
import trimesh
from datetime import datetime
from utils.get_models import get_handmodel
from utils_model.HandModel import HandModel
from utils.constants import ROOT_PATH, DATA_PATH
from utils.rot6d import q_euler_to_q_rot6d

def main():


    device = 'cuda'
    dataset='multidex'
    date_str = datetime.now().strftime('%m%d%Y')
    # date_str = '09112025'

    # log_path = os.path.join(ROOT_PATH, 'logs_inference_grasps', f'{date_str}')
    log_path_isaac = os.path.join(ROOT_PATH, 'logs_isaac', f'{date_str}')
    
    vis_info = []
    for robot_name in ['allegro', 'barrett', 'shadowhand']:
        try:
            info = torch.load(
                os.path.join(log_path_isaac, f'{robot_name}_validation_results_{dataset}_r01.pt'),
                map_location=device,
                weights_only=True
            )
            info = [{'robot_name': robot_name, **item} for item in info]
            vis_info += info
        except Exception:
            continue


    assert vis_info != [], "No valid visualization information found."

    # Filter only successful grasps
    # filtered_vis_info = []
    # for info in vis_info:
    #     mask = info['success']
    #     info['predicted_q'] = info['predicted_q'][mask]
    #     info['q_isaac'] = info['q_isaac'][mask]
    #     # Add the filtered dictionary to our new list
    #     filtered_vis_info.append(info)
    # vis_info = filtered_vis_info


    def on_update(idx):

        invalid = True
        for info in vis_info:
            if idx >= info['predicted_q'].shape[0]:
                idx -= info['predicted_q'].shape[0]
            else:
                invalid = False
                break
        if invalid:
            print('Invalid index!')
            return

        robot_name = info['robot_name']

        object_name = info['object_name']
        if ('+' in object_name):    # For multidex dataset
            object_name_split = object_name.split('+')
            object_path = os.path.join(DATA_PATH, 'urdf/objects', f'{dataset}/{object_name_split[0]}/{object_name_split[1]}/{object_name_split[1]}.stl')
        else:
            try:
                object_path = os.path.join(DATA_PATH, 'urdf/objects', f'{dataset}/{object_name}.obj')
            except:
                object_path = os.path.join(DATA_PATH, 'urdf/objects', f'{dataset}/{object_name}.ply')
        object_trimesh = trimesh.load_mesh(object_path)
        server.scene.add_mesh_simple(
            'object',
            object_trimesh.vertices,
            object_trimesh.faces,
            color=(239, 132, 167),
            material='toon3',
            opacity=0.8
        )

        # print(f"Visualizing: {robot_name} - {object_name} - grasp {idx+1}/{info['predicted_q'].shape[0]} (isaac: {info['q_isaac'].shape[0]}) - Success: {info['success'][idx]}")
        print(f"Visualizing: {robot_name} - {object_name} - grasp {idx+1}/{info['predicted_q'].shape[0]}")
        

        # string = "{"
        # for v in info['q_isaac'][idx].cpu().numpy():
        #     string += f'{v}, '
        # string = string[:-2] + '}'
        # # print(f"Predicted q: {info['predicted_q'][idx].cpu().numpy()}")
        # print(f'q_isaac : {string}')

        hand_model : HandModel = get_handmodel(robot=robot_name, batch_size=1, device=device)

        pred_q = q_euler_to_q_rot6d(info['predicted_q'][idx].unsqueeze(0))
        robot_trimesh = hand_model.get_trimesh_data(pred_q)
        server.scene.add_mesh_simple(
            'robot_predict',
            robot_trimesh.vertices,
            robot_trimesh.faces,
            color=(102, 192, 255),
            opacity=0.7,
            visible=True,
        )

        isaac_q = q_euler_to_q_rot6d(info['q_isaac'][idx].unsqueeze(0))
        robot_trimesh = hand_model.get_trimesh_data(isaac_q)
        server.scene.add_mesh_simple(
            'robot_isaac',
            robot_trimesh.vertices,
            robot_trimesh.faces,
            color=(255, 102, 102),
            opacity=0.7
        )

    server = viser.ViserServer(host='127.0.0.1', port=6006)

    grasp_num = 0
    for info in vis_info:
        grasp_num += info['predicted_q'].shape[0]

    server.scene.world_axes.axes_length = 0.05
    server.scene.world_axes.axes_radius = 0.002
    server.scene.world_axes.origin_radius = 0.005
    server.scene.world_axes.origin_color = (236, 236, 0)
    server.scene.world_axes.visible = True

    slider = server.gui.add_slider(
        label='grasp_idx',
        min=0,
        max=grasp_num,
        step=1,
        initial_value=0
    )
    slider.on_update(lambda _: on_update(slider.value))
    on_update(0)

    while True:
        time.sleep(1)

if __name__ == '__main__':
    main()