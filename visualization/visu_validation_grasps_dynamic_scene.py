import os, sys
from datetime import datetime
import torch
import viser
import time
import trimesh
import numpy as np
from math import pi as PI
from scipy.spatial.transform import Rotation as R
import math
import random

from utils.get_models import get_handmodel
from utils_model.HandModel import HandModel
from utils.constants import ROOT_PATH, DATA_PATH
from utils.rot6d import q_euler_to_q_rot6d


device = 'cuda'
dataset = 'multidex'

def load_visu_info(grasps_per_obj=5):
    date_str = datetime.now().strftime('%m%d%Y')
    # date_str = '09112025'
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

    # Filter only successful grasps and limit to grasps_per_obj
    filtered_vis_info = []
    for info in vis_info:
        mask = info['success']
        info['predicted_q'] = info['predicted_q'][mask][:grasps_per_obj]
        info['q_isaac'] = info['q_isaac'][mask][:grasps_per_obj]
        # Add the filtered dictionary to our new list
        filtered_vis_info.append(info)
    vis_info = filtered_vis_info

    return vis_info

def load_single_grasp():   
    # object_name = 'ycb+potted_meat_can'
    # q_string = '{-0.05212658271193504, 0.006612053606659174, 0.07994189858436584, -1.5044550895690918, 0.21188464760780334, 1.3776154518127441, 0.42924484610557556, 0.46444016695022583, 0.6170399188995361, 0.5269085764884949, 0.11469155550003052, 1.0281413793563843, 0.8160648345947266, 0.54860919713974, 0.23220312595367432, 1.0052032470703125, 0.6952430605888367, 0.8499584794044495, 1.0372591018676758, 0.8302886486053467, 0.6797212958335876, 0.4978525936603546}'
    
    # object_name = 'ycb+tomato_soup_can'
    # q_string = '{0.004611401818692684, -0.046945057809352875, 0.02985387109220028, 1.1761835813522339, -1.3470265865325928, 2.7820863723754883, -0.34157460927963257, 1.0231032371520996, 0.715347170829773, 1.433565378189087, -0.35530081391334534, 0.9703208208084106, 0.7884077429771423, 1.5136690139770508, -0.21763938665390015, 0.6451674699783325, 0.5535324215888977, 0.5472055673599243, 1.1279971599578857, 0.8969464302062988, 0.904589831829071, 0.9443775415420532}'

    # object_name = 'ycb+baseball'
    # q_string = '{-0.012179823592305183, -0.038536906242370605, 0.027416834607720375, -0.7677930593490601, -1.1143131256103516, 1.3879581689834595, -0.1164468377828598, 1.273723840713501, 1.050407886505127, 0.9861451387405396, 0.09016821533441544, 1.051138162612915, 0.9331105351448059, 0.9412038922309875, 0.24434803426265717, 1.0632102489471436, 0.8889516592025757, 0.8454704284667969, 0.9223315119743347, 0.6758972406387329, 0.7597538828849792, 1.149966835975647}'
    
    object_name = 'ycb+bowl'
    q_string = '{-0.007151256315410137, 0.01990089938044548, 0.07096679508686066, -1.7277506589889526, -0.1873166263103485, 1.6885319948196411, 0.4708242416381836, 1.0258872509002686, 1.1496142148971558, 1.0876705646514893, -0.23218387365341187, 0.2621847987174988, 0.8629964590072632, 0.7467960715293884, 0.07832244038581848, 0.21708127856254578, 1.141969919204712, 0.8024187684059143, 0.5562161207199097, 1.0092312097549438, 0.5822309255599976, 0.8344998955726624}'

    cleaned_string = q_string.strip().replace("{", "").replace("}", "").replace("\n", "").replace(" ", "")
    number_strings = cleaned_string.split(",")
    q_list = [float(num) for num in number_strings if num]
    q_eul = torch.tensor(q_list, device=device).float().unsqueeze(0)

    visu_info = [{'robot_name': 'allegro', 'object_name': object_name, 'q_isaac': q_eul}]
    return visu_info

def get_meshes(object_name, robot_name, q_eul):

    if ('+' in object_name):    # For multidex dataset
        object_name_split = object_name.split('+')
        object_path = os.path.join(DATA_PATH, 'urdf/objects', f'{dataset}/{object_name_split[0]}/{object_name_split[1]}/{object_name_split[1]}.stl')
    else:
        try:
            object_path = os.path.join(DATA_PATH, 'urdf/objects', f'{dataset}/{object_name}.obj')
        except:
            object_path = os.path.join(DATA_PATH, 'urdf/objects', f'{dataset}/{object_name}.ply')
    object_trimesh = trimesh.load_mesh(object_path)
    object_trimesh.visual.material = trimesh.visual.texture.SimpleMaterial(
        metallic=1.0,  # A value between 0.0 and 1.0 for metallic finish
        roughness=0.1  # A value between 0.0 and 1.0 for roughness
    )

    hand_model : HandModel = get_handmodel(robot=robot_name, batch_size=1, device=device)
    isaac_q = q_euler_to_q_rot6d(q_eul.unsqueeze(0))
    robot_trimesh = hand_model.get_trimesh_data(isaac_q)
    if robot_name == 'barrett':
        robot_trimesh.visual.face_colors = np.array([153, 255, 204, 255]) # R, G, B, Alpha
    elif robot_name == 'allegro':
        robot_trimesh.visual.face_colors = np.array([255, 153, 204, 255]) # R, G, B, Alpha
    elif robot_name == 'shadowhand':
        robot_trimesh.visual.face_colors = np.array([153, 153, 255, 255]) # R, G, B, Alpha

    return object_trimesh, robot_trimesh

def main():

    vis_info = load_visu_info(grasps_per_obj=3)
    # vis_info = load_single_grasp()

    assert vis_info != [], "No valid visualization information found."

    # random.shuffle(vis_info)

    total_grasps = sum([info['q_isaac'].shape[0] for info in vis_info])
    print(total_grasps)

    server = viser.ViserServer(host='127.0.0.1', port=6006)
    server.scene.configure_default_lights()

    rows = int(math.sqrt(total_grasps))
    cols = int(math.ceil(total_grasps / rows))
    spacing_rows = 0.25
    spacing_cols = 0.25


    print(f"Adding {total_grasps} grasps to the scene...")
    with server.gui.add_folder("Scene Controls"):
        # Slider to control the rotation speed.
        rotation_speed = server.gui.add_slider("Rotation Speed", min=0.0, max=2.0, step=0.01, initial_value=0.25)
        # Checkbox to toggle rotation.
        is_rotating = server.gui.add_checkbox("Enable Rotation", initial_value=True)

    grasp_handles_pairs = []
    total_grasp_index = 0
    for info in vis_info:
        for grasp_q in info['q_isaac']:
            object_mesh, gripper_mesh = get_meshes(info['object_name'], info['robot_name'], grasp_q)
            
            # Calculate grid position for this specific grasp.
            row_index = total_grasp_index // cols
            col_index = total_grasp_index % cols
            
            # Calculate the world position for this grasp. We'll center the grid.
            x = (col_index - (cols - 1) / 2) * spacing_cols
            y = (row_index - (rows - 1) / 2) * spacing_rows
            z = 0.0
            position = np.array([x, y, z])
            # position = np.array([x, z, y])

            initial_transform = np.eye(4)
            initial_transform[:3, 3] = position
            
            # Add the object mesh to the scene. Each object needs a unique name.
            obj_handle = server.scene.add_mesh_trimesh(
                f"grasp_{total_grasp_index}/object",
                mesh=object_mesh,
                position=position,
            )

            # Add the gripper mesh to the scene. Each gripper needs a unique name.
            gripper_handle = server.scene.add_mesh_trimesh(
                f"grasp_{total_grasp_index}/gripper",
                mesh=gripper_mesh,
                position=position,
            )
            grasp_handles_pairs.append((obj_handle, gripper_handle))

            total_grasp_index += 1

    # while True:
    #     time.sleep(0)

    # --- Real-time rotation loop ---
    while True:

        # Check if rotation is enabled.
        if is_rotating.value:
            # Calculate the rotation angle based on time and the rotation speed slider.
            angle = time.time() * rotation_speed.value
            
            # Create a rotation quaternion for a rotation around the Z-axis.
            rotation_quaternion = trimesh.transformations.quaternion_about_axis(angle, [0, 0, 1])

            # Update the orientation for each grasp by updating both mesh handles.
            for obj_handle, gripper_handle in grasp_handles_pairs:
                obj_handle.wxyz = rotation_quaternion
                gripper_handle.wxyz = rotation_quaternion

        time.sleep(0.01)

if __name__ == '__main__':
    main()