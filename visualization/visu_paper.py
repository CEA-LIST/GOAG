import viser
import torch
import os
import numpy as np
import trimesh
from scipy.spatial import KDTree
import matplotlib as mpl
import time
import colorsys
import open3d as o3d
from utils.constants import DATA_PATH, ALLEGRO_GRASPS_LABELS
from utils.get_models import get_handmodel
from utils.rot6d import q_euler_to_q_rot6d
from utils.tools import get_contact_map, get_cp_from_grasp, get_cp_clusters_from_grasp
from scipy.spatial.transform import Rotation as R

def apply_contact_map_on_trimesh(trimesh_obj, contact_map_values, handprints_points, cmap):
    # For each vertex in the robot mesh, find the closest handprint point
    mesh_vertices = trimesh_obj.vertices  # (N, 3)
    tree = KDTree(handprints_points)
    dists, idxs = tree.query(mesh_vertices)  # idxs: (N,)

    # Remove indices where the distance is greater than a certain threshold
    mask = dists <= 0.01
    filtered_idxs = idxs[mask]
    # If you want to color only the filtered vertices, update vertex_colors and robot_trimesh accordingly
    vertex_colors_float = cmap(contact_map_values[filtered_idxs])[:, :3]
    vertex_colors = (vertex_colors_float * 255).astype(np.uint8)
    # Add alpha channel if needed
    if trimesh_obj.visual.vertex_colors.shape[1] == 4:
        alpha = np.full((vertex_colors.shape[0], 1), 255, dtype=np.uint8)
        vertex_colors = np.concatenate([vertex_colors, alpha], axis=1)

    # Set all vertices to a default color first
    default_color = np.array([200, 200, 200, 20], dtype=np.uint8) if trimesh_obj.visual.vertex_colors.shape[1] == 4 else np.array([200, 200, 200], dtype=np.uint8)
    trimesh_obj.visual.vertex_colors = np.tile(default_color, (mesh_vertices.shape[0], 1))

    # Update only the filtered vertices with the colormap
    trimesh_obj.visual.vertex_colors[mask] = vertex_colors

    return trimesh_obj

def project_mesh_to_pcd(mesh, pcd):
    mesh_vertices = mesh.vertices
    mesh_colors = mesh.visual.vertex_colors

    tree = KDTree(mesh_vertices)
    dists, idxs = tree.query(pcd)
    mask = dists < 0.01
    filtered_idxs = idxs[mask]

    # Assign mesh colors to matched pcd points
    pcd_colors = mesh_colors[filtered_idxs][:, :3]
    pcd = pcd[mask]
    # Remove points where the color is not valid
    mask_valid = ~np.all(pcd_colors == np.array([200, 200, 200]), axis=1)
    pcd = pcd[mask_valid]
    pcd_colors = pcd_colors[mask_valid]
    return pcd, pcd_colors

def main():

    robot_name = 'allegro'
    object_name = 'ycb+tomato_soup_can' # Allegro

    device = 'cuda' 

    server = viser.ViserServer(host='127.0.0.1', port=6006)
    cmap_robot = mpl.colormaps.get_cmap('Reds')

    object_name_split = object_name.split('+')
    object_path = os.path.join(DATA_PATH, 'urdf/objects/multidex', f'{object_name_split[0]}/{object_name_split[1]}/{object_name_split[1]}.stl')
    object_trimesh = trimesh.load_mesh(object_path)
    object_pcd_dense = trimesh.sample.sample_surface_even(object_trimesh, count=20000)[0]  # (10000, 3)
    object_pcd_dense = torch.tensor(object_pcd_dense, dtype=torch.float32, device=device)  # (10000, 3)

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(object_pcd_dense.squeeze(0).cpu().numpy())
    pcd_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd_o3d.orient_normals_consistent_tangent_plane(k=30)

    object_pcd = torch.tensor(np.asarray(pcd_o3d.points), dtype=torch.float32, device=device)  # (10000, 3)
    object_normals = torch.tensor(np.asarray(pcd_o3d.normals), dtype=torch.float32, device=device)  # (10000, 3)
    object_pcd_normals = torch.cat([object_pcd, object_normals], dim=-1)  # (10000, 6)

    hand_model = get_handmodel(robot_name, batch_size=1, device=device, num_points=4096)

    # ALLEGRO - TOMATO SOUP CAN
    q = torch.tensor([-0.03067869, -0.04470987, -0.01299424, -0.6554664, -1.4952718, 0.9188983] +
                    [-0.0956918,   1.4636681, 0.7709917,   0.6016648] +              # Little
                    [0.05174683,  1.4636681, 0.7709917,   0.6016648] +              # Middle
                    [0.05983057,  0.8852127,   1.4871141,   0.2887494] +              # Index
                    [0.5,   1.5,   1.0,  1.4],               # Thumb
                    dtype=torch.float, device='cuda').unsqueeze(0)
    
    q = q_euler_to_q_rot6d(q)  # Convert to 6D representation
    robot_trimesh = hand_model.get_trimesh_data(q, show_color_map=False)
    robot_trimesh_link_map = hand_model.get_trimesh_data(q, show_color_map=True)

    handprints_lbl = hand_model.get_handprint_points(q, label=True)
    handprints, labels = torch.split(handprints_lbl, [6, 1], dim=-1)
    handprints_points, handprints_normals = torch.split(handprints, [3, 3], dim=-1)  # (1, 2048, 3), (1, 2048, 3)
    handprints_points, labels = handprints_points.squeeze(0), labels.squeeze(0)  # (2048, 3), (2048, 1)

    grasp_mask_c1 = get_cp_from_grasp(handprints_points, labels, list_grasp_labels=ALLEGRO_GRASPS_LABELS['c1'], ratio=1.0)
    grasp_c1 = handprints_points[grasp_mask_c1 == 1.0]

    grasp_mask_c6 = get_cp_from_grasp(handprints_points, labels, list_grasp_labels=ALLEGRO_GRASPS_LABELS['c6'], ratio=1.0)
    grasp_c6 = handprints_points[grasp_mask_c6 == 1.0]

    grasp_mask_f23 = get_cp_from_grasp(handprints_points, labels, list_grasp_labels=ALLEGRO_GRASPS_LABELS['f23'], ratio=1.0)
    grasp_f23 = handprints_points[grasp_mask_f23 == 1.0]

    grasp_mask_f27 = get_cp_from_grasp(handprints_points, labels, list_grasp_labels=ALLEGRO_GRASPS_LABELS['f27'], ratio=1.0)
    grasp_f27 = handprints_points[grasp_mask_f27 == 1.0]

    grasp_mask_29 = get_cp_from_grasp(handprints_points, labels, list_grasp_labels=ALLEGRO_GRASPS_LABELS['f29'], ratio=1.0)
    grasp_f29 = handprints_points[grasp_mask_29 == 1.0]

    grasp_mask_f34 = get_cp_from_grasp(handprints_points, labels, list_grasp_labels=ALLEGRO_GRASPS_LABELS['f34'], ratio=1.0)
    grasp_f34 = handprints_points[grasp_mask_f34 == 1.0]

    contact_points_mask = get_cp_clusters_from_grasp(handprints_points, labels, list_grasp_labels=ALLEGRO_GRASPS_LABELS['c1'], ratio=0.4)  # (N_cp, 3)
    contact_points = handprints_points[contact_points_mask == 1.0]
    contact_labels = labels[contact_points_mask == 1.0].squeeze(-1).cpu().numpy()

    generate_color_map = lambda N: torch.tensor([colorsys.hsv_to_rgb(i / N, torch.rand(1).item() * 0.5 + 0.5, torch.rand(1).item() * 0.5 + 0.5) for i in torch.randperm(N)])
    color_map = lambda my_list: generate_color_map(int(np.max(my_list)) + 1)[np.array(my_list, dtype=int)]
    contact_labels_colors = color_map(contact_labels + np.abs(np.min(contact_labels)))

    workspace = torch.load(os.path.join(DATA_PATH, f'workspaces/{robot_name}_workspace.pt'), map_location='cuda', weights_only=True)[:, :3]     # (8192, 3)
    # Apply translation and euler rotation
    translation = torch.tensor([-0.03067869, -0.04470987, -0.01299424], device=workspace.device)
    euler_angles = torch.tensor([-0.6554664, -1.4952718, 0.9188983 + torch.pi], device=workspace.device)
    # Convert euler angles to rotation matrix (ZYX order)
    rot = torch.tensor(R.from_euler('zyx', euler_angles.cpu().numpy()).as_matrix(), device=workspace.device, dtype=workspace.dtype)
    workspace = ((workspace @ rot.T) + translation).cpu().numpy()

    # handprint_contact_map, _ = get_contact_map(handprints_points, object_pcd_normals)
    # robot_trimesh_cmap = apply_contact_map_on_trimesh(robot_trimesh.copy(), handprint_contact_map.squeeze(0).cpu().numpy(), handprints_points.cpu().numpy(), cmap_robot)
    handprint_contact_map, _ = get_contact_map(contact_points, object_pcd_normals)
    robot_trimesh_cmap = apply_contact_map_on_trimesh(robot_trimesh.copy(), handprint_contact_map.squeeze(0).cpu().numpy(), contact_points.cpu().numpy(), cmap_robot)
    w_hand_cmap_pcd, w_hand_cmap_colors = project_mesh_to_pcd(robot_trimesh_cmap, workspace)

    w_hand_link_map, w_hand_link_map_colors = project_mesh_to_pcd(robot_trimesh_link_map, workspace)

    # server.scene.add_point_cloud('workspace', workspace, colors=(200,200,200), point_shape='rounded', point_size=0.0007)
    server.scene.add_mesh_trimesh('robot', robot_trimesh, visible=True)
    server.scene.add_point_cloud('handprints', handprints_points.cpu().numpy(), colors=(0, 0, 255), point_shape='rounded', point_size=0.001)
    server.scene.add_point_cloud('grasp_pcd_c1', grasp_c1.cpu().numpy(), colors=(0, 0, 0), point_shape='rounded', point_size=0.001)
    server.scene.add_point_cloud('grasp_pcd_c6', grasp_c6.cpu().numpy(), colors=(0, 0, 0), point_shape='rounded', point_size=0.001)
    server.scene.add_point_cloud('grasp_pcd_f23', grasp_f23.cpu().numpy(), colors=(0, 0, 0), point_shape='rounded', point_size=0.001)
    server.scene.add_point_cloud('grasp_pcd_f27', grasp_f27.cpu().numpy(), colors=(0, 0, 0), point_shape='rounded', point_size=0.001)
    server.scene.add_point_cloud('grasp_pcd_f29', grasp_f29.cpu().numpy(), colors=(0, 0, 0), point_shape='rounded', point_size=0.001)
    server.scene.add_point_cloud('grasp_pcd_f34', grasp_f34.cpu().numpy(), colors=(0, 0, 0), point_shape='rounded', point_size=0.001)
    
    server.scene.add_point_cloud('contact_points', contact_points.cpu().numpy(), colors=(255, 0, 0), point_shape='rounded', point_size=0.001)
    # server.scene.add_point_cloud('workspace_hand_cmap', w_hand_cmap_pcd, colors=w_hand_cmap_colors, point_shape='rounded', point_size=0.0025, visible=False)
    # server.scene.add_point_cloud('workspace_hand_link_map', w_hand_link_map, colors=w_hand_link_map_colors, point_shape='rounded', point_size=0.0025, visible=False)
    server.scene.add_point_cloud('contact_points', contact_points.cpu().numpy(), colors=(150, 0, 0), point_shape='rounded', point_size=0.002)
    server.scene.add_point_cloud('contact_labels', contact_points.cpu().numpy(), colors=contact_labels_colors, point_shape='rounded', point_size=0.002)

    while True:
        time.sleep(1)

if __name__ == '__main__':
    main()