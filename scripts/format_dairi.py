import os
import json
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import open3d as o3d

def json_reader(json_file):
    f = open(json_file, 'rb')
    txt = f.read().decode('utf-8')
    f.close()
    x = json.loads(txt)
    return x

def calib_parser(sample_name, calib_path):
    intrinsic_path = calib_path + 'camera_intrinsic/' + sample_name + '.json'
    extrinsic_path = calib_path + 'virtuallidar_to_camera/' + sample_name + '.json'
    intrinsic_info = json_reader(intrinsic_path)
    P = intrinsic_info['P']
    cam_K = intrinsic_info['cam_K']
    cam_D = intrinsic_info['cam_D']
    extrinsic_info = json_reader(extrinsic_path)
    R = extrinsic_info['rotation']  # 3x3
    T = extrinsic_info['translation']  # 3x1
    P = np.array(P, dtype=np.float32)
    if len(P)==12:
        P = P.reshape((3, 4))
        P = P[:3,:3]
    elif len(P)==9:
        P = P.reshape((3, 3))
    cam_K = np.array(cam_K, dtype=np.float32)
    cam_K = cam_K.reshape((3, 3))
    cam_D = np.array(cam_D, dtype=np.float32)
    R = np.array(R, dtype=np.float32)
    T = np.array(T, dtype=np.float32)
    return cam_K, cam_D, P, R, T

def label_parser(sample_name, label_path):
    label_list = json_reader(label_path + sample_name + '.json')
    det3d_list = []
    for label in label_list:
        if label['type'] not in ['Car', 'Truck', 'Van', 'Bus', 'Pedestrian', 'Cyclist', 'Tricyclist', 'Motorcyclist'] :
            continue
        # 'vehicle', 'non-motor', 'person', 'tricycle'
        if label['type'] in ['Car', 'Truck', 'Van', 'Bus']:
            agent_type = 'vehicle'
        elif label['type'] in ['Pedestrian']:
            agent_type = 'person'
        elif label['type'] in ['Cyclist', 'Motorcyclist']:
            agent_type = 'non-motor'
        elif label['type'] in ['Tricyclist']:
            agent_type = 'tricycle'
        
        cur_dim   = label['3d_dimensions']
        cur_dim   = [cur_dim['l'], cur_dim['w'], cur_dim['h']]
        cur_dim   = [float(vv) for vv in cur_dim]
        cur_loc   = label['3d_location']
        cur_loc   = [cur_loc['x'], cur_loc['y'], cur_loc['z']]
        cur_loc   = [float(vv) for vv in cur_loc]
        cur_yaw   = float(label['rotation'])

        cur_3dinfo = cur_loc + cur_dim + [cur_yaw]
        cur_label  = {'type': agent_type, '3dinfo': cur_3dinfo}
        det3d_list.append(cur_label)
    return det3d_list

def obtain_filename_list(filepath):
    nlist = os.listdir(filepath)
    filename_list = []
    for name in nlist:
        if 'jpg' in name:
            filename_list.append(name[:-4])
    return filename_list

def rotation_matrix(euler_angles):
    tx, ty, tz = euler_angles
    Rx = np.array([[1,0,0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
    Ry = np.array([[np.cos(ty), 0, np.sin(ty)], [0, 1, 0], [-np.sin(ty), 0, np.cos(ty)]])
    Rz = np.array([[np.cos(tz), -np.sin(tz), 0], [np.sin(tz), np.cos(tz), 0], [0,0,1]])
    return np.dot(Rx, np.dot(Rz,Ry))

def RT_converter(sample_name, datapath):
    cam_K, cam_D, P, DAIRI_R, DAIRI_T = calib_parser(sample_name, datapath + 'calib/')
    cur_sample_labels = label_parser(sample_name, datapath + 'label/camera/')
    cur_frame_lidar_height = []
    for sample_label in cur_sample_labels:
        if sample_label['type'] in ['vehicle']:
            cur_frame_lidar_height.append(sample_label['3dinfo'][2]-sample_label['3dinfo'][5]/2.0)
    if len(cur_frame_lidar_height)<1:
        for sample_label in cur_sample_labels:
            cur_frame_lidar_height.append(sample_label['3dinfo'][2]-sample_label['3dinfo'][5]/2.0)
    if len(cur_frame_lidar_height)>0:
        cur_frame_lidar_height_mid = np.median(np.array(cur_frame_lidar_height))
    else:
        cur_frame_lidar_height_mid = 0
    
    R_GROUND2DAIR = rotation_matrix([0,0,-np.pi/2])
    T_GROUND2DAIR = np.array([[0],[0],[cur_frame_lidar_height_mid]], dtype=np.float32)

    GROUND_R = DAIRI_R.dot(R_GROUND2DAIR)
    GROUND_T = DAIRI_T + DAIRI_R.dot(T_GROUND2DAIR)

    update_sample_labels = []
    for sample_label in cur_sample_labels:
        obj_type = sample_label['type']
        obj_anno = sample_label['3dinfo']
        if True:
            obj_anno[0], obj_anno[1] = -obj_anno[1], obj_anno[0]
            obj_anno[2] = obj_anno[2] - cur_frame_lidar_height_mid
            obj_anno[6] += np.pi/2
        else:
            obj_anno[2] = obj_anno[2] - obj_anno[5]/2.0
        update_sample = {'type':obj_type, '3dinfo':obj_anno}
        update_sample_labels.append(update_sample)
    return update_sample_labels, GROUND_R, GROUND_T, cam_K, cam_D, P,  -cur_frame_lidar_height_mid

def rot_angle2matrix(euler_angles):
    tx, ty, tz = euler_angles
    Rx = np.array([[1,0,0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
    Ry = np.array([[np.cos(ty), 0, np.sin(ty)], [0, 1, 0], [-np.sin(ty), 0, np.cos(ty)]])
    Rz = np.array([[np.cos(tz), -np.sin(tz), 0], [np.sin(tz), np.cos(tz), 0], [0,0,1]])
    return np.dot(Rx, np.dot(Rz,Ry))

def lidarAnno2lidarCorners(lidar_anno):
    x, y, z, L, W, H, yaw = lidar_anno
    if True:
        obj_corners = [[ L/2.0,  W/2.0, H], 
                        [ L/2.0,  W/2.0, 0],
                        [ L/2.0, -W/2.0, H], 
                        [ L/2.0, -W/2.0, 0], 
                        [-L/2.0,  W/2.0, H], 
                        [-L/2.0,  W/2.0, 0],
                        [-L/2.0, -W/2.0, H], 
                        [-L/2.0, -W/2.0, 0]]
    else:
        obj_corners = [[ L/2.0,  W/2.0, H/2.0], 
                        [ L/2.0,  W/2.0, -H/2.0],
                        [ L/2.0, -W/2.0, H/2.0], 
                        [ L/2.0, -W/2.0, -H/2.0], 
                        [-L/2.0,  W/2.0, H/2.0], 
                        [-L/2.0,  W/2.0, -H/2.0],
                        [-L/2.0, -W/2.0, H/2.0], 
                        [-L/2.0, -W/2.0, -H/2.0]]
    obj_corners = np.array(obj_corners)
    euler_angles = [0, 0, yaw]
    R = rot_angle2matrix(euler_angles)
    loc = [x, y, z]
    lidar_corners = np.dot(R, obj_corners.T) + np.expand_dims(np.array(loc), 1)
    lidar_corners = lidar_corners.T
    return lidar_corners

def lidarAnno2pixCorners(lidar_anno, R, T, K):
    lidar_corners = lidarAnno2lidarCorners(lidar_anno)
    cam_corners_t = np.dot(R, lidar_corners.T) + T
    pixel_corners = np.dot(K, cam_corners_t)
    pixel_corners = pixel_corners[:2] / pixel_corners[2]
    pixel_corners = (pixel_corners.T).astype(np.int64)  # 8x2
    return pixel_corners

def save_GROUNDstyle_json(sample_labels, cur_R, cur_T, cur_cam_K, anno_savepath, calib_savepath):
    calib_dict = OrderedDict()
    cur_cam_K  = list(cur_cam_K)
    cur_cam_K  = [list(subArr) for subArr in cur_cam_K]
    cur_cam_K = [[float(vv) for vv in sublist] for sublist in cur_cam_K]
    calib_dict['P'] = cur_cam_K
    cur_RT = np.concatenate((cur_R, cur_T), 1)
    cur_RT_ = np.array([[0, 0, 0, 1]], dtype=np.float32)
    cur_RT = np.concatenate((cur_RT, cur_RT_), 0)
    cur_RT = list(cur_RT)
    cur_RT = [list(subArr) for subArr in cur_RT]
    cur_RT = [[float(vv) for vv in sublist] for sublist in cur_RT]
    calib_dict['RT'] = cur_RT
    f = open(calib_savepath, 'w')
    f.write(json.dumps(calib_dict))
    f.write('\n')
    f.close()
    label_dict = OrderedDict()
    label_dict["meta"] = {"width": 1920, "height": 1080}
    anno_list = []
    for sample_label in sample_labels:
        obj_type = sample_label['type']
        obj_anno = sample_label['3dinfo']
        cur_anno = OrderedDict()
        cur_anno['type'] = obj_type
        cur_anno['lidar'] = {"position": {"x": obj_anno[0], "y": obj_anno[1], "z": obj_anno[2]}, "angles": {"rx": 0, "ry": 0, "rz": obj_anno[6]}, "size": {"length": obj_anno[3], "width": obj_anno[4], "height": obj_anno[5]}}
        anno_list.append(cur_anno)
    label_dict['annotation'] = anno_list
    f = open(anno_savepath, 'w')
    f.write(json.dumps(label_dict))
    f.write('\n')
    f.close()
    return cur_cam_K, cur_RT

def read_pcd(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points)
    return pts

def project_cam2img(pt, cam_to_img):

    point = np.array(pt)

    point = np.dot(cam_to_img, point)

    point = point[:2]/point[2]
    point = point.astype(np.int64)

    return point

def obtain_point_depth(pts, K, RT):
    H = 1080
    W = 1920
    
    R = RT[:3, :3]
    T = RT[:3, 3].reshape([3, 1])
    N = pts.shape[0]
    pts_cam = R.dot(pts.T) + T
    pts_pix = project_cam2img(pts_cam, K)
    
    mask = np.ones(pts_cam.shape[1], dtype=bool)
    mask = np.logical_and(mask, pts_cam[2] > 0)
    mask = np.logical_and(mask, pts_pix[0, :] > 1)
    mask = np.logical_and(mask, pts_pix[0, :] < W - 1)
    mask = np.logical_and(mask, pts_pix[1, :] > 1)
    mask = np.logical_and(mask, pts_pix[1, :] < H - 1)
    points = pts_pix[:, mask]
    depth = pts_cam[2, mask]
    return points, depth

def obtain_point_height(pts, K, RT):
    H = 1080
    W = 1920
    
    R = RT[:3, :3]
    T = RT[:3, 3].reshape([3, 1])
    N = pts.shape[0]
    pts_cam = R.dot(pts.T) + T
    pts_T = pts.T
    pts_pix = project_cam2img(pts_cam, K)
    
    mask = np.ones(pts_cam.shape[1], dtype=bool)
    mask = np.logical_and(mask, pts_cam[2] > 0)
    mask = np.logical_and(mask, pts_pix[0, :] > 1)
    mask = np.logical_and(mask, pts_pix[0, :] < W - 1)
    mask = np.logical_and(mask, pts_pix[1, :] > 1)
    mask = np.logical_and(mask, pts_pix[1, :] < H - 1)
    points = pts_pix[:, mask]
    height = pts_T[2, mask]
    return points, height

def pcd2bin(path_pcd, K, RT, sample_name, shift_Z, path_depth, path_height):
    K = np.array(K, dtype=np.float32)
    RT = np.array(RT, dtype=np.float32)
    pts = read_pcd(path_pcd)
    if True:
        pts_x = -pts[:,1]
        pts_y = pts[:,0]
        pts_z = pts[:,2] + shift_Z
        pts = np.stack((pts_x, pts_y, pts_z), 1)
    
    points, depth = obtain_point_depth(pts, K, RT)
    ply_arr = np.concatenate([points.T, depth[:, None]], axis=1).astype(np.float32)
    ply_arr.flatten().tofile(path_depth)

    points, height = obtain_point_height(pts, K, RT)
    ply_arr = np.concatenate([points.T, height[:, None]], axis=1).astype(np.float32)
    ply_arr.flatten().tofile(path_height)

def process(datapath, outpath):
    os.makedirs(outpath, exist_ok=True)
    os.makedirs(outpath+'label/', exist_ok=True)
    os.makedirs(outpath+'calib/', exist_ok=True)
    os.makedirs(outpath+'depth_gt/', exist_ok=True)
    os.makedirs(outpath+'height_gt/', exist_ok=True)
    filename_list = obtain_filename_list(datapath+'image/')
    total_samples = len(filename_list)
    for i in tqdm(range(total_samples)):
        sample_name = filename_list[i]
        cur_sample_labels, cur_R, cur_T, cur_cam_K, cur_cam_D, cur_P, shift_Z = RT_converter(sample_name, datapath)
        cur_anno_savepath = outpath+'label/' + sample_name + '.json'
        cur_calib_savepath = outpath+'calib/' + sample_name + '.json'
        cur_calib_K, cur_calib_RT = save_GROUNDstyle_json(cur_sample_labels, cur_R, cur_T, cur_cam_K, cur_anno_savepath, cur_calib_savepath)
        path_depth = outpath+'depth_gt/'+sample_name+'.bin'
        path_height = outpath+'height_gt/'+sample_name+'.bin'
        pcd2bin(datapath+'velodyne/'+sample_name+'.pcd', cur_calib_K, cur_calib_RT, sample_name, shift_Z, path_depth,path_height)

process('../single-infrastructure-side/', '../GROUND_DAIRI/')
