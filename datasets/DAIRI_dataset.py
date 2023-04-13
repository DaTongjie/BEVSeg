from cmath import atanh
from configparser import Interpolation
import cv2
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import json
import mmcv
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes

def obtain_filenames(filenames_path):
    with open(filenames_path, 'r') as f:
        lines = f.readlines()
        filenames_list = [line.split()[0] for line in lines]
    return filenames_list

def parse_calib_file(path):
    f = open(path, 'rb')
    txt = f.read().decode('utf-8')
    f.close()
    x = json.loads(txt)
    intrin = x['P']
    lidar2cam = x['RT']
    return np.array(intrin), np.array(lidar2cam)

def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])

def img_transform(img, resize, resize_dims, crop, flip, rotate):
    ida_rot = torch.eye(2)
    ida_tran = torch.zeros(2)
    # adjust image
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)

    # post-homography transformation
    if len(resize) > 1:
        resize_w, resize_h = resize
        ida_rot = ida_rot.matmul(torch.Tensor([[resize_w, 0], [0, resize_h]]))
    else:
        ida_rot *= resize
    ida_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
    A = get_rot(rotate / 180 * np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    ida_rot = A.matmul(ida_rot)
    ida_tran = A.matmul(ida_tran) + b
    ida_mat = ida_rot.new_zeros(4, 4)
    ida_mat[3, 3] = 1
    ida_mat[2, 2] = 1
    ida_mat[:2, :2] = ida_rot
    ida_mat[:2, 3] = ida_tran
    return img, ida_mat

def bev_transform(gt_boxes, rotate_angle, scale_ratio, flip_dx, flip_dy):
    rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
    rot_sin = torch.sin(rotate_angle)
    rot_cos = torch.cos(rotate_angle)
    rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                            [0, 0, 1]])
    scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                              [0, 0, scale_ratio]])
    flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if flip_dx:
        flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if flip_dy:
        flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    rot_mat = flip_mat @ (scale_mat @ rot_mat)
    if gt_boxes.shape[0] > 0:
        gt_boxes[:, :3] = (rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
        gt_boxes[:, 3:6] *= scale_ratio
        gt_boxes[:, 6] += rotate_angle
        if flip_dx:
            gt_boxes[:, 6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:, 6]
        if flip_dy:
            gt_boxes[:, 6] = -gt_boxes[:, 6]
    return gt_boxes, rot_mat

def depth_transform(cam_depth, resize, resize_dims, crop, flip, rotate):
    """Transform depth based on ida augmentation configuration.

    Args:
        cam_depth (np array): Nx3, 3: x,y,d.
        resize (float): Resize factor.
        resize_dims (list): Final dimension.
        crop (list): x1, y1, x2, y2
        flip (bool): Whether to flip.
        rotate (float): Rotation value.

    Returns:
        np array: [h/down_ratio, w/down_ratio, d]
    """

    H, W = resize_dims
    if len(resize) > 1:
        resize_w, resize_h = resize
        cam_depth[:, 0] = cam_depth[:, 0] * resize_w
        cam_depth[:, 1] = cam_depth[:, 1] * resize_h
    else:
        cam_depth[:, :2] = cam_depth[:, :2] * resize
    cam_depth[:, 0] -= crop[0]
    cam_depth[:, 1] -= crop[1]
    if flip:
        cam_depth[:, 0] = resize_dims[1] - cam_depth[:, 0]

    cam_depth[:, 0] -= W / 2.0
    cam_depth[:, 1] -= H / 2.0

    h = rotate / 180 * np.pi
    rot_matrix = [
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ]
    cam_depth[:, :2] = np.matmul(rot_matrix, cam_depth[:, :2].T).T

    cam_depth[:, 0] += W / 2.0
    cam_depth[:, 1] += H / 2.0

    depth_coords = cam_depth[:, :2].astype(np.int16)

    depth_map = np.zeros(resize_dims)
    valid_mask = ((depth_coords[:, 1] < resize_dims[0])
                  & (depth_coords[:, 0] < resize_dims[1])
                  & (depth_coords[:, 1] >= 0)
                  & (depth_coords[:, 0] >= 0))
    depth_map[depth_coords[valid_mask, 1],
              depth_coords[valid_mask, 0]] = cam_depth[valid_mask, 2]

    return torch.Tensor(depth_map)  

def height_transform(cam_height, resize, resize_dims, crop, flip, rotate):
    """Transform height based on ida augmentation configuration.

    Args:
        cam_height (np array): Nx3, 3: x,y,d.
        resize (float): Resize factor.
        resize_dims (list): Final dimension.
        crop (list): x1, y1, x2, y2
        flip (bool): Whether to flip.
        rotate (float): Rotation value.

    Returns:
        np array: [h/down_ratio, w/down_ratio, d]
    """

    H, W = resize_dims
    if len(resize) > 1:
        resize_w, resize_h = resize
        cam_height[:, 0] = cam_height[:, 0] * resize_w
        cam_height[:, 1] = cam_height[:, 1] * resize_h
    else:
        cam_height[:, :2] = cam_height[:, :2] * resize
    cam_height[:, 0] -= crop[0]
    cam_height[:, 1] -= crop[1]
    if flip:
        cam_height[:, 0] = resize_dims[1] - cam_height[:, 0]

    cam_height[:, 0] -= W / 2.0
    cam_height[:, 1] -= H / 2.0

    h = rotate / 180 * np.pi
    rot_matrix = [
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ]
    cam_height[:, :2] = np.matmul(rot_matrix, cam_height[:, :2].T).T

    cam_height[:, 0] += W / 2.0
    cam_height[:, 1] += H / 2.0

    height_coords = cam_height[:, :2].astype(np.int16)

    height_map = np.zeros(resize_dims)
    valid_mask = ((height_coords[:, 1] < resize_dims[0])
                  & (height_coords[:, 0] < resize_dims[1])
                  & (height_coords[:, 1] >= 0)
                  & (height_coords[:, 0] >= 0))
    height_map[height_coords[valid_mask, 1],
              height_coords[valid_mask, 0]] = cam_height[valid_mask, 2]

    return torch.Tensor(height_map) 

class DAIRIDataset(data.Dataset):
    def __init__(self, ida_aug_conf, bda_aug_conf, classes, image_path, label_path, filenames_list, depth_path=None, height_path=None, calib_path=None, training=True, 
                    img_conf=dict(img_mean=[123.675, 116.28, 103.53],
                                   img_std=[58.395, 57.12, 57.375],
                                   to_rgb=True)):
        if True:
            if image_path[-1]!='/':
                image_path = image_path + '/'
            if label_path[-1]!='/':
                label_path = label_path + '/'
            self.image_path = [image_path + name + '.jpg' for name in filenames_list]
            self.label_path = [label_path + name + '.json' for name in filenames_list]
            if depth_path is not None:
                if depth_path[-1]!='/':
                    depth_path = depth_path + '/'
                self.depth_path = [depth_path + name + '.bin' for name in filenames_list]
            else:
                self.depth_path = None
            if height_path is not None:
                if height_path[-1]!='/':
                    height_path = height_path + '/'
                self.height_path = [height_path + name + '.bin' for name in filenames_list]
            else:
                self.height_path = None
            if calib_path is not None:
                if calib_path[-1]!='/':
                    calib_path = calib_path + '/'
                self.calib_path = [calib_path + name + '.json' for name in filenames_list]
            else:
                self.calib_path = None
        
        self.ida_aug_conf = ida_aug_conf
        self.bda_aug_conf = bda_aug_conf
        self.classes = classes
        self.img_mean = np.array(img_conf['img_mean'], np.float32)
        self.img_std = np.array(img_conf['img_std'], np.float32)
        self.to_rgb = img_conf['to_rgb']
        self.is_train = training

    def __getitem__(self, index):
        cur_image_path = self.image_path[index]
        cur_label_path = self.label_path[index]
        if self.depth_path is not None:
            cur_depth_path = self.depth_path[index]
        if self.height_path is not None:
            cur_height_path = self.height_path[index]        
        if self.calib_path is not None:
            cur_calib_path = self.calib_path[index]

        input_list = self.format_input(cur_image_path, cur_depth_path, cur_height_path, cur_calib_path)
        sweep_imgs, sweep_sensor2ego_mats, sweep_intrins, sweep_ida_mats, gt_depths, gt_heights, sweep_imgs_tb = input_list
        gt_boxes, gt_labels = self.get_gt(cur_label_path)

        if True:  # TODO:
            rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation()
            bda_mat = sweep_imgs.new_zeros(4, 4)
            bda_mat[3, 3] = 1
            gt_boxes, bda_rot = bev_transform(gt_boxes, rotate_bda, scale_bda, flip_dx, flip_dy)
            bda_mat[:3, :3] = bda_rot

        token = cur_image_path.split('/')[-1].split('.')[0]
        img_metas = dict()
        img_metas['box_type_3d'] = LiDARInstance3DBoxes
        img_metas['token'] = token

        ret_list = [sweep_imgs, sweep_sensor2ego_mats, sweep_intrins, sweep_ida_mats, bda_mat, gt_boxes, gt_labels, gt_depths, gt_heights, sweep_imgs_tb, img_metas]
        return ret_list

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy
    
    def get_gt(self, label_path):
        f = open(label_path, 'rb')
        txt = f.read().decode('utf-8')
        f.close()
        label_info = json.loads(txt)
        annos = label_info['annotation']
        gt_boxes = list()
        gt_labels = list()
        for anno in annos:
            if anno['type'] not in self.classes:
                continue
            cur_pos = [anno['lidar']['position']['x'], anno['lidar']['position']['y'], anno['lidar']['position']['z']]
            cur_angles = [anno['lidar']['angles']['rx'], anno['lidar']['angles']['ry'], anno['lidar']['angles']['rz']]
            cur_size = [anno['lidar']['size']['length'], anno['lidar']['size']['width'], anno['lidar']['size']['height']]
            box_xyz = np.array(cur_pos)
            box_dxdydz = np.array(cur_size)
            box_yaw = np.array([cur_angles[2], 0, 0])
            gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw])
            gt_boxes.append(gt_box)
            gt_labels.append(self.classes.index(anno['type']))

        return torch.Tensor(gt_boxes), torch.Tensor(gt_labels)

    def sample_ida_augmentation(self, W, H):
        """Generate ida augmentation values based on ida_config."""
        #H, W = self.ida_aug_conf['H'], self.ida_aug_conf['W']
        fH, fW = self.ida_aug_conf['final_dim']
        if self.is_train:
            if False:
                resize = np.random.uniform(*self.ida_aug_conf['resize_lim'])
                resize_dims = (int(W * resize), int(H * resize))
            else:
                resize = np.random.uniform(*self.ida_aug_conf['resize_lim'], 2)
                resize_W, resize_H = resize
                resize_dims = (int(resize_W * W), int(resize_H * H))
                resize = (float(resize_dims[0])/float(W), float(resize_dims[1]/float(H)))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.ida_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(np.random.uniform(max(0, newW/2 - fW), min(newW/2 + fW, newW-fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.ida_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate_ida = np.random.uniform(*self.ida_aug_conf['rot_lim'])
        else:
            if False:
                resize = max(fH / H, fW / W)
                resize_dims = (int(W * resize), int(H * resize))
            else:
                resize_dims = (fW, fH)
                resize = (float(fW) / float(W), float(fH) / float(H))
            
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.ida_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate_ida = 0
        return resize, resize_dims, crop, flip, rotate_ida

    def format_input(self, image_path, depth_path, height_path, calib_path):
        sweep_imgs = list()
        sweep_imgs_tb = list()
        sweep_sensor2ego_mats = list()
        sweep_intrin_mats = list()
        sweep_ida_mats = list()
        gt_depth = list()
        gt_height = list()

        imgs = list()
        imgs_tb = list()
        sensor2ego_mats = list()
        intrin_mats = list()
        ida_mats = list()

        img = Image.open(image_path)
        imgW, imgH = img.size
        resize, resize_dims, crop, flip, rotate_ida = self.sample_ida_augmentation(imgW, imgH)
        img, ida_mat = img_transform(img, resize=resize, resize_dims=resize_dims, crop=crop, flip=flip, rotate=rotate_ida)
        ida_mats.append(ida_mat)

        intrin, lidar2cam = parse_calib_file(calib_path)
        intrin_mat = torch.zeros((4,4))
        intrin_mat[3, 3] = 1
        intrin_mat[:3, :3] = torch.Tensor(intrin)

        ego2cam = torch.Tensor(lidar2cam)
        sweepsensor2keyego = ego2cam.inverse()
        sensor2ego_mats.append(sweepsensor2keyego)

        point_depth = np.fromfile(depth_path, dtype=np.float32, count=-1).reshape(-1, 3)
        point_depth_augmented = depth_transform(point_depth, resize, self.ida_aug_conf['final_dim'], crop, flip, rotate_ida)
        gt_depth.append(point_depth_augmented)

        point_height = np.fromfile(height_path, dtype=np.float32, count=-1).reshape(-1, 3)
        point_height_augmented = height_transform(point_height, resize, self.ida_aug_conf['final_dim'], crop, flip, rotate_ida)
        gt_height.append(point_height_augmented)        

        img_tb = torch.from_numpy(np.array(img.resize((240, 136)))).permute(2, 0, 1)
        imgs_tb.append(img_tb)
        img = mmcv.imnormalize(np.array(img), self.img_mean, self.img_std, self.to_rgb)
        img = torch.from_numpy(img).permute(2, 0, 1)

        imgs.append(img)
        intrin_mats.append(intrin_mat)

        sweep_imgs.append(torch.stack(imgs))
        sweep_imgs_tb.append(torch.stack(imgs_tb))
        sweep_intrin_mats.append(torch.stack(intrin_mats))
        sweep_sensor2ego_mats.append(torch.stack(sensor2ego_mats))
        sweep_ida_mats.append(torch.stack(ida_mats))

        ret_list = [
            torch.stack(sweep_imgs).permute(1, 0, 2, 3, 4), 
            torch.stack(sweep_sensor2ego_mats).permute(1, 0, 2, 3), 
            torch.stack(sweep_intrin_mats).permute(1, 0, 2, 3), 
            torch.stack(sweep_ida_mats).permute(1, 0, 2, 3), 
            torch.stack(gt_depth), 
            torch.stack(gt_height),
            torch.stack(sweep_imgs_tb).permute(1, 0, 2, 3, 4)
        ]
        return ret_list
    
    def __len__(self):
        return len(self.image_path)

def collate_fn(data):
    imgs_batch = list()
    imgs_tb_batch = list()
    sensor2ego_mats_batch = list()
    intrin_mats_batch = list()
    ida_mats_batch = list()
    bda_mat_batch = list()
    gt_boxes_batch = list()
    gt_labels_batch = list()
    depth_labels_batch = list()
    height_labels_batch = list()
    img_metas_batch = list()
    for iter_data in data:
        (
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            bda_mat,
            gt_boxes,
            gt_labels,
            gt_depth,
            gt_height,
            sweep_imgs_tb,
            img_metas 
        ) = iter_data
        depth_labels_batch.append(gt_depth)
        height_labels_batch.append(gt_height)
        imgs_batch.append(sweep_imgs)
        sensor2ego_mats_batch.append(sweep_sensor2ego_mats)
        intrin_mats_batch.append(sweep_intrins)
        ida_mats_batch.append(sweep_ida_mats)
        bda_mat_batch.append(bda_mat)
        gt_boxes_batch.append(gt_boxes)
        gt_labels_batch.append(gt_labels)
        imgs_tb_batch.append(sweep_imgs_tb)
        img_metas_batch.append(img_metas)
    mats_dict = dict()
    mats_dict['sensor2ego_mats'] = torch.stack(sensor2ego_mats_batch)
    mats_dict['intrin_mats'] = torch.stack(intrin_mats_batch)
    mats_dict['ida_mats'] = torch.stack(ida_mats_batch)
    mats_dict['bda_mat'] = torch.stack(bda_mat_batch)

    ret_list = [
        torch.stack(imgs_batch),
        mats_dict,
        gt_boxes_batch,
        gt_labels_batch,
        torch.stack(depth_labels_batch),
        torch.stack(height_labels_batch),
        torch.stack(imgs_tb_batch),
        img_metas_batch
    ]
    return ret_list






