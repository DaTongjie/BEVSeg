import os
import os.path as osp
import tempfile
import math

import mmcv
import numpy as np
import json
import pyquaternion
from tqdm import tqdm
from nuscenes.utils.data_classes import Box
from evaluators.result2kitti import kitti_evaluation

category_map_dair = {"vehicle": "Car", "non-motor": "Cyclist", "person": "pedestrian"}

def convert_point(point, matrix):
    pos =  matrix @ point
    return pos[0], pos[1], pos[2]

def normalize_angle(angle):
    alpha_tan = np.tan(angle)
    alpha_arctan = np.arctan(alpha_tan)
    if np.cos(angle) < 0:
        alpha_arctan = alpha_arctan + math.pi
    return alpha_arctan

def get_camera_3d_8points(obj_size, yaw_lidar, center_lidar, center_in_cam, r_velo2cam, t_velo2cam):
    liadr_r = np.matrix([[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], [0, 0, 1]])
    l, w, h = obj_size
    corners_3d_lidar = np.matrix(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0, 0, 0, 0, h, h, h, h],
        ]
    )
    corners_3d_lidar = liadr_r * corners_3d_lidar + np.matrix(center_lidar).T
    corners_3d_cam = r_velo2cam * corners_3d_lidar + t_velo2cam
    
    x0, z0 = corners_3d_cam[0, 0], corners_3d_cam[2, 0]
    x3, z3 = corners_3d_cam[0, 3], corners_3d_cam[2, 3]
    dx, dz = x0 - x3, z0 - z3
    yaw = math.atan2(-dz, dx)
    alpha = yaw - math.atan2(center_in_cam[0], center_in_cam[2])
    if alpha > math.pi:
        alpha = alpha - 2.0 * math.pi
    if alpha <= (-1 * math.pi):
        alpha = alpha + 2.0 * math.pi
    alpha_arctan = normalize_angle(alpha)
    return alpha_arctan, yaw

def get_lidar_3d_8points(obj_size, yaw_lidar, center_lidar):
    center_lidar = [float(center_lidar[0]), float(center_lidar[1]), float(center_lidar[2])]
    lidar_r = np.matrix([[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], [0, 0, 1]])
    l, w, h = obj_size
    center_lidar[2] = center_lidar[2] - h / 2
    corners_3d_lidar = np.matrix(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0, 0, 0, 0, h, h, h, h],
        ]
    )
    corners_3d_lidar = lidar_r * corners_3d_lidar + np.matrix(center_lidar).T
    return corners_3d_lidar.T

def bbbox2bbox(box3d, Tr_velo_to_cam, camera_intrinsic, img_size=[1920, 1080]):
    corners_3d = np.array(box3d)
    corners_3d_extend = np.concatenate(
        [corners_3d, np.ones((corners_3d.shape[0], 1), dtype=np.float32)], axis=1) 
    corners_3d_extend = np.matmul(Tr_velo_to_cam, corners_3d_extend.transpose(1, 0))
        
    corners_2d = np.matmul(camera_intrinsic, corners_3d_extend)
    corners_2d = corners_2d[:2] / corners_2d[2]
    box2d = np.array([min(corners_2d[0]), min(corners_2d[1]),
                      max(corners_2d[0]), max(corners_2d[1])])
    
    # [xmin, ymin, xmax, ymax]
    box2d[0] = max(box2d[0], 0.0)
    box2d[1] = max(box2d[1], 0.0)
    box2d[2] = min(box2d[2], img_size[0])
    box2d[3] = min(box2d[3], img_size[1])
    return box2d

def write_kitti_in_txt(pred_lines, path_txt):
    wf = open(path_txt, "w")
    for line in pred_lines:
        line_string = " ".join(line) + "\n"
        wf.write(line_string)
    wf.close()

def result2kitti(results_file, results_path, dair_root):
    with open(results_file,'r',encoding='utf8')as fp:
        results = json.load(fp)["results"]
    for sample_token in tqdm(results.keys()):
        cur_calib_path = osp.join(dair_root,'SMAIMM_DAIRI/calib',sample_token+'.json')
        with open(cur_calib_path ,'r',encoding = 'utf-8') as fp:
            calib = json.load(fp)
        calib_intrinsic = np.array(calib['P'])
        camera_intrinsic = np.concatenate([calib_intrinsic, np.zeros((calib_intrinsic.shape[0], 1))], axis=1)
        Tr_velo_to_cam = np.array(calib['RT'])
        r_velo2cam = np.array(calib['RT'])[:3,:3]
        t_velo2cam = np.array(calib['RT'])[:3,3:4]

        preds = results[sample_token]
        pred_lines = []
        bboxes = []
        for pred in preds:
            loc = pred["translation"]
            dim = pred["size"]
            yaw_lidar = pred["box_yaw"]
            detection_score = pred["detection_score"]
            class_name = pred["detection_name"]
            
            w, l, h = dim[0], dim[1], dim[2]
            x, y, z = loc[0], loc[1], loc[2]            
            bottom_center = [x, y, z]
            obj_size = [l, w, h]
            bottom_center_in_cam = r_velo2cam * np.matrix(bottom_center).T + t_velo2cam
            alpha, yaw = get_camera_3d_8points(
                obj_size, yaw_lidar, bottom_center, bottom_center_in_cam, r_velo2cam, t_velo2cam
            )
            yaw  = 0.5 * np.pi - yaw_lidar

            cam_x, cam_y, cam_z = convert_point(np.array([x, y, z, 1]).T, Tr_velo_to_cam)
            box = get_lidar_3d_8points([w, l, h], yaw_lidar, [x, y, z + h/2])
            box2d = bbbox2bbox(box, Tr_velo_to_cam, camera_intrinsic)
            if detection_score > 0.45 and class_name in category_map_dair.keys():
                i1 = category_map_dair[class_name]
                i2 = str(0)
                i3 = str(0)
                i4 = str(round(alpha, 4))
                i5, i6, i7, i8 = (
                    str(round(box2d[0], 4)),
                    str(round(box2d[1], 4)),
                    str(round(box2d[2], 4)),
                    str(round(box2d[3], 4)),
                )
                i9, i11, i10 = str(round(h, 4)), str(round(w, 4)), str(round(l, 4))
                i12, i13, i14 = str(round(cam_x, 4)), str(round(cam_y, 4)), str(round(cam_z, 4))
                i15 = str(round(yaw, 4))
                line = [i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, str(round(detection_score, 4))]
                pred_lines.append(line)
                bboxes.append(box)
        os.makedirs(os.path.join(results_path, "data"), exist_ok=True)
        write_kitti_in_txt(pred_lines, os.path.join(results_path, "data",  sample_token + ".txt"))       
    return os.path.join(results_path, "data")

class RoadSideEvaluator():
    DefaultAttribute = {
        'vehicle': 'vehicle.moving',
        'person': 'person.moving',
        'non-motor': 'non-motor.moving',
        'tricycle': 'tricycle.moving',

    }
    def __init__(self,
                class_names,
                current_classes,
                data_root,
                modality=dict(use_lidar=False,
                            use_camera=True,
                            use_radar=False,
                            use_map=False,
                            use_external=False),
                output_dir=None,) -> None:    
        self.class_names = class_names
        self.current_classes = current_classes
        self.data_root = data_root
        self.modality = modality
        self.output_dir = output_dir

    def evaluate(self,
                    results,
                    img_metas,
                    metric='bbox',
                    logger=None,
                    jsonfile_prefix=None,
                    result_names=['img_bbox'],
                    show=False,
                    out_dir=None,
                    pipeline=None):
        result_files, tmp_dir = self.format_results(results, img_metas,
                                                    result_names,
                                                    jsonfile_prefix)
        results_path = "/workspace/mnt/storage/guangcongzheng/zju_fbz_backup/temp" 
        gt_label_path = '/workspace/mnt/storage/guangcongzheng/zju_fbz_backup/DAIR-V2X-I/infrastructure-side-kitti/training/label_2'
        pred_label_path = result2kitti(result_files["img_bbox"], results_path, self.data_root)
        return kitti_evaluation(pred_label_path, gt_label_path, current_classes=self.current_classes, metric_path="outputs/metrics")


    def format_results(self,
                        results,
                        img_metas,
                        result_names=['img_bbox'],
                        jsonfile_prefix=None,
                        **kwargs):
        assert isinstance(results, list), 'results must be a list'

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = dict()
        for rasult_name in result_names:
            if '2d' in rasult_name:
                continue
            print(f'\nFormating bboxes of {rasult_name}')
            tmp_file_ = osp.join(jsonfile_prefix, rasult_name)
            if self.output_dir:
                result_files.update({
                    rasult_name:
                    self._format_bbox(results, img_metas, self.output_dir)
                })
            else:
                result_files.update({
                    rasult_name:
                    self._format_bbox(results, img_metas, tmp_file_)
                })
        return result_files, tmp_dir
    
    def _format_bbox(self, results, img_metas, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.class_names

        print('Start to convert detection format...')

        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            boxes, scores, labels = det
            boxes = boxes
            sample_token = img_metas[sample_id]['token']
            # trans = np.array(img_metas[sample_id]['ego2global_translation'])
            # rot = Quaternion(img_metas[sample_id]['ego2global_rotation'])
            annos = list()
            for i, box in enumerate(boxes):
                name = mapped_class_names[labels[i]]
                center = box[:3]
                wlh = box[[4, 3, 5]]
                box_yaw = box[6]
                box_vel = box[7:].tolist()
                box_vel.append(0)
                quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw)
                nusc_box = Box(center, wlh, quat, velocity=box_vel)

                attr = self.DefaultAttribute[name]
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=nusc_box.center.tolist(),
                    size=nusc_box.wlh.tolist(),
                    rotation=nusc_box.orientation.elements.tolist(),
                    box_yaw=box_yaw,
                    velocity=nusc_box.velocity[:2],
                    detection_name=name,
                    detection_score=float(scores[i]),
                    attribute_name=attr,
                )
                annos.append(nusc_anno)
            # other views results of the same frame should be concatenated
            if sample_token in nusc_annos:
                nusc_annos[sample_token].extend(annos)
            else:
                nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }
        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path