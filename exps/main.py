# Copyright (c) Megvii Inc. All rights reserved.
from argparse import ArgumentParser, Namespace

import mmcv
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from pytorch_lightning.core import LightningModule
from torch.cuda.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import MultiStepLR
from callbacks.ema import EMACallback
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets.DAIRI_dataset_seg import DAIRIDatasetSeg, collate_fn
from evaluators.kitti_evaluator import RoadSideEvaluator
from models.bev_seg import BEVSeg
from utils.torch_dist import all_gather_object, get_rank, synchronize
from pytorch_lightning.strategies import DDPStrategy

H = 1080
W = 1920
final_dim = (864, 1536)
backbone_conf = {
    'x_bound': [-51.2, 51.2, 0.8],
    'y_bound': [0, 102.4, 0.8],
    'z_bound': [-1, 4, 5],
    'h_bound': [-1, 4, 80],
    'final_dim':
        final_dim,
    'output_channels':
        80,
    'downsample_factor':
        16,
    'img_backbone_conf':
        dict(
            type='ResNet',
            depth=101,
            frozen_stages=0,
            out_indices=[0, 1, 2, 3],
            norm_eval=False,
            init_cfg=dict(type='Pretrained',
                          checkpoint='torchvision://resnet101'),
        ),
    'img_neck_conf':
        dict(
            type='SECONDFPN',
            in_channels=[256, 512, 1024, 2048],
            upsample_strides=[0.25, 0.5, 1, 2],
            out_channels=[128, 128, 128, 128],
        ),
    'height_net_conf':
        dict(in_channels=512, mid_channels=512),
    'softmax_height_pred': False
}
ida_aug_conf = {
    'resize_lim': (0.386, 1.2),
    'final_dim':
        final_dim,
    'rot_lim': (-5.4, 5.4),
    'H':
        H,
    'W':
        W,
    'rand_flip':
        True,
    'bot_pct_lim': (0.0, 0.0),
}
bda_aug_conf = {
    'rot_lim': (-22.5, 22.5),
    'scale_lim': (0.95, 1.05),
    'flip_dx_ratio': 0.5,
    'flip_dy_ratio': 0.5
}
bev_backbone = dict(
    type='ResNet',
    in_channels=80 + 1,
    depth=18,
    num_stages=3,
    strides=(1, 2, 2),
    dilations=(1, 1, 1),
    out_indices=[0, 1, 2],
    norm_eval=False,
    base_channels=160,
)
bev_neck = dict(type='SECONDFPN',
                in_channels=[81, 160, 320, 640],
                upsample_strides=[1, 2, 4, 8],
                out_channels=[64, 64, 64, 64])
CLASSES = [
    'vehicle',
    'tricycle',
    'non-motor',
    'person',
]
TASKS = [
    dict(num_class=1, class_names=['vehicle']),
    dict(num_class=1, class_names=['tricycle']),
    dict(num_class=1, class_names=['non-motor']),
    dict(num_class=1, class_names=['person']),
]
common_heads = dict(reg=(2, 2),
                    height=(1, 2),
                    dim=(3, 2),
                    rot=(2, 2))
bbox_coder = dict(
    type='CenterPointBBoxCoder',
    post_center_range=[-61.2, 0, -1, 61.2, 122.4, 4],
    max_num=500,
    score_threshold=0.1,
    # score_threshold=0.001,
    out_size_factor=4,
    voxel_size=[0.2, 0.2, 5],
    pc_range=[-51.2, 0, -1, 51.2, 104.4, 4],
    code_size=9,
)
train_cfg = dict(
    point_cloud_range=[-51.2, 0, -1, 51.2, 102.4, 4],
    grid_size=[512, 512, 1],
    voxel_size=[0.2, 0.2, 5],
    out_size_factor=4,
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
    code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
)
test_cfg = dict(
    post_center_limit_range=[-61.2, 0, -1, 61.2, 122.4, 4.0],
    max_per_img=500,
    max_pool_nms=False,
    min_radius=[4, 12, 10, 1, 0.85, 0.175],
    score_threshold=0.1,
    out_size_factor=4,
    voxel_size=[0.2, 0.2, 5],
    nms_type='circle',
    pre_max_size=1000,
    post_max_size=83,
    nms_thr=0.2,
)
head_conf = {
    'bev_backbone_conf': bev_backbone,
    'bev_neck_conf': bev_neck,
    'tasks': TASKS,
    'common_heads': common_heads,
    'bbox_coder': bbox_coder,
    'train_cfg': train_cfg,
    'test_cfg': test_cfg,
    'in_channels': 256,  # Equal to bev_neck output_channels.
    'loss_cls': dict(type='GaussianFocalLoss', reduction='mean'),
    'loss_bbox': dict(type='L1Loss', reduction='mean', loss_weight=0.25),
    'gaussian_overlap': 0.1,
    'min_radius': 2,
}
img_conf = dict(img_mean=[123.675, 116.28, 103.53],
                img_std=[58.395, 57.12, 57.375],
                to_rgb=True)

class BEVSegLightningModel(LightningModule):
    MODEL_NAMES = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith('__')
                         and callable(models.__dict__[name]))    

    def __init__(self,
                 gpus: int = 1,
                 data_root='./data/',
                 eval_interval=1,
                 batch_size_per_device=8,
                 class_names=CLASSES,
                 backbone_conf=backbone_conf,
                 head_conf=head_conf,
                 ida_aug_conf=ida_aug_conf,
                 bda_aug_conf=bda_aug_conf,
                 default_root_dir='./outputs/',
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.gpus = gpus
        self.eval_interval = eval_interval
        self.batch_size_per_device = batch_size_per_device
        self.data_root = data_root
        self.basic_lr_per_img = 2e-4 / 64
        self.class_names = class_names
        self.backbone_conf = backbone_conf
        self.head_conf = head_conf
        self.ida_aug_conf = ida_aug_conf
        self.bda_aug_conf = bda_aug_conf
        mmcv.mkdir_or_exist(default_root_dir)
        self.default_root_dir = default_root_dir

        self.model = BEVSeg(self.backbone_conf,
                              self.head_conf,
                              is_train_height=True, train_height_only=False)
        self.evaluator = RoadSideEvaluator(class_names=self.class_names,
                                           current_classes=["Car", "Pedestrian", "Cyclist"],
                                           data_root=data_root,
                                           output_dir=self.default_root_dir)
        self.mode = 'valid'
        self.img_conf = img_conf
        self.data_use_cbgs = False
        self.num_sweeps = 1
        self.sweep_idxes = list()
        self.key_idxes = list()
        self.data_return_height = True
        self.downsample_factor = self.backbone_conf['downsample_factor']
        self.hbound = self.backbone_conf['h_bound']
        self.height_channels = int(self.hbound[2])

        self.image_path = self.data_root + 'SMAIMM_DAIRI/image'
        self.label_path = self.data_root + 'SMAIMM_DAIRI/label'
        self.calib = self.data_root + 'SMAIMM_DAIRI/calib'
        self.seg_path = self.data_root + 'DAIR-I-SEG/seg_images'
        self.height_path = self.data_root + 'SMAIMM_DAIRI/height_gt'

    def forward(self, sweep_imgs, mats):
        return self.model(sweep_imgs, mats)
    
    def training_step(self, batch):
        (sweep_imgs, mats, gt_boxes, gt_labels, seg_labels, height_labels, _, _) = batch

        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
            gt_boxes = [gt_box.cuda() for gt_box in gt_boxes]
            gt_labels = [gt_label.cuda() for gt_label in gt_labels]
        preds, height_preds, seg_preds = self(sweep_imgs, mats)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            targets = self.model.module.get_targets(gt_boxes, gt_labels)
            detection_loss, loss_bbox, loss_heatmap = self.model.module.loss(targets, preds)
        else:
            targets = self.model.get_targets(gt_boxes, gt_labels)
            detection_loss, loss_bbox, loss_heatmap = self.model.loss(targets, preds)

        if len(height_labels.shape) == 5:
            # only key-frame will calculate height loss
            height_labels = height_labels[:, 0, ...]
        if len(seg_labels.shape) == 5:
            # only key-frame will calculate seg loss
            seg_labels = seg_labels[:, 0, ...]        


        height_preds_softmax = F.softmax(height_preds, 1)
        height_loss = self.get_height_loss(height_labels.cuda(), height_preds_softmax)

        seg_preds_sigmod = F.sigmoid(seg_preds)
        seg_loss = self.get_seg_loss(seg_labels.cuda(), seg_preds_sigmod)

        return detection_loss + height_loss + seg_loss

    def get_seg_loss(self, seg_labels, seg_preds):
        seg_labels = torch.where(seg_labels > 0,torch.ones_like(seg_labels),torch.zeros_like(seg_labels))
        B, N, H, W = seg_labels.shape
        seg_labels = seg_labels.view(
            B * N,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )
        seg_labels = seg_labels.permute(0, 1, 3, 5, 2, 4).contiguous()
        seg_labels = seg_labels.view( -1, self.downsample_factor * self.downsample_factor)
        seg_labels = torch.max(seg_labels, dim=-1).values
        seg_labels = seg_labels.view(-1, 1).float()
    
        seg_preds = seg_preds.permute(0, 2, 3, 1).contiguous().view(-1, 1)
        fg_mask = torch.max(seg_labels, dim=1).values > -100

        with autocast(enabled=False):
            seg_loss = (F.binary_cross_entropy(
                seg_preds[fg_mask],
                seg_labels[fg_mask],
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum()))        

        return 3.0 * seg_loss

    def get_height_loss(self, height_labels, height_preds):
        height_labels = self.get_downsampled_gt_height(height_labels)
        height_preds = height_preds.permute(0, 2, 3, 1).contiguous().view(
            -1, self.height_channels)

        fg_mask = torch.max(height_labels, dim=1).values > -100

        with autocast(enabled=False):
            height_loss = (F.binary_cross_entropy(
                height_preds[fg_mask],
                height_labels[fg_mask],
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum()))

        return 3.0 * height_loss

    def get_downsampled_gt_height(self, gt_heights):
        """
        Input:
            gt_heights: [B, N, H, W]
        Output:
            gt_heights: [B*N*h*w, d]
        """
        B, N, H, W = gt_heights.shape
        gt_heights = gt_heights.view(B*N,H,W)
        gt_heights = torch.where(((gt_heights<self.hbound[1])&(gt_heights>self.hbound[0])),gt_heights,torch.zeros_like(gt_heights))
        gt_heights = gt_heights.view(
            B * N,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )
        gt_heights = gt_heights.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_heights = gt_heights.view(
            -1, self.downsample_factor * self.downsample_factor)
        gt_heights = torch.max(gt_heights, dim=-1).values
        gt_heights = gt_heights.view(B * N, H // self.downsample_factor,
                                   W // self.downsample_factor)

        gt_heights = ((gt_heights -self.hbound[0])/(self.hbound[1] - self.hbound[0])) * self.hbound[2]
        gt_heights = torch.where(
            (gt_heights < self.height_channels + 1) & (gt_heights >= 0.0),
            gt_heights, torch.zeros_like(gt_heights))
        gt_heights = F.one_hot(gt_heights.long(),
                              num_classes=self.height_channels + 1).view(
            -1, self.height_channels + 1)[:, 1:]

        return gt_heights.float()
    
    def eval_step(self, batch, batch_idx, prefix: str):
        (sweep_imgs, mats, _, _, _, _, _, img_metas) = batch
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
        preds, _, _ = self.model(sweep_imgs, mats)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            results = self.model.module.get_bboxes(preds, img_metas)
        else:
            results = self.model.get_bboxes(preds, img_metas)
        
        for i in range(len(results)):
            results[i][0] = results[i][0].tensor.detach().cpu().numpy()
            results[i][1] = results[i][1].detach().cpu().numpy()
            results[i][2] = results[i][2].detach().cpu().numpy()
            results[i].append(img_metas[i])
        return results

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'val')

    def validation_epoch_end(self, validation_step_outputs):
        all_pred_results = list()
        all_img_metas = list()       
        for validation_step_output in validation_step_outputs:
            for i in range(len(validation_step_output)):
                all_pred_results.append(validation_step_output[i][:3])
                all_img_metas.append(validation_step_output[i][3])          
        synchronize()

        len_dataset = len(self.val_dataloader().dataset)
        all_pred_results = sum(
            map(list, zip(*all_gather_object(all_pred_results))),
            [])[:len_dataset]
        all_img_metas = sum(
            map(list, zip(*all_gather_object(all_img_metas))),
            [])[:len_dataset]
        
        if get_rank() == 0:
            self.evaluator.evaluate(all_pred_results, all_img_metas)

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'test')

    def test_epoch_end(self, test_step_outputs):
        all_pred_results = list()
        all_img_metas = list()
        for test_step_output in test_step_outputs:
            for i in range(len(test_step_output)):
                all_pred_results.append(test_step_output[i][:3])
                all_img_metas.append(test_step_output[i][3])
        synchronize()

        len_dataset = len(self.test_dataloader().dataset)
        all_pred_results = sum(
            map(list, zip(*all_gather_object(all_pred_results))),
            [])[:len_dataset]
        all_img_metas = sum(
            map(list, zip(*all_gather_object(all_img_metas))),
            [])[:len_dataset]
        
        if get_rank() == 0:
            self.evaluator.evaluate(all_pred_results, all_img_metas)

    def configure_optimizers(self):
        lr = self.basic_lr_per_img * \
             self.batch_size_per_device * self.trainer.devices
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=lr,
                                      weight_decay=1e-7)
        scheduler = MultiStepLR(optimizer, [100])
        return [[optimizer], [scheduler]]

    def train_dataloader(self):
        root_path = self.data_root
        image_path = self.image_path
        label_path = self.label_path 
        calib = self.calib 
        seg_path = self.seg_path 
        height_path = self.height_path

        f = open(root_path + "SMAIMM_DAIRI/DAIRI_Train.txt", encoding="utf-8")
        out = f.readlines()

        file_name_list = []
        for name in out:
            file_name_list.append(name.replace("\n", ""))
        train_dataset = DAIRIDatasetSeg(
            ida_aug_conf=self.ida_aug_conf,
            bda_aug_conf=self.bda_aug_conf,
            classes=self.class_names,
            image_path=image_path,
            label_path=label_path,
            filenames_list=file_name_list,
            seg_path=seg_path,
            height_path=height_path,
            calib_path=calib,
            training=True
        )
        from functools import partial
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=4,
            drop_last=True,
            shuffle=True,
            collate_fn=partial(collate_fn, ),
            sampler=None,
        )
        return train_loader

    def val_dataloader(self):
        root_path = self.data_root
        image_path = self.image_path
        label_path = self.label_path 
        calib = self.calib 
        seg_path = self.seg_path 
        height_path = self.height_path

        f = open(root_path + "SMAIMM_DAIRI/DAIRI_Val.txt", encoding="utf-8")
        out = f.readlines()

        file_name_list = []
        for name in out:
            file_name_list.append(name.replace("\n", ""))

        val_dataset = DAIRIDatasetSeg(
            ida_aug_conf=self.ida_aug_conf,
            bda_aug_conf=self.bda_aug_conf,
            classes=self.class_names,
            image_path=image_path,
            label_path=label_path,
            filenames_list=file_name_list,
            seg_path=seg_path,
            height_path=height_path,
            calib_path=calib,
            training=False,
        )
        from functools import partial
        eval_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=4,
            drop_last=True,
            shuffle=False,
            collate_fn=partial(collate_fn, ),
            sampler=None,
        )
        return eval_loader

    def test_dataloader(self):
        root_path = self.data_root
        image_path = self.image_path
        label_path = self.label_path 
        calib = self.calib 
        seg_path = self.seg_path 
        height_path = self.height_path

        f = open(root_path + "SMAIMM_DAIRI/DAIRI_Val.txt", encoding="utf-8")
        out = f.readlines()

        file_name_list = []
        for name in out:
            file_name_list.append(name.replace("\n", ""))

        test_dataset = DAIRIDatasetSeg(
            ida_aug_conf=self.ida_aug_conf,
            bda_aug_conf=self.bda_aug_conf,
            classes=self.class_names,
            image_path=image_path,
            label_path=label_path,
            filenames_list=file_name_list,
            seg_path=seg_path,
            height_path = height_path,
            calib_path=calib,
            training=False,
        )
        from functools import partial
        eval_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=4,
            drop_last=True,
            shuffle=False,
            collate_fn=partial(collate_fn, ),
            sampler=None,
        )
        return eval_loader

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        return parent_parser


def main(args: Namespace, use_ema=False) -> None:
    if args.seed is not None:
        pl.seed_everything(args.seed)
    model = BEVSegLightningModel(**vars(args))

    if use_ema:
        train_dataloader = model.train_dataloader()
        ema_callback = EMACallback(
            len(train_dataloader.dataset) * args.max_epochs)
        trainer = pl.Trainer.from_argparse_args(args, callbacks=[ema_callback])
    else:
        trainer = pl.Trainer.from_argparse_args(args)
    if args.evaluate:
        trainer.test(model, ckpt_path=args.ckpt_path)
    else:
        trainer.fit(model, val_dataloaders=model.val_dataloader())

def run_cli(use_ema=False):
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument('-e',
                               '--evaluate',
                               dest='evaluate',
                               action='store_true',
                               help='evaluate model on validation set')
    parent_parser.add_argument('-b', '--batch_size_per_device', type=int)
    parent_parser.add_argument('--seed',
                               type=int,
                               default=0,
                               help='seed for initializing training.')
    parent_parser.add_argument('--ckpt_path', type=str)
    parent_parser.add_argument('--data_root', type=str,default='data/')

    parser = BEVSegLightningModel.add_model_specific_args(parent_parser)

    parser.set_defaults(
        profiler='simple',
        deterministic=False,
        max_epochs=20, # 800
        accelerator='gpu',
        strategy="ddp_find_unused_parameters_false",
        devices=1,
        num_sanity_val_steps=0,
        gradient_clip_val=5,
        gradient_clip_algorithm="norm",
        sync_batchnorm=False,
        track_grad_norm=-1,

        # limit_val_batches=1.0,
        enable_checkpointing=True,
        precision=32,
        default_root_dir='./outputs/BEVSeg',
        check_val_every_n_epoch=2, # 5
    )
    args = parser.parse_args()
    main(args, use_ema=use_ema)

if __name__ == '__main__':
    run_cli(use_ema=False)    








