# CUDA_VISIBLE_DEVICES=1 python exps/bevheight/main.py --amp_backend native -b 4 --devices 1 --max_epochs 50 --check_val_every_n_epoch 2 --resume_from_checkpoint outputs/BEVHeight/lightning_logs/version_6/checkpoints/epoch=43-step=55440.ckpt
CUDA_VISIBLE_DEVICES=1 python exps/bevseg/main.py --amp_backend native -b 4 --devices 1 --max_epochs 30 --check_val_every_n_epoch 2 --resume_from_checkpoint outputs/BEVSeg/epoch=19-step=25200.ckpt








