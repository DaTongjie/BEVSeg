pip install -i https://pypi.tuna.tsinghua.edu.cn/simple openmim
mim install -i https://pypi.tuna.tsinghua.edu.cn/simple mmcv-full==1.6.2
mim install -i https://pypi.tuna.tsinghua.edu.cn/simple mmdet==2.28.2
mim install -i https://pypi.tuna.tsinghua.edu.cn/simple mmsegmentation==0.30.0
git clone https://github.com/open-mmlab/mmdetection3d.git -b 1.0
cd mmdetection3d
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e .

# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv_python==3.4.10.37
