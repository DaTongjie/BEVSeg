/opt/conda/bin/conda init
source ~/.bashrc
/opt/conda/bin/conda activate
source /opt/conda/etc/porfile.d/conda.sh
conda activate

chmod 777 ./install_mmd3D.sh
./install_mmd3D.sh

cd $workspace
pip uninstall -y opencv-python 
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python==4.5.1.48
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "opencv-python-headless<4.3"
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
python setup.py develop


chmod 777 ./shells/train.sh
./shells/train.sh

