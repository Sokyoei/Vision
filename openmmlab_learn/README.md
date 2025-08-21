# [OpenMMLab](https://openmmlab.com)

## 环境搭建

```bash
# torch 2.1.x with mmcv 2.1.0 on Ubuntu 22.04
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# fix environment
pip install "numpy<2"  # mmcv 2.1.0 requires numpy<2
pip install opencv-python==4.11.0.86  #  next version requires numpy>2

# install with min
pip install -U openmim
mim install mmcv==2.1.0 --trusted-host download.openmmlab.com
mim install mmdet==3.2.0 --trusted-host download.openmmlab.com
mim install mmpose==1.3.2 --trusted-host download.openmmlab.com
```
