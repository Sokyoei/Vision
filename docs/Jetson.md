# Jetson

## Python wheel for Jetson

TensorRT Python 位置在 `/usr/lib/pythonX.Y/dist-packages/tensorrt`

- [Jetson Zoo](https://elinux.org/Jetson_Zoo)
- [PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)

## Jetson FAQ

### 未找到 `nvidia-smi`

使用 `jtop` 查看

```shell
# install jtop
sudo apt install python3-pip
sudo pip install jetson-stats
# or build from source
git clone https://github.com/rbonghi/jetson_stats
cd jetson_stats
python setup.py install

reboot

sudo jtop
```

使用 `nvtop` 查看

```shell
# using cmake build
git clone https://github.com/Syllo/nvtop
mkdir build && cd build
cmake ..
make
make install

nvtop
```
