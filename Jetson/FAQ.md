# Jetson FAQ

## 未找到 `nvidia-smi`

使用 `jtop` 查看

```shell
# install jtop
pip install jetson-stats
# or build from source
git clone https://github.com/rbonghi/jetson_stats
cd jetson_stats
python setup.py install

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
