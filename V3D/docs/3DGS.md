# 3DGS

3D Gaussian Splatting 3D 高斯飞溅

- [awesome-3D-gaussian-splatting](https://github.com/MrNeRF/awesome-3D-gaussian-splatting)
- [paper](https://github.com/graphdeco-inria/gaussian-splatting)
- [website](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

特点：机器学习、CG、线性代数

## 渲染

- Ray-casting(NeRF), 被动，计算出每个像素点受到的发光粒子的影响生成图像
- Splatting(3DGS), 主动，计算每个发光粒子如何影响像素点

椭球高斯公式：
$$
G(x)=
\frac{1}{\sqrt{(2\pi)^k\begin{vmatrix}\Sigma\end{vmatrix}}}
e^{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)}
$$
$\Sigma$ 是协方差矩阵，半正定，$\begin{vmatrix}\Sigma\end{vmatrix}$是其行列式

## tools

[SIBR_viewer](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/binaries/viewers.zip)

球谐函数
