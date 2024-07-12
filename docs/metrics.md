# 指标

## SSIM

structural similarity 结构相似性

$$
SSIM(x, y) =
\begin{bmatrix}
l(x, y)^{\alpha} \cdot c(x, y)^{\beta} \cdot s(x, y)^{\gamma}
\end{bmatrix}
$$

## PSNR

$$
PSNR =
10 \cdot \log_{10}(\frac{MAX^2_I}{MSE})
$$

## LPIPS

Learned Perceptual Image Patch Similarity 学习感知图像块相似度

$$
d(x, x_0) =
\sum\limits_{l} \frac{1}{H_l W_l}
\sum\limits_{h, w}
\begin{Vmatrix}
w_l \bigodot (\hat{y}_{hw}^l - \hat{y}_{0hw}^l)
\end{Vmatrix}_2^2
$$
