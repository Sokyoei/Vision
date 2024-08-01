# 坐标转换

## 缩放

$$
P^{\prime} =
\begin{bmatrix}
s_1 & 0   \\
0   & s_2 \\
\end{bmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix}
= S \cdot P
$$

$$
S_H =
\begin{bmatrix}
s_1 & 0   & 0 \\
0   & s_2 & 0 \\
0   & 0   & 1
\end{bmatrix}
$$

## 旋转

$$
P^{\prime} =
\begin{bmatrix}
cos\theta & -sin\theta \\
sin\theta & cos\theta  \\
\end{bmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix}
= R \cdot P
$$

$$
P_H =
\begin{bmatrix}
cos\theta & -sin\theta & 0 \\
sin\theta & cos\theta  & 0 \\
0         & 0          & 1
\end{bmatrix}
$$

## 平移

欧氏坐标转换为齐次坐标

$$
P_E =
\begin{bmatrix}
x \\
y
\end{bmatrix}
\rightarrow
P_H =
\begin{bmatrix}
x \\
y \\
1
\end{bmatrix}
$$

$$
P_H^{\prime} =
\begin{bmatrix}
1 & 0 & x_0 \\
0 & 1 & y_0 \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
x \\
y \\
1
\end{bmatrix}
= T \cdot P_H
$$

## 四元数

$$
q =
\begin{bmatrix}
w, x, y, z
\end{bmatrix}
= w + xi + yj + zk
$$

w 控制旋转角度($2\arccos(w)$)，x, y, z 控制旋转方向
