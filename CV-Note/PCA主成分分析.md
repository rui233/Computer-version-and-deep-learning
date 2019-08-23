PCA主成分分析数学推导

$w^Tw=I$

协方差：
$$
Cov = E[X-E(X)][Y-E(Y)]
$$
均方差：
$$
\sigma(X)=\sqrt{D(X)}
$$
方差：
$$
D(X) = E[X- E(X)]^2
$$
期望：
$$
E(X) = \sum_{k=1}^{\infty}x_kp_k
$$

$$
E(X) =\int_{-\infty}^{+\infty}xf(x)dx
$$

$$
z^{(n)}=w^TX^{n}
$$

$$
\bar x=\frac{1}{N}\sum_{n=1}^{N}X^{n}
$$

$$
D(x,w)=\frac{1}{N}\sum_{n=1}^{N}(w^TX^{n}-w^T\bar x)^2=w^T\frac{1}{N}(X-\bar X)(X-\bar X)^Tw=w^TSw
$$

$\bar X=\bar x[1]_d^T$是d列$\bar x$组成的矩阵

利用拉格朗日乘子法：
$$
max\quad w^TSw + \lambda(I-w^Tw)
$$
求导得$Sw=\lambda w$,w是协方差矩阵S的特征向量，$\lambda$是特征值

则$D(x,w)=\lambda$

