SURF (Speeded Up Robust Features)也是一种类似于SIFT的兴趣点检测及描述子算法。其通过Hessian矩阵的行列式来确定兴趣点位置，再根据兴趣点邻域点的Haar小波响应来确定描述子，其描述子大小只有64维（也可以扩展到128维，效果更好），是一种非常优秀的兴趣点检测算法。本文主要从SURF原文出发，结合自己一些理解，并比较sift方法，对其算法原理进行总结。

关键词：SURF SIFT 尺度空间 Haar小波 Hessian 兴趣点检测描述子

### 一、FAST-Hessian检测

首先同SIFT方法一样，SURF也必须考虑如何确定兴趣点位置，不过SIFT采用是DOG来代替LOG算子，找到其在尺度和图像内局部极值视为特征点，而SURF方法是基于Hessian矩阵的，而它通过积分图像极大地减少运算时间，并称之为FAST-Hessian。（这里提一下，SIFT通过DOG来近似LOG，也实际上相当于计算Laplacian，即可以视为Hessian矩阵的迹，而SURF则利用的近似Hessian矩阵的行列式）

首先我们考虑一个Hessian矩阵：

![img](https://img-blog.csdn.net/20151028181426193?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

这里的Lxx是指图像经过高斯二阶梯度模板卷积之后得到的，像素点关于x方向的二阶梯度。SURF方法考虑将高斯二阶梯度模板用盒函数来近似，如下图：

![img](https://img-blog.csdn.net/20151028181456547?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

如此以来，我们可以通过积分图像非常方便地计算高斯二阶梯度，得到其近似：

![img](https://img-blog.csdn.net/20151028181527613?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

因为是近似，我们也需要平衡两者之间的相关比，这里我们假设，尺度为1.2的高斯模板可以用9*9的盒函数模板代替，然后计算下式归一化尺度的模板比值，这里的是指Frobenius范数：

![img](https://img-blog.csdn.net/20151028181641875?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

最后Hessian矩阵的行列式，我们可以近似为：

![img](https://img-blog.csdn.net/20151028181703218?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

由此，这里的0.9是归一化比值，所以在任何尺度下，我们都可以通过这个比来补偿近似造成的误差，因此任何尺度下，我们都可以计算近似Hessian行列式的值。

### 二、SURF的尺度空间

尺度空间通常通过高斯金字塔来实施，图像需要被重复高斯平滑，然后经过不断子采样，一层一层直到塔顶，如sift方法。而SUFR通过盒函数和积分图像，我们就不需要像SIFT方法那样，每组之间需要采样，而且还需要对当前组应用同上层组相同的滤波器，而SURF方法不需要进行采样操作，直接应用不同大小的滤波器就可以了。

为什么可以这样呢？因为都是为了得到不同尺度的图像，而sift通过采样操作比图像卷积操作计算量更少，而对于SURF来说，不存在这样的问题，因为盒函数和积分图像的操作计算量也非常小。另一方面，因为不需要采样，所以也不会出现混叠现象。

下图说明了这一情况，左图是sift算法，其是图像大小减少，而模板不变（这里只是指每组间，组内层之间还是要变的）。而SURF算法（右图）刚好相反，其是图像大小不变，而模板大小扩大。

![img](https://img-blog.csdn.net/20151028181858361?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

SURF也是将金字塔分为组（Octaves），而每组分为若干层。其将9*9大小的滤波器结果作为初始尺度组，即指的高斯尺度为1.2。那么接下来的每组，是通过逐渐增大的模板来进行滤波图像，一般情况下，滤波器的大小以9*9，15*15，21*21，27*27等变化，随着尺度增加，滤波器大小之间的差别也在增加。因此，对于每组来说，其滤波器大小增加数（15-9）是以双倍增长的（如6到12再24）。与此同时，提取兴趣点的采样间隔也是在以双倍增长的（这样可以获得小的尺度变化范围）。下面是模板的变化图：

![img](https://img-blog.csdn.net/20151028181919355?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

下图反映了组及层之间尺度变化，及滤波模板长度变化过程，我们可以发现层间采样间隔以2倍扩大，所以随着层尺度增加，其尺度变化的粒度减少了，但是我们发现第一组每一层的尺度变化粒度太大了，所以在这里我们需要引入尺度空间更为精细的插值操作。

![img](https://img-blog.csdn.net/20151028181952246?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

尺度空间搭建好了之后，同sift运算一样，我们找到在尺度及图像空间的3*3*3的范围内进行非极大值抑制，找到局部极值点（Hessian行列式），最后再应用尺度和图像空间的插值操作，以获得精确的兴趣点位置（原文用的是Brown的方法，也可以参考sift方法，不再详细讲解了）

### 三、兴趣点主方向获得

为了获得旋转不变性，我们需要识别兴趣点区域的一个主方向。SIFT方法采用的是计算兴趣点附近3*1.5Ϭ大小的圆形区域内方向直方图，选择最大的方向为主方向。而SURF方法则是通过计算其在x,y方向上的haar-wavelet响应，这是在兴趣点周围一个6s半径大小的圆形区域内。当然小波变换的大小也同尺度参数s有关，其步长为s，其大小为4s。

一旦区域内所有小波响应被计算，再对所有小波响应进行高斯加权（以兴趣点为中心，尺度为2.5s），然后建立小波响应dx,dy的坐标系（dx是小波在x方向上的响应，而dy是小波在y方向上的响应），将区域内的每点在这个坐标系来表示，如下图所示，选择一个60度的扇区（下图灰色区域），统计这个扇区所有响应的总和，就获得了一个总的方向（下图红箭头），旋转整个扇区，找到最长的矢量方向即为主方向。

​                                                           ![img](https://img-blog.csdn.net/20151028182040674?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

### 四、SURF描述子

同sift算法一样，SURF也是通过建立兴趣点附近区域内的信息来作为描述子的，不过sift是利用邻域点的方向，而SURF则是利用Haar小波响应。

SURF首先在兴趣点附近建立一个20s大小的方形区域，为了获得旋转不变性，同sift算法一样，我们需要将其先旋转到主方向，然后再将方形区域划分成16个（4*4）子域。对每个子域（其大小为5s*5s）我们计算25（5*5）个空间归一化的采样点的Haar小波响应dx和dy。

之后我们将每个子区域（共4*4）的dx,dy相加，因此每个区域都有一个描述子（如下式），为了增加鲁棒性，我们可以给描述子再添加高斯权重（尺度为3.3s，以兴趣点为中心）

![img](https://img-blog.csdn.net/20151028182210662?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

![img](https://img-blog.csdn.net/20151028182250394?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

所以最后在所有的16个子区域内的四位描述子结合，将得到该兴趣点的64位描述子

![img](https://img-blog.csdn.net/20151028182305708?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

由于小波响应对于光流变化偏差是不变的，所以描述子具有了光流不变性，而对比性不变可以通过将描述子归一化为单位向量得到。

另外也建立128位的SURF描述子，其将原来小波的结果再细分，比如dx的和将根据dy的符号，分成了两类，所以此时每个子区域内都有8个分量，SURF-128有非常好效果，如下图所示。

![img](https://img-blog.csdn.net/20151028182325947?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

### 五、快速索引匹配

我们发现兴趣点其Laplacian（Hessian矩阵的迹）的符号（即正负）可以将区分相同对比形状的不同区域，如黑暗背景下的白斑和相反的情况。

![img](https://img-blog.csdn.net/20151028182357704?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

考虑到检测时，我们容易遇到两个这样类似的结构（因为特征兴趣点经常是这样的斑点形状），原来我们必须为这两个结构分别建立描述子，现在我们只需要为其中一个建立描述子，而给另一个索引，而在匹配过程中，只要比较一个描述子，就能确定两个位置，是不是属于这两个结构中的一种，如果是，再通过迹来判断其是这两个结构中的那一种。

因为引入特性不需要额外的计算（检测过程中已经计算了）。所以在匹配过程中，可以达到快速的效果。

---------------------
作者：tostq 
来源：CSDN 
原文：https://blog.csdn.net/tostq/article/details/49472709 
版权声明：本文为博主原创文章，转载请附上博文链接！