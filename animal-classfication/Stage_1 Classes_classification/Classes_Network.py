import torch.nn as nn
import torch.nn.functional as F
# torch.nn是专门为神经网络设计的模块化接口。nn构建于autograd之上，可以用来定义和运行神经网络。
# nn.Module是nn中十分重要的类,包含网络各层的定义及forward方法。

# 定义自已的网络：
#     需要继承nn.Module类，并实现forward方法。
#     一般把网络中具有可学习参数的层放在构造函数__init__()中，
#     不具有可学习参数的层(如ReLU)可放在构造函数中，也可不放在构造函数中(而在forward中使用nn.functional来代替)

# 只要在nn.Module的子类中定义了forward函数，backward函数就会被自动实现(利用Autograd)。
#     在forward函数中可以使用任何Variable支持的函数，毕竟在整个pytorch构建的图中，是Variable在流动。还可以使用
#     if,for,print,log等python语法.
#
#     注：Pytorch基于nn.Module构建的模型中，只支持mini-batch的Variable输入方式，
#     比如，只有一张输入图片，也需要变成 N x C x H x W 的形式：
#
#     input_image = torch.FloatTensor(1, 28, 28)
#     input_image = Variable(input_image)
#     input_image = input_image.unsqueeze(0)   # batch（128） x 3 x 500 x 500（Tensor）

class Net(nn.Module):
    def __init__(self):
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__() # 等价与nn.Module.__init__()

        # nn.Conv2d返回的是一个Conv2d class的一个对象，该类中包含forward函数的实现
        # 当调用self.conv1(input)的时候，就会调用该类的forward函数

        # nn.Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True))
        # 参数：
        #   in_channel:　输入数据的通道数，例RGB图片通道数为3；
        #   out_channel: 输出数据的通道数，这个根据模型调整；
        #   kennel_size: 卷积核大小，可以是int，或tuple；kennel_size=2,意味着卷积大小2， kennel_size=（2,3），意味着卷积在第一维度大小为2，在第二维度大小为3；
        #   stride：步长，默认为1，与kennel_size类似，stride=2,意味在所有维度步长为2， stride=（2,3），意味着在第一维度步长为2，意味着在第二维度步长为3；
        #   padding：　零填充

        self.conv1 = nn.Conv2d(3, 3, 3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(3, 6, 3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU(inplace=True)

        # 6 * 123 * 123, 150是指维度
        self.fc1 = nn.Linear(6 * 123 * 123, 150)
        self.relu3 = nn.ReLU(inplace=True)

        # torch.nn.Dropout对所有元素中每个元素按照概率更改为零
        # 而torch.nn.Dropout2d是对每个通道按照概率置为0
        self.drop = nn.Dropout2d()

        self.fc2 = nn.Linear(150, 2)
        # Softmax简单的说就是把一个N*1的向量归一化为（0，1）之间的值
        # 举个例子：假设你的Out=[2,3]，
        # 那么经过softmax层后就会得到[0.24,0.67]，
        # 这三个数字表示这个样本属于第1,2,3类的概率分别是0.24,0.67。
        # 取概率最大的0.67，所以这里得到的预测值就是第二类。
        self.softmax1 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.relu2(x)

        # print(x.shape)
        # 它相当于numpy中resize（）的功能，进行shape改变。
        x = x.view(-1, 6 * 123 * 123)
        x = self.fc1(x)
        x = self.relu3(x)

        # 使用F.dropout ( nn.functional.dropout )
        # 的时候需要设置它的training这个状态参数与模型整体的一致.默认为False
        x = F.dropout(x, training=self.training)

        x_classes = self.fc2(x)  # 128*1*2
        # print(x_classes.shape)
        x_classes = self.softmax1(x_classes)

        return x_classes