#!/usr/bin/env python
# coding: utf-8

# 
# ## 说明：
# 
# * 本代码是用yolov3或yolo-tiny来识别害虫；
# * 数据集是虫子,格式是xml;
# * 代码中数据集的解析过程是先把xml转换成了txt 
# 
# * 可能报错的地方：GPU配置不对应：可能需要在后面代码中修改use_GPU，能用/不能用GPU对应True/False.
# 

# ## 解压数据集

# In[1]:


# 解压bugs data
get_ipython().system('cd data/data7085/ && unzip -qo train.zip && unzip -q test.zip && unzip -q val.zip ')


# ## 解析xml
# 
# * 本次的数据集是用labelImg标注的，所以生成的是XML格式；
# * xml文件一般不能直接读取，一般要先解析，转换成“X + label”这种形式（比如txt等），才能送进网络来训练；
# * 运行下面代码之后，就生成了train.txt、test.txt、label_list.txt 、label_list 四个文件；

# In[77]:


# 按要求处理数据
#!/usr/bin/evn python 
#coding:utf-8 
import os

try: 
  import xml.etree.cElementTree as ET 
except ImportError: 
  import xml.etree.ElementTree as ET 
import sys 

# 分别制作 train  test  val ， 一共要跑三次哦
for set_ in ['train', 'test', 'val']:
    xml_ROOT = 'data/data7085/{}/annotations/xmls'.format(set_)
    jpg_ROOT = 'data/data7085/{}/images'.format(set_)
    out_txt_path = 'data/data7085/{}.txt'.format(set_)
    
    print(set_)
    
    xml_list = os.listdir(xml_ROOT)  #其中包含所有待计算的文件名
    
    if os.path.exists(out_txt_path):
        os.remove(out_txt_path)
    
    txt = open(out_txt_path, 'w')
    
    for xml_n in xml_list:
        xml_path = os.path.join(xml_ROOT, xml_n)
        tree = ET.parse(xml_path)     #打开xml文档 
        root = tree.getroot()         #获得root节点  
        # print ("*"*10)
        filename = root.find('filename').text
        filename = os.path.join(jpg_ROOT, filename)
        # print (filename)
    
        all_box_str = filename+'\t'
        box_count = 0
        for object in root.findall('object'): #找到root节点下的所有object节点 
            name = object.find('name').text   #子节点下节点name的值 
            if name!= 'Leconte':
                continue
            
            box_count += 1
            bndbox = object.find('bndbox')      #子节点下属性bndbox的值 
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            # 图片路径\t	{"value":"bolt","coordinate":[[769.459,241.819],[947.546,506.167]]}\t{...}... 注意，每个字段值之间用\t分割
            box_str = '{{\"value\":\"Leconte\",\"coordinate\":[[{:.3f},{:.3f}],[{:.3f},{:.3f}]]}}\t'.                                                            format(xmin,  ymin,  xmax, ymax)
            all_box_str += box_str
            pass
        
        if box_count==0:
            continue
        
        all_box_str += '\n'
        # print(all_box_str)
        txt.write(all_box_str)
    txt.close()
    print ('{}.txt is ok '.format(set_))

txt = open('data/data7085/label_list', 'w')
txt.write('Leconte')
txt.close()
print ('label_list is ok')
txt = open('data/data7085/label_list.txt', 'w')
txt.write('0\tLeconte')
txt.close()
print ('label_list.txt is ok')


# ## 特别提醒：
# * 运行完上面代码之后，要重启一下Kernel，然后再运行下面往后的所有代码，这样不会报错。
# 
# * 如果，偶尔出现cant call...once之类的问题，刷新下页面，重启Kernel。问题可以解决。

# ## 设置Yolov3模型的配置项
# * 设置训练Yolov3模型的配置项，此代码没有预训练模型。可以选择是否启用tiny版本，tiny版本体积小，适合部署在移动设备。
# 
# * 如果不熟悉yolov3模型，请不要随便更改图片的尺寸和anchors的尺寸，两者相互关联。

# In[16]:


# -*- coding: UTF-8 -*-
"""
训练常基于dark-net的YOLOv3网络，目标检测
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import uuid
import numpy as np
import time
import six
import math
import random
import paddle
import paddle.fluid as fluid
import logging
import xml.etree.ElementTree
import codecs
import json

from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
from PIL import Image, ImageEnhance, ImageDraw

logger = None
train_parameters = {
    "data_dir": "data/data7085/",
    "file_list": "train.txt",
    "class_dim": -1,
    "label_dict": {},
    "image_count": -1,
    "continue_train": False,     # 是否加载前一次的训练参数，接着训练
    "pretrained": False,
    "pretrained_model_dir": "./pretrained-model",
    "save_model_dir": "./yolo-model",
    "model_prefix": "yolo-v3",
    "use_tiny": True,          # 是否使用 裁剪 tiny 模型
    "max_box_num": 20,          # 一幅图上最多有多少个目标
    "num_epochs": 120,
    "train_batch_size": 5,      # 对于完整 yolov3，每一批的训练样本不能太多，内存会炸掉
    "use_gpu":False,
    "yolo_cfg": {
        "input_size": [3, 608, 608],
        "anchors": [10, 13,  16, 30,  33, 23,  30, 61,  62, 45,  59, 119,  116, 90,  156, 198,  373, 326],
        "anchor_mask": [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    },
    "yolo_tiny_cfg": {
        "input_size": [3, 256, 256],
        "anchors": [6, 8, 13, 15, 22, 34, 48, 50, 81, 100, 205, 191],
        "anchor_mask": [[3, 4, 5], [0, 1, 2]]
    },
    "ignore_thresh": 0.7,
    "mean_rgb": [127.5, 127.5, 127.5],
    "mode": "train",
    "multi_data_reader_count": 4,
    "apply_distort": True,
    "valid_thresh": 0.01,
    "nms_thresh": 0.45,
    "image_distort_strategy": {
        "expand_prob": 0.5,
        "expand_max_ratio": 4,
        "hue_prob": 0.5,
        "hue_delta": 18,
        "contrast_prob": 0.5,
        "contrast_delta": 0.5,
        "saturation_prob": 0.5,
        "saturation_delta": 0.5,
        "brightness_prob": 0.5,
        "brightness_delta": 0.125
    },
    "rsm_strategy": {
        "learning_rate": 0.001,
        "lr_epochs": [20, 40, 60, 80, 100],
        "lr_decay": [1, 0.5, 0.25, 0.1, 0.05, 0.01],
    },
    "momentum_strategy": {
        "learning_rate": 0.1,
        "decay_steps": 2 ** 7,
        "decay_rate": 0.8
    },
    "early_stop": {
        "sample_frequency": 50,
        "successive_limit": 3,
        "min_loss": 2.5,
        "min_curr_map": 0.84
    }
}

print('ok lalalalala')


# ## 用paddle搭建yolov3模型
# 定义两个类，分别代表 Yolo-v3 和 Yolo-v3-tiny 两个模型。跟随其后的是模型选择函数，根据配置使用不同的模型

# In[17]:


# -*- coding: UTF-8 -*-
"""
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class YOLOv3(object):
    def __init__(self, class_num, anchors, anchor_mask):
        self.outputs = []
        self.downsample_ratio = 1
        self.anchor_mask = anchor_mask
        self.anchors = anchors
        self.class_num = class_num

        self.yolo_anchors = []
        self.yolo_classes = []
        for mask_pair in self.anchor_mask:
            mask_anchors = []
            for mask in mask_pair:
                mask_anchors.append(self.anchors[2 * mask])
                mask_anchors.append(self.anchors[2 * mask + 1])
            self.yolo_anchors.append(mask_anchors)
            self.yolo_classes.append(class_num)

    def name(self):
        return 'YOLOv3'

    def get_anchors(self):
        return self.anchors

    def get_anchor_mask(self):
        return self.anchor_mask

    def get_class_num(self):
        return self.class_num

    def get_downsample_ratio(self):
        return self.downsample_ratio

    def get_yolo_anchors(self):
        return self.yolo_anchors

    def get_yolo_classes(self):
        return self.yolo_classes

    def conv_bn(self,
                input,
                num_filters,
                filter_size,
                stride,
                padding,
                use_cudnn=True):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02)),
            bias_attr=False)

        # batch_norm中的参数不需要参与正则化，所以主动使用正则系数为0的正则项屏蔽掉
        # 在batch_norm中使用 leaky 的话，只能使用默认的 alpha=0.02；如果需要设值，必须提出去单独来
        out = fluid.layers.batch_norm(
            input=conv, act=None, 
            param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02), regularizer=L2Decay(0.)),
            bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), regularizer=L2Decay(0.)))
        out = fluid.layers.leaky_relu(out, 0.1)
        return out

    def downsample(self, input, num_filters, filter_size=3, stride=2, padding=1):
        self.downsample_ratio *= 2
        return self.conv_bn(input, 
                num_filters=num_filters, 
                filter_size=filter_size, 
                stride=stride, 
                padding=padding)

    def basicblock(self, input, num_filters):
        conv1 = self.conv_bn(input, num_filters, filter_size=1, stride=1, padding=0)
        conv2 = self.conv_bn(conv1, num_filters * 2, filter_size=3, stride=1, padding=1)
        out = fluid.layers.elementwise_add(x=input, y=conv2, act=None)
        return out

    def layer_warp(self, input, num_filters, count):
        res_out = self.basicblock(input, num_filters)
        for j in range(1, count):
            res_out = self.basicblock(res_out, num_filters)
        return res_out

    def upsample(self, input, scale=2):
        # get dynamic upsample output shape
        shape_nchw = fluid.layers.shape(input)
        shape_hw = fluid.layers.slice(shape_nchw, axes=[0], starts=[2], ends=[4])
        shape_hw.stop_gradient = True
        in_shape = fluid.layers.cast(shape_hw, dtype='int32')
        out_shape = in_shape * scale
        out_shape.stop_gradient = True

        # reisze by actual_shape
        out = fluid.layers.resize_nearest(
            input=input,
            scale=scale,
            actual_shape=out_shape)
        return out
    
    def yolo_detection_block(self, input, num_filters):
        assert num_filters % 2 == 0, "num_filters {} cannot be divided by 2".format(num_filters)
        conv = input
        for j in range(2):
            conv = self.conv_bn(conv, num_filters, filter_size=1, stride=1, padding=0)
            conv = self.conv_bn(conv, num_filters * 2, filter_size=3, stride=1, padding=1)
        route = self.conv_bn(conv, num_filters, filter_size=1, stride=1, padding=0)
        tip = self.conv_bn(route, num_filters * 2, filter_size=3, stride=1, padding=1)
        return route, tip

    def net(self, img): 
        # darknet
        stages = [1,2,8,8,4]
        assert len(self.anchor_mask) <= len(stages), "anchor masks can't bigger than downsample times"
        # 256x256
        conv1 = self.conv_bn(img, num_filters=32, filter_size=3, stride=1, padding=1)
        downsample_  = self.downsample(conv1, conv1.shape[1] * 2)
        blocks = []

        for i, stage_count in enumerate(stages):
            block = self.layer_warp(downsample_, 32 *(2**i), stage_count)
            blocks.append(block)
            if i < len(stages) - 1:
                downsample_ = self.downsample(block, block.shape[1]*2)
        blocks = blocks[-1:-4:-1]   # 取倒数三层，并且逆序，后面跨层级联需要

        # yolo detector
        for i, block in enumerate(blocks):
            # yolo 中跨视域链接
            if i > 0:
                block = fluid.layers.concat(input=[route, block], axis=1)
            route, tip = self.yolo_detection_block(block, num_filters=512 // (2**i))
            block_out = fluid.layers.conv2d(
                input=tip,
                num_filters=len(self.anchor_mask[i]) * (self.class_num + 5),      # 5 elements represent x|y|h|w|score
                filter_size=1,
                stride=1,
                padding=0,
                act=None,
                param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02)),
                bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), regularizer=L2Decay(0.)))
            self.outputs.append(block_out)
            # 为了跨视域链接，差值方式提升特征图尺寸
            if i < len(blocks) - 1:
                route = self.conv_bn(route, 256//(2**i), filter_size=1, stride=1, padding=0)
                route = self.upsample(route)

        return self.outputs


class YOLOv3Tiny(object):
    def __init__(self, class_num, anchors, anchor_mask):
        self.outputs = []
        self.downsample_ratio = 1
        self.anchor_mask = anchor_mask
        self.anchors = anchors
        self.class_num = class_num

        self.yolo_anchors = []
        self.yolo_classes = []
        for mask_pair in self.anchor_mask:
            mask_anchors = []
            for mask in mask_pair:
                mask_anchors.append(self.anchors[2 * mask])
                mask_anchors.append(self.anchors[2 * mask + 1])
            self.yolo_anchors.append(mask_anchors)
            self.yolo_classes.append(class_num)

    def name(self):
        return 'YOLOv3-tiny'

    def get_anchors(self):
        return self.anchors

    def get_anchor_mask(self):
        return self.anchor_mask

    def get_class_num(self):
        return self.class_num

    def get_downsample_ratio(self):
        return self.downsample_ratio

    def get_yolo_anchors(self):
        return self.yolo_anchors

    def get_yolo_classes(self):
        return self.yolo_classes

    def conv_bn(self,
                input,
                num_filters,
                filter_size,
                stride,
                padding,
                num_groups=1,
                use_cudnn=True):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            act=None,
            groups=num_groups,
            use_cudnn=use_cudnn,
            param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02)),
            bias_attr=False)

        # batch_norm中的参数不需要参与正则化，所以主动使用正则系数为0的正则项屏蔽掉
        out = fluid.layers.batch_norm(
            input=conv, act='relu', 
            param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02), regularizer=L2Decay(0.)),
            bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), regularizer=L2Decay(0.)))

        return out

    def depthwise_conv_bn(self, input, filter_size=3, stride=1, padding=1):
        num_filters = input.shape[1]
        return self.conv_bn(input, 
                num_filters=num_filters, 
                filter_size=filter_size, 
                stride=stride, 
                padding=padding, 
                num_groups=num_filters)

    def downsample(self, input, pool_size=2, pool_stride=2):
        self.downsample_ratio *= 2
        return fluid.layers.pool2d(input=input, pool_type='max', pool_size=pool_size,
                                    pool_stride=pool_stride)

    def basicblock(self, input, num_filters):
        conv1 = self.conv_bn(input, num_filters, filter_size=3, stride=1, padding=1)
        out = self.downsample(conv1)
        return out


    def upsample(self, input, scale=2):
        # get dynamic upsample output shape
        shape_nchw = fluid.layers.shape(input)
        shape_hw = fluid.layers.slice(shape_nchw, axes=[0], starts=[2], ends=[4])
        shape_hw.stop_gradient = True
        in_shape = fluid.layers.cast(shape_hw, dtype='int32')
        out_shape = in_shape * scale
        out_shape.stop_gradient = True

        # reisze by actual_shape
        out = fluid.layers.resize_nearest(
            input=input,
            scale=scale,
            actual_shape=out_shape)
        return out
    
    def yolo_detection_block(self, input, num_filters):
        route = self.conv_bn(input, num_filters, filter_size=1, stride=1, padding=0)
        tip = self.conv_bn(route, num_filters * 2, filter_size=3, stride=1, padding=1)
        return route, tip

    def net(self, img): 
        # darknet-tiny
        stages = [16, 32, 64, 128, 256, 512]
        assert len(self.anchor_mask) <= len(stages), "anchor masks can't bigger than downsample times"
        # 256x256
        tmp = img
        blocks = []
        for i, stage_count in enumerate(stages):
            if i == len(stages) - 1:
                block = self.conv_bn(tmp, stage_count, filter_size=3, stride=1, padding=1)
                blocks.append(block)
                block = self.depthwise_conv_bn(blocks[-1])
                block = self.depthwise_conv_bn(blocks[-1])
                block = self.conv_bn(blocks[-1], stage_count * 2, filter_size=1, stride=1, padding=0)
                blocks.append(block)
            else:
                tmp = self.basicblock(tmp, stage_count)
                blocks.append(tmp)
        
        blocks = [blocks[-1], blocks[3]]

        # yolo detector
        for i, block in enumerate(blocks):
            # yolo 中跨视域链接
            if i > 0:
                block = fluid.layers.concat(input=[route, block], axis=1)
            if i < 1:
                route, tip = self.yolo_detection_block(block, num_filters=256 // (2**i))
            else:
                tip = self.conv_bn(block, num_filters=256, filter_size=3, stride=1, padding=1)
            block_out = fluid.layers.conv2d(
                input=tip,
                num_filters=len(self.anchor_mask[i]) * (self.class_num + 5),      # 5 elements represent x|y|h|w|score
                filter_size=1,
                stride=1,
                padding=0,
                act=None,
                param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02)),
                bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), regularizer=L2Decay(0.)))
            self.outputs.append(block_out)
            # 为了跨视域链接，差值方式提升特征图尺寸
            if i < len(blocks) - 1:
                route = self.conv_bn(route, 128 // (2**i), filter_size=1, stride=1, padding=0)
                route = self.upsample(route)

        return self.outputs
        

def get_yolo(is_tiny, class_num, anchors, anchor_mask):
    if is_tiny:
        print('USE YOLOv3Tiny')
        return YOLOv3Tiny(class_num, anchors, anchor_mask)
    else:
        print('USE YOLOv3')
        return YOLOv3(class_num, anchors, anchor_mask)


# ## 定义初始化相关函数
# * init_train_parameters()主要作用是得到本数据集的class_dim（种类）,还有本数据集的总训练样本数image_count；
# 
# * init_log_config():初始化日志的相关配置；
#  
# 

# In[18]:


def init_train_parameters():
    """
    初始化训练参数，主要是初始化图片数量，类别数
    :return:
    """
    file_list = os.path.join(train_parameters['data_dir'], train_parameters['file_list'])
    label_list = os.path.join(train_parameters['data_dir'], "label_list")
    index = 0
    with codecs.open(label_list, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        for line in lines:
            train_parameters['label_dict'][line.strip()] = index
            index += 1
        train_parameters['class_dim'] = index
    with codecs.open(file_list, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        train_parameters['image_count'] = len(lines)


def init_log_config():
    """
    初始化日志相关配置
    :return:
    """
    global logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_path = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(log_path, 'train.log')
    sh = logging.StreamHandler()
    fh = logging.FileHandler(log_name, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)


# 图像增强处理的系列函数

# In[19]:


def box_to_center_relative(box, img_height, img_width):
    """
    Convert COCO annotations box with format [x1, y1, w, h] to 
    center mode [center_x, center_y, w, h] and divide image width
    and height to get relative value in range[0, 1]
    """
    assert len(box) == 4, "box should be a len(4) list or tuple"
    x, y, w, h = box

    x1 = max(x, 0)
    x2 = min(x + w - 1, img_width - 1)
    y1 = max(y, 0)
    y2 = min(y + h - 1, img_height - 1)

    x = (x1 + x2) / 2 / img_width
    y = (y1 + y2) / 2 / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height

    return np.array([x, y, w, h])


def resize_img(img, sampled_labels, input_size):
    target_size = input_size
    img = img.resize((target_size[1], target_size[2]), Image.BILINEAR)
    return img


def box_iou_xywh(box1, box2):
    assert box1.shape[-1] == 4, "Box1 shape[-1] should be 4."
    assert box2.shape[-1] == 4, "Box2 shape[-1] should be 4."

    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_y2 = np.minimum(b1_y2, b2_y2)
    inter_w = inter_x2 - inter_x1 + 1
    inter_h = inter_y2 - inter_y1 + 1
    inter_w[inter_w < 0] = 0
    inter_h[inter_h < 0] = 0

    inter_area = inter_w * inter_h
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    return inter_area / (b1_area + b2_area - inter_area)


def box_crop(boxes, labels, crop, img_shape):
    x, y, w, h = map(float, crop)
    im_w, im_h = map(float, img_shape)

    boxes = boxes.copy()
    boxes[:, 0], boxes[:, 2] = (boxes[:, 0] - boxes[:, 2] / 2) * im_w, (boxes[:, 0] + boxes[:, 2] / 2) * im_w
    boxes[:, 1], boxes[:, 3] = (boxes[:, 1] - boxes[:, 3] / 2) * im_h, (boxes[:, 1] + boxes[:, 3] / 2) * im_h

    crop_box = np.array([x, y, x + w, y + h])
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    mask = np.logical_and(crop_box[:2] <= centers, centers <= crop_box[2:]).all(axis=1)

    boxes[:, :2] = np.maximum(boxes[:, :2], crop_box[:2])
    boxes[:, 2:] = np.minimum(boxes[:, 2:], crop_box[2:])
    boxes[:, :2] -= crop_box[:2]
    boxes[:, 2:] -= crop_box[:2]

    mask = np.logical_and(mask, (boxes[:, :2] < boxes[:, 2:]).all(axis=1))
    boxes = boxes * np.expand_dims(mask.astype('float32'), axis=1)
    labels = labels * mask.astype('float32')
    boxes[:, 0], boxes[:, 2] = (boxes[:, 0] + boxes[:, 2]) / 2 / w, (boxes[:, 2] - boxes[:, 0]) / w
    boxes[:, 1], boxes[:, 3] = (boxes[:, 1] + boxes[:, 3]) / 2 / h, (boxes[:, 3] - boxes[:, 1]) / h

    return boxes, labels, mask.sum()


def random_brightness(img):
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_distort_strategy']['brightness_prob']:
        brightness_delta = train_parameters['image_distort_strategy']['brightness_delta']
        delta = np.random.uniform(-brightness_delta, brightness_delta) + 1
        img = ImageEnhance.Brightness(img).enhance(delta)
    return img


def random_contrast(img):
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_distort_strategy']['contrast_prob']:
        contrast_delta = train_parameters['image_distort_strategy']['contrast_delta']
        delta = np.random.uniform(-contrast_delta, contrast_delta) + 1
        img = ImageEnhance.Contrast(img).enhance(delta)
    return img


def random_saturation(img):
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_distort_strategy']['saturation_prob']:
        saturation_delta = train_parameters['image_distort_strategy']['saturation_delta']
        delta = np.random.uniform(-saturation_delta, saturation_delta) + 1
        img = ImageEnhance.Color(img).enhance(delta)
    return img


def random_hue(img):
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_distort_strategy']['hue_prob']:
        hue_delta = train_parameters['image_distort_strategy']['hue_delta']
        delta = np.random.uniform(-hue_delta, hue_delta)
        img_hsv = np.array(img.convert('HSV'))
        img_hsv[:, :, 0] = img_hsv[:, :, 0] + delta
        img = Image.fromarray(img_hsv, mode='HSV').convert('RGB')
    return img


def distort_image(img):
    prob = np.random.uniform(0, 1)
    # Apply different distort order
    if prob > 0.5:
        img = random_brightness(img)
        img = random_contrast(img)
        img = random_saturation(img)
        img = random_hue(img)
    else:
        img = random_brightness(img)
        img = random_saturation(img)
        img = random_hue(img)
        img = random_contrast(img)
    return img


def random_crop(img, boxes, labels, scales=[0.3, 1.0], max_ratio=2.0, constraints=None, max_trial=50):
    if random.random() > 0.6:
        return img, boxes, labels
    if len(boxes) == 0:
        return img, boxes, labels

    if not constraints:
        constraints = [
                (0.1, 1.0),
                (0.3, 1.0),
                (0.5, 1.0),
                (0.7, 1.0),
                (0.9, 1.0),
                (0.0, 1.0)]

    w, h = img.size
    crops = [(0, 0, w, h)]
    for min_iou, max_iou in constraints:
        for _ in range(max_trial):
            scale = random.uniform(scales[0], scales[1])
            aspect_ratio = random.uniform(max(1 / max_ratio, scale * scale),                                           min(max_ratio, 1 / scale / scale))
            crop_h = int(h * scale / np.sqrt(aspect_ratio))
            crop_w = int(w * scale * np.sqrt(aspect_ratio))
            crop_x = random.randrange(w - crop_w)
            crop_y = random.randrange(h - crop_h)
            crop_box = np.array([[
                (crop_x + crop_w / 2.0) / w,
                (crop_y + crop_h / 2.0) / h,
                crop_w / float(w),
                crop_h /float(h)
                ]])

            iou = box_iou_xywh(crop_box, boxes)
            if min_iou <= iou.min() and max_iou >= iou.max():
                crops.append((crop_x, crop_y, crop_w, crop_h))
                break

    while crops:
        crop = crops.pop(np.random.randint(0, len(crops)))
        crop_boxes, crop_labels, box_num = box_crop(boxes, labels, crop, (w, h))
        if box_num < 1:
            continue
        img = img.crop((crop[0], crop[1], crop[0] + crop[2], 
                        crop[1] + crop[3])).resize(img.size, Image.LANCZOS)
        return img, crop_boxes, crop_labels
    return img, boxes, labels


def random_expand(img, gtboxes, keep_ratio=True):
    if np.random.uniform(0, 1) < train_parameters['image_distort_strategy']['expand_prob']:
        return img, gtboxes

    max_ratio = train_parameters['image_distort_strategy']['expand_max_ratio']    
    w, h = img.size
    c = 3
    ratio_x = random.uniform(1, max_ratio)
    if keep_ratio:
        ratio_y = ratio_x
    else:
        ratio_y = random.uniform(1, max_ratio)
    oh = int(h * ratio_y)
    ow = int(w * ratio_x)
    off_x = random.randint(0, ow -w)
    off_y = random.randint(0, oh -h)

    out_img = np.zeros((oh, ow, c), np.uint8)
    for i in range(c):
        out_img[:, :, i] = train_parameters['mean_rgb'][i]

    out_img[off_y: off_y + h, off_x: off_x + w, :] = img
    gtboxes[:, 0] = ((gtboxes[:, 0] * w) + off_x) / float(ow)
    gtboxes[:, 1] = ((gtboxes[:, 1] * h) + off_y) / float(oh)
    gtboxes[:, 2] = gtboxes[:, 2] / ratio_x
    gtboxes[:, 3] = gtboxes[:, 3] / ratio_y

    return Image.fromarray(out_img), gtboxes


def preprocess(img, bbox_labels, input_size, mode):
    img_width, img_height = img.size
    sample_labels = np.array(bbox_labels)
    if mode == 'train':
        if train_parameters['apply_distort']:
            img = distort_image(img)
        img, gtboxes = random_expand(img, sample_labels[:, 1:5])
        img, gtboxes, gtlabels = random_crop(img, gtboxes, sample_labels[:, 0])
        sample_labels[:, 0] = gtlabels
        sample_labels[:, 1:5] = gtboxes
    img = resize_img(img, sample_labels, input_size)
    img = np.array(img).astype('float32')
    img -= train_parameters['mean_rgb']
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img *= 0.007843
    return img, sample_labels


# In[20]:


def custom_reader(file_list, data_dir, input_size, mode):
    def reader():
        np.random.shuffle(file_list)
        for line in file_list:
            if mode == 'train' or mode == 'eval':
                
                parts = line.split('\t')
                image_path = parts[0]
                img = Image.open(image_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                im_width, im_height = img.size
                # bbox 的列表，每一个元素为这样
                # layout: label | x-center | y-cneter | width | height | difficult
                bbox_labels = []
                for object_str in parts[1:]:
                    if len(object_str) <= 1:
                        continue
                    bbox_sample = []
                    object = json.loads(object_str)
                    bbox_sample.append(float(train_parameters['label_dict'][object['value']]))
                    bbox = object['coordinate']
                    box = [bbox[0][0], bbox[0][1], bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1]]
                    bbox = box_to_center_relative(box, im_height, im_width)
                    bbox_sample.append(float(bbox[0]))
                    bbox_sample.append(float(bbox[1]))
                    bbox_sample.append(float(bbox[2]))
                    bbox_sample.append(float(bbox[3]))
                    difficult = float(0)
                    bbox_sample.append(difficult)
                    bbox_labels.append(bbox_sample)
                
                if len(bbox_labels) == 0: continue
                img, sample_labels = preprocess(img, bbox_labels, input_size, mode)
                # sample_labels = np.array(sample_labels)
                if len(sample_labels) == 0: continue
                boxes = sample_labels[:, 1:5]
                lbls = sample_labels[:, 0].astype('int32')
                difficults = sample_labels[:, -1].astype('int32')
                max_box_num = train_parameters['max_box_num']
                cope_size = max_box_num if len(boxes) >= max_box_num else len(boxes)
                ret_boxes = np.zeros((max_box_num, 4), dtype=np.float32)
                ret_lbls = np.zeros((max_box_num), dtype=np.int32)
                ret_difficults = np.zeros((max_box_num), dtype=np.int32)
                ret_boxes[0: cope_size] = boxes[0: cope_size]
                ret_lbls[0: cope_size] = lbls[0: cope_size]
                ret_difficults[0: cope_size] = difficults[0: cope_size]
                yield img, ret_boxes, ret_lbls, ret_difficults
            elif mode == 'test':
                img_path = os.path.join(line)
                yield Image.open(img_path)

    return reader


# 
# ## 训练准备
# * 定义异步数据读取
# 
# * 定义优化器
# 
# * 参数创建完成后，我们需要定义一个优化器optimizer，为了改善模型的训练速度以及效果，学术界先后提出了很多优化算法，包括： Momentum、RMSProp、Adam 等，已经被封装在fluid内部，读者可直接调用。
# 
# * 构建 program 和损失函数
# 

# In[21]:


def multi_process_custom_reader(file_path, data_dir, num_workers, input_size, mode):
    file_path = os.path.join(data_dir, file_path)
    readers = []
    images = [line.strip() for line in open(file_path)]
    n = int(math.ceil(len(images) // num_workers))
    image_lists = [images[i: i + n] for i in range(0, len(images), n)]
    for l in image_lists:
        readers.append(paddle.batch(custom_reader(l, data_dir, input_size, mode), 
                                    batch_size=train_parameters['train_batch_size']))
    return paddle.reader.multiprocess_reader(readers, False)


def create_eval_reader(file_path, data_dir, input_size, mode):
    file_path = os.path.join(data_dir, file_path)
    images = [line.strip() for line in open(file_path)]
    return paddle.batch(custom_reader(images, data_dir, input_size, mode), 
                        batch_size=train_parameters['train_batch_size'],
                        drop_last=True)


def optimizer_momentum_setting():
    learning_strategy = train_parameters['momentum_strategy']
    learning_rate = fluid.layers.exponential_decay(learning_rate=learning_strategy['learning_rate'],
                                                   decay_steps=learning_strategy['decay_steps'],
                                                   decay_rate=learning_strategy['decay_rate'])
    optimizer = fluid.optimizer.MomentumOptimizer(learning_rate=learning_rate, momentum=0.1)
    return optimizer


def optimizer_rms_setting():
    batch_size = train_parameters["train_batch_size"]
    iters = train_parameters["image_count"] // batch_size
    learning_strategy = train_parameters['rsm_strategy']
    lr = learning_strategy['learning_rate']

    boundaries = [i * iters for i in learning_strategy["lr_epochs"]]
    values = [i * lr for i in learning_strategy["lr_decay"]]

    optimizer = fluid.optimizer.RMSProp(
        learning_rate=fluid.layers.piecewise_decay(boundaries, values),
        regularization=fluid.regularizer.L2Decay(0.00005))

    return optimizer


def build_train_program_with_async_reader(main_prog, startup_prog):
    max_box_num = train_parameters['max_box_num']
    ues_tiny = train_parameters['use_tiny']
    yolo_config = train_parameters['yolo_tiny_cfg'] if ues_tiny else train_parameters['yolo_cfg']
    with fluid.program_guard(main_prog, startup_prog):
        img = fluid.layers.data(name='img', shape=yolo_config['input_size'], dtype='float32')
        gt_box = fluid.layers.data(name='gt_box', shape=[max_box_num, 4], dtype='float32', lod_level=0)
        gt_label = fluid.layers.data(name='gt_label', shape=[max_box_num], dtype='int32', lod_level=0)
        difficult = fluid.layers.data(name='difficult', shape=[max_box_num], dtype='int32', lod_level=0)
        data_reader = fluid.layers.create_py_reader_by_data(capacity=train_parameters['train_batch_size'],
                                                            feed_list=[img, gt_box, gt_label, difficult],
                                                            name='train')
        multi_reader = multi_process_custom_reader(train_parameters['file_list'],
                                                   train_parameters['data_dir'],
                                                   train_parameters['multi_data_reader_count'],
                                                   yolo_config['input_size'],
                                                   'train')
        data_reader.decorate_paddle_reader(multi_reader)
        with fluid.unique_name.guard():
            img, gt_box, gt_label, difficult = fluid.layers.read_file(data_reader)
            model = get_yolo(ues_tiny, train_parameters['class_dim'], yolo_config['anchors'], yolo_config['anchor_mask'])
            outputs = model.net(img)
            losses = []
            downsample_ratio = model.get_downsample_ratio()
            with fluid.unique_name.guard('train'):
                for i, out in enumerate(outputs):
                    logger.info("{0} downsample_ratio: {1} output:{2}".format(i, downsample_ratio, out))
                    loss = fluid.layers.yolov3_loss(
                            x=out,
                            gtbox=gt_box,
                            gtlabel=gt_label,
                            anchors=model.get_anchors(),
                            anchor_mask=model.get_anchor_mask()[i],
                            class_num=model.get_class_num(),
                            ignore_thresh=train_parameters['ignore_thresh'],
                            downsample_ratio=downsample_ratio)
                    losses.append(fluid.layers.reduce_mean(loss))
                    downsample_ratio //= 2
                loss = sum(losses)
                optimizer = optimizer_rms_setting()
                optimizer.minimize(loss)
                return data_reader, loss


def build_eval_program_with_feeder(main_prog, startup_prog, place):
    ues_tiny = train_parameters['use_tiny']
    yolo_config = train_parameters['yolo_tiny_cfg'] if ues_tiny else train_parameters['yolo_cfg']
    with fluid.program_guard(main_prog, startup_prog):
        img = fluid.layers.data(name='img', shape=yolo_config['input_size'], dtype='float32')
        gt_box = fluid.layers.data(name='gt_box', shape=[4], dtype='float32', lod_level=1)
        gt_label = fluid.layers.data(name='gt_label', shape=[1], dtype='int32', lod_level=1)
        difficult = fluid.layers.data(name='difficult', shape=[1], dtype='int32', lod_level=1)
        feeder = fluid.DataFeeder(feed_list=[img, gt_box, gt_label, difficult], place=place, program=main_prog)
        reader = create_eval_reader(train_parameters['file_list'], train_parameters['data_dir'], 
                                    yolo_config['input_size'], 'eval')
        with fluid.unique_name.guard():
            model = get_yolo(ues_tiny, train_parameters['class_dim'], yolo_config['anchors'], yolo_config['anchor_mask'])
            outputs = model.net(img)
            return feeder, reader, outputs, gt_box, gt_label, difficult


# ## 导入已有模型
# 定义一个函数load_pretrained_params()，来做两个选择：
# 
# ①如果train_parameters['continue_train'] = true 则在上次保存的模型基础上继续训练；
# 
# ②如果train_parameters['continue_train'] = False，且train_parameters['pretrained'] = True，则使用预训练模型；
# 
# ③但是要注意，因为代码中使用的是if ...elif ...所以，如果train_parameters['continue_train'] = True，则不会执行elif，也就是不会加载预训练模型；

# In[22]:


def load_pretrained_params(exe, program):
    if train_parameters['continue_train'] and os.path.exists(train_parameters['save_model_dir']):
        logger.info('load param from retrain model')
        fluid.io.load_persistables(executor=exe,
                                   dirname=train_parameters['save_model_dir'],
                                   main_program=program)
    elif train_parameters['pretrained'] and os.path.exists(train_parameters['pretrained_model_dir']):
        logger.info('load param from pretrained model')
        def if_exist(var):
            return os.path.exists(os.path.join(train_parameters['pretrained_model_dir'], var.name))

        fluid.io.load_vars(exe, train_parameters['pretrained_model_dir'], main_program=program,
                           predicate=if_exist)


# ## 开始训练
# 接下来，我们就定义一个train()方法，然后就可以调用这个方法，来执行训练了； train()方法里有几个工作要做，比如：
# 
# １．定义训练场所：
# 
# ２．定义执行器： 为了能够运行开发者定义的网络拓扑结构和优化器，需要定义执行器。由执行器来真正的执行参数的初始化和网络的训练过程。fulid使用了一个C++类Executor用于运行一个程序，Executor类似一个解释器，Fluid将会使用这样一个解析器来训练和测试模型。 

# In[23]:


def train():
    init_log_config()
    init_train_parameters()
    logger.info("start train YOLOv3, train params:%s", str(train_parameters))

    logger.info("create place, use gpu:" + str(train_parameters['use_gpu']))
    place = fluid.CUDAPlace(0) if train_parameters['use_gpu'] else fluid.CPUPlace()

    logger.info("build network and program")
    train_program = fluid.Program()
    start_program = fluid.Program()
    eval_program = fluid.Program()
    start_program = fluid.Program()
    train_reader, loss = build_train_program_with_async_reader(train_program, start_program)
    eval_feeder, eval_reader, outputs, gt_box, gt_label, difficult = build_eval_program_with_feeder(eval_program, start_program, place)
    eval_program = eval_program.clone(for_test=True)

    logger.info("build executor and init params")
    exe = fluid.Executor(place)
    exe.run(start_program)
    train_fetch_list = [loss.name]
    eval_fetch_list = [v.name for v in outputs]
    load_pretrained_params(exe, train_program)


    stop_strategy = train_parameters['early_stop']
    successive_limit = stop_strategy['successive_limit']
    sample_freq = stop_strategy['sample_frequency']
    min_curr_map = stop_strategy['min_curr_map']
    min_loss = stop_strategy['min_loss']
    stop_train = False
    successive_count = 0
    total_batch_count = 0
    valid_thresh = train_parameters['valid_thresh']
    nms_thresh = train_parameters['nms_thresh']
    for pass_id in range(train_parameters["num_epochs"]):
        logger.info("current pass: %d, start read image", pass_id)
        batch_id = 0
        train_reader.start()
        try:
            while True:
                t1 = time.time()
                loss = exe.run(train_program, fetch_list=train_fetch_list)
                period = time.time() - t1
                loss = np.mean(np.array(loss))
                batch_id += 1
                total_batch_count += 1

                if batch_id % 10 == 0:
                    logger.info(
                        "Pass {0}, trainbatch {1}, loss {2} time {3}".format(pass_id, batch_id, loss, "%2.2f sec" % period))
                # 采用简单的定时采样停止办法，可以调整为更精细的保存策略
                if total_batch_count % 100 == 0:
                    logger.info("temp save {0} batch train result".format(total_batch_count))
                    fluid.io.save_persistables(dirname=train_parameters['save_model_dir'],
                                               main_program=train_program,
                                               executor=exe)
        except fluid.core.EOFException:
            train_reader.reset()

    logger.info("training till last epcho, end training")
    fluid.io.save_persistables(dirname=train_parameters['save_model_dir'], main_program=train_program, executor=exe)


if __name__ == '__main__':
    train()


# ## 模型压缩
# 我们可以将训练好的模型进行压缩带一个压缩包里，这样方便复制和移动；
# 
# 注意： 这一条是linux命令，和编程是没有关系的。

# In[78]:


get_ipython().system('tar -cf yolo-model.tar yolo-model/')


# ## 模型固化
# 接下来我们把训练好的模型进行固化，为什么需要把模型进行固化呢？是因为：
# 
# 1.训练后保存的模型，有很多保留项，比如有优化器的、BN等，（比如本次yolo-model里面有1034项）；
# 
# 2.固化后的模型，只保留和主模型有关的，和预测有关的项的参数，（比如本次固化后的freeze_model里面有367项）；

# In[81]:



label_dict = {}
with codecs.open('data/data7085/label_list.txt') as f:
    for line in f:
        parts = line.strip().split()
        label_dict[float(parts[0])] = parts[1]
print(label_dict)
class_dim = len(label_dict)

def freeze_model():

    path = "./yolo-model"
    exe = fluid.Executor(fluid.CPUPlace())

    ues_tiny = train_parameters['use_tiny']
    yolo_config = train_parameters['yolo_tiny_cfg'] if ues_tiny else train_parameters['yolo_cfg']
    model = model = get_yolo(ues_tiny, class_dim, yolo_config['anchors'], yolo_config['anchor_mask'])
    image = fluid.layers.data(name='image', shape=yolo_config['input_size'], dtype='float32')
    pred = model.net(image)
    
    freeze_program = fluid.default_main_program()
    fluid.io.load_persistables(exe, path, freeze_program)
    freeze_program = freeze_program.clone(for_test=True)

    fluid.io.save_inference_model("./freeze_model", ['image'], pred, exe, freeze_program)


if __name__ == '__main__':
    freeze_model()


# In[ ]:


#接下来这一条还是linux命令；
#把固化的模型就行压缩
get_ipython().system('tar -cf freeze_yolov3_model.tar freeze_yolov3_model/')


# ## 预测阶段
# 接下来我们开始预测阶段，此阶段我们主要有以下工作：
# 
# 1.定义几个用于画图的方法，目的是在预测出的结果上可视化边界框，并标出类别；
# 
# 2.定义预测方法infer(),我们可以直接调用这个infer()方法来执行预测； 
# 

# In[73]:


import codecs
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import os


ues_tiny = train_parameters['use_tiny']
yolo_config = train_parameters['yolo_tiny_cfg'] if ues_tiny else train_parameters['yolo_cfg']

target_size = yolo_config['input_size']
anchors = yolo_config['anchors']
anchor_mask = yolo_config['anchor_mask']

nms_threshold = 0.4
valid_thresh = 0.4
confs_threshold = 0.5
label_dict = {}
with codecs.open('data/data7085/label_list.txt') as f:
    for line in f:
        parts = line.strip().split()
        label_dict[str(float(parts[0]))] = parts[1]
print(label_dict)
class_dim = len(label_dict)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
path = "./freeze_yolov3_model"
[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=path, executor=exe)


def get_yolo_anchors_classes(class_num, anchors, anchor_mask):
    yolo_anchors = []
    yolo_classes = []
    for mask_pair in anchor_mask:
        mask_anchors = []
        for mask in mask_pair:
            mask_anchors.append(anchors[2 * mask])
            mask_anchors.append(anchors[2 * mask + 1])
        yolo_anchors.append(mask_anchors)
        yolo_classes.append(class_num)
    return yolo_anchors, yolo_classes


def draw_bbox_image(img, boxes, labels, save_name):
    """
    给图片画上外接矩形框
    :param img:
    :param boxes:
    :param save_name:
    :param labels
    :return:
    """
    # font = ImageFont.truetype("font.ttf", 25)
    img_width, img_height = img.size
    draw = ImageDraw.Draw(img)
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        draw.rectangle((xmin, ymin, xmax, ymax), None, 'red')
        draw.text((xmin, ymin), label_dict[str(label)], (255, 255, 0))
    img.save(save_name)
    return img


def clip_bbox(bbox):
    """
    截断矩形框
    :param bbox:
    :return:
    """
    xmin = max(min(bbox[0], 1.), 0.)
    ymin = max(min(bbox[1], 1.), 0.)
    xmax = max(min(bbox[2], 1.), 0.)
    ymax = max(min(bbox[3], 1.), 0.)
    return xmin, ymin, xmax, ymax


def resize_img(img, target_size):
    """
    保持比例的缩放图片
    :param img:
    :param target_size:
    :return:
    """
    img = img.resize(target_size[1:], Image.ANTIALIAS)
    return img


def read_image(img_path):
    """
    读取图片
    :param img_path:
    :return:
    """
    origin = Image.open(img_path)
    img = resize_img(origin, target_size)
    resized_img = img.copy()
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = np.array(img).astype('float32').transpose((2, 0, 1))  # HWC to CHW
    img -= 127.5
    img *= 0.007843
    img = img[np.newaxis, :]
    return origin, img, resized_img


def sigmoid(x):
    """Perform sigmoid to input numpy array"""
    return 1.0 / (1.0 + np.exp(-1.0 * x))


def box_xywh_to_xyxy(box):
    """
    bbox 两种形式的转换，左上角和宽高---->左上角|右下角
    :param box:
    :return:
    """
    shape = box.shape
    assert shape[-1] == 4, "Box shape[-1] should be 4."

    box = box.reshape((-1, 4))
    box[:, 0], box[:, 2] = box[:, 0] - box[:, 2] / 2, box[:, 0] + box[:, 2] / 2
    box[:, 1], box[:, 3] = box[:, 1] - box[:, 3] / 2, box[:, 1] + box[:, 3] / 2
    box = box.reshape(shape)
    return box


def box_iou_xyxy(box1, box2):
    assert box1.shape[-1] == 4, "Box1 shape[-1] should be 4."
    assert box2.shape[-1] == 4, "Box2 shape[-1] should be 4."

    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_y2 = np.minimum(b1_y2, b2_y2)
    inter_w = inter_x2 - inter_x1
    inter_h = inter_y2 - inter_y1
    inter_w[inter_w < 0] = 0
    inter_h[inter_h < 0] = 0

    inter_area = inter_w * inter_h
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    return inter_area / (b1_area + b2_area - inter_area)


def rescale_box_in_input_image(boxes, im_shape, input_size):
    """Scale (x1, x2, y1, y2) box of yolo output to input image"""
    h, w = im_shape
    fx = w / input_size
    fy = h / input_size
    boxes[:, 0] *= fx
    boxes[:, 1] *= fy
    boxes[:, 2] *= fx
    boxes[:, 3] *= fy
    boxes[boxes<0] = 0
    boxes[:, 2][boxes[:, 2] > (w - 1)] = w - 1
    boxes[:, 3][boxes[:, 3] > (h - 1)] = h - 1
    return boxes


def get_yolo_detection(preds, anchors, class_num, img_height, img_width):
    """Get yolo box, confidence score, class label from Darknet53 output"""
    preds_n = np.array(preds)
    n, c, h, w = preds_n.shape
    print(preds_n.shape, anchors)
    anchor_num = len(anchors) // 2
    preds_n = preds_n.reshape([n, anchor_num, class_num + 5, h, w]).transpose((0, 1, 3, 4, 2))
    preds_n[:, :, :, :, :2] = sigmoid(preds_n[:, :, :, :, :2])
    preds_n[:, :, :, :, 4:] = sigmoid(preds_n[:, :, :, :, 4:])

    pred_boxes = preds_n[:, :, :, :, :4]
    pred_confs = preds_n[:, :, :, :, 4]
    pred_scores = preds_n[:, :, :, :, 5:] * np.expand_dims(pred_confs, axis=4)

    grid_x = np.tile(np.arange(w).reshape((1, w)), (h, 1))
    grid_y = np.tile(np.arange(h).reshape((h, 1)), (1, w))
    anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
    anchors_s = np.array([(an_w, an_h) for an_w, an_h in anchors])
    anchor_w = anchors_s[:, 0:1].reshape((1, anchor_num, 1, 1))
    anchor_h = anchors_s[:, 1:2].reshape((1, anchor_num, 1, 1))

    pred_boxes[:, :, :, :, 0] += grid_x
    pred_boxes[:, :, :, :, 1] += grid_y
    pred_boxes[:, :, :, :, 2] = np.exp(pred_boxes[:, :, :, :, 2]) * anchor_w
    pred_boxes[:, :, :, :, 3] = np.exp(pred_boxes[:, :, :, :, 3]) * anchor_h
    
    pred_boxes[:, :, :, :, 0] = pred_boxes[:, :, :, :, 0] * img_width / w
    pred_boxes[:, :, :, :, 1] = pred_boxes[:, :, :, :, 1] * img_height / h
    pred_boxes[:, :, :, :, 2] = pred_boxes[:, :, :, :, 2]
    pred_boxes[:, :, :, :, 3] = pred_boxes[:, :, :, :, 3]

    pred_boxes = box_xywh_to_xyxy(pred_boxes)
    pred_boxes = np.tile(np.expand_dims(pred_boxes, axis=4), (1, 1, 1, 1, class_num, 1))
    pred_labels = np.zeros_like(pred_scores) + np.arange(class_num)

    return pred_boxes.reshape((n, -1, 4)), pred_scores.reshape((n, -1)), pred_labels.reshape((n, -1))


def get_all_yolo_pred(outputs, yolo_anchors, yolo_classes, input_shape):
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_labels = []
    for output, anchors, classes in zip(outputs, yolo_anchors, yolo_classes):
        pred_boxes, pred_scores, pred_labels = get_yolo_detection(output, anchors, classes, input_shape[0], input_shape[1])
        all_pred_boxes.append(pred_boxes)
        all_pred_labels.append(pred_labels)
        all_pred_scores.append(pred_scores)
    pred_boxes = np.concatenate(all_pred_boxes, axis=1)
    pred_scores = np.concatenate(all_pred_scores, axis=1)
    pred_labels = np.concatenate(all_pred_labels, axis=1)

    return pred_boxes, pred_scores, pred_labels


def calc_nms_box(pred_boxes, pred_scores, pred_labels, valid_thresh=0.4, nms_thresh=0.45, nms_topk=400):
    output_boxes = np.empty((0, 4))
    output_scores = np.empty(0)
    output_labels = np.empty(0)
    for boxes, labels, scores in zip(pred_boxes, pred_labels, pred_scores):
        valid_mask = scores > valid_thresh
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        labels = labels[valid_mask]

        score_sort_index = np.argsort(scores)[::-1]
        boxes = boxes[score_sort_index][:nms_topk]
        scores = scores[score_sort_index][:nms_topk]
        labels = labels[score_sort_index][:nms_topk]

        for c in np.unique(labels):
            c_mask = labels == c
            c_boxes = boxes[c_mask]
            c_scores = scores[c_mask]

            detect_boxes = []
            detect_scores = []
            detect_labels = []
            while c_boxes.shape[0]:
                detect_boxes.append(c_boxes[0])
                detect_scores.append(c_scores[0])
                detect_labels.append(c)
                if c_boxes.shape[0] == 1:
                    break
                iou = box_iou_xyxy(detect_boxes[-1].reshape((1, 4)), c_boxes[1:])
                c_boxes = c_boxes[1:][iou < nms_thresh]
                c_scores = c_scores[1:][iou < nms_thresh]

            output_boxes = np.append(output_boxes, detect_boxes, axis=0)
            output_scores = np.append(output_scores, detect_scores)
            output_labels = np.append(output_labels, detect_labels)
    return output_boxes, output_scores, output_labels


def infer(image_path):
    """
    预测，将结果保存到一副新的图片中
    :param image_path:
    :return:
    """
    origin, tensor_img, resized_img = read_image(image_path)
    t1 = time.time()
    batch_outputs = exe.run(inference_program,
                        feed={feed_target_names[0]: tensor_img},
                        fetch_list=fetch_targets)
    period = time.time() - t1
    print("predict cost time:{0}".format("%2.2f sec" % period))
    input_w, input_h = origin.size[0], origin.size[1]
    yolo_anchors, yolo_classes = get_yolo_anchors_classes(class_dim, anchors, anchor_mask)
    pred_boxes, pred_scores, pred_labels = get_all_yolo_pred(batch_outputs, yolo_anchors, yolo_classes, (target_size[1], target_size[2]))
    boxes, scores, labels = calc_nms_box(pred_boxes, pred_scores, pred_labels, valid_thresh, nms_threshold)
    boxes = rescale_box_in_input_image(boxes, [input_h, input_w], target_size[1])
    print("result boxes: ", boxes)
    print("result scores:", scores)
    print("result labels:", labels)
    last_dot_index = image_path.rfind('.')
    out_path = image_path[:last_dot_index]
    out_path += '-reslut.jpg'
    draw_bbox_image(origin, boxes, labels, out_path)


if __name__ == '__main__':
    # image_path = sys.argv[1]
    name_list = os.listdir('data/data7085/test/images/')
    for name_ in name_list:
        image_path = os.path.join('data/data7085/test/images', name_)
        infer(image_path)


# In[ ]:




