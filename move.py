#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 将一个文件夹下图片按比例分在三个文件夹下
import xml.dom.minidom
import os
import random
import shutil
from shutil import copy2

# datadir_01 = "/home/lfw/cocodata/data/JCOCO/train2017/"
# datadir_02 = '/home/lfw/cocodata/data/JCOCO/val2017/'
# # datadir_03 = '/home/lfw/cocodata/data/coco/test2017/'
# xmlpath = "/home/lfw/data/TVOCdevkit/VOC2007/Annotations/"
datadir_01 = "E:/efficientdet/Yet-Another-EfficientDet-Pytorch-master/datasets/TT100K/train2017/"
datadir_02 = 'E:/efficientdet/Yet-Another-EfficientDet-Pytorch-master/datasets/TT100K/val2017/'
# datadir_03 = '/home/lfw/cocodata/data/coco/test2017/'
xmlpath = "E:/efficientdet/Yet-Another-EfficientDet-Pytorch-master/datasets/TT100K/Annotations2/"

trainDir = "E:/new_dataclass/train/"  # （将训练集放在这个文件夹下）
if not os.path.exists(trainDir):
    os.mkdir(trainDir)

validDir = "E:/new_dataclass/val/"  # （将验证集放在这个文件夹下）
if not os.path.exists(validDir):
    os.mkdir(validDir)

# testDir = '/home/lfw/xml/test/'  # （将测试集放在这个文件夹下）
# if not os.path.exists(testDir):
#     os.mkdir(testDir)
# 移动train文件
train_data = os.listdir(datadir_01)  # （图片文件夹）
num_train_data = len(train_data)
for train_file in train_data:
    a1, b1 = os.path.splitext(train_file)  # 分离出文件名a
    s_train = a1+".xml"
    fileName1 = os.path.join(xmlpath, s_train)
    copy2(fileName1, trainDir)

# 移动VAL文件
val_data = os.listdir(datadir_02)  # （图片文件夹）
num_val_data = len(val_data)
for val_file in val_data:
    a2, b2 = os.path.splitext(val_file)  # 分离出文件名a
    s_val = a2+".xml"
    fileName2 = os.path.join(xmlpath, s_val)
    copy2(fileName2, validDir)

# 移动Test文件

# test_data = os.listdir(datadir_03)  # （图片文件夹）
# num_test_data = len(test_data)
# for test_file in test_data:
#     a3, b3 = os.path.splitext(test_file)  # 分离出文件名a
#     s_test = a3+".xml"
#     fileName3 = os.path.join(xmlpath, s_test)
#     copy2(fileName3, testDir)
#
#
#

