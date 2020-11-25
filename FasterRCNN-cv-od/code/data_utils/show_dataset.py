# %%
# Annotations/xml文件
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from os import truncate
from bs4 import BeautifulSoup
import numpy as np

xml_path = "../../data/VOCdevkit/VOC2007/Annotations/000001.xml"
img_path = "../../data/VOCdevkit/VOC2007/JPEGImages/000001.jpg"

with open(xml_path) as f:
    soup = BeautifulSoup(f, 'xml')
    # print(soup)
    folder = soup.folder.text
    filename = soup.filename.text
    boxs = []
    for obj in soup.find_all('object'):
        # name = obj.find('name').text

        obj_name = obj.find('name').text  # 注: obj.name代表object本身的名字,而不是标签名字.
        obj_pose = obj.pose.text
        obj_truncated = int(obj.truncated.text)
        obj_difficult = int(obj.difficult.text)
        bndbox_xmin = int(obj.bndbox.xmin.text)
        bndbox_ymin = int(obj.bndbox.ymin.text)
        bndbox_xmax = int(obj.bndbox.xmax.text)
        bndbox_ymax = int(obj.bndbox.ymax.text)

        box = []  # bndbox的坐标和目标分类
        box.append(bndbox_xmin)  # voc数据的坐标从1开始,这里修正为0开始
        box.append(bndbox_ymin)
        box.append(bndbox_xmax)
        box.append(bndbox_ymax)
        box.append(obj_name)
        boxs.append(box)

        print(obj_name, obj_pose, obj_truncated, obj_difficult)
        print("bndbox", bndbox_xmin, bndbox_ymin, bndbox_xmax, bndbox_ymax)
        print()


# JPEGImages/jpg文件
img = Image.open(img_path)
plt.figure(figsize=(12, 8))
plt.axis('off')
plt.imshow(img)

# 将目标分类对应到数字序列上
classes = ['aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

class_to_idx = dict(zip(classes, range(len(classes))))
# print(class_to_idx)

# 可视化 在图片上显示box
# 初始化boxs
for box in boxs:
    box[0] = box[0] - 1 # voc数据的坐标从1开始,这里修正为0开始
    box[1] = box[1] - 1
    box[2] = box[2] - 1
    box[3] = box[3] - 1
    box[-1] = class_to_idx[box[-1]] # 将目标分类转为对应数字
# 加载boxs

for box in boxs:
    xmin = box[0]
    ymin = box[1]
    xmax = box[2]
    ymax = box[3]
    obj_name = classes[int(box[-1])]
    current_axis = plt.gca()
    current_axis.add_patch(plt.Rectangle(
        (xmin, ymin), xmax-xmin, ymax-ymin, color='yellow', fill=False, linewidth=2))
    current_axis.text(xmin, ymin, obj_name, size='x-large',
                    color='black', bbox={'facecolor': 'white', 'alpha': 0.8})
# %%
