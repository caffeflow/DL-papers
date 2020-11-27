import numpy as np
import torch
from torch._C import ListType, TensorType
import torch.nn as nn
from torch.utils import data
import torchvision.transforms as tvtsf
import torchvision.transforms.functional as tvtsfunc
import PIL.Image as Image
from torchvision.transforms.functional import scale
# img = Image.open('faster-speed.jpg')
# print(img.size)
# img = tvtsfunc.to_tensor(img)
# img = tvtsfunc.resize(img,[50,100])
# print(img.shape)


class Dataset(data.Dataset):
    def __init__(self,):
        super(Dataset, self).__init__()

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult =
        if self.aug is not None:
            pass

    def __len__(self):
        pass


class Transform(object):
    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    @ staticmethod
    def inverse_normalize(img: torch.Tensor, mean: list = [0.485, 0.456, 0.406], std: list = [0.229, 0.224, 0.225]) -> torch.Tensor:
        ### 去正则化, img输出维度为[[R,G,B],H,W]
        img = img + np.array(mean).reshape(-1, 1, 1)  # 维度一致才相加(h,w方向为广播复制)
        img = img * np.array(std).reshape(-1, 1, 1)
        img = img.clip(min=0, max=1) * 255  # 裁剪边缘
        return img

    # img缩放,正则化,张量化
    @ staticmethod
    def preprocess(self, img: Image.Image, mean: list = [0.485, 0.456, 0.406], std: list = [0.229, 0.224, 0.225], min_size: int = 600, max_size: int = 1000) -> torch.Tensor:
        w, h = img.size  # 注意返回值顺序
        scale1 = min_size / min(w, h)
        scale2 = max_size / max(w, h)
        scale = min(scale1, scale2)
        img = tvtsfunc.resize(img, [h*scale, w*scale])
        img = tvtsfunc.to_tensor(img)
        img = img / 255.
        img = tvtsfunc.normalize(img, mean=mean, std=std)
        return img

    def __call__(self, img, bbox, label):
        _, H, W = img.shape
