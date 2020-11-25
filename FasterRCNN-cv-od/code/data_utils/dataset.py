"""
    transform 图片和box是同步的进行的.

    """

from numpy.lib.shape_base import take_along_axis
from torch.utils import data
import torchvision as tv
import torchvision.transforms.functional as tvfunc
import PIL.Image as Image
# img = Image.open('faster-speed.jpg')
# print(img.size)
# img = tvfunc.to_tensor(img)
# img = tvfunc.resize(img,[50,100])
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

    def transform_func(self, img, bbox, min_size=600, max_size=1000):
        # PIL Image(RGB) or Tensor
        # to Tensor
        img = tvfunc.to_tensor(img)
        bbox = tvfunc.to_tensor(bbox)
        # random
        import random
        if random.choice([True, False]):
            img = tvfunc.vflip(img)
            bbox = tvfunc.vflip(bbox)
        if random.choice([True, False]):
            img = tvfunc.hflip(img)
            bbox = tvfunc.hflip(bbox)
        # resize
        c, h, w = img.shape
        scale = min(min_size/min(h, w), max_size/max(h, w))
        img = tvfunc.resize(img, [h*scale, w*scale])
        
        # normalize
        tvfunc.normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])



# ######## bbox ##############
def resize_bbox(bbox,in_size,out_size):
    