import os
import torch.utils.data as data
from torchvision.transforms import transforms
import PIL.Image as Image
import h5py
import random
import numpy as np

# dataset返回img和embedding


class Dataset(data.Dataset):
    def __init__(self, img_dir, img_endwith, embedd_dir, embedd_endwith,):
        super(Dataset, self).__init__()
        self.img_dir = img_dir
        self.img_endwith = img_endwith
        self.embedd_dir = embedd_dir
        self.embedd_endwith = embedd_endwith

        self.img_paths = self.get_path(self.img_dir, self.img_endwith)
        self.embedd_paths = self.get_path(self.embedd_dir, self.embedd_endwith)

        # 排序
        self.img_paths.sort(key=lambda x: x.split(os.altsep)[-1].split('.')[0])
        self.embedd_paths.sort(
            key=lambda x: x.split(os.altsep)[-1].split('.')[0])
        # 获得文件名(不含路径和后缀)
        self.img_filenames = [x.split(os.altsep)[-1].split('.')[0]
                              for x in self.img_paths]
        self.embedd_filenames = [x.split(os.altsep)[-1].split('.')[0]
                                 for x in self.embedd_paths]
    # 换行符 os.linesep \r\n
    # 分隔符 os.altsep /

    @staticmethod
    def get_path(root_dir, endswith):
        root_dir = os.path.join(*root_dir.split("/"))
        paths = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith(endswith):
                    paths.append(os.path.join(dirpath, filename))
        print(paths[0])
        return paths

    def load_img(self, img_path):
        img = Image.open(img_path).convert('RGB')
        return img

    def load_embedd(self, embedd_path, max_length=300):
        """[summary]
        数据并非文本, 而是多个词向量,我们随机选择一个词向量.
        Args:
            embedd_path (str): 该图片对应的词向量路径 
        """
        with h5py.File(embedd_path, mode='r') as f:
            key = random.choice(list(f.keys()))
            embedd = [int(v) for v in f[key]]
            vec = np.zeros(max_length, np.float32)
            vec[:len(embedd)] = embedd
            return vec

    def __getitem__(self, idx):
        assert self.img_filenames[idx] == self.embedd_filenames[idx]
        img_path = self.img_paths[idx]
        embedd_path = self.embedd_paths[idx]
        img = self.load_img(img_path)
        embedd = self.load_embedd(embedd_path)
        return img, embedd

    def __len__(self):
        return len(self.filenames)


img_dir = "../data/102flowers"
embedd_dir = "../data/102flowers_text10"

# # 测试
# dataset = Dataset(img_dir, "jpg", embedd_dir, "h5")
# img = dataset.__getitem__(1)[0]
# img = np.array(img)
# emb = dataset.__getitem__(1)[1]
# print(img.shape)  # (500,625,3)
# print(emb.shape)  # (300,)

# import matplotlib.pyplot as plt
# plt.imshow(img)
