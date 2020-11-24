import os
import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Dataset name: flowers, birds
__C.DATASET_NAME = 'birds'
__C.EMBEDDING_TYPE = 'cnn-rnn'
__C.CONFIG_NAME = ''
__C.GPU_ID = '0'
__C.CUDA = True
__C.WORKERS = 6

__C.NET_G = ''
__C.NET_D = ''
__C.STAGE1_G = ''
__C.DATA_DIR = ''
__C.VIS_COUNT = 64

__C.Z_DIM = 100
__C.IMSIZE = 64
__C.STAGE = 1


# Training options
__C.TRAIN = edict()
__C.TRAIN.FLAG = True
__C.TRAIN.BATCH_SIZE = 64
__C.TRAIN.MAX_EPOCH = 600
__C.TRAIN.SNAPSHOT_INTERVAL = 50
__C.TRAIN.PRETRAINED_MODEL = ''
__C.TRAIN.PRETRAINED_EPOCH = 600
__C.TRAIN.LR_DECAY_EPOCH = 600
__C.TRAIN.DISCRIMINATOR_LR = 2e-4
__C.TRAIN.GENERATOR_LR = 2e-4

__C.TRAIN.COEFF = edict()
__C.TRAIN.COEFF.KL = 2.0

# Modal options
__C.GAN = edict()
__C.GAN.CONDITION_DIM = 128
__C.GAN.DF_DIM = 64
__C.GAN.GF_DIM = 128
__C.GAN.R_NUM = 4

__C.TEXT = edict()
__C.TEXT.DIMENSION = 1024


def _merge_a_into_b(a, b):
    """[summary]
    用b去更新a, 更新之前要检查key和type是否匹配.
    Args:
        a ([type]): [description] - EasyDict类型,待更新字典
        b ([type]): [description] - EasyDict类型,用于更新的字典

    Raises:
        KeyError: [description] - key不存在
        ValueError: [description] - 类型不匹配.
    """
    if type(a) is not edict:
        # a must type of EasyDict
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError(f'{k} is not a valid config key')

        # the types must match, too
        if type(b[k]) is not type(v):
            raise ValueError(
                f'Type mismatch for config key: {k} (need {type(b[k])},but get {type(v)}) ')

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print(f"Error under config key: {k}")
        else:
            b[k] = v


def cfg_from_file(filename):
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(a=yaml_cfg, b=__C)
