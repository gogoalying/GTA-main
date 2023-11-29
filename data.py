import os, sys, shutil, csv
import random as rd
from PIL import Image
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss
import random
import cv2
import imgaug as ia
import imgaug.augmenters as iaa

RootDir = {'C':'/ssd2/baozenghao/data/Age/CACD/CACD2000_arccropped/',
           'E':'/ssd1/data/face/age_data/data/MegaAge/megaage_asian_arccropped/',
           'I':'/ssd1/baozenghao/data/IMDB-WIKI/',
           'LO':'/ssd2/baozenghao/data/Age/CLAP16/CLAP16_arccrop/',
           'L':'/ssd2/baozenghao/data/Age/CLAP16/CLAP16_arccrop/train/',
           'LT':'/ssd2/baozenghao/data/Age/CLAP16/train/',
            'MS': '/ssd2/data/face/MS_Celeb_1M/imgs',
            'G': '/ssd2/data/face/Glint360k/imgs',
           'M':'/ssd1/data/face/age_data/data/Morph/Album2_arccropped/',
           'U': '/ssd1/data/face/age_data/data/UTKFace/UTKFACE_arccropped/'}
#RootDir字典分别代表驱动器路径
AllTrain = {'C': '/ssd2/baozenghao/data/Age/CACD/txt/big_noise_images_shuffle_renamed.txt',
            'E': '/ssd1/data/face/age_data/data/MegaAge/txt/MegaAge_Asian_train.txt',
            'I': '/ssd1/baozenghao/data/IMDB-WIKI/txt/imdb_wiki_CLEAN_train.txt',
            'L': '/ssd2/baozenghao/data/Age/CLAP16/txt/train.txt',
            'LT': '/ssd2/baozenghao/data/Age/CLAP16/txt/train.txt',
            'MS': '/ssd2/data/face/MS_Celeb_1M/txt/list.txt',
            'G': '/ssd2/data/face/Glint360k/txt/list.txt',
            'M': '/ssd1/data/face/age_data/data/Morph/txt/RANDOM_80_20/morph_random_80_20_train.txt',
            'U': '/ssd1/data/face/age_data/data/UTKFace/txt/utkface_train.txt'}
#AllTrain字典包含训练文件的路径
AllTest = {'C': '/ssd2/baozenghao/data/Age/CACD/txt/small_noise_images_rank345_renamed.txt',
            'E': '/ssd1/data/face/age_data/data/MegaAge/txt/MegaAge_Asian_test.txt',
            'I': '/ssd1/baozenghao/data/IMDB-WIKI/txt/imdb_wiki_CLEAN_test.txt',
            'LO': '/ssd2/baozenghao/data/Age/CLAP16/txt/chalearn16_AG_test.txt',
            'LT': '/ssd2/baozenghao/data/Age/CLAP16/txt/chalearn16_AG_test.txt',
            'M': '/ssd1/data/face/age_data/data/Morph/txt/RANDOM_80_20/morph_random_80_20_test.txt',
            'U': '/ssd1/data/face/age_data/data/UTKFace/txt/utkface_test.txt'}
#AllTest包含测试文件路径

rootdir = '/ssd2/baozenghao/data/Age/MIVIA/caip_arccropped'#main中使用的，accropped是裁剪后数据集
trainlist = '/ssd2/baozenghao/data/Age/MIVIA/MIVIA_train.csv'
# trainlist = '/ssd2/baozenghao/data/Age/MIVIA/training_caip_contest.csv'
testlist = '/ssd2/baozenghao/data/Age/MIVIA/MIVIA_test.csv'
# testlist = '/bzh/test.csv'

#cutout transform
class CutoutDefault(object):
    """
    Apply cutout transformation.
    Code taken from: https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def loadcsv(data_dir, file):
    imgs = list()
    with open(file, mode='r') as csv_file:
        gt = csv.reader(csv_file, delimiter=',')
        for row in gt: # 逐行读取csv文件
            img_name, age = row[0], row[1]# 赋值name和age
            img_path = os.path.join(data_dir, img_name)
            # data_dir= "/path/to/data"， img_name "image1.jpg"，经过 os.path.join(data_dir, img_name) 操作后，img_path 的值将是 "/path/to/data/image1.jpg"
            age = int(round(float(age)))
            imgs.append((img_path, age))
    return imgs

# def loadclass(data_dir, file):


def loadrank(data_dir, file, rank):
    imgs = list()
    with open(file, mode='r') as csv_file:
        gt = csv.reader(csv_file, delimiter=',')
        for row in gt:
            img_name, age = row[0], row[1]
            img_path = os.path.join(data_dir, img_name)
            age = int(round(float(age)))
            if age > 10 * rank and age <= 10 * (rank + 1) and rank != 7:
                imgs.append((img_path, age))
            if rank == 7 and age > 10 * rank:
                imgs.append((img_path, age))
    return imgs



def loadage(data_dir, file, shuffle=True):
    imgs = list()
    with open(file) as f:
        for eachline in f:
            contents = eachline.strip().split(' ')
            img_name, age = contents[0], contents[1]
            img_path = os.path.join(data_dir, img_name)
            age = int(round(float(age)))
            # if age > 15 and age < 61:#16--60
            imgs.append((img_path, age))
    if shuffle:
        random.shuffle(imgs)
    return imgs

def loadface(data_dir, image_list_file):
    imgs = list()
    with open(image_list_file) as f:
        for eachline in f:
            contents = eachline.strip().split('/')
            label, img_name = contents[0], contents[1]
            img_path = os.path.join(data_dir, label, img_name)
            label = int(label[3:])
            imgs.append((img_path, label))
    return imgs

def normal_sampling(mean, label_k, std=1):
    return math.exp(-(label_k-mean)**2/(2*std**2))/(math.sqrt(2*math.pi)*std)

class Balance(data.Dataset):#main
    def __init__(self, transform):
        imgs = loadcsv(rootdir, trainlist) #调用loadcsv函数
        self.transform = transform
        self.class_dict = self._get_class_dict()#子调用get_class_dict获取类别字典
        self.imgs = imgs
    def _get_class_dict(self):
        class_dict = dict()#创建空字典
        for i in range(1,82):
            class_dict[str(i)] = []
            # 初始化每个类别对应的索引列表为空列表class_dict={"1":[]}由于字典是不可变的，所以要转化为str
        with open(trainlist, mode='r') as csv_file:
            gt = csv.reader(csv_file, delimiter=',')
            for i, row in enumerate(gt):#遍历csv文件中的数据行
                age = int(round(float(row[1])))# 提取年龄信息转化为整数
                for j in range(1, 82):
                    if age == j:
                        class_dict[str(j)].append(i)# 将索引添加到对硬类别的索引列表中
        return class_dict

    def __getitem__(self, index):
        sample_class = random.randint(1, 81)
        sample_indexes = self.class_dict[str(sample_class)]
        index = random.choice(sample_indexes)
        
        img_path, age = self.imgs[index]# 根据索引获取图像路径和年龄
        age = int(round(float(age)))# 年龄转化整数
        img_path = os.path.join(rootdir, img_path)# 构建完整的路径

        img = Image.open(img_path).convert("RGB")

        # 将年龄生成标签分布
        # 目标年龄和标准差，然后根据正态分布生成一个长度为101的标签分布列表。这里使用了一个循环，从0到100迭代，将每个年龄对应的标签设置为正态分布的概率值。
        label = [normal_sampling(int(age), i) for i in range(101)]
        label = [i if i > 1e-15 else 1e-15 for i in label]
        label = torch.Tensor(label)

        seq_rand = iaa.Sequential([iaa.RandAugment(n=2, m=10)])

        cv_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        cv_img = seq_rand.augment_image(image=cv_img)
        img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

        img = self.transform(img)
        return img, age, label

    def __len__(self):
        return len(self.imgs)


class TrainM(data.Dataset):#main数据增强
    def __init__(self, transform):
        imgs = loadcsv(rootdir, trainlist) 
        random.shuffle(imgs)
        self.imgs = imgs
        self.transform = transform
    def __getitem__(self, item):#使用train_dataset和dataloader调用每次dataloader会使用一次
        img_path, age = self.imgs[item]
        img = Image.open(img_path).convert("RGB")#打开图像转化成RGB格式

        label = [normal_sampling(int(age), i) for i in range(101)]#标签分布
        label = [i if i > 1e-15 else 1e-15 for i in label]
        label = torch.Tensor(label)#转化tensor方便后续处理

        seq_rand = iaa.Sequential([iaa.RandAugment(n=2, m=10)])#n为数量m为强度

        cv_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        cv_img = seq_rand.augment_image(image=cv_img)
        img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))#数据增强

        # self.transform.transforms.append(CutoutDefault(20))

        img = self.transform(img)
        return img, age, label
    def __len__(self):
        return len(self.imgs)

class TestM(data.Dataset):#main
    def __init__(self, transform):
        imgs = loadcsv(rootdir, testlist) 
        self.imgs = imgs
        self.transform = transform
    def __getitem__(self, item):
        img_path, age = self.imgs[item]
        img = Image.open(img_path).convert("RGB")
        # img2 = img.transpose(Image.FLIP_LEFT_RIGHT)向右旋转
        img = self.transform(img)
        # img2 = self.transform(img2)
        return img, age
    def __len__(self):
        return len(self.imgs)

class Face(data.Dataset):
    def __init__(self, dataset, InTrain, transform):
        if InTrain:
            imgs = loadface(RootDir[dataset], AllTrain[dataset]) #如果在训练则导入loadface函数
            UsedImages = imgs
            random.shuffle(UsedImages)
        else:
            imgs = loadface(RootDir[dataset], AllTest[dataset])#不在训练则在Test导入loadface函数
            UsedImages = imgs
        self.imgs = UsedImages
        self.transform = transform
        self.InTrain = InTrain#实例化赋予
    def __getitem__(self, item):
        img_path, label = self.imgs[item]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label
    def __len__(self):
        return len(self.imgs)

class AAR(data.Dataset):
    def __init__(self, transform, rank):
        imgs = loadrank(rootdir, testlist, rank) 
        self.imgs = imgs
        self.transform = transform
    def __getitem__(self, item):
        img_path, age = self.imgs[item]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, age
    def __len__(self):
        return len(self.imgs)

class Train(data.Dataset):
    def __init__(self, dataset, transform):
        imgs = loadage(RootDir[dataset], AllTrain[dataset]) 
        UsedImages = imgs
        random.shuffle(UsedImages)
        self.imgs = UsedImages
        self.transform = transform
    def __getitem__(self, item):
        img_path, age = self.imgs[item]
        img = Image.open(img_path).convert("RGB")

        label = [normal_sampling(int(age), i) for i in range(101)]
        label = [i if i > 1e-15 else 1e-15 for i in label]
        label = torch.Tensor(label)

        # seq_rand = iaa.Sequential([iaa.RandAugment(n=2, m=9)])

        # cv_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        # cv_img = seq_rand.augment_image(image=cv_img)
        # img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

        img = self.transform(img)
        return img, age, label
    def __len__(self):
        return len(self.imgs)

class Test(data.Dataset):
    def __init__(self, dataset, transform):
        imgs = loadage(RootDir[dataset], AllTest[dataset])
        UsedImages = imgs
        self.imgs = UsedImages
        self.transform = transform
    def __getitem__(self, item):
        img_path, age = self.imgs[item]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, age
    def __len__(self):
        return len(self.imgs)
