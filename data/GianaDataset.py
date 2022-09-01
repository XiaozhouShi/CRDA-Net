import os
from PIL import Image, ImageFilter
import multiprocessing as mp
import io
import random
import numpy as np
import shutil

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import data.data_utils.data_transforms as dt
import glob

import data.data_utils.data_transforms as dt



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


class TrainDataset(Dataset):
    def __init__(self, data, args):
        super(TrainDataset, self).__init__()
        self.trans_img = dt.RandomChoiceOrder([
            dt.RandomDownResizeUpResize((0.1, 1.0)),
            transforms.ColorJitter(0.5,0.5,0.5,0.2),
            #dt.AddPepperNoise(0.999, 0.2),
            dt.AddGaussianNoise(0, 0.05),
            transforms.GaussianBlur(kernel_size = 5, sigma=(0.05, 5.0)),
            transforms.RandomPosterize(bits=2, p = 0.1),
            transforms.RandomSolarize(threshold = 245, p = 0.5),
            transforms.RandomAdjustSharpness(0.1, p=0.5)
            ], p = 0.5)
        self.trans_img2tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5))
            ]) 
        self.trans_mask = transforms.Compose([
            transforms.ToTensor()
            ]) 
        self. trans_all = dt.Compose([
            #dt.RandomCrop(args.crop_size)
            dt.RandomResizedCrop(args.crop_size, scale = (0.2, 1))
            ])
        self.trans_all_random = dt.RandomChoiceOrderImgMask([
            dt.RandomPerspective(0.1, 0.1),
            dt.RandomAffine(60, translate = (0.05, 0.05), scale = (0.8, 1.0)),
            dt.RandomHorizontalFlip(),
            dt.RandomVerticalFlip()
            ], p = 0.5)
        self.imgfiles = data

    def __getitem__(self, index):
        img_path = self.imgfiles[index]
        label_path = img_path.replace('images', 'masks')
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path).convert('L')
        image, mask = self.trans_all(image, mask)
        image, mask = self.trans_all_random(image, mask)
        image = self.trans_img(image)
        mask = self.trans_mask(mask)
        image = self.trans_img2tensor(image)
        return image, mask 

    def __len__(self):
        return len(self.imgfiles)


class ValDataset(Dataset):
    def __init__(self, data, args = None): 
        super().__init__()
        self.trans_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5))
            ]) 
        self.trans_mask = transforms.Compose([
            transforms.ToTensor()
            ]) 
        self.trans = None
        self.imgfiles = data

    def __getitem__(self, index):
        img_path = self.imgfiles[index]
        label_path = img_path.replace('images', 'masks')
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path).convert('L')
        if self.trans:
            image, mask = self.trans(image, mask)
        if self.trans_img:
            image = self.trans_img(image)
        if self.trans_mask:
            mask = self.trans_mask(mask)
        return image, mask 

    def __len__(self):
        return len(self.imgfiles)

class TestDataset(Dataset):
    def __init__(self, args): 
        super().__init__()
        image_path = os.path.join(args.data_path, 'images')
        print('---', image_path)
        self.trans_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5))
            ]) 
        self.trans_mask = transforms.Compose([
            transforms.ToTensor()
            ]) 
        self.trans = None
        self.imgfiles = glob.glob(image_path + '/*.png')

    def __getitem__(self, index):
        img_path = self.imgfiles[index]
        label_path = img_path.replace('images', 'masks')
        image = Image.open(img_path).convert('RGB')
        #image = image.filter(ImageFilter.GaussianBlur(3))
        mask = Image.open(label_path).convert('L')
        name = os.path.basename(img_path)
        if self.trans:
            image, mask = self.trans(image, mask)
        if self.trans_img:
            image = self.trans_img(image)
        if self.trans_mask:
            mask = self.trans_mask(mask)
        return image, mask, name 

    def __len__(self):
        return len(self.imgfiles)

def splitdata5folds(path, folds_for_val = None):
    print('**************', path)
    img_list = glob.glob(os.path.join(path, '*/images/*.png'))
    img_list += glob.glob(os.path.join(path, 'images/*.png'))
    img_list.sort()
    lenght = len(img_list)
    one_folds_lenght = lenght // 5
    train_list, val_list = [], []
    if folds_for_val == None:
        return img_list
    for i in range(5):
        if i == folds_for_val:
            val_list += img_list[i*one_folds_lenght:(i+1)*one_folds_lenght]
        else:
            train_list += img_list[i*one_folds_lenght:(i+1)*one_folds_lenght]
    return train_list, val_list

class DatasetTemp(Dataset):
    def __init__(self, path):
        super(DatasetTemp, self).__init__()
        self.trans_img = dt.RandomChoiceOrder([
            transforms.ColorJitter(0.2,0.2,0.2,0.2),
            dt.AddPepperNoise(0.95),
            dt.AddGaussianNoise(0, 0.005),
            transforms.GaussianBlur(kernel_size = 5, sigma=(0.1, 2.0)),
            transforms.RandomPosterize(bits=2),
            transforms.RandomSolarize(threshold = 245, p = 0.5),
            transforms.RandomAdjustSharpness(0.8, p=0.1)
            ], p = 0.5)
        self.trans_img2tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5))
            ]) 
        self.trans_mask = transforms.Compose([
            transforms.ToTensor()
            ]) 
        self. trans_all = dt.Compose([
            #dt.RandomCrop(args.crop_size)
            dt.RandomResizedCrop((512,512), scale = (0.1, 1))
            ])
        self.trans_all_random = dt.RandomChoiceOrderImgMask([
            dt.RandomPerspective(0.5, 0.5),
            dt.RandomAffine(60, translate = (0.4, 0.4), scale = (0.5, 1.0)),
            dt.RandomHorizontalFlip(),
            dt.RandomVerticalFlip()
            ], p = 0.5)
        self.imgfiles = glob.glob(path + '/*.bmp')

    def __getitem__(self, index):
        img_path = self.imgfiles[index]
        label_path = img_path.replace('images', 'masks')
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path).convert('L')
        image, mask = self.trans_all(image, mask)
        image, mask = self.trans_all_random(image, mask)
        image = self.trans_img(image)
        mask = self.trans_mask(mask)
        image = self.trans_img2tensor(image)
        return image, mask, os.path.basename(img_path)

    def __len__(self):
        return len(self.imgfiles)
