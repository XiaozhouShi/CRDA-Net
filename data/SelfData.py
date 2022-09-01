import os
from PIL import Image
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
        self.trans_img = transforms.Compose([
            transforms.ColorJitter(0.2,0.2,0.2,0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5))
            ]) 
        self.trans_mask = transforms.Compose([
            transforms.ToTensor()
            ]) 
        self.trans = dt.Compose([
            dt.RandomCrop(args.crop_size),
            dt.RandomHorizontalFlip(),
            dt.RandomVerticalFlip()
            ]) 
        self.imgfiles = data 
        self.dataset = []
        for img_path in self.imgfiles:
            label_path = img_path.replace('img_file_256', 'mask_file_256')
            label_path = label_path.split('.')[0] + '_mask.jpg'
            if not os.path.exists(label_path):
                continue
            self.dataset.append([img_path, label_path])

    def __getitem__(self, index):
        img_path, label_path = self.dataset[index] 
        image = Image.open(img_path)
        mask = Image.open(label_path)
        if self.trans:
            image, mask = self.trans(image, mask)
        if self.trans_img:
            image = self.trans_img(image)
        if self.trans_mask:
            mask = self.trans_mask(mask)
        return image, mask 

    def __len__(self):
        return len(self.dataset)


class ValDataset(Dataset):
    def __init__(self, data, args): 
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
        self.dataset = []
        for img_path in self.imgfiles:
            label_path = img_path.replace('img_file_val', 'mask_file_val')
            label_path = label_path.split('.')[0] + '_mask.jpg'
            if not os.path.exists(label_path):
                continue
            self.dataset.append([img_path, label_path])

    def __getitem__(self, index):
        img_path, label_path = self.dataset[index] 
        image = Image.open(img_path)
        mask = Image.open(label_path)
        if self.trans:
            image, mask = self.trans(image, mask)
        if self.trans_img:
            image = self.trans_img(image)
        if self.trans_mask:
            mask = self.trans_mask(mask)
        return image, mask 

    def __len__(self):
        return len(self.dataset)

class TestDataset(Dataset):
    def __init__(self, args): 
        super().__init__()
        image_path = os.path.join(args.data_path, 'img_file_test')
        self.trans_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5))
            ]) 
        self.trans_mask = transforms.Compose([
            transforms.ToTensor()
            ]) 
        self.trans = None
        self.imgfiles = glob.glob(image_path + '/*.jpg')
        self.data = []
        for img_path in self.imgfiles:
            label_path = img_path.replace('img_file_test', 'mask_file_test')
            label_path = label_path.split('.')[0] + '_mask.jpg'
            if not os.path.exists(label_path):
                continue
            self.data.append([img_path, label_path])

    def __getitem__(self, index):
        img_path = self.data[index][0]
        label_path = self.data[index][1]
        name = os.path.basename(img_path)
        image = Image.open(img_path)
        mask = Image.open(label_path)
        if self.trans:
            image, mask = self.trans(image, mask)
        if self.trans_img:
            image = self.trans_img(image)
        if self.trans_mask:
            mask = self.trans_mask(mask)
        return image, mask, name 
    
    def __len__(self):
        return len(self.data)

def splitdata5folds(path, folds_for_val = None):
    img_list = glob.glob(os.path.join(path, '*.jpg'))
    img_list.sort()
    lenght = len(img_list)
    one_folds_lenght = lenght //5
    train_list, val_list = [], []
    if folds_for_val == None:
        print('train len', len(img_list))
        return img_list
