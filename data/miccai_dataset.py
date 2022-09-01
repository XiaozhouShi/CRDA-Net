""" Quick n Simple Image Folder, Tarfile based DataSet

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch.utils.data as data
import os
import torch
import torch.nn as nn
import logging

from PIL import Image
import glob
import numpy as np

_logger = logging.getLogger(__name__)
import torchvision.transforms as transforms
from .data_utils.data_transforms import * 
import copy



class TrainDataset(data.Dataset):
    def __init__(
            self,
            root,
            args,
            transform=None,
            img_transform=None,
            target_transform=None,
    ):
        self.root = root
        self.args = args
        self.sup_image_list = glob.glob(self.root+'/supervised/*/images/*.png')
        self.sup_image_list += glob.glob(self.root+'/supervised/*/images/*.jpg')
        self.unsup_image_list = glob.glob(self.root + '/unsupervised/*/images/*.png')
        self.transform = transform
        self.img_transform = img_transform
        self.target_transform = target_transform

        self.trans_img = RandomChoiceOrder([
            RandomDownResizeUpResize((0.5, 1.0)),
            transforms.ColorJitter(0.5,0.5,0.5,0.2),
            #dt.AddPepperNoise(0.999, 0.2),
            AddGaussianNoise(0, 0.05),
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
        self. trans_all = Compose([
            RandomCrop(args.crop_size)
            #RandomResizedCrop(args.crop_size, scale = (0.2, 1))
            ])
        self.trans_all_random = RandomChoiceOrderImgMask([
            RandomPerspective(0.8, 0.1),
            RandomAffine(60, translate = (0.05, 0.05), scale = (0.8, 1.0)),
            RandomHorizontalFlip(),
            RandomVerticalFlip()
            ], p = 0.5)
        
        self.trans_unsup = transforms.Compose([
            transforms.RandomCrop((args.crop_size[0] * 2, args.crop_size[1] * 2))
        ])
        self.trans_unsup_input = RandomChoiceOrder([
            RandomDownResizeUpResize((0.5, 1.0)),
            transforms.GaussianBlur(kernel_size = 7, sigma=(0.05, 5.0)),
            AddGaussianNoise(0, 0.01,0.2)
        ], p=0.9)

    def __getitem__(self, index):
        sup_index = index % len(self.sup_image_list)
        unsup_index = index % len(self.unsup_image_list)
        sup_img_path = self.sup_image_list[sup_index]
        sup_mask_path = sup_img_path.replace('/images/', '/masks/')
        unsup_img_path = self.unsup_image_list[unsup_index]

        sup_image =  Image.open(sup_img_path).convert('RGB')
        target = Image.open(sup_mask_path).convert('L')
        unsup_image =  Image.open(unsup_img_path).convert('RGB')

        image, mask = self.trans_all(sup_image, target)
        image, mask = self.trans_all_random(image, mask)
        image = self.trans_img(image)
        target = self.trans_mask(mask)
        img = self.trans_img2tensor(image)

        unsup_image = self.trans_unsup(unsup_image)
        unsup_target = copy.deepcopy(unsup_image)
        unsup_image, unsup_target = self.trans_all_random(unsup_image, unsup_target)
        unsup_image = self.trans_unsup_input(unsup_image)
        unsup_image = self.trans_img2tensor(unsup_image)
        unsup_target = self.trans_img2tensor(unsup_target)
        return img, target, unsup_image, unsup_target

    def __len__(self):
        return len(self.sup_image_list)


class ValDataset(data.Dataset):
    def __init__(
            self,
            root,
            args,
            transform=None,
            img_transform=None,
            target_transform=None,
    ):
        self.root = root
        image_list = glob.glob(self.root+'/*/images/*.png')
        image_list += glob.glob(self.root+'/*/images/*.jpg')
        self.transform = transform
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.dataset = []
        for item in image_list:
            mask = item.replace('images', 'masks')
            if os.path.exists(mask):
                self.dataset.append([item, mask])
        self._consecutive_errors = 0

        self.trans_img2tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5))
            ]) 
        self.trans_mask = transforms.Compose([
            transforms.ToTensor()
            ]) 

    def __getitem__(self, index):
        img, target = self.dataset[index]
        try:
            img =  Image.open(img).convert('RGB')
            target = Image.open(target).convert('L')
        except Exception as e:
            raise e
        target = self.trans_mask(target)
        img = self.trans_img2tensor(img)
        return img, target

    def __len__(self):
        return len(self.dataset)


class TestDataset(data.Dataset):
    def __init__(
            self,
            root,
            args,
            transform=None,
            img_transform=None,
            target_transform=None,
    ):
        self.root = root
        image_list = glob.glob(self.root+'/images/*.png')
        image_list += glob.glob(self.root+'/images/*.jpg')
        self.transform = transform
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.dataset = []
        for item in image_list:
            mask = item.replace('images', 'masks')
            if os.path.exists(mask):
                self.dataset.append([item, mask])
        self._consecutive_errors = 0
        print('------', len(self.dataset))

        self.trans_img2tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5))
            ]) 
        self.trans_mask = transforms.Compose([
            transforms.ToTensor()
            ]) 

    def __getitem__(self, index):
        img, target = self.dataset[index]
        name = os.path.basename(img)
        try:
            img =  Image.open(img).convert('RGB')
            target = Image.open(target).convert('L')
        except Exception as e:
            raise e
        target = self.trans_mask(target)
        img = self.trans_img2tensor(img)
        return img, target, name

    def __len__(self):
        return len(self.dataset)
