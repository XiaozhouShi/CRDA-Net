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

from dataloader import KVReader


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def split_dataset(path, shuffle=True, ratio=0.9):
    dataset = []
    for img_root, _, img_names in os.walk(path):
        tmp = img_root.split('/')
        label = None
        for item in tmp[::-1]:
            if item == '0' or item == '1' or item == '2':
                label = item
        if label is None:
            continue
        label = int(label)
        data_tmp = [[os.path.join(img_root, name), label] for name in img_names if is_image_file(name)]
        dataset += data_tmp
    n_total = len(dataset)
    print('data set lenght:', n_total)
    offset = int(n_total*ratio)
    if n_total ==0 or offset < 1:
        return dataset, []
    if shuffle:
        random.shuffle(dataset)
    train_list = dataset[:offset]
    val_list = dataset[offset:]
    return train_list, val_list
    

class TrainDataset(Dataset):
    def __init__(self, dataset, transforms):
        super(TrainDataset, self).__init__()
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.dataset[index][0]
        label = self.dataset[index][1]
        image = self.transforms(Image.open(img_path))
        if label == 2:
            label = 0
        label = torch.from_numpy(np.array(label)).float()
        return image, label

    def __len__(self):
        return len(self.dataset)


class ValDataset(Dataset):
    def __init__(self, dataset, transforms):
        super(ValDataset, self).__init__()
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.dataset[index][0]
        label = self.dataset[index][1]
        image = self.transforms(Image.open(img_path))
        if label == 2:
            label = 0
        label = torch.from_numpy(np.array(label)).float()
        return image, label

    def __len__(self):
        return len(self.dataset)

class EvalDataset(Dataset):
    def __init__(self, dataset, transforms):
        super(EvalDataset, self).__init__()
        self.dataset = dataset
        self.transforms = transforms
    
    def __getitem__(self, index):
        img_path = self.dataset[index][0]
        label = self.dataset[index][1]
        image = self.transforms(Image.open(img_path))
        if label == 2:
            label = 0
        label = torch.from_numpy(np.array(label)).float()
        return image, label, img_path
    
    def __len__(self):
        return len(self.dataset)



class TestDataset(Dataset):
    def __init__(self, dataset, transforms):
        super(TestDataset, self).__init__()
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.dataset[index][0]
        image = self.transforms(Image.open(img_path))
        return image, img_path

    def __len__(self):
        return len(self.dataset)


def get_keys(args):
    return KVReader(*args).list_keys()


# 数据集定义
class KVDataset(torch.utils.data.Dataset):
    def __init__(self, path, num_readers):
        self.path = path
        self.num_readers = num_readers
        with mp.Pool(1) as p:
            self.keys = p.map(get_keys, [(path, num_readers)])[0]

        classes = sorted(list(set([key.split('/')[1] for key in self.keys])))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        
        # 预处理定义
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        # 返回一个 batch 的数据
        keys = [self.keys[i] for i in index]
        labels = [self.class_to_idx[key.split('/')[1]] for key in keys]
        images = self.reader.read_many(keys)
        images = [
            self.transform(Image.open(io.BytesIO(image)).convert('RGB'))
            for image in images
        ]
        return torch.stack(images), torch.Tensor(labels)


# 组一个 batch
def chunk(iterable, chunk_size):
    ret = []
    for record in iterable:
        ret.append(record)
        if len(ret) == chunk_size:
            yield ret
            ret = []
    if ret:
        yield ret

# 自定义分布式采样
class KVSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, dataset, batch_size, num_replicas, rank, shuffle=True):
        super(KVSampler, self).__init__(dataset,
                                        num_replicas=num_replicas,
                                        rank=rank,
                                        shuffle=shuffle)
        self.batch_size = batch_size
        self.iter = super(KVSampler, self).__iter__

    def __iter__(self):
        iterable = self.iter()
        return chunk(iterable, self.batch_size)


if __name__ == '__main__':
    path = '/home/tiger/dataset/class_train'
    out_path = '/home/tiger/dataset/class_test'
    random.seed(10)
    _, test = split_dataset(path, shuffle = True, ratio = 0.95)
    for item in test:
        img_path = item[0]
        label = str(item[1])
        img_2path = os.path.join(out_path, label)
        if not os.path.exists(img_2path):
            os.makedirs(img_2path)
        img_save_path = os.path.join(img_2path, os.path.basename(img_path))
        shutil.copy(img_path, img_save_path)
        os.remove(img_path)
