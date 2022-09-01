import torch
import torchvision.transforms.functional as F
import torch.nn as nn
import cv2
import numpy as np


class ImageSegMerge(nn.Module):

    def __init__(self, crop_size, mode, out_channel):
        super().__init__()
        self.crop_size = crop_size 
        self.mode = mode
        self.out_channel = out_channel

    def forward(self, img):
        self.batch, _, self.height, self.width = img.size() 
        self.crop_h, self.crop_w = self.crop_size[0], self.crop_size[1]
        leg_h, leg_w = self.height % self.crop_h, self.width % self.crop_w
        pad_h, pad_w = self.crop_h - leg_h, self.crop_w - leg_w
        padding = [0, 0, pad_w, pad_h]
        img_pad = F.pad(img, padding, fill = -1)

        height_img_pad, width_img_pad = img_pad.size()[2:]
        assert(height_img_pad % self.crop_h == 0 and width_img_pad % self.crop_w == 0), 'pad error'

        self.n_h, self.n_w = height_img_pad // self.crop_h, width_img_pad // self.crop_w
        result = torch.zeros(self.batch, self.out_channel, height_img_pad, width_img_pad)
        for row in range(self.n_h):
            for col in range(self.n_w):
                start_h, end_h, start_w, end_w = self.crop_h * row, self.crop_h * (row + 1), \
                     self.crop_w * col, self.crop_w * (col + 1)
                input_img = img_pad[...,start_h:end_h,start_w:end_w]
                out = self.mode(input_img)
                result[...,start_h:end_h,start_w:end_w] = out
        return F.crop(result, 0, 0, self.height, self.width)


class ImagePadToSize(nn.Module):

    def __init__(self, up_num, mode):
        super().__init__()

        self.basic_size = 2**up_num 
        self.mode = mode

    def forward(self, img):
        self.batch, _, self.height, self.width = img.size() 
        leg_h, leg_w = self.height % self.basic_size, self.width % self.basic_size
        pad_h, pad_w = self.basic_size - leg_h, self.basic_size - leg_w
        padding = [0, 0, pad_w, pad_h]
        img_pad = F.pad(img, padding, fill = -1)
        out = self.mode(img_pad)
        return F.crop(out, 0, 0, self.height, self.width)


class CropBlack:
    def __init__(self, img, args):
        self.img = img
        self.img_array = img.detach().permute(0, 2, 3, 1).squeeze().cpu().numpy()
        self.img_array = (self.img_array / 2 + 0.5) * 255
        self.img_array = self.img_array.astype(np.uint8)
        b, c, h, w = self.img.size()
        self.pre_img = torch.zeros(b, 1, h, w).to(args.device)

    def get_start_end(self, array, y1, y2):
        for i in range(len(array)):
            if array[i] > 125:
                y1 = min(y1, i)
                y2 = max(y2, i)
        return y1, y2

    def crop(self):
        threshold = np.mean(self.img_array[:10,:10]) + 2
        self.img_array = cv2.cvtColor(self.img_array,cv2.COLOR_RGB2GRAY)
        self.img_array = cv2.blur(self.img_array, (5, 5))
        self.img_array[self.img_array < threshold] = 0
        self.img_array[self.img_array >= threshold] = 255
        h, w = self.img_array.shape
        start, end = max(w//2 -20, 0), min(w//2 +20, w)
        
        self.y1, self.y2 = h, 0 
        for i in range(start, end):
            self.y1, self.y2 = self.get_start_end(self.img_array[:, i], self.y1, self.y2)

        start, end = max(h//2 -20, 0), min(h//2 +20, h)
        self.x1, self.x2 = w, 0
        for i in range(start, end):
            self.x1, self.x2 = self.get_start_end(self.img_array[i, :], self.x1, self.x2)

        self.crop_img = self.img[:, :,self.y1: self.y2, self.x1: self.x2]
        return self.crop_img
    
    def merge(self, crop_img):
        self.pre_img[:, :, self.y1:self.y2, self.x1:self.x2] = crop_img
        return self.pre_img

if __name__ == '__main__':
    imageseg = ImageSegMerge((512, 512), None, 3)
    import torchvision.transforms as tf
    from PIL import Image
    import numpy as np
    img = Image.open('/mnt/bd/aurora-mtrc-data/codes/GIANA_challenge/self_seg/tmp/2.bmp')
    out = '/mnt/bd/aurora-mtrc-data/codes/GIANA_challenge/self_seg/tmp/2_r.bmp'
    img = tf.ToTensor()(img).unsqueeze(0)
    print('img size', img.size())
    img = imageseg(img)
    img = img.detach().permute(0, 2, 3, 1).squeeze().cpu().numpy()
    img = img * 255
    img = Image.fromarray(img.astype(np.uint8))
    img.save(out)
