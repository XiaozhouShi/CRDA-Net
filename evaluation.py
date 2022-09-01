import argparse
import os
from PIL import Image
import numpy as np

import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms

#from data.GianaDataset import TestDataset
from data.miccai_dataset import TestDataset
import utils.utils as utils
from utils.patch import ImageSegMerge, ImagePadToSize, CropBlack 



def parse_args():
    parser = argparse.ArgumentParser(description='test Super Resolution Models')

    # data
    parser.add_argument('--data_path', default = '/mnt/bd/aurora-mtrc-data/datas/GIANA_challenge/segmentation/val', type = str, help = ' val image path')
    parser.add_argument('--model_upsample_num', default=5, type=int, help='val images size')
    parser.add_argument('--crop_black', default = True, type=bool)
    #test 
    parser.add_argument('--batch_size', default = 1, type = int, help = 'val batch size')
    parser.add_argument('--num_workers', default = 4, type = int, help = ' data loader num workers')
    parser.add_argument('--threshold', default = 0.5, type = int)
    parser.add_argument('--device', default = 'cuda:0', type = str, help = 'device')
    parser.add_argument('--result_save_path', default = './result', type = str)
    parser.add_argument('--model_path', default = './checkpoint/20210823_214049/model_best.pth.tar', type = str)
    parser.add_argument('--use_connect_domain', default = True, type = bool)
    # model
    parser.add_argument('--num_classes', default =2, type = int, help = 'number of classes')
    parser.add_argument('--model', default = None, type = str)
    

    args = parser.parse_args()
    return args


def main():
    args = parse_args() 
    print(args)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(args.result_save_path):
        os.mkdir(args.result_save_path)
    

    test_set = TestDataset(args.data_path,args)
    
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, \
        num_workers = args.num_workers)

    if args.model == 'segformer':
        from model.segformer.segformer import Segformer
        model = Segformer()
    elif args.model == 'deeplabv3plus':
        from model.deeplabv3Plus.deeplab import DeepLab
        model = DeepLab()
    elif args.model == 'unet':
        from model.unet.unet_model import UNet
        model = UNet()
    elif args.model == 'swin_unet':
        from model.swin_unet.vision_transformer import SwinUnet
        model = SwinUnet(args)
    elif args.model == 'unetpp':
        from model.unetpp.unetpp import NestedUNet
        model = NestedUNet()
    elif args.model == 'pranet':
        from model.pranet.PraNet_Res2Net import PraNet
        model = PraNet()
    elif args.model == 'ccbanet':
        from model.ccbanet.CCBANet import CCBANet
        model = CCBANet()
    elif args.model == 'resunetpp':
        from model.resunetpp.res_unet_plus import ResUnetPlusPlus
        model = ResUnetPlusPlus()
    else:
        raise('model error')
    print('# model parameters:', sum(param.numel() for param in model.parameters()))

    checkpoint = torch.load(args.model_path)['model']
    print('epoch:', torch.load(args.model_path)['epoch'])
    print('best eval:', torch.load(args.model_path)['best_eval'])
    model.load_state_dict(checkpoint)
    
    model.to(args.device)
    model.eval()
    with torch.no_grad():
        iou_list, dice_list = [], [] 
        metrics_all = utils.IOUMetric(args.num_classes, args.threshold)
        for i, (imgs, labels, name_list) in enumerate(test_loader):
            metrics = utils.IOUMetric(args.num_classes, args.threshold)
            imgs = imgs.to(args.device)
            labels = labels.to(args.device)
            #path_seg = ImageSegMerge(args.model_upsample_num, model, 1)
            #pred = path_seg(imgs)
            #if args.crop_black:
            #    crop_black = CropBlack(imgs, args)
            #    crop_img = crop_black.crop()
            #    img_forward = ImagePadToSize(args.model_upsample_num, model)
            #    pred = img_forward(crop_img)
            #    pred = crop_black.merge(pred)
            #else:
            #    img_forward = ImagePadToSize(args.model_upsample_num, model)
            #    pred = img_forward(imgs)
            pred = model(imgs)
            _, _, h, w = labels.size()
            pred = nn.functional.interpolate(pred, size = (h, w), mode = 'bilinear')
            if args.use_connect_domain:
                #pred = utils.max_connected_domain(pred, args.threshold)
                pred = utils.max_prob_domain(pred, args.threshold)
            else:
                pred = pred.detach().permute(0,2,3,1).squeeze().cpu().numpy()
            metrics.add_batch(pred, labels)
            metrics_all.add_batch(pred, labels)
            temp = args.model_path.split('/')[-2]
            name = name_list[0].split('.')[0] +  '_'+temp+'.jpg'
            pred *= 255
            result = Image.fromarray(pred.astype(np.uint8))
            save_path = os.path.join(args.result_save_path, name)
            result.save(save_path)
            print('save ', name)
            iou, dice = metrics.evaluate_front()
            iou_list.append(iou)
            dice_list.append(dice)
        avg_iou, avg_dice = sum(iou_list) / len(iou_list), sum(dice_list) / len(dice_list)
        print('mean iou: {}, mean dice: {}'.format(avg_iou, avg_dice))
        mean_all_iou, mean_all_dice = metrics_all.evaluate_front()
        #print('maen all iou: {}, mean all dice: {}'.format(mean_all_iou, mean_all_dice))
    print('Done.')


if __name__ == '__main__':
    main() 
