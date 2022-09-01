import argparse
import os
from datetime import datetime
import time
import random
import warnings
import numpy as np

import torch
import pandas as pd
import torch.optim as optim
import torch.utils.data
from torch.utils.data import sampler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from tensorboardX import SummaryWriter 
import torchvision.utils as vutils
from torch import nn

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from data.data_utils.data_transforms import Mixup
from data.miccai_dataset import TrainDataset, ValDataset
import utils.utils as utils
from utils.loss import focal_loss, dice_loss, PerceptionLoss
from utils.patch import ImagePadToSize, ImageSegMerge, CropBlack



def parse_args():
    parser = argparse.ArgumentParser(description='Train Models')

    # data
    parser.add_argument('--data_path', default ='/mnt/bd/aurora-mtrc-data/data/giana/segmentation/train_crop', type = str, help = ' training image path')
    parser.add_argument('--val_path', default ='/mnt/bd/aurora-mtrc-data/data/giana/segmentation/test_crop/hdtest_crop', type = str, help = ' training image path')
    parser.add_argument('--save_path', default = './checkpoint', type = str, help = 'model save path')
    parser.add_argument('--crop_size', default=(256,256), type=int, help='training images size')
    parser.add_argument('--folds', default = None, type=int, help = 'validation for folds')
    # train
    parser.add_argument('--num_epochs', default=1000, type=int, help='train epoch number')
    parser.add_argument('--batch_size', default = 2, type = int, help = 'training batch size')
    parser.add_argument('--lr', default = 0.0001, type = float, help='learning rate')
    parser.add_argument('--print_freq', default = 10, type = int, help = 'the frequency with which messages are printed')
    parser.add_argument('--tensorboard_freq', default = 200, type = int, help = 'the frequency with tensorboard')
    parser.add_argument('--save_freq', default = 10, type = int, help = 'save model according to epoch')
    parser.add_argument('--best_eval', default = 0, type = float, help = 'best auc of model')
    parser.add_argument('--num_workers', default = 4, type = int, help = ' data loader num workers')
    parser.add_argument('--decay_gamma', default = 1, type = int)
    parser.add_argument('--threshold', default = 0.5, type = int)
    parser.add_argument('--benchmark', default = True, type = bool)
    parser.add_argument('--cutmix_alpha', default = 0.5, type = float)
    # distributed
    parser.add_argument('--gpus', default = 2, type = int, help = 'number of gpus of per node')
    parser.add_argument('--local_rank', type = int, help = 'rank of current process')
    parser.add_argument('--init_method', default = 'env://')
    parser.add_argument('--device', type = str)
    # model
    parser.add_argument('--model', type =str, default = 'segformer')
    parser.add_argument('--num_classes', default =2, type = int, help = 'number of classes')
    parser.add_argument('--upsample_num', default=5, type = int)
    # other
    parser.add_argument('--seed', default=10, type = int, help = 'seed for initializing training. ')
    parser.add_argument('--reuse', default = None) 
    parser.add_argument('--eval_all', default = True, type = bool)
    parser.add_argument('--crop_black', default = True)
    parser.add_argument('--use_connect_domain', default = True)
    parser.add_argument('--test_frequent', default = 5, type = int)
    

    args = parser.parse_args()
    return args


def main():
    args = parse_args() 
    if not isinstance(args.crop_size, tuple):
        args.crop_size = (args.crop_size, args.crop_size)
    print(args)

    dist.init_process_group(backend='nccl', init_method=args.init_method)
    args.local_rank = dist.get_rank()
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device('cuda', args.local_rank)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                        'This will turn on the CUDNN deterministic setting, '
                        'which can slow down your training considerably! '
                        'You may see unexpected behavior when restarting '
                        'from checkpoints.')
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if args.folds is None:
        args.save_path = os.path.join(args.save_path, args.model, timestamp)
    else:
        args.save_path = os.path.join(args.save_path, args.model, str(args.folds) + timestamp)


    if args.local_rank == 0:
        print('print device:', args.local_rank)
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        writer = SummaryWriter(args.save_path)

    train_set = TrainDataset(args.data_path, args)
    val_set = ValDataset(args.val_path, args)
    print('train set lenght:', train_set.__len__(), 'val set lenght:', val_set.__len__())
    
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, \
        sampler = DistributedSampler(train_set), num_workers = args.num_workers, drop_last = True)
    val_loader = DataLoader(dataset=val_set, batch_size=1, \
        sampler = DistributedSampler(val_set),num_workers = args.num_workers)

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
        model = SwinUnet(args.crop_size[0])
    elif args.model == 'segformer_2':
        from model.segformer_2.segformer_2 import Segformer_2
        model = Segformer_2()
    else:
        raise('model error')
    print('# model parameters:', sum(param.numel() for param in model.parameters()))
    from model.unet.unet_res_model import UNet_res
    SRNet = UNet_res()
    from model.generator.model import Discriminator
    DomainNet = Discriminator()
    if args.reuse is not None:
        dic = torch.load(args.reuse)
        model.load_state_dict(dic['model'])
        SRNet.load_state_dict(dic['SRNet'])
        DomainNet.load_state_dict(dic['DomainNet'])

    model.to(args.device)
    SRNet.to(args.device)
    DomainNet.to(args.device)
    if torch.cuda.device_count() > 1:
        print('use', torch.cuda.device_count(), 'gpus')
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids = [args.local_rank], output_device= args.local_rank)

        SRNet = torch.nn.parallel.DistributedDataParallel(SRNet, device_ids = [args.local_rank], output_device= args.local_rank)
        DomainNet = torch.nn.parallel.DistributedDataParallel(DomainNet, device_ids = [args.local_rank], output_device= args.local_rank)


    optimizer_model = optim.AdamW(model.parameters(), lr = args.lr)
    optimizer_SRNet = optim.AdamW(SRNet.parameters(), lr = args.lr)
    optimizer_DomainNet = optim.AdamW(DomainNet.parameters(), lr = args.lr)

    expLR_model = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_model, T_max = 80, eta_min = 1e-5)
    expLR_SRNet = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_SRNet, T_max = 80, eta_min = 1e-5)
    expLR_DomainNet = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_DomainNet, T_max = 80, eta_min = 1e-5)

    start = datetime.now()
    steps = train_loader.__len__()
    cutmix = Mixup(mixup_alpha=0., cutmix_alpha=args.cutmix_alpha, cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                        mode='pair', correct_lam=True, label_smoothing=0.1, num_classes=args.num_classes)
    step = 0
    L1 = nn.L1Loss()
    percep_loss = PerceptionLoss(args.device)
    loss_func = nn.BCELoss()
    
    amp_autocast = torch.cuda.amp.autocast
    loss_scaler_domain_sr = utils.NativeScaler()
    loss_scaler_domain_D = utils.NativeScaler()
    loss_scaler_re = utils.NativeScaler()
    loss_scaler_seg = utils.NativeScaler()
    for epoch in range(args.num_epochs):
        model.train()
        SRNet.train()
        DomainNet.train()
        metrics = utils.IOUMetric(args.num_classes, args.threshold)
        for i, (imgs, labels, unsup_image, unsup_target) in enumerate(train_loader):
            if args.cutmix_alpha is not None:
                imgs, labels = cutmix(imgs, labels)
            imgs_l = imgs.to(args.device)
            labels_l = labels.to(args.device)
            unsup_image = unsup_image.to(args.device)
            unsup_target = unsup_target.to(args.device)

            imgs_h_input = nn.functional.interpolate(imgs_l, scale_factor=2.0, mode = 'bilinear')
            labels_h = nn.functional.interpolate(labels_l, scale_factor=2.0,mode='nearest')

            with amp_autocast():
                pred_unsup = SRNet(unsup_image)
                unsup_l1 = 100 * L1(pred_unsup, unsup_target)
                percep = 2 * percep_loss(pred_unsup, unsup_target)
                G_loss = unsup_l1 + percep
            loss_scaler_re(G_loss, optimizer_SRNet)

            # train domain adaption
            with amp_autocast():
                imgs_h = SRNet(imgs_h_input)
                d_fake = DomainNet(imgs_h)
                G_loss = 0.1 * nn.functional.softplus(-d_fake.mean())
            loss_scaler_domain_sr(G_loss, optimizer_SRNet)

            with amp_autocast():
                d_real = DomainNet(pred_unsup.detach())
                d_fake = DomainNet(imgs_h.detach())
                gp = utils.calc_gradient_penalty(DomainNet, pred_unsup.detach(), imgs_h.detach(), args.device)
                wasserstein = nn.functional.softplus(d_fake.mean()) + nn.functional.softplus(-d_real.mean())
                D_loss = wasserstein+gp
            loss_scaler_domain_D(D_loss, optimizer_DomainNet)

            # train segmentaion
            #with amp_autocast():
            #    pred = model(imgs)
            #    loss = focal_loss(pred, labels, alpha = 0.9,gamma = 2, reduction = 'mean') + dice_loss(pred, labels)
            #    # loss = loss_func(pred, labels)
            #    # loss = focal_loss(pred, labels, alpha = 0.7, reduction = 'mean') + loss_func(pred, labels)
            #    # loss = dice_loss(pred, labels) + loss_func(pred, labels)
            #    # loss = L1(pred, labels) + dice_loss(pred, labels)
            #loss_scaler(loss, optimizer)

            with amp_autocast():
                pred_l = model(imgs_l)
                pred_h = model(imgs_h.detach())

                loss_l = focal_loss(pred_l, labels_l, alpha = 0.9,gamma = 2, reduction = 'mean') + dice_loss(pred_l, labels_l)
                loss_h = focal_loss(pred_h, labels_h, alpha = 0.9,gamma = 2, reduction = 'mean') + dice_loss(pred_h, labels_h)
                loss = loss_l + loss_h
            loss_scaler_seg(loss, optimizer_model)

            preds = pred_l.detach().permute(0,2,3,1).squeeze().cpu().numpy()
            metrics.add_batch(preds, labels)
            acc, acc_cls, iu, mean_iu, fwavacc = metrics.evaluate()
            acc = utils.dist_mean_param(acc, args.device)
            iu = [utils.dist_mean_param(x, args.device) for x in iu]
            mean_iu = utils.dist_mean_param(mean_iu, args.device)
            if step % args.print_freq == 0 and args.local_rank == 0:
                print("epoch: {}|{}, unsup l1: {:.5f}, G: {:.5f}, seg: {:.5f}, lowseg: {:.5f}, acc: {:.2f}, mean_iu: {}, time: {}".format(epoch, args.num_epochs, unsup_l1.item(), G_loss.item(), loss.item(), loss_l.item(), acc, mean_iu, datetime.now() - start))
                writer.add_scalar('train/loss', loss.item(), step)
                writer.add_scalar('train/mean_iu', mean_iu, step)
                start = datetime.now()
            
            if step % args.tensorboard_freq == 0 and args.local_rank == 0:
                imgs = vutils.make_grid(imgs_l, normalize=True, scale_each=True)
                writer.add_image('trian/inputs', imgs, step)
                gts = vutils.make_grid(labels_l, normalize=True, scale_each=True)
                writer.add_image('train/gts', gts, step)
                preds= vutils.make_grid(pred_l, normalize=True, scale_each=True)
                writer.add_image('train/preds', preds, step)
                imgs = vutils.make_grid(imgs_h, normalize=True, scale_each=True)
                writer.add_image('train/fake', imgs, step)
            step += 1
        expLR_model.step()
        expLR_SRNet.step()
        expLR_DomainNet.step()
        model.eval()
        is_best = False
        if epoch % args.test_frequent == 0 or epoch == args.num_epochs-1:
            with torch.no_grad():
                iou_list, dice_list = [], [] 
                for i, (imgs, labels) in enumerate(val_loader):
                    imgs = imgs.to(args.device)
                    labels = labels.to(args.device)
                    if args.crop_black:
                        crop_black = CropBlack(imgs, args)
                        crop_img = crop_black.crop()
                        img_forward = ImagePadToSize(args.upsample_num, model)
                        pred = img_forward(crop_img)
                        pred = crop_black.merge(pred)
                    else:
                        img_forward = ImagePadToSize(args.upsample_num, model)
                        pred = img_forward(imgs)
                    loss = loss_func(pred, labels)
                    if args.local_rank == 0 and i == 0:
                        imgs = vutils.make_grid(imgs, normalize=True, scale_each=True)
                        writer.add_image('val/inputs', imgs, epoch)
                        gts = vutils.make_grid(labels, normalize=True, scale_each=True)
                        writer.add_image('val/gts', gts, epoch)
                        preds= vutils.make_grid(pred, normalize=True, scale_each=True)
                        writer.add_image('val/preds', preds, epoch)
                    if args.use_connect_domain:
                        #pred = utils.max_connected_domain(pred, args.threshold)
                        pred = utils.max_prob_domain(pred, args.threshold)
                    else:
                        pred = pred.detach().permute(0,2,3,1).squeeze().cpu().numpy()
                    metrics = utils.IOUMetric(args.num_classes, args.threshold)
                    metrics.add_batch(pred, labels)
                    iou, dice = metrics.evaluate_front()
                    iou_list.append(iou)
                    dice_list.append(dice)

                avg_iou, avg_dice = sum(iou_list) / len(iou_list), sum(dice_list) / len(dice_list)
                torch.distributed.barrier()
                avg_iou = utils.dist_mean_param(avg_iou, args.device)
                avg_dice = utils.dist_mean_param(avg_dice, args.device)
                if args.local_rank == 0:
                    writer.add_scalar('val/mean_iou', avg_iou, epoch)
                    writer.add_scalar('val/mean_dice', avg_dice, epoch)
                    print('--------test  mean_iou: {}, mean_dice: {}'.format(avg_iou, avg_dice))

                if avg_iou > args.best_eval:
                    args.best_eval = avg_iou
                    is_best = True
                # save model parameters
                if args.local_rank == 0 and (epoch % args.save_freq == 0 or is_best):
                    print('save checkpoint.')
                    #os.system("hdfs dfs -put -f ./run $ARNOLD_OUTPUT")
                    utils.save_checkpoint({'epoch': epoch, 'model': model.module.state_dict(), 'best_eval': args.best_eval, 'sr': SRNet.module.state_dict(), 'd': DomainNet.module.state_dict()}, is_best, args.save_path)

    print('Done.')


if __name__ == '__main__':
    main() 
