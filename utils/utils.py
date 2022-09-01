import torch
import torch.autograd as autograd

import numpy as np
import math
import os
import shutil
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
import skimage.measure as measure
import copy


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def AUC(pred, label):
    pred, label = pred.squeeze().detach().cpu().numpy(), label.squeeze().detach().cpu().numpy()
    fpr, tpr, thresholds = roc_curve(label, pred)
    return auc(fpr, tpr)

def acc(pred, label, threshold):
    pred, label = pred.squeeze().detach().cpu().numpy(), label.squeeze().detach().cpu().numpy()
    pred[pred >= threshold] = 1
    pred[pred < threshold] = 0
    pred = pred.astype(np.uint8)
    label = label.astype(np.uint8)
    acc = accuracy_score(label, pred)
    return acc
    
def acc_precision_f1score(pred, label, threshold):
    pred, label = pred.squeeze().detach().cpu().numpy(), label.squeeze().detach().cpu().numpy()
    pred[pred >= threshold] = 1
    pred[pred < threshold] = 0
    pred = pred.astype(np.uint8)
    label = label.astype(np.uint8)
    acc = accuracy_score(label, pred)
    precision = precision_score(label, pred, zero_division = 1)
    f1 = f1_score(label, pred, zero_division = 1)
    return acc, precision, f1

def evaluation(pred, label, threshold):
    fpr, tpr, _ = roc_curve(label, pred)
    Auc = auc(fpr, tpr)
    pred[pred >= threshold] = 1
    pred[pred < threshold] = 0
    pred = pred.astype(np.uint8)
    label = label.astype(np.uint8)

    acc = accuracy_score(label, pred)
    precision = precision_score(label, pred, zero_division = 1)
    f1 = f1_score(label, pred, zero_division = 1)
    recall = recall_score(label, pred)
    matrix = confusion_matrix(label, pred)
    return Auc, acc, precision, f1, recall, matrix

def save_checkpoint(state, is_best, file_path):
    torch.save(state, os.path.join(file_path, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(file_path, 'checkpoint.pth.tar'), os.path.join(file_path,'model_best.pth.tar'))


class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """
    def __init__(self, num_classes, threshold = None):
        self.threshold = threshold
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist
    def add_batch(self, predictions, gts):
        gts = gts.detach().permute(0,2,3,1).squeeze().cpu().numpy()
        predictions[predictions >= self.threshold] = 1
        predictions[predictions < self.threshold] = 0
        predictions = predictions.astype(np.uint8)
        gts = gts.astype(np.uint8)
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())
    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc

    def evaluate_front(self):
        iou = self.hist[1,1] / (self.hist[1].sum() + self.hist[:,1].sum() - self.hist[1, 1])
        dice = 2 * self.hist[1,1] / (self.hist[1].sum() + self.hist[:,1].sum())
        return iou, dice

def dist_mean_param(param, device):
    tensor_param = torch.tensor(param).to(device)
    size = float(torch.distributed.get_world_size())
    torch.distributed.all_reduce(tensor_param, op = torch.distributed.ReduceOp.SUM)
    tensor_param = tensor_param / size
    return tensor_param.item()

def max_connected_domain(pred, threshold):
    pred = pred.detach().permute(0, 2, 3, 1).squeeze().cpu().numpy()
    pred[pred >= threshold] = 255
    pred[pred < threshold] = 0
    pred = pred.astype(np.uint8)

    #label_img, num = measure.label(img, neighbors=8,background=0,return_num = True)
    label_img = measure.label(pred, connectivity=2)
    max_label, max_area = 0, 0
    properties = measure.regionprops(label_img)
    for prop in properties:
        if prop.area > max_area:
            max_label = prop.label
            max_area = prop.area
    if max_label != 0:
        pred[label_img == max_label] = 1 
        pred[label_img != max_label] = 0
    return pred
    

def max_prob_domain(pred, threshold):
    pred = pred.detach().permute(0, 2, 3, 1).squeeze().cpu().numpy()
    orig_pred = copy.deepcopy(pred) 
    pred[pred >= threshold] = 255
    pred[pred < threshold] = 0
    pred = pred.astype(np.uint8)

    #label_img, num = measure.label(img, neighbors=8,background=0,return_num = True)
    label_img = measure.label(pred, connectivity=2)
    max_label, max_avg_prob = 0, 0
    properties = measure.regionprops(label_img)
    for prop in properties:
        coords = prop.coords
        sum_num = sum([orig_pred[i,j] for i, j in coords])
        avg_prob = sum_num / prop.area
        if avg_prob > max_avg_prob:
            max_label = prop.label
            max_avg_prob = avg_prob 
    if max_label != 0:
        pred[label_img == max_label] = 1 
        pred[label_img != max_label] = 0
    return pred

class NativeScaler:
    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad = None, clip_mode='norm', parameters=None, create_graph=False):
        optimizer.zero_grad()
        self._scaler.scale(loss).backward(create_graph=create_graph)
        # if clip_grad is not None:
        #     assert parameters is not None
        #     self._scaler.unscale_(optimizer)
        #     dispatch_clip_grad(parameters, clip_grad, mode = clip_mode)
        self._scaler.step(optimizer)
        self._scaler.update()
    
    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def calc_gradient_penalty(D, real, fake, device):
    fake.requires_grad=True
    b,c,h,w = real.size()
    alpha = torch.rand((b, 1, 1, 1))
    alpha = alpha.to(device)
    interpolates = alpha * real + (1-alpha)*fake
    disc_interolates = D(interpolates)
    gradients = autograd.grad(
            outputs = disc_interolates, inputs=interpolates, \
            grad_outputs=torch.ones(disc_interolates.size()).to(device), \
            create_graph=True, retain_graph=True,only_inputs=True
        )[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()*10
    return gradient_penalty
