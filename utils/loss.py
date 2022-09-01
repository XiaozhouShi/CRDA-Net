import torch
from torch import nn
from torchvision.models.vgg import vgg16
import torch.nn.functional as F



class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss

class PerceptionLoss(nn.Module):
    def __init__(self, device):
        super(PerceptionLoss, self).__init__()
        vgg = vgg16(pretrained=True).to(device)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred, target):
        return self.mse_loss(self.loss_network(pred), self.loss_network(target))



class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class FocalLoss(nn.Module):
  """
  This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
  'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
    Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
  :param num_class:
  :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
  :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
          focus on hard misclassified example
  :param smooth: (float,double) smooth value when cross entropy
  :param balance_index: (int) balance class index, should be specific when alpha is float
  :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
  """
  
  def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
    super(FocalLoss, self).__init__()
    self.num_class = num_class
    self.alpha = alpha
    self.gamma = gamma
    self.smooth = smooth
    self.size_average = size_average
  
    if self.alpha is None:
      self.alpha = torch.ones(self.num_class, 1)
    elif isinstance(self.alpha, (list, np.ndarray)):
      assert len(self.alpha) == self.num_class
      self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
      self.alpha = self.alpha / self.alpha.sum()
    elif isinstance(self.alpha, float):
      alpha = torch.ones(self.num_class, 1)
      alpha = alpha * (1 - self.alpha)
      alpha[balance_index] = self.alpha
      self.alpha = alpha
    else:
      raise TypeError('Not support alpha type')
  
    if self.smooth is not None:
      if self.smooth < 0 or self.smooth > 1.0:
        raise ValueError('smooth value should be in [0,1]')
  
  def forward(self, input, target):
    logit = F.softmax(input, dim=1)
  
    if logit.dim() > 2:
      # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
      logit = logit.view(logit.size(0), logit.size(1), -1)
      logit = logit.permute(0, 2, 1).contiguous()
      logit = logit.view(-1, logit.size(-1))
    target = target.view(-1, 1)
  
    # N = input.size(0)
    # alpha = torch.ones(N, self.num_class)
    # alpha = alpha * (1 - self.alpha)
    # alpha = alpha.scatter_(1, target.long(), self.alpha)
    epsilon = 1e-10
    alpha = self.alpha
    if alpha.device != input.device:
      alpha = alpha.to(input.device)
  
    idx = target.cpu().long()
    one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
    one_hot_key = one_hot_key.scatter_(1, idx, 1)
    if one_hot_key.device != logit.device:
      one_hot_key = one_hot_key.to(logit.device)
  
    if self.smooth:
      one_hot_key = torch.clamp(
        one_hot_key, self.smooth, 1.0 - self.smooth)
    pt = (one_hot_key * logit).sum(1) + epsilon
    logpt = pt.log()
  
    gamma = self.gamma
  
    alpha = alpha[idx]
    loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
  
    if self.size_average:
      loss = loss.mean()
    else:
      loss = loss.sum()
    return loss


def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
              balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                'none': No reduction will be applied to the output.
                'mean': The output will be averaged.
                'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # p = torch.sigmoid(inputs)
    p = inputs
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def dice_loss(predict, target, ep = 1e-8):
    intersection = 2 * torch.sum(predict * target) + ep
    union = torch.sum(predict) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss


if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
