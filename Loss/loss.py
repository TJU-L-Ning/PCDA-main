from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))       
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target) 

def DANN(features, ad_net):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    标签软化
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        #epsilon 控制软化效果
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss

class ProtoLoss(nn.Module):

    def __init__(self, nav_t: float):
        super(ProtoLoss, self).__init__()
        self.nav_t = nav_t

    def pairwise_cosine_dist(self, x, y):
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        cos_dis = 1 - torch.matmul(x, y.T)
        return cos_dis

    def forward(self, mu_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        sim_mat = torch.matmul(mu_s, f_t.T)
        ##  (17x256) * (256xN) = (17xN) 
        real_dist = F.softmax(sim_mat/self.nav_t, dim=0) ##单个样本的置信度
        fake_dist = F.softmax(sim_mat/self.nav_t, dim=1) ##单个样本的置信度


        # mask = hard_psd_label > 0  # Check if pseudo-label is greater than 0
        # mask = mask.unsqueeze(1).expand(-1, f_t.size(1))  # Expand the mask to match the dimensions of f_t

        cost_mat = self.pairwise_cosine_dist(mu_s, f_t) ##17*N

        source_loss = (0.3*cost_mat*real_dist).sum(0).mean() ##sum(0) 将保持维度为 (N)
        target_loss = (0.7*cost_mat*fake_dist).sum(1).mean() ##sum(1) 将保持维度为 (17)
        loss = source_loss + target_loss
        return loss

# class PseudoLabelLoss(nn.Module):
#     def init(self,)
    
#     def forward(self, mu_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
#         sim_mat = torch.matmul(mu_s, f_t.T)