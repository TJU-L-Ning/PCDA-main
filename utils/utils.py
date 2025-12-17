import os
import sys 
import logging 
import torch 
import random 
import numpy as np 
import torch.nn as nn 
import pdb

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def log_args(args):
    s = "\n==========================================\n"
    s += ("python" + " ".join(sys.argv) + "\n")
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    s += "==========================================\n"
    return s

def set_logger(args, log_name="train_log.txt"):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    if args.test:
        file_handler = logging.FileHandler(os.path.join(args.save_dir, log_name), mode="a")
        file_format = logging.Formatter("%(message)s")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_format)
    else:
        file_handler = logging.FileHandler(os.path.join(args.save_dir, log_name), mode="w")
        file_format = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_format)
    
    terminal_handler = logging.StreamHandler()
    terminal_format = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
    terminal_handler.setLevel(logging.INFO)
    terminal_handler.setFormatter(terminal_format)
    
    logger.addHandler(file_handler)
    logger.addHandler(terminal_handler)
    if not args.test:
        logger.debug(log_args(args))
    
    return logger

def get_acc(args, class_list, gt_label_all, pred_cls_all, open_flag=True,pred_unc_all=None):
    
    per_class_num = np.zeros((len(class_list)))
    per_class_correct = np.zeros_like(per_class_num)
    pred_label_all = torch.max(pred_cls_all, dim=1)[1] #[N]
    print(pred_label_all.shape)
    
    

    
    if open_flag:
        cls_num = pred_cls_all.shape[1]
        if pred_unc_all is None:
            pred_unc_all = Entropy(pred_cls_all)/np.log(cls_num)# [N]
        unc_idx = torch.where(pred_unc_all > 0.9)[0]
        pred_label_all[unc_idx] = cls_num 
    for i, label in enumerate(class_list):
        label_idx = torch.where(gt_label_all == label)[0]
        correct_idx = torch.where(pred_label_all[label_idx] == label)[0]
        per_class_num[i] = float(len(label_idx))
        per_class_correct[i] = float(len(correct_idx))
    per_class_acc = per_class_correct / (per_class_num + 1e-5)
    known_acc = per_class_acc[:-1].mean()
    unknown_acc = per_class_acc[-1]
    h_score = 2 * known_acc * unknown_acc / (known_acc + unknown_acc + 1e-5)
    return h_score, known_acc, unknown_acc, per_class_acc
    
class CrossEntropyLabelSmooth(nn.Module):
    
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """      

    def __init__(self, num_classes, epsilon=0.1, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, targets, applied_softmax=True):
        """
        Args:
            inputs: prediction matrix (after softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size, num_classes).
        """
        if applied_softmax:
            log_probs = torch.log(inputs)
        else:
            log_probs = self.logsoftmax(inputs)
        if inputs.shape != targets.shape:
            targets = torch.zeros_like(inputs).scatter(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
