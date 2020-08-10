import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.loss import _WeightedLoss

def add_margin_softmax_loss(logits, target, one_hot_target, add_margin, loss_fn):
    # target: not one-hot label

    row = range(target.shape[0])
    col = target.cpu().numpy()

    target_logits = logits[row, col]
    margin_flag = (target_logits - add_margin)>0
    margin_logits = target_logits - (add_margin * margin_flag.float())
    logits[row, col] = margin_logits

    # loss = (-torch.log(logits + 1e-10) * one_hot_target).sum() / logits.shape[0]
    loss = loss_fn(logits, target)

    return loss


def confuse_loss(alpha=10, output=None, label=None):
    prob = torch.softmax(output, dim=1)
    batch_size = prob.shape[0]
    if batch_size % 2 != 0:
        batch_size = batch_size - 1
    single_batch = int(batch_size / 2)
    batch_1 = prob[0:single_batch]
    batch_2 = prob[single_batch:batch_size]

    label_1 = label[0:single_batch]
    label_2 = label[single_batch:batch_size]

    mix_flag = 1 - torch.eq(label_1, label_2)
    mix_flag = mix_flag.type(torch.cuda.FloatTensor)
    mix_number = mix_flag.sum() + 1  # in case mix_number equal 0

    mix_loss = (torch.norm((batch_1 - batch_2).abs(), 2, 1) * mix_flag).sum() / mix_number
    loss = alpha * mix_loss

    return loss


def binary_cross_entropy(logit, target, cls_weight, eps=1e-10):
    '''if not (target.size() == input.size()):
        warnings.warn("Using a target size ({}) that is different to the input size ({}) is deprecated. "
                      "Please ensure they have the same size.".format(target.size(), input.size()))
    if input.nelement() != target.nelement():
        raise ValueError("Target and input must have the same number of elements. target nelement ({}) "
                         "!= input nelement ({})".format(target.nelement(), input.nelement()))

    if weight is not None:
        new_size = _infer_size(target.size(), weight.size())
        weight = weight.expand(new_size)
        if torch.is_tensor(weight):
            weight = Variable(weight)
    '''
    logit = torch.sigmoid(logit)
    if cls_weight is None:
        loss = -(target * torch.log(logit + eps) + (1 - target) * torch.log(1 - logit + eps))
    else:
        loss = -(target * torch.log(logit + eps) + (1 - target) * torch.log(1 - logit + eps))
        loss = loss * cls_weight
    # print(loss)
    # assert False
    return loss


class MultiLabelLoss(_WeightedLoss):
    def __init__(self, cls_weight=None):
        super(MultiLabelLoss, self).__init__()
        if cls_weight is None:
            self.cls_weight = None
        else:
            self.cls_weight = torch.from_numpy(np.array(cls_weight).astype(np.float32))
            self.cls_weight = (self.cls_weight / self.cls_weight.sum() * len(cls_weight)).view(1, -1).cuda()

    def forward(self, logit, target):
        return binary_cross_entropy(logit, target, self.cls_weight)


def multi_label_smooth(target, eps, num_classes):
    target_num = target.sum()
    target_index = (target == 1.)
    new_target = target + (target_num * eps / (num_classes - target_num))
    new_target[target_index] = 1. - eps
    return new_target


def multi_label_focal_loss(logit, target, eps=1e-10):
    logit = torch.sigmoid(logit)

    # Calculate focal loss
    pt_index = (target > 0.8)
    pt = logit.clone()
    pt[pt_index] = 1 - logit[pt_index]
    focal_coff = torch.pow(pt, 2)

    loss = -(target * torch.log(logit + eps) + (1 - target) * torch.log(1 - logit + eps))
    loss = focal_coff * loss

    return loss


class MultiLabelFocalLoss(_WeightedLoss):
    def forward(self, logit, target):
        return multi_label_focal_loss(logit, target)


def we_are_diff_loss(logit):
    result = F.relu(torch.sigmoid(logit) - 0.5)
    loss = 0
    for i in range(result.shape[1] - 1):
        loss += torch.sum(result[:, -1] * result[:, i])
    loss = loss / (result.shape[0] * (result.shape[1] - 1))

    return loss


def we_must_have_one_loss(logit):
    result = torch.max(logit, dim=1)[0] - 0.5
    result = torch.sum(result) / result.shape[0]
    loss = -result
    return loss


# def generate_mask()
def get_class_balanced_mask(logit, target):
    n_data = np.ones((logit.shape[0], logit.shape[1]))
    n_target = 1 - target.cpu().detach().numpy()
    n_output = logit.cpu().detach().numpy()
    index = np.where(n_output < -20)
    if len(index) == 0:
        return np.ones((logit.shape[0], logit.shape[1]))
    n_data[index] = 0
    n_data = 1 - n_data
    n_data = torch.from_numpy(n_data).float().cuda()
    n_target = torch.from_numpy(n_target).cuda()
    mask = 1 - torch.mul(n_data, n_target)

    return mask


class MultiLabelCBLoss(_WeightedLoss):
    def __init__(self, cls_nums=[]):
        super(MultiLabelCBLoss, self).__init__()
        self.cls_nums = cls_nums
        self.belta = (np.array(self.cls_nums).sum() - 1) / np.array(self.cls_nums).sum()
        self.cls_weights = [(1 - self.belta) / (1 - self.belta.__pow__(cls_num)) for cls_num in self.cls_nums]
        self.cls_weights = torch.from_numpy(np.array(self.cls_weights).astype(np.float32)).cuda().view(1, len(cls_nums))
        self.cls_weights = self.cls_weights / self.cls_weights.sum() * len(self.cls_nums)

    def forward(self, logit, target, eps=1e-10):
        logit = torch.sigmoid(logit)
        loss = -(target * torch.log(logit + eps) + (1 - target) * torch.log(1 - logit + eps))
        loss = self.cls_weights * loss

        return loss


class MultiLabelCBFocalLoss(_WeightedLoss):
    def __init__(self, cls_nums=[]):
        super(MultiLabelCBFocalLoss, self).__init__()
        self.cls_nums = cls_nums
        self.belta = (np.array(self.cls_nums).sum() - 1) / np.array(self.cls_nums).sum()
        self.cls_weights = [(1 - self.belta) / (1 - self.belta.__pow__(cls_num)) for cls_num in self.cls_nums]
        self.cls_weights = torch.from_numpy(np.array(self.cls_weights).astype(np.float32)).cuda().view(1, len(cls_nums))
        self.cls_weights = self.cls_weights / self.cls_weights.sum() * len(self.cls_nums)

    def forward(self, logit, target, eps=1e-10):
        logit = torch.sigmoid(logit)

        # Calculate focal loss
        pt_index = (target > 0.8)
        pt = logit.clone()
        pt[pt_index] = 1 - logit[pt_index]
        focal_coff = torch.pow(pt, 2)

        loss = -(target * torch.log(logit + eps) + (1 - target) * torch.log(1 - logit + eps))
        loss = self.cls_weights * focal_coff * loss

        return loss

