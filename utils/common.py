from utils import ramps
import torch

def adjust_learning_rate(args, optimizer, epoch, step_in_epoch, total_steps_in_epoch):

    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.learning_rate - args.initial_lr) + args.initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_one_hot_label(target, num_classes):
    final_target = torch.zeros(target.shape[0], num_classes)
    for idx in range(target.shape[0]):
        final_target[idx, :][target[idx]] = 1
    return final_target

def l2_normalize(x):
    l2 = torch.norm(x, 2, dim=1)
    x = x / torch.sprt(l2 + 1e-10)
    return x
