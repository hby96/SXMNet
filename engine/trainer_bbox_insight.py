import torch
import torch.nn.functional as F
import json
import os
import torchvision.transforms as transforms
from utils.common import adjust_learning_rate, get_one_hot_label
from model.losses.loss import MultiLabelLoss, MultiLabelFocalLoss
from tqdm import tqdm
import numpy as np
import shutil


def fit(args, train_loader, test_loader, model, loss_fn, ap_meter, optimizer, scheduler, cuda, state=None,
        start_epoch=0, writer=None):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """

    n_epochs = state['epochs']

    for epoch in range(0, start_epoch):
        scheduler.step()

    best_map = 0.0
    best_epoch = 0
    best_single_ap = list(0. for i in range(args.num_classes))
    for epoch in range(start_epoch, n_epochs):
        print('epoch:', int(epoch))
        scheduler.step()

        # Train stage
        train_epoch(args, train_loader, model, loss_fn, ap_meter, optimizer, cuda, state, epoch, writer=writer)

        # Test stage
        # test_epoch(args, test_loader, model, loss_fn, ap_meter, state, epoch)

        if (args.model_save_path is not None) and (((epoch+1) % 100) == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
            }, True, args.model_save_path, epoch + 1)

    writer.close()


def covert2png(tensor):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    t_mean = torch.FloatTensor(mean).view(3, 1, 1).cuda()
    t_std = torch.FloatTensor(std).view(3, 1, 1).cuda()
    img = (tensor * t_std + t_mean)
    return img


def train_epoch(args, train_loader, model, criterion, ap_meter, optimizer, cuda, state, n_epoch, writer=None):

    model.train()
    loss_avg = 0.0

    # ap_meter.reset()

    batch_num = len(train_loader)
    cls_loss_func = MultiLabelLoss()

    for batch_idx, (data, target, total_mask_56) in tqdm(enumerate(train_loader), desc="Training"):

        if cuda:
            data = data.cuda()
            target = target.cuda()
            total_mask_56 = total_mask_56.cuda()

        # get model output
        optimizer.zero_grad()

        heat_map_28, heat_map_56 = model(data)

        # get gt heatmap
        total_gt_heat_map_56 = total_mask_56.sum(dim=1, keepdim=True) / 3.
        total_gt_heat_map_28 = F.avg_pool2d(total_gt_heat_map_56, kernel_size=2, stride=2)

        # total_gt_heat_map_56 = F.upsample(total_gt_heat_map_56, size=(56, 56), mode='bilinear')
        for i in range(total_gt_heat_map_28.shape[0]):
            if total_gt_heat_map_28[i, :, :, :].max() > 0.2:
                total_gt_heat_map_28[i, :, :, :] = total_gt_heat_map_28[i, :, :, :] / total_gt_heat_map_28[i, :, :, :].max()
            if total_gt_heat_map_56[i, :, :, :].max() > 0.2:
                total_gt_heat_map_56[i, :, :, :] = total_gt_heat_map_56[i, :, :, :] / total_gt_heat_map_56[i, :, :, :].max()

        # get classification loss
        htmap_loss_28 = criterion(heat_map_28, total_gt_heat_map_28)
        htmap_loss_28 = torch.sum(htmap_loss_28) / (htmap_loss_28.shape[0])
        htmap_loss_56 = criterion(heat_map_56, total_gt_heat_map_56)
        htmap_loss_56 = torch.sum(htmap_loss_56) / (htmap_loss_56.shape[0])

        total_loss = htmap_loss_28 + htmap_loss_56

        writer.add_image('image/input_pic', covert2png(data[0, :, :, :]))
        writer.add_image('image/total_gt_htmap_28', total_gt_heat_map_28[0, :, :, :])
        writer.add_image('image/pred_htmap_28', heat_map_28[0, 0, :, :].expand(3, -1, -1))
        writer.add_image('image/total_gt_htmap_56', total_gt_heat_map_56[0, :, :, :])
        writer.add_image('image/pred_htmap_56', heat_map_56[0, 0, :, :].expand(3, -1, -1))
        writer.add_scalar('scalar/train_htmap_loss_28', htmap_loss_28, n_epoch*batch_num + batch_idx)
        writer.add_scalar('scalar/train_htmap_loss_56', htmap_loss_56, n_epoch*batch_num + batch_idx)
        writer.add_scalar('scalar/train_total_loss', total_loss, n_epoch * batch_num + batch_idx)

        # optimize
        total_loss.backward()
        optimizer.step()

        loss_avg = loss_avg * 0.2 + float(total_loss) * 0.8

    state['train_loss'] = loss_avg


def test_epoch(args, test_loader, model, loss_fn, ap_meter, state, n_epoch):

    model.eval()
    loss_avg = 0.0

    ap_meter.reset()

    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(test_loader), desc='Testing'):
            data, target = data.cuda(), target.cuda()

            # forward
            _, logit = model(data)
            # logit = model(data)

            # loss = torch.mean(loss_fn(layer4_logit, target)) + torch.mean(
            #     loss_fn(layer3_logit, target)) + torch.mean(loss_fn(layer2_logit, target))
            loss = torch.mean(loss_fn(logit, target))

            # output = torch.sigmoid(layer4_logit) + torch.sigmoid(layer3_logit) + torch.sigmoid(layer2_logit)
            # output = output / 3
            output = torch.sigmoid(logit)

            ap_meter.add(output.data, target)

            # test loss average
            loss_avg += float(loss)

    ap = 100 * ap_meter.value()
    map = ap.mean()

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_map'] = map
    for index in range(args.num_classes-1):
        state[args.classes[index] + '_ap'] = ap[index]


def save_checkpoint(state, is_best, dirpath, epoch):
    filename = 'checkpoint.{}.ckpt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best.ckpt')
    torch.save(state, checkpoint_path)
    print("--- checkpoint saved to %s ---" % checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)
        print("--- checkpoint copied to %s ---" % best_path)
