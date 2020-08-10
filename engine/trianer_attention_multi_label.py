import torch
import torch.nn.functional as F
import json
import os
from utils.common import adjust_learning_rate, get_one_hot_label
from model.losses.loss import we_are_diff_loss, we_must_have_one_loss, get_class_balanced_mask
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
        scheduler.step()

        # Train stage
        train_epoch(args, train_loader, model, loss_fn, ap_meter, optimizer, cuda, state, epoch)

        # Test stage
        test_epoch(args, test_loader, model, loss_fn, ap_meter, state, epoch)

        if state['test_map'] > best_map:
            best_map = state['test_map']
            best_epoch = epoch
            for index in range(args.num_classes):
                best_single_ap[index] = state[args.classes[index] + '_ap']
            # # torch.save(model, os.path.join(state['model_save_path'], 'epoch' + str(epoch) + '_model.pkl'))
            if (args.model_save_path is not None) and (best_map > 95.0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec': best_map,
                }, True, args.model_save_path, epoch + 1)

        # print(state)
        print("epoch:", int(epoch))
        print('----------Test----------')
        print("Best test epoch: %f" % best_epoch)
        print("Best test map: %f" % best_map)
        for index in range(args.num_classes):
            print('Test ' + args.classes[index] + ': %f' % best_single_ap[index])
        print('----------Test----------')


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_epoch(args, train_loader, model, loss_fn, ap_meter, optimizer, cuda, state, n_epoch):

    model.train()
    loss_avg = 0.0

    ap_meter.reset()

    for batch_idx, (data, target) in tqdm(enumerate(train_loader), desc="Training"):
    # for batch_idx, (data, target, total_mask_28, total_mask_56) in tqdm(enumerate(train_loader), desc="Training"):

        if cuda:
            data = data.cuda()
            target = target.cuda()

        # mix up
        data, targets_a, targets_b, lam = mixup_data(data, target, 1.0, True)

        # get model output
        optimizer.zero_grad()
        logit = model(data, n_epoch)

        # get classification loss
        # cls_loss = (torch.mean(loss_fn(layer4_logit, target)) + torch.mean(loss_fn(layer3_logit, target)) + torch.mean(loss_fn(layer2_logit, target))) / 3
        # cls_loss = torch.mean(loss_fn(logit, target))
        cls_loss = torch.mean(mixup_criterion(loss_fn, logit, targets_a, targets_b, lam))

        loss = cls_loss

        # output = torch.sigmoid(layer4_logit) + torch.sigmoid(layer3_logit) + torch.sigmoid(layer2_logit)
        # output = output / 3
        output = torch.sigmoid(logit)

        ap_meter.add(output.data, target)

        # optimize
        loss.backward()

        optimizer.step()

        loss_avg = loss_avg * 0.2 + float(loss) * 0.8

    state['train_loss'] = loss_avg
    ap = 100 * ap_meter.value()
    map = ap.mean()

    print('----------Train----------')
    print("Train map: %f" % map)
    for index in range(args.num_classes):
        print('Train ' + args.classes[index] + ': %f' % ap[index])
    print('----------Train----------')


def test_epoch(args, test_loader, model, loss_fn, ap_meter, state, n_epoch):

    model.eval()
    loss_avg = 0.0

    ap_meter.reset()

    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(test_loader), desc='Testing'):
            data, target = data.cuda(), target.cuda()

            # forward
            # layer4_logit, layer3_logit, layer2_logit = model(data)
            logit = model(data, n_epoch)

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
    for index in range(args.num_classes):
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
