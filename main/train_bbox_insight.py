import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import sys

sys.path.append('../')
import argparse
from config import bbox_config
from torch.optim import lr_scheduler
from model.architectures.bbox_insight import My_Counting_Net_Insight
from utils.mean_average_precisioin import AveragePrecisionMeter
# from model.architectures.resnet50 import ResNet50_Pre_Trained_5
from model.losses.loss import MultiLabelLoss, MultiLabelFocalLoss
from engine.trainer_bbox_insight import fit
from data_set.bbox_dataset import prepare_data
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
from torchvision import models


os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
# Margin set
add_margin = -0.0

def main():
    # args = config.create_parser().parse_args()
    print(args.exp_name)

    # Init tensorboard
    writer = SummaryWriter(comment=args.exp_name)
    # writer = None

    # Init logger
    state = {k: v for k, v in args._get_kwargs()}

    # Training data config
    train_loader = prepare_data(args)

    # Init model
    model = My_Counting_Net_Insight(num_classes=args.num_classes, pretrained=True)
    # model = My_Counting_Net_CHR(num_classes=args.num_classes, pretrained=True)

    # Set optimizer
    criterion = torch.nn.MSELoss(reduction='none')

    ap_meter = AveragePrecisionMeter(False)

    layer2_params = list(map(id, model.layer2.parameters()))
    layer3_params = list(map(id, model.layer3.parameters()))
    layer4_params = list(map(id, model.layer4.parameters()))

    special_params = filter(lambda p: id(p) not in layer2_params + layer3_params + layer4_params, model.parameters())

    params = [
        {"params": special_params, "lr": 5 * state['learning_rate']},
        {"params": model.layer2.parameters(), "lr": state['learning_rate']},
        {"params": model.layer3.parameters(), "lr": state['learning_rate']},
        {"params": model.layer4.parameters(), "lr": state['learning_rate']},
    ]

    # params = [
    #     {"params": model.layer2_out.parameters(), "lr": state['learning_rate']},
    #     {"params": model.layer3_out.parameters(), "lr": state['learning_rate']},
    #     {"params": model.layer4_out.parameters(), "lr": state['learning_rate']},
    #     {"params": model.layer4_fc.parameters(), "lr": 5 * state['learning_rate']},
    #     {"params": model.layer3_cut_fuse.parameters(), "lr": 5 * state['learning_rate']},
    # ]

    optimizer = torch.optim.SGD(params, momentum=state['momentum'], lr=state['learning_rate'],
                                weight_decay=state['decay'], nesterov=True)

    scheduler = lr_scheduler.StepLR(optimizer, args.schedule, gamma=args.gamma, last_epoch=-1)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

    fit(args, train_loader, None, model, criterion, ap_meter, optimizer, scheduler,
        state=state, cuda=True, writer=writer)


if __name__ == "__main__":
    args = bbox_config.parse_commandline_args()
    main()
