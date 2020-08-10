import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import sys

sys.path.append('../')
import argparse
from config import attention_multi_label_config
from torch.optim import lr_scheduler
from model.architectures.attention_multi_label import My_Counting_Net_Insight, My_Counting_Net_Insight_Gated
# from utils.mean_average_precisioin import AveragePrecisionMeter
from utils.my_metric import Correct_Num
from model.losses.loss import MultiLabelLoss, MultiLabelFocalLoss, MultiLabelCBLoss, MultiLabelCBFocalLoss
from engine.trianer_attention_multi_label import fit
from data_set.multi_label_attention_dataset import prepare_data
from tensorboardX import SummaryWriter
from torchvision import models

# Margin set
add_margin = -0.0


def main():
    # args = config.create_parser().parse_args()
    print(args.exp_name)

    # Init tensorboard
    # writer = SummaryWriter(comment=args.exp_name)
    writer = None

    # Init logger
    state = {k: v for k, v in args._get_kwargs()}

    # Training data config
    train_loader, test_loader = prepare_data(args)

    # Init model
    # model = My_Res50(num_classes=args.num_classes, pretrained=True)
    # model = My_Counting_Net_Insight(num_classes=args.num_classes, pretrained=True)
    model = My_Counting_Net_Insight_Gated(num_classes=args.num_classes, pretrained=True)

    # Set optimizer
    criterion = MultiLabelLoss()
    # criterion = MultiLabelFocalLoss()
    # criterion = MultiLabelCBLoss(cls_nums=[482, 3048, 253, 155, 1203, 1202, 3465, 2373])
    # criterion = MultiLabelCBFocalLoss(cls_nums=[482, 3048, 253, 155, 1203, 1202, 3465, 2373])

    # ap_meter = AveragePrecisionMeter(False)
    # ap_meter = Correct_Num(thre=[0.5, 0.5, 0.5, 0.3, 0.5, 0.5, 0.7, 0.5])
    ap_meter = Correct_Num(thre=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    # fc_params = list(map(id, model.fc.parameters()))
    fc_1_params = list(map(id, model.fc_1.parameters()))
    fc_2_params = list(map(id, model.fc_2.parameters()))
    fc_3_params = list(map(id, model.fc_3.parameters()))
    # base_params = filter(lambda p: id(p) not in fc_params, model.parameters())
    base_params = filter(lambda p: id(p) not in fc_1_params+fc_2_params+fc_3_params, model.parameters())
    params = [
        {"params": base_params, "lr": state['learning_rate']},
        # {"params": model.fc.parameters(), "lr": state['learning_rate']},  # 10
        {"params": model.fc_1.parameters(), "lr": 5 * state['learning_rate']},  # 10
        {"params": model.fc_2.parameters(), "lr": 5 * state['learning_rate']},
        {"params": model.fc_3.parameters(), "lr": 5 * state['learning_rate']},
    ]
    optimizer = torch.optim.SGD(params, momentum=state['momentum'], lr=state['learning_rate'],
                                weight_decay=state['decay'], nesterov=True)

    scheduler = lr_scheduler.StepLR(optimizer, args.schedule, gamma=args.gamma, last_epoch=-1)

    # if args.n_gpu > 1:
    #     model = nn.DataParallel(model).cuda()
    model = model.cuda()

    if args.resume_path is not None:
        checkpoint = torch.load(args.resume_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print('success! load model from {}'.format(args.resume_path))

    fit(args, train_loader, test_loader, model, criterion, ap_meter, optimizer, scheduler,
        state=state, cuda=True)


if __name__ == "__main__":
    args = attention_multi_label_config.parse_commandline_args()
    main()
