import torch
import torch.nn as nn
import os
import sys

sys.path.append('../')
import argparse
from config import config
from model.architectures.bbox import My_Counting_Net_Insight
from data_set.xray_bbox_folder import MyFolder
import torchvision.transforms as transforms
from tqdm import tqdm
import cv2
from utils.mean_average_precisioin import AveragePrecisionMeter
import torch.nn.functional as F
import numpy as np

object_categories = ['Gun', 'Knife', 'Pliers', 'Scissors', 'Wrench']

model_path = '/data/Experiments/My_CHR/attention/attention_models/best.ckpt'

test_set_path = '../data_set/attention_dataset/train_test_split/test'

save_path = '../data_set/attention_dataset/train_test_split/test/pred_heatmap/'

img_num = 0


def covert2image(tensor):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    t_mean = torch.FloatTensor(mean).view(3, 1, 1)
    t_std = torch.FloatTensor(std).view(3, 1, 1)

    img = (tensor.cpu() * t_std + t_mean)

    return img


def show_cam_on_image(img, mask):
    global img_num
    heatmap = cv2.applyColorMap(np.uint8(255*mask.cpu()), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(covert2image(img).permute(1, 2, 0))
    cam = cam / np.max(cam)
    cv2.imwrite(save_path + str(img_num) + '.jpg', np.uint8(255 * cam))
    # cv2.imwrite(save_path + str(img_num) + '.jpg', np.uint8(255 * heatmap))
    img_num += 1
    print(img_num)


def visualization(args, test_loader, model, state):

    model.eval()
    pred_list = []
    ap_meter = AveragePrecisionMeter(False)

    ap_meter.reset()
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(test_loader)):
            data, target = data.cuda(), target.cuda()

            ht_map_28, ht_map_56 = model(data)

            ht_map_224 = F.upsample(ht_map_56, size=(224, 224), mode='bilinear')

            for i in range(ht_map_56.shape[0]):
                show_cam_on_image(data[i, :, :, :], ht_map_224[i, 0, :, :])



def main():

    print('--- start evalute and save wrong cases ---')

    state = {k: v for k, v in args._get_kwargs()}

    # Test data config
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.Resize(args.image_size + args.image_size // 7),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        normalize,
    ])
    test_folder = MyFolder(test_set_path, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_folder,
        batch_size=64, shuffle=False,
        num_workers=1, pin_memory=True)

    # Init model
    model = My_Counting_Net_Insight(num_classes=args.num_classes, pretrained=True)
    model = model.cuda()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    # Save wrong cases
    visualization(args, test_loader, model, state)


if __name__ == "__main__":
    args = config.parse_commandline_args()
    main()
