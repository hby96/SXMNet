import torch
import torchvision.transforms as transforms

import json

import sys
sys.path.append('../')

from tqdm import tqdm
from baseline import My_Res50
from folder import MyFolder

object_categories = ['battery', 'bottle', 'firecracker', 'grenade', 'gun', 'hammer', 'knife', 'scissors']

# model_path = '/data/Experiments/Xray-Multi-Label/final_divide/add_safe_baseline_models_v2/best.ckpt'
# model_path = '/data/Experiments/Siamese-Xray-Multi-Label/base_models/checkpoint.41.ckpt'
# model_path = '/data/Experiments/Siamese-Xray-Multi-Label/attention_fpn_models/best.ckpt'
# model_path = '/data/Experiments/Siamese-Xray-Multi-Label/max_entropy_models/0.2/best.ckpt'
model_path = '/data/Experiments/Xray-Multi-Label/attention_cls_models/best.ckpt'

test_set_path = '../data_set/final_divide/data_train_test_split/test'

result_save_path = '/home/hubenyi/Multi-Label-Series/Xray-Multi-Label/test_offline/result.json'


def prepare_data(test_set_path):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
        ])
    test_folder = MyFolder(test_set_path, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_folder,
        batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True)
    return test_loader


def eval_model(test_loader, model):
    model.eval()
    pred_list = []
    target_list = []
    info_dicts = list()

    with torch.no_grad():
        for batch_idx, (data, target, path) in tqdm(enumerate(test_loader), desc='Testing'):

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            logit = model(data, batch_idx)

            pred = torch.sigmoid(logit)

            for i in range(data.shape[0]):
                info_dict = dict()
                info_dict['img_name'] = path[i].split('/')[-1]
                info_dict['label'] = target[i].tolist()
                info_dict['pred'] = pred[i].numpy().tolist()
                info_dicts.append(info_dict)

    with open(result_save_path, 'w') as f:
        json.dump(info_dicts, f)


def main():
    print('--- start evalute ---')

    # Training data config
    test_loader = prepare_data(test_set_path)

    # Init model
    model = My_Res50(num_classes=len(object_categories))
    if torch.cuda.is_available():
        model = model.cuda()
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    # Save wrong cases
    eval_model(test_loader, model)


if __name__ == "__main__":
    main()
