import torch
import torchvision.transforms as transforms
from data_set.xray_bbox_folder import MyFolder
import data_set.my_transform as my_transforms


def prepare_data(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size + args.image_size // 7, args.image_size + args.image_size // 7)),
        # transforms.Resize(args.image_size + args.image_size//7),
        # my_transforms.LiubaiResize(args.image_size + args.image_size//7),
        # my_transforms.YinhuaResize(args.image_size + args.image_size // 7),
        # transforms.RandomRotation(30),  # 可选
        transforms.RandomCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        normalize,
        ])
    ht_transform = transforms.Compose([
        transforms.Resize((args.image_size + args.image_size // 7, args.image_size + args.image_size // 7)),
        # transforms.Resize(args.image_size + args.image_size // 7),
        # my_transforms.LiubaiResize(args.image_size + args.image_size // 7),
        # my_transforms.YinhuaResize(args.image_size + args.image_size // 7),
        # transforms.RandomRotation(30),  # 可选
        transforms.RandomCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize(112),
        transforms.ToTensor(),
    ])
    train_folder = MyFolder(root=args.train_set_path, transform=train_transform, ht_transform=ht_transform)
    train_loader = torch.utils.data.DataLoader(
        train_folder,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    return train_loader

