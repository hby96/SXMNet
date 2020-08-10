import torch
import torchvision.transforms as transforms
from data_set.xray_multi_label_attention_folder import MyFolder
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
    test_transform = transforms.Compose([
        transforms.Resize((args.image_size + args.image_size // 7, args.image_size + args.image_size // 7)),
        # transforms.Resize((args.image_size, args.image_size)),
        # my_transforms.LiubaiResize(args.image_size + args.image_size // 7),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        normalize,
        ])
    train_folder = MyFolder(root=args.train_set_path, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_folder,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    test_folder = MyFolder(args.test_set_path, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_folder,
        batch_size=64, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    return train_loader, test_loader

