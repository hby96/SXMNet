import os
import os.path

import numpy as np
from PIL import Image

import xml.dom.minidom as xmldom

import torch.utils.data as data

from model.losses.loss import multi_label_smooth


object_categories = ['battery', 'bottle', 'firecracker', 'grenade', 'gun', 'hammer', 'knife', 'scissors']


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def read_xml_label(path):

    label = 0.1 + np.zeros(len(object_categories)).astype(np.float32)

    domobj = xmldom.parse(path)
    elementobj = domobj.documentElement
    objects = elementobj.getElementsByTagName("object")
    for obj in objects:
        name = obj.getElementsByTagName('name')[0].childNodes[0].data

        if name in object_categories:
            idx = object_categories.index(name)
            label[idx] = 1.0
    return label


def make_dataset_from_xml(dir):
    images = []
    img_list = os.listdir(os.path.join(dir, 'Images'))
    for img in img_list:
        img_path = os.path.join(dir, 'Images', img)
        if 'safe' in img:
            label = np.array([0.1] * len(object_categories)).astype(np.float32)
        else:
            anno_path = os.path.join(dir, 'Annotations', img.split('.')[0] + '.xml')
            label = read_xml_label(anno_path)
            if label.sum() < 1.:
                continue
        item = (img_path, label)
        images.append(item)
    return images


class ToSpaceBGR(object):

    def __init__(self, is_bgr):
        self.is_bgr = is_bgr

    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor


class ToRange255(object):

    def __init__(self, is_255):
        self.is_255 = is_255

    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor


class FeatureFolder(data.Dataset):

    def __init__(self, feature, label, transform=None, target_transform=None):
        self.feature = np.load(feature)
        self.label = np.load(label)

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        return self.feature[index], self.label[index]  # , path

    def __len__(self):
        return len(self.label)


class MyDatasetFolder(data.Dataset):

    def __init__(self, root, loader, extensions, transform=None, target_transform=None):

        classes = object_categories
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        samples = make_dataset_from_xml(root)
        print(len(samples))
        battery_xray_num = 0
        bottle_xray_num = 0
        firecracker_xray_num = 0
        grenade_xray_num = 0
        gun_xray_num = 0
        hammer_xray_num = 0
        knife_xray_num = 0
        scissors_xray_num = 0
        safe_xray_num = 0
        for (img, label) in samples:
            if label[0] >= 0.89:
                battery_xray_num += 1
            if label[1] >= 0.89:
                bottle_xray_num += 1
            if label[2] >= 0.89:
                firecracker_xray_num += 1
            if label[3] >= 0.89:
                grenade_xray_num += 1
            if label[4] >= 0.89:
                gun_xray_num += 1
            if label[5] >= 0.89:
                hammer_xray_num += 1
            if label[6] >= 0.89:
                knife_xray_num += 1
            if label[7] >= 0.89:
                scissors_xray_num += 1
            if label.sum() < 1.0:
                safe_xray_num += 1

        print('battery_xray_num:', battery_xray_num)
        print('bottle_xray_num:', bottle_xray_num)
        print('firecracker_xray_num:', firecracker_xray_num)
        print('grenade_xray_num:', grenade_xray_num)
        print('gun_xray_num:', gun_xray_num)
        print('hammer_xray_num:', hammer_xray_num)
        print('knife_xray_num:', knife_xray_num)
        print('scissors_xray_num:', scissors_xray_num)
        print('safe_xray_num:', safe_xray_num)

        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                                                                            "Supported extensions are: " + ",".join(
                extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.transform = transform
        self.target_transform = target_transform

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target  # , path

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.JPEG']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class MyFolder(MyDatasetFolder):

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(MyFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                       transform=transform,
                                       target_transform=target_transform)
        self.imgs = self.samples


def default_image_loader(path):
    return Image.open(path).convert('RGB')
