import torch
import torchvision.transforms as transforms
import data_set.functional as F
from PIL import Image
import collections
import cv2
import numpy as np
import random
import math
import numbers


class RandomGaussianBlur(object):
    def __init__(self, kernel_scale=[1, 3, 5, 7, 9], sigma_scale=(0.5, 2)):
        self.kernel_scale = kernel_scale
        self.sigma_scale = sigma_scale

    def __call__(self, img):
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        choice = np.random.randint(10)
        if choice <= 1:
            kernel_size = self.kernel_scale[np.random.randint(5)]
            sigma = (self.sigma_scale[1] - self.sigma_scale[0]) * np.random.rand(1)[0] + self.sigma_scale[0]
            out = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
        else:
            out = img
        img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        return img


class LiubaiResize(object):
    def __init__(self, size, interpolation=1):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def liubairesize(self, img):
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        new_img = cv2.cvtColor(np.zeros((self.size, self.size, 3), dtype=np.uint8), cv2.COLOR_RGB2BGR)

        scale = min(self.size / img.shape[0], self.size / img.shape[1])
        img = cv2.resize(img, (round(img.shape[1] * scale), round(img.shape[0] * scale)), interpolation=self.interpolation)

        if img.shape[0] == self.size:
            new_img[:, (self.size // 2 - img.shape[1] // 2):(self.size // 2 - img.shape[1] // 2 + img.shape[1]),
            :] = img
        else:
            new_img[(self.size // 2 - img.shape[0] // 2):(self.size // 2 - img.shape[0] // 2 + img.shape[0]), :,
            :] = img

        img = Image.fromarray(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))

        return img

    def __call__(self, img):

        # img.save("/home/hubenyi/Semi-Supervised-FG-series/SSSS/" + 'ori' + ".png", "PNG")

        img = self.liubairesize(img)

        # img.save("/home/hubenyi/Semi-Supervised-FG-series/SSSS/" + 'yinhua' + ".png", "PNG")

        return img

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class YinhuaResize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def yinhuaresize(self, img):
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        new_img = cv2.cvtColor(np.zeros((self.size, self.size, 3), dtype=np.uint8), cv2.COLOR_RGB2BGR)

        scale = min(self.size / img.shape[0], self.size / img.shape[1])
        img = cv2.resize(img, (round(img.shape[1] * scale), round(img.shape[0] * scale)))

        if img.shape[0] == self.size:
            yinhua_num = self.size // img.shape[1]
            for i in range(yinhua_num):
                new_img[:, i * img.shape[1]:(i + 1) * img.shape[1], :] = img
            res = new_img.shape[1] - yinhua_num * img.shape[1]
            new_img[:, yinhua_num * img.shape[1]:, :] = img[:, 0:res, :]
        else:
            yinhua_num = self.size // img.shape[0]
            for i in range(yinhua_num):
                new_img[i * img.shape[0]:(i + 1) * img.shape[0], :, :] = img
            res = new_img.shape[0] - yinhua_num * img.shape[0]
            new_img[yinhua_num * img.shape[0]:, :, :] = img[0:res, :, :]

        img = Image.fromarray(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))

        return img

    def __call__(self, img):

        # img.save("/home/hubenyi/Semi-Supervised-FG-series/SSSS/" + 'ori' + ".png", "PNG")

        img = self.yinhuaresize(img)

        # img.save("/home/hubenyi/Semi-Supervised-FG-series/SSSS/" + 'yinhua' + ".png", "PNG")

        return img

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class MiddleYinhuaResize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def middle_yinhua_resize(self, img):
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        new_img = cv2.cvtColor(np.zeros((self.size, self.size, 3), dtype=np.uint8), cv2.COLOR_RGB2BGR)
        new_img[:, :, 0] = int(255 * 0.406)
        new_img[:, :, 1] = int(255 * 0.456)
        new_img[:, :, 2] = int(255 * 0.485)

        scale = min(self.size / img.shape[0], self.size / img.shape[1])
        img = cv2.resize(img, (round(img.shape[1] * scale), round(img.shape[0] * scale)))

        if img.shape[0] == self.size:
            middle_pos = self.size // 2
            left_pos = (middle_pos - img.shape[1] // 2)
            right_pos = (left_pos + img.shape[1])

            new_img[:, left_pos:right_pos, :] = img

            left_yinhua_num = left_pos // img.shape[1]
            for i in range(left_yinhua_num):
                new_img[:, left_pos - (i + 1) * img.shape[1]:left_pos - i * img.shape[1], :] = img
            res = left_pos - left_yinhua_num * img.shape[1]
            new_img[:, 0:res, :] = img[:, img.shape[1]-res:img.shape[1], :]

            right_yinhua_num = (self.size - right_pos) // img.shape[1]

            for i in range(right_yinhua_num):
                new_img[:, right_pos + i * img.shape[1]:right_pos + (i + 1) * img.shape[1], :] = img
            res = self.size - right_pos - right_yinhua_num * img.shape[1]
            new_img[:, self.size - res:self.size, :] = img[:, 0:res, :]
        else:
            middle_pos = self.size // 2
            up_pos = middle_pos - img.shape[0] // 2
            down_pos = up_pos + img.shape[0]
            new_img[up_pos:down_pos, :, :] = img

            up_yinhua_num = up_pos // img.shape[0]
            for i in range(up_yinhua_num):
                new_img[up_pos - (i + 1) * img.shape[0]:up_pos - i * img.shape[0], :, :] = img
            res = up_pos - up_yinhua_num * img.shape[0]
            new_img[0:res, :, :] = img[img.shape[0] - res:img.shape[0], :, :]

            down_yinhua_num = (self.size - down_pos) // img.shape[0]
            for i in range(down_yinhua_num):
                new_img[down_pos + i * img.shape[0]:down_pos + (i + 1) * img.shape[0], :, :] = img
            res = self.size - down_pos - down_yinhua_num * img.shape[0]
            new_img[self.size-res:self.size, :, :] = img[0:res, :, :]

        img = Image.fromarray(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))

        return img

    def __call__(self, img):

        # img.save("/home/hubenyi/Semi-Supervised-FG-series/SSSS/" + 'ori' + ".png", "PNG")

        img = self.middle_yinhua_resize(img)

        # img.save("/home/hubenyi/Semi-Supervised-FG-series/SSSS/" + 'yinhua' + ".png", "PNG")

        return img

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class YinhuaRandomResizedCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3./4., 4./3.), interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

            # Fallback
        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w

    def yinhuaresize(self, img):
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        new_img = cv2.cvtColor(np.zeros((self.size, self.size, 3), dtype=np.uint8), cv2.COLOR_RGB2BGR)

        scale = min(self.size / img.shape[0], self.size / img.shape[1])
        img = cv2.resize(img, (round(img.shape[1] * scale), round(img.shape[0] * scale)))

        if img.shape[0] == self.size:
            yinhua_num = self.size // img.shape[1]
            for i in range(yinhua_num):
                new_img[:, i * img.shape[1]:(i + 1) * img.shape[1], :] = img
            res = new_img.shape[1] - yinhua_num * img.shape[1]
            new_img[:, yinhua_num * img.shape[1]:, :] = img[:, 0:res, :]
        else:
            yinhua_num = self.size // img.shape[0]
            for i in range(yinhua_num):
                new_img[i * img.shape[0]:(i + 1) * img.shape[0], :, :] = img
            res = new_img.shape[0] - yinhua_num * img.shape[0]
            new_img[yinhua_num * img.shape[0]:, :, :] = img[0:res, :, :]

        img = Image.fromarray(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))

        return img

    def __call__(self, img):

        # img.save("/home/hubenyi/Semi-Supervised-FG-series/SSSS/" + 'ori' + ".png", "PNG")

        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        crop_img = F.crop(img, i, j, h, w)
        img = self.yinhuaresize(crop_img)

        # img.save("/home/hubenyi/Semi-Supervised-FG-series/SSSS/" + 'yinhua' + ".png", "PNG")

        return img

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class Randomswap(object):
    def __init__(self, size, radius=1):
        self.size = size
        self.radius = radius
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img):
        choice = np.random.randint(10)
        if choice <= 1:
            # size = self.size_scale[np.random.randint(2)]
            out = F.swap(img, self.size)
        else:
            out = img

        # add_edge_img = copy_edge(img, swap_img, self.size)

        # add_edge_img = add_edge_img.filter(MyGaussianBlur(radius=self.radius))
        # swap_img = swap_img.filter(MyGaussianBlur(radius=self.radius))

        return out


    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
