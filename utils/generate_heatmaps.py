import numpy as np
import os
import xml.dom.minidom as xmldom
from PIL import Image
import cv2

# object_categories = ['Gun', 'Knife', 'Pliers', 'Scissors', 'Wrench']
object_categories = ['battery', 'bottle', 'firecracker', 'grenade', 'gun', 'hammer', 'knife', 'scissors']

train_dir = '../data_set/final_divide/data_train_test_split/train/'
# htmap_save_dir = '../data_set/final_dataset/train_test_split/train/Heatmaps/'
htmap_save_dir = '../data_set/final_divide/data_train_test_split/train/Ellipse_Heatmaps/'


def make_gaussian_point(height, width, sigma=45, center=None):
    x = np.arange(0, height, 1, float)
    y = np.arange(0, width, 1, float)[:, np.newaxis]
    sigma_2 = (min(height, width)**2)/8
    if center is None:
        x0 = height // 2
        y0 = width // 2
    else:
        x0 = center[1]
        y0 = center[0]
    result = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (sigma_2))
    return result / result.max()


def make_gaussian_point_on_image(size, height, width, center=None):
    x = np.arange(0, size[1], 1, float)
    y = np.arange(0, size[0], 1, float)[:, np.newaxis]

    # sigma_2 = (min(height, width) ** 2) / 8
    # sigma_2 = (((height + width)/2) ** 2) / 8
    sigma_2 = max(3000, (min(height, width) ** 2) / 8)
    if center is None:
        x0 = height // 2
        y0 = width // 2
    else:
        x0 = center[1]
        y0 = center[0]
    result = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / sigma_2)
    return result


def make_point_on_image(size, height, width, center=None):
    x = np.arange(0, size[1], 1, float)
    y = np.arange(0, size[0], 1, float)[:, np.newaxis]

    # sigma_2 = (min(height, width) ** 2) / 8
    sigma_2 = (((height + width)/2) ** 2) / 8
    # sigma_2 = max(3000, (min(height, width) ** 2) / 8)
    if center is None:
        x0 = height // 2
        y0 = width // 2
    else:
        x0 = center[1]
        y0 = center[0]
    result = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / sigma_2)
    result[result > 0.3] = 1.
    return result


def make_gaussian_rect(height, width, center=None):
    rect = np.ones((width, height))
    return rect


def make_ellipse(height, width):
    result = np.zeros((width, height))
    a = width / 2
    b = height / 2
    for i in range(width):
        for j in range(height):
            if ((((i-a)**2) / (a**2)) + (((j-b)**2) / (b**2))) <= 1:
                result[i][j] = 1.
    return result


def loader_heat_map(size, bbox_list):
    gt_ht_map = 1 + np.zeros((size[0], size[1], len(object_categories)))
    for (name, bbox) in bbox_list:
        height = bbox[3] - bbox[1]
        width = bbox[2] - bbox[0]
        # gt_ht_map[:, :, object_categories.index(name)] += 255 * make_point_on_image(gt_ht_map.shape, height, width, center=(((bbox[2] + bbox[0])//2), (bbox[3] + bbox[1])//2))
        # gt_ht_map[:, :, object_categories.index(name)] += 255 * make_gaussian_point_on_image(gt_ht_map.shape, height, width, center=(((bbox[2] + bbox[0])//2), (bbox[3] + bbox[1])//2))
        # gt_ht_map[bbox[0]:bbox[2], bbox[1]:bbox[3], object_categories.index(name)] += 255 * make_gaussian_point(height, width)
        # gt_ht_map[bbox[0]:bbox[2], bbox[1]:bbox[3], object_categories.index(name)] += 255 * make_gaussian_rect(height, width)
        gt_ht_map[bbox[0]:bbox[2], bbox[1]:bbox[3], object_categories.index(name)] += 255 * make_ellipse(height, width)
    return gt_ht_map


def read_xml_label_bbox(path):
    label = 0.1 + np.zeros(len(object_categories)).astype(np.float32)
    bbox_list = []
    domobj = xmldom.parse(path)
    elementobj = domobj.documentElement
    objects = elementobj.getElementsByTagName("object")
    for obj in objects:
        name = obj.getElementsByTagName('name')[0].childNodes[0].data
        # if name == 'gun':
        #     name = 'Gun'
        # elif name == 'knife':
        #     name = 'Knife'
        # # elif name == 'liquid':
        # #     name = 'Liquid'
        # # elif name == 'bottle':
        # #     name = 'Liquid'
        # elif name == 'scissors':
        #     name = 'Scissors'
        # elif name == 'wrench':
        #     name = 'Wrench'
        # elif name == 'pliers':
        #     name = 'Pliers'

        if name in object_categories:
            idx = object_categories.index(name)
            label[idx] = 1.0
            xmin = float(obj.getElementsByTagName('bndbox')[0].getElementsByTagName('xmin')[0].childNodes[0].data)
            ymin = float(obj.getElementsByTagName('bndbox')[0].getElementsByTagName('ymin')[0].childNodes[0].data)
            xmax = float(obj.getElementsByTagName('bndbox')[0].getElementsByTagName('xmax')[0].childNodes[0].data)
            ymax = float(obj.getElementsByTagName('bndbox')[0].getElementsByTagName('ymax')[0].childNodes[0].data)
            bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]
            bbox_list.append((name, bbox))
    return label, bbox_list


def save_np_as_png(nparray, path):
    cv2.imwrite(path, nparray.T)


def make_dataset_from_xml(dir):
    images = []
    print('[dataset] read', dir)
    img_list = os.listdir(os.path.join(dir, 'Images'))
    print(len(img_list))
    pic_num = 0
    for img in img_list:
        img_path = os.path.join(dir, 'Images', img)
        if 'safe' in img:
            gt_ht_map = 1 + np.zeros((Image.open(img_path).size[0], Image.open(img_path).size[1]))
            for idx in range(len(object_categories)):
                save_np_as_png(gt_ht_map, os.path.join(htmap_save_dir, object_categories[idx]) + '/' + img)
            save_np_as_png(gt_ht_map, os.path.join(htmap_save_dir, 'ZZ-Total') + '/' + img)
        else:
            anno_path = os.path.join(dir, 'Annotations', img.split('.')[0] + '.xml')
            label, bbox_list = read_xml_label_bbox(anno_path)

            if label.sum() < 1.:
                gt_ht_map = 1 + np.zeros((Image.open(img_path).size[0], Image.open(img_path).size[1]))
                for idx in range(len(object_categories)):
                    save_np_as_png(gt_ht_map, os.path.join(htmap_save_dir, object_categories[idx]) + '/' + img)
                save_np_as_png(gt_ht_map, os.path.join(htmap_save_dir, 'ZZ-Total') + '/' + img)
            else:
                heat_maps = loader_heat_map(Image.open(img_path).size, bbox_list)
                for idx in range(heat_maps.shape[2]):
                    save_np_as_png(heat_maps[:, :, idx], os.path.join(htmap_save_dir, object_categories[idx]) + '/' + img)
                total_heatmap = np.sum(heat_maps, axis=2)
                save_np_as_png(total_heatmap, os.path.join(htmap_save_dir, 'ZZ-Total') + '/' + img)
                pic_num += 1
            print('idx:', pic_num)
        item = (img_path, label, bbox_list)
        images.append(item)
    return images


if __name__ == "__main__":

    make_dataset_from_xml(train_dir)
