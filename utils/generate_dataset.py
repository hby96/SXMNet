import os, shutil
import random

total_root_path = '/Users/hby/Documents/XJTU/场景生成与离线测试/Projects/Mulit-Branch-Recognition/data_set/Xray-Total/OP44'

split_path = '/Users/hby/Documents/XJTU/场景生成与离线测试/Projects/Mulit-Branch-Recognition/data_set/Xray-Total/train_test_split/1_1'

ratio = 1 # train : test

train_num_ratio = 1 / (ratio + 1)

for root, dirs, files in os.walk(total_root_path):
    for dir in dirs:
        print(dir)
        new_root_path = os.path.join(root, dir)
        for _, _, pics in os.walk(new_root_path):
            class_length = len(pics)
            train_num = int(class_length * train_num_ratio)
            random.shuffle(pics)
            train_list = pics[:train_num]
            test_list = pics[train_num:]
            for pic in train_list:
                ori_pic_root = os.path.join(new_root_path, pic)
                new_pic_root = os.path.join(os.path.join(split_path+'/train', dir),
                                            total_root_path.split('/')[-1] + '_' + pic)
                shutil.copyfile(ori_pic_root, new_pic_root)
            for pic in test_list:
                ori_pic_root = os.path.join(new_root_path, pic)
                new_pic_root = os.path.join(os.path.join(split_path + '/test', dir),
                                            total_root_path.split('/')[-1] + '_' + pic)
                shutil.copyfile(ori_pic_root, new_pic_root)
    print('have done!')
    break

