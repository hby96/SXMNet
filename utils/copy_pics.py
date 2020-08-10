import os, shutil

root_path = '/Users/hby/Documents/XJTU/场景生成与离线测试/Projects/Mulit-Branch-Recognition/data_set/Xray-Total/OP2'

dest_save_path = '/Users/hby/Documents/XJTU/场景生成与离线测试/Projects/Mulit-Branch-Recognition/data_set/Xray-Total/total'

for root, dirs, files in os.walk(root_path):
    for dir in dirs:
        print(dir)
        new_root_path = os.path.join(root, dir)
        for _, _, pics in os.walk(new_root_path):
            for pic in pics:
                ori_pic_root = os.path.join(new_root_path, pic)
                new_pic_root = os.path.join(os.path.join(dest_save_path, dir), root_path.split('/')[-1]+'_'+pic)
                shutil.copyfile(ori_pic_root, new_pic_root)
    print('have done!')
    break

