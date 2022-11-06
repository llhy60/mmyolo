import os
import argparse
from glob import glob
from os.path import join
from tqdm import tqdm
from sklearn.model_selection import train_test_split


'''
Create VOC dataset

Usage: python preprocess.py --root_path /home/data/
'''

# 根据自己的数据类别修改下面classes列表
classes = ['electric_scooter', 'person', 'bicycle', 'others']

class2id = {name:i for i, name in enumerate(classes)}


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Merge data")
    parser.add_argument ('--root_path', required=True, default=None, type=str, help="root file path")
    args = parser.parse_args()
    return args

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def merge_data(data_path_list, dst_path):
    create_dir(dst_path)
    for src_path in data_path_list:
        if os.path.exists(src_path):
            _, name = os.path.split(src_path)
            os.symlink(src_path, join(dst_path, name))
    print('[INFO] Soft links in the %s has been completed! ' % (dst_path))


def catrainlist(imgs_path, trainval_path):
    create_dir(trainval_path)
    with open(os.path.join(trainval_path, "trainval.txt"), 'w') as f:
        with tqdm(total=len(imgs_path), desc="ProgressBar") as pbar:
            for name in imgs_path:
                f.write(os.path.split(name)[1][:-4] + '\n')
                pbar.update(1)
    print(f"[INFO] Trainval.txt has been writed! Total of data: {len(imgs_path)}")
    train_images_path, val_images_path = train_test_split(imgs_path, test_size=0.15, random_state=42)
    # 生成train和val文件夹，存放各自图片，以便后续转coco格式
    merge_data(train_images_path, train_data)
    merge_data(val_images_path, val_data)
    with open(os.path.join(trainval_path, 'train.txt'), 'w') as f:
        with tqdm(total=len(train_images_path), desc="ProgressBar") as pbar:
            for name in train_images_path:
                f.write(os.path.split(name)[1][:-4] + '\n')
                pbar.update(1)
    print(f'[INFO] Train.txt has been writed! Total of data: {len(train_images_path)}')
    with open(os.path.join(trainval_path, 'val.txt'), 'w') as f:
        with tqdm(total=len(val_images_path), desc="ProgressBar") as pbar:
            for name in val_images_path:
                f.write(os.path.split(name)[1][:-4] + '\n')
                pbar.update(1)
    print(f'[INFO] Val.txt has been writed! Total of data: {len(val_images_path)}')


if __name__ == "__main__":
    args = parse_args()
    root = args.root_path
    Annotations = join(root, 'VOCdevkit2007/VOC2007/Annotations')
    JPEGImages = join(root, 'VOCdevkit2007/VOC2007/JPEGImages')
    ImageSets = join(root, 'VOCdevkit2007/VOC2007/ImageSets/Main')
    # 初始化train和val文件夹，存放各自图片，以便后续转coco格式
    train_data = join(root, 'train_data')
    val_data = join(root, 'val_data')
    
    xml_files = glob(join(root + '*/*.xml'))
    img_files = glob(join(root + '*/*.jpg'))
    assert len(img_files) == len(xml_files), '[INFO] Error: The number of pictures is inconsistent with the xml files!'
    merge_data(xml_files, Annotations)
    merge_data(img_files, JPEGImages)
    catrainlist(img_files, ImageSets)