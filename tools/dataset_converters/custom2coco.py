import os
import shutil
import random
import os.path as osp

import mmcv
import mmengine


def split_train_val(ann_file, image_prefix, split_rate=0.95):
    data_infos = mmengine.load(ann_file)
    data_nums = len(data_infos)
    data_nums_filter = 0
    data_infos_filter = []
    for info in data_infos:
        if len(info['region']) != 0:
            data_nums_filter += 1
            data_infos_filter.append(info)
    print(f"[INFO] 过滤没有标签的图片, 一共有{data_nums - data_nums_filter}张.")
    random.shuffle(data_infos_filter)
    train_data_infos = data_infos_filter[:int(split_rate * data_nums_filter)]
    val_data_infos   = data_infos_filter[int(split_rate * data_nums_filter):]
    print(f"[INFO] 训练集有{len(train_data_infos)}张图片.")
    print(f"[INFO] 验证集有{len(val_data_infos)}张图片.")

    val_imgs_path = osp.join(image_prefix, 'val')
    train_imgs_path = osp.join(image_prefix, 'train')
    val_label_path = osp.join(image_prefix, 'label_val.json')
    train_label_path = osp.join(image_prefix, 'label_train.json')
    orig_imgs_path = osp.join(image_prefix, 'orig_data/images')
    if not osp.exists(val_imgs_path):
        os.makedirs(val_imgs_path)
    if not osp.exists(train_imgs_path):
        os.makedirs(train_imgs_path)

    mmengine.dump(val_data_infos, val_label_path)
    mmengine.dump(train_data_infos, train_label_path)

    for data_info in val_data_infos:
        file_name = data_info['id']
        src_path = osp.join(orig_imgs_path, file_name)
        dst_path = osp.join(val_imgs_path, file_name)
        shutil.copy(src_path, dst_path)

    for data_info in train_data_infos:
        file_name = data_info['id']
        src_path = osp.join(orig_imgs_path, file_name)
        dst_path = osp.join(train_imgs_path, file_name)
        shutil.copy(src_path, dst_path)


def convert_custom_to_coco(ann_file, out_file, image_prefix):
    '''此函数用于转标注文件(.json)为[{'id': 'train_8359.jpg', 'region': [[130.0, 72.0, 207.0, 88.0]]}, ...]的数据集.
        Args:
            ann_file(str): 标注文件的地址
            out_file(str): 输出coco格式的标注文件的地址
            image_prefix(str): 图像数据的路径
        Returns:
            None
    '''
    data_infos = mmengine.load(ann_file)

    annotations = []
    images = []
    obj_count = 0
    for idx, v in enumerate(mmengine.track_iter_progress(data_infos)):
        filename = v['id']
        img_path = osp.join(image_prefix, filename)
        height, width = mmcv.imread(img_path).shape[:2]

        images.append(
            dict(id=idx, file_name=filename, height=height, width=width))

        for obj in v['region']:
            x1, y1, x2, y2 = obj
            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=0,
                bbox=[x1, y1, x2 - x1, y2 - y1],
                area=(x2 - x1) * (y2 - y1),
                segmentation=[[x1, y1, x2, y1, x2, y2, x1, y2]],
                iscrowd=0)
            annotations.append(data_anno)
            obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{
            'id': 0,
            'name': 'forge'
        }])
    mmengine.dump(coco_format_json, out_file)


if __name__ == '__main__':
#   ann_file = '/home/liulinhai-ghq/datasets/forge_data/orig_data/label_train.json'
#   image_prefix = '/home/liulinhai-ghq/datasets/forge_data/'
#   split_train_val(ann_file, image_prefix, split_rate=0.96)

#   convert_custom_to_coco('/home/liulinhai-ghq/datasets/forge_data/label_train.json',
#                           '/home/liulinhai-ghq/datasets/forge_data/train.json', 
#                           '/home/liulinhai-ghq/datasets/forge_data/train/')
#   convert_custom_to_coco('/home/liulinhai-ghq/datasets/forge_data/label_val.json',
#                           '/home/liulinhai-ghq/datasets/forge_data/val.json', 
#                           '/home/liulinhai-ghq/datasets/forge_data/val/')
    convert_custom_to_coco(ann_file='/Users/llhy/Downloads/submit.json',
                           out_file='/Users/llhy/Downloads/datasets/tamper_detection/val.json',
                           image_prefix='/Users/llhy/Downloads/datasets/tamper_detection/val_images/')