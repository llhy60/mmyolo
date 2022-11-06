import os
import sys
import json
import time
import torch
import numpy as np
from glob import glob
from typing import List
from dataclasses import dataclass


@dataclass
class CFG:
    device: str
    target: List[str]
    score_thr: float
    yolox_path: str
    config: str
    checkpoint: str

cfgs = CFG(device='cuda:0', 
          target=['electric_scooter'],
          score_thr=0.3,
          yolox_path='/project/train/src_repo/mmyolo',
          config='/project/train/src_repo/mmyolo/configs/yolov5/cvmart/yolov5_s-v61_syncbn_fast_1xb4-300e_baseline.py',
          checkpoint='/project/train/src_repo/mmyolo/work_dirs/yolov5_s-v61_syncbn_fast_1xb4-300e_baseline/best_coco/*.pth'
          
         )
sys.path.insert(1, cfgs.yolox_path)
from mmdet.apis import (async_inference_detector, inference_detector, init_detector)
from mmyolo.utils import register_all_modules

def init():
    # register all modules in mmdet into the registries
    register_all_modules()
    # Initialize
    checkpoint = glob(cfgs.checkpoint)[0]
    model = init_detector(cfgs.config, checkpoint, device=cfgs.device)
    return model

@torch.no_grad()
def process_image(model, input_image=None, args=None, **kwargs):
    classes = list(model.dataset_meta['CLASSES'])
    data_sample = inference_detector(model, input_image)
    
    data_sample = data_sample.cpu()
    pred_instances = data_sample.pred_instances
    pred_instances = pred_instances[pred_instances.scores > cfgs.score_thr]
    bboxes = pred_instances.bboxes
    labels = pred_instances.labels
    scores = pred_instances.scores
    
    fake_result = {}

    fake_result["algorithm_data"] = {
       "is_alert": False,
       "target_count": 0,
       "target_info": []
   }
    fake_result["model_data"] = {"objects": []}
    # Process detections
    cnt = 0
    for cls, bbox, conf in zip(labels, bboxes, scores):
        name, bbox, conf = classes[cls], bbox[:4], round(float(conf),4)
        x1, y1, x2, y2 = bbox.astype(np.int32)
        fake_result["model_data"]['objects'].append({
                "x": x1,
                "y": y1,
                "height": y2 - y1,
                "width": x2 - x1,
                "confidence": conf,
                "name": name
            }
        )
        if name in cfgs.target:
            cnt += 1
            fake_result["algorithm_data"]["target_info"].append({
                "x": x1,
                "y": y1,
                "height": y2 - y1,
                "width": x2 - x1,
                "confidence": conf,
                "name": name
            }
        )
    if cnt:
        fake_result["algorithm_data"]["is_alert"] = True
        fake_result["algorithm_data"]["target_count"] = cnt
    return json.dumps(fake_result, indent = 4)


if __name__ == '__main__':
    # Test API
    image_names = glob('/home/data/*/*.jpg')[:10]
    predictor = init()
    s = 0
    for image_name in image_names:
        t1 = time.time()
        res = process_image(predictor, image_name)
        print(res)
        t2 = time.time()
        s += t2 - t1
    print(1/(s/100))