#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 10:10:25 2020

@author: sam
"""

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger
setup_logger()
import os
import cv2
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.structures import BoxMode
import codecs, json 
from uuid import uuid1
import matplotlib.pyplot as plt

def my_dataset_function(dset):
    return dset

annotationFolder='/home/sam/bucket/annotations/quadCam/'
segmentation_titles=['octopus']


currDataset_name= str(uuid1())
#load existing dataset
obj_text = codecs.open(annotationFolder + 'annotations_coco.json', 'r', encoding='utf-8').read()
dataset = json.loads(obj_text)
#boxmode isn't preserved in json file
for index,i in enumerate(dataset):
    for obj in range(len(i['annotations'])):
        dataset[index]['annotations'][obj]['bbox_mode']=BoxMode.XYWH_ABS
    dataset[index]['file_name']=dataset[index]['file_name'] 
        
        
#train test split
from sklearn.model_selection import train_test_split
TEST_SIZE = 0.1
RANDOM_SEED = 99
X_train, X_test= train_test_split(dataset, test_size=TEST_SIZE, random_state=RANDOM_SEED)
DatasetCatalog.register(currDataset_name + '_train', lambda: my_dataset_function(X_train))
DatasetCatalog.register(currDataset_name + '_test', lambda: my_dataset_function(X_test))


meta=MetadataCatalog.get(currDataset_name + '_train').set(thing_classes=segmentation_titles)


# # #train the model
cfg = get_cfg()
cfg.merge_from_file("/home/sam/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = (currDataset_name + '_train',)
cfg.DATASETS.TEST = (currDataset_name + '_test',)   # no metrics implemented for this dataset
cfg.OUTPUT_DIR = annotationFolder + 'output'
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 3000
cfg.SOLVER.STEPS = (1000, 1500)
cfg.SOLVER.GAMMA = 0.05
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128#64   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(segmentation_titles)+1  
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85   # set the testing threshold for this model
DatasetCatalog.register(str(uuid1()), lambda: my_dataset_function(dataset))
cfg.DATASETS.TEST = (currDataset_name + '_test', )
predictor = DefaultPredictor(cfg)

#save test images
os.makedirs(annotationFolder + "annotated_results", exist_ok=True)
test_image_paths=[]
for i in X_test:
    test_image_paths.append(i['file_name'])

for num,imageName in enumerate(test_image_paths):
    file_path = imageName
    im = cv2.imread(file_path)
    outputs = predictor(im)
    v = Visualizer(
      im[:, :, ::-1],
      metadata=meta, 
      scale=1., 
      instance_mode=ColorMode.IMAGE
    )
    instances = outputs["instances"].to("cpu")
    v = v.draw_instance_predictions(instances)
    result = v.get_image()[:, :, ::-1]
    write_res = cv2.imwrite(annotationFolder + 'annotated_results/' + str(num) + '.png', result)


    v = Visualizer(im[:, :, ::-1],
                    metadata=meta, 
                    scale=1, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(v.get_image())




