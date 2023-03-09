#!/bin/bash
#SBATCH -n 1
#SBATCH -t 0-96
#SBATCH -c 32
#SBATCH --mem=96G
#SBATCH --partition=powernv
#SBATCH --gres=gpu:1
#SBATCH --job-name chunking

source ~/.bashrc;
srun /home/s/samuel-reiter/octoSleep/detectron/detectronOnOctopus.py \
--video /bucket/ReiterU/octopus/sleep_rem_cycle/oct_06/oct_06_day1.avi \
--labels /home/sam/bucket/annotations/octopus_sleep \
--config /home/s/samuel-reiter/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
--outputPath /work/ReiterU/





