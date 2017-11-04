#!/bin/bash

HOST=arc-titan:
# HOST=arc-titan@147.46.125.108:
DARKNET_DIR=$HOST~/darknet-gemm

# YOLO
wget https://pjreddie.com/media/files/yolo-voc.weights
wget https://pjreddie.com/media/files/tiny-yolo-voc.weights

mkdir -p ~/ssd 
scp -r $HOST~/VOCdevkit ~/ssd/
ln -s ~/ssd/VOCdevkit ./data/


# Imagenet
# scp -r $DARKNET_DIR/data/imgs ./data/
# scp -r $DARKNET_DIR/data/val ./data/
scp -r $DARKNET_DIR/data/labelled ./data/

cd data


# git
# data/imagenet_label.sh
# data/imagenet.labels.list
# data/imagenet.shortnames.list
# data/inet.val.list
