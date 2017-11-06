#!/bin/bash

HOST=
# HOST=arc-titan:
# HOST=arc-titan@147.46.125.108:
DARKNET_DIR=$HOST~/darknet-gemm
REMOTE=0

# YOLO
wget https://pjreddie.com/media/files/yolo-voc.weights
wget https://pjreddie.com/media/files/tiny-yolo-voc.weights

mkdir -p ~/ssd
scp -r $HOST~/ssd/VOCdevkit ~/ssd/
ln -s ~/ssd/VOCdevkit ./data/


# Imagenet
# scp -r $DARKNET_DIR/data/imgs ./data/
# scp -r $DARKNET_DIR/data/val ./data/
scp -r $DARKNET_DIR/data/labelled ./data/

# Download darknet refernce (28 MB)
wget http://pjreddie.com/media/files/darknet.weights


# git
# data/imagenet_label.sh
# data/imagenet.labels.list
# data/imagenet.shortnames.list
# data/inet.val.list
