#!/bin/bash

validation_list=./data/imagenet1k.valid.list
ROOT=`pwd`

NTEST=100
echo $ROOT
head $validation_list.template -n $NTEST > $validation_list
sed -i -E "s#(.*.JPEG)#${ROOT}\1#g" $validation_list

./darknet classifier valid cfg/imagenet1k.data cfg/darknet.cfg darknet.weights -gpus 0 > inet-$NTEST

./scripts/avg_top1.py inet-$NTEST
