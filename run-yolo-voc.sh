#/bin/bash

result_path=./results # darknet default

if [[ -d $result_path ]]
then
  rm -rf $result_path 
fi

mkdir -p $result_path

NTEST=-1
# NTEST=10
validation_list=./data/2007_test.txt
test_list=test.txt
if [ $NTEST != -1 ];then
  head $validation_list.template -n $NTEST > $validation_list
  head ./data/$test_list.template -n $NTEST > ./data/VOCdevkit/VOC2007/ImageSets/Main/$test_list
else
  cp $validation_list.template $validation_list
  cp ./data/$test_list.template ./data/VOCdevkit/VOC2007/ImageSets/Main/$test_list
fi

ROOT=`pwd`
sed -i -E "s#(.*.JPEG)#${ROOT}\1#g" $validation_list


###############################################
# YOLO2

# ./darknet detector valid ./cfg/voc.data cfg/yolo-voc.cfg yolo-voc.weights -gpus 0 2> $result_path/yolo.log > /dev/null
#
# rm ./data/VOCdevkit/annotation_cache -rf
# python reval_voc.py $result_path 2> /dev/null 1> mAP-yolo.log
# sed -E "s@Mean AP = (0.d+)@\1@g" mAP-yolo.log

###############################################
# Tiny-YOLO

time ./darknet detector valid ./cfg/voc.data cfg/tiny-yolo-voc.cfg tiny-yolo-voc.weights -gpus 0 2> $result_path/tiny-yolo.log > /dev/null

rm ./data/VOCdevkit/annotation_cache -rf
python reval_voc.py $result_path 2> /dev/null | tee mAP-tiny.log
