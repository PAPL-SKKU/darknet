result_path=./results # darknet default

if [[ -d $result_path ]]
then
  rm -rf $result_path 
fi

NTEST=2000
validation_list=./data/2007_test.txt
head $validation_list.template -n $NTEST > $validation_list

ROOT=`pwd`
sed -i -E "s#(.*.JPEG)#${ROOT}\1#g" $validation_list


# mkdir -p $result_path

# def
# (time ./darknet detector valid ./cfg/voc.data cfg/yolo-voc.cfg yolo-voc.weights -gpus 0) > $result_path/yolo-voc-2007.log 2>&1 &
# pid=$!
# tail -f $result_path/yolo-voc-2007.log
# wait $pid
# kill $pid
#
# rm ./data/VOCdevkit/annotation_cache -rf
# python reval_voc.py $result_path > $result_path/mAP-def.log 2>&1
# mv $result_path $result_path\_def.voc.2007.val


###############################################

mkdir -p $result_path

# tiny
(time ./darknet detector valid ./cfg/voc.data cfg/tiny-yolo-voc.cfg tiny-yolo-voc.weights -gpus 0) > $result_path/tiny-yolo-voc-2007.log 2>&1 &
pid=$!
tail -f $result_path/tiny-yolo-voc-2007.log
wait $pid
kill $pid
# 
rm ./data/VOCdevkit/annotation_cache -rf
python reval_voc.py $result_path > $result_path/mAP-tiny.log 2>&1
mv $result_path $result_path\_tiny.voc.2007.val
