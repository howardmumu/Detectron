EXAMPLE=examples/dogs : where we are going to store LMDB
DATA=data/dogs/dogs_data : folder with dogs train.txt, val.txt
TRAIN_DATA_ROOT : folder with train images
VAL_DATA_ROOT : folder with test images (with script above it’s same folder)
RESIZE=false

GLOG_logtostderr=1 $TOOLS/convert_imageset \
 — resize_height=$RESIZE_HEIGHT \
— resize_width=$RESIZE_WIDTH \
— shuffle \
$TRAIN_DATA_ROOT \
$DATA/train.txt \
$EXAMPLE/dogs_train_lmdb
echo “Creating val lmdb…”
GLOG_logtostderr=1 $TOOLS/convert_imageset \
— resize_height=$RESIZE_HEIGHT \
— resize_width=$RESIZE_WIDTH \
— shuffle \
$VAL_DATA_ROOT \
$DATA/val.txt \
$EXAMPLE/dogs_val_lmdb