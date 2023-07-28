CUDA='0'
BATCH=64
DATA=/home/zy/datasets/ImageNetS50
IMAGENETS=/home/zy/datasets/ImageNetS50

DUMP_PATH=./weights/pass50
DUMP_PATH_FINETUNE=${DUMP_PATH}/pixel_attention
DUMP_PATH_SEG=${DUMP_PATH}/pixel_finetuning
QUEUE_LENGTH=2048
QUEUE_LENGTH_PIXELATT=3840
HIDDEN_DIM=512
NUM_PROTOTYPE=500
ARCH=resnet18
NUM_CLASSES=50
EPOCH=2
EPOCH_PIXELATT=2
EPOCH_SEG=2
FREEZE_PROTOTYPES=1001
FREEZE_PROTOTYPES_PIXELATT=0

mkdir -p ${DUMP_PATH_FINETUNE}
mkdir -p ${DUMP_PATH_SEG}

CUDA_VISIBLE_DEVICES=${CUDA} python inference.py -a ${ARCH} \
--pretrained ${DUMP_PATH_SEG}/checkpoint.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_SEG} \
-c ${NUM_CLASSES} \
--mode test \
--match_file ${DUMP_PATH_SEG}/validation/match.json

# readme
# 1. download test data to ImageNetS50
# wget https://cloud.tsinghua.edu.cn/f/166f0b12f2eb4399b997/\?dl\=1 && mv index.html\?dl=1 test.zip && unzip test.zip && mv img test

# 2. mv test data to ImageNetS50
# mv test 'ImageNetS50 path'

# 3. generate test.zip
# sh scripts/luss50_pass/8_test.sh
# path in weights/pass50/pixel_finetuning/test.zip
# submit
