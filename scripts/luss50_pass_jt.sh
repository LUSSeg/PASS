CUDA='0,1,2,3'
N_GPU=4
BATCH=64
DATA=/data/ImageNetS/ImageNetS50
IMAGENETS=/data/ImageNetS/ImageNetS50

DUMP_PATH=./weights/pass50
DUMP_PATH_FINETUNE=${DUMP_PATH}/pixel_attention
DUMP_PATH_SEG=${DUMP_PATH}/pixel_finetuning
QUEUE_LENGTH=2048
QUEUE_LENGTH_PIXELATT=3840
HIDDEN_DIM=512
NUM_PROTOTYPE=500
ARCH=resnet18
NUM_CLASSES=50
EPOCH=200
EPOCH_PIXELATT=20
EPOCH_SEG=20
FREEZE_PROTOTYPES=1001
FREEZE_PROTOTYPES_PIXELATT=0

mkdir -p ${DUMP_PATH_FINETUNE}
mkdir -p ${DUMP_PATH_SEG}

CUDA_VISIBLE_DEVICES=${CUDA} mpirun -np ${N_GPU} --allow-run-as-root python main_pretrain.py \
--arch ${ARCH} \
--data_path ${DATA}/train \
--dump_path ${DUMP_PATH} \
--nmb_crops 2 6 \
--size_crops 224 96 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--crops_for_assign 0 1 \
--temperature 0.1 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--feat_dim 128 \
--hidden_mlp ${HIDDEN_DIM} \
--nmb_prototypes ${NUM_PROTOTYPE} \
--queue_length ${QUEUE_LENGTH} \
--epoch_queue_starts 15 \
--epochs ${EPOCH} \
--batch_size ${BATCH} \
--base_lr 0.6 \
--final_lr 0.0006  \
--freeze_prototypes_niters ${FREEZE_PROTOTYPES} \
--wd 0.000001 \
--warmup_epochs 0 \
--workers 10 \
--seed 31 \
--shallow 3 \
--weights 1 1

CUDA_VISIBLE_DEVICES=${CUDA} mpirun -np ${N_GPU} --allow-run-as-root python main_pixel_attention.py \
--arch ${ARCH} \
--data_path ${IMAGENETS}/train \
--dump_path ${DUMP_PATH_FINETUNE} \
--nmb_crops 2 \
--size_crops 224 \
--min_scale_crops 0.08 \
--max_scale_crops 1. \
--crops_for_assign 0 1 \
--temperature 0.1 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--feat_dim 128 \
--hidden_mlp ${HIDDEN_DIM} \
--nmb_prototypes ${NUM_PROTOTYPE} \
--queue_length ${QUEUE_LENGTH_PIXELATT} \
--epoch_queue_starts 0 \
--epochs ${EPOCH_PIXELATT} \
--batch_size ${BATCH} \
--base_lr 6.0 \
--final_lr 0.0006  \
--freeze_prototypes_niters ${FREEZE_PROTOTYPES_PIXELATT} \
--wd 0.000001 \
--warmup_epochs 0 \
--workers 10 \
--seed 31 \
--pretrained ${DUMP_PATH}/checkpoint.pth.tar

CUDA_VISIBLE_DEVICES=${CUDA} python cluster.py -a ${ARCH} \
--pretrained ${DUMP_PATH_FINETUNE}/checkpoint.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_FINETUNE} \
-c ${NUM_CLASSES}

### Evaluating the pseudo labels on the validation set.
CUDA_VISIBLE_DEVICES=${CUDA} python inference_pixel_attention.py -a ${ARCH} \
--pretrained ${DUMP_PATH_FINETUNE}/checkpoint.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_FINETUNE} \
-c ${NUM_CLASSES} \
--mode validation \
--test \
--centroid ${DUMP_PATH_FINETUNE}/cluster/centroids.npy

CUDA_VISIBLE_DEVICES=${CUDA} python evaluator.py \
--predict_path ${DUMP_PATH_FINETUNE} \
--data_path ${IMAGENETS} \
-c ${NUM_CLASSES} \
--mode validation \
--curve \
--min 20 \
--max 80

CUDA_VISIBLE_DEVICES=${CUDA} mpirun -np ${N_GPU} --allow-run-as-root python inference_pixel_attention.py -a ${ARCH} \
--pretrained ${DUMP_PATH_FINETUNE}/checkpoint.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_FINETUNE} \
-c ${NUM_CLASSES} \
--mode train \
--centroid ${DUMP_PATH_FINETUNE}/cluster/centroids.npy \
-t 0.26

CUDA_VISIBLE_DEVICES=${CUDA} mpirun -np ${N_GPU} --allow-run-as-root python main_pixel_finetuning.py \
--arch ${ARCH} \
--data_path ${DATA}/train \
--dump_path ${DUMP_PATH_SEG} \
--epochs ${EPOCH_SEG} \
--batch_size ${BATCH} \
--base_lr 0.6 \
--final_lr 0.0006 \
--wd 0.000001 \
--warmup_epochs 0 \
--workers 8 \
--num_classes ${NUM_CLASSES} \
--pseudo_path ${DUMP_PATH_FINETUNE}/train \
--pretrained ${DUMP_PATH}/checkpoint.pth.tar

CUDA_VISIBLE_DEVICES=${CUDA} python inference.py -a ${ARCH} \
--pretrained ${DUMP_PATH_SEG}/checkpoint.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_SEG} \
-c ${NUM_CLASSES} \
--mode validation \
--match_file ${DUMP_PATH_SEG}/validation/match.json

CUDA_VISIBLE_DEVICES=${CUDA} python evaluator.py \
--predict_path ${DUMP_PATH_SEG} \
--data_path ${IMAGENETS} \
-c ${NUM_CLASSES} \
--mode validation