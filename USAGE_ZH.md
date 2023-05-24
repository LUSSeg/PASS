# PASS的使用
[安装](#1)

[训练](#2)

[评估](#3)

<div id="1"></div>

# 安装
PASS模型的环境要求：

* System: **Linux**(e.g. Ubuntu/CentOS/Arch), **macOS**, or **Windows Subsystem of Linux (WSL)**
* Python version >= 3.7
* CPU compiler (require at least one of the following)
    * g++ (>=5.4.0)
    * clang (>=8.0)

## 第一步: 安装计图
计图的安装可以参考以下文档[Jittor install](https://github.com/Jittor/jittor#install)

## 第二步: 安装依赖
```shell
python -m pip install scikit-learn
python -m pip install pandas
python -m pip install munkres
python -m pip install tqdm
python -m pip install pillow
python -m pip install opencv-python
python -m pip install faiss-gpu
```



# 训练
我们提供了训练脚本[luss50_pass_jt.sh](scripts/luss50_pass_jt.sh)，下文解释了训练脚本中每个部分的功能。

## 步骤1：无监督的表征学习
首先进行非对比像素到像素表示对齐和深度到浅层监督进行预训练。
```shell
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
```

## 步骤2：使用像素注意力生成像素标签
### 步骤2.1：微调像素注意力
在这一部分中，您应该将"--pretrained"设置为步骤1中获得的预训练权重。
```shell

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
```

### 步骤2.2：聚类
将"--pretrained"设置为步骤2.1中获得的预训练权重。

在本部分中，每个聚类的中心将生成并保存在`${DUMP_PATH_FINETUNE}/cluster/centroids.npy`中。
```shell
CUDA_VISIBLE_DEVICES=${CUDA} python cluster.py -a ${ARCH} \
--pretrained ${DUMP_PATH_FINETUNE}/checkpoint.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_FINETUNE} \
-c ${NUM_CLASSES}
```

### 步骤2.3：选择生成伪标签的阈值。
“centroid”是一个保存聚类中心的npy文件。并且“pretrained”应该被设置为在步骤2.1中获得的预训练权重。

在此步骤中，将显示不同阈值下的val mIoUs。
```shell
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
```

### 步骤2.4：为训练集生成伪标签
请将“t”设置为在步骤2.3中获得的最佳阈值。
```shell
CUDA_VISIBLE_DEVICES=${CUDA} mpirun -np ${N_GPU} --allow-run-as-root python inference_pixel_attention.py -a ${ARCH} \
--pretrained ${DUMP_PATH_FINETUNE}/checkpoint.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_FINETUNE} \
-c ${NUM_CLASSES} \
--mode train \
--centroid ${DUMP_PATH_FINETUNE}/cluster/centroids.npy \
-t 0.26
```

## 步骤3：使用像素级伪标签进行微调
请将“pseudo_path”设置为保存步骤2.4中生成的伪标签的路径。
```shell
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
```

## 步骤4：推理
如果想评估测试集的性能，请将“mode”设置为“test”，从而生成对应的标签。
```shell
CUDA_VISIBLE_DEVICES=${CUDA} python inference.py -a ${ARCH} \
--pretrained ${DUMP_PATH_SEG}/checkpoint.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_SEG} \
-c ${NUM_CLASSES} \
--mode validation \
--match_file ${DUMP_PATH_SEG}/validation/match.json
```

<div id="2"></div>

# 评估

## Fully unsupervised protocol
```shell
CUDA_VISIBLE_DEVICES=${CUDA} python evaluator.py \
--predict_path ${DUMP_PATH_SEG} \
--data_path ${IMAGENETS} \
-c ${NUM_CLASSES} \
--mode validation
```
