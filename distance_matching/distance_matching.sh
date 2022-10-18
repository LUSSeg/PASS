ARCH=$1
PRETRAINED=$2
SAVE=$3
DATA=$4
NUMCLASSES=$5
MODE=$6

python distance_matching/get_instance.py --data_path ${DATA} \
--dump_path ${SAVE} \
--arch ${ARCH} \
--pretrained ${PRETRAINED} \
--num_classes ${NUMCLASSES}

python distance_matching/inference_distance_matching.py --data_path ${DATA} \
--dump_path ${SAVE} \
--arch ${ARCH} \
--pretrained ${PRETRAINED} \
--num_classes ${NUMCLASSES} \
--mode ${MODE}

if [ ${MODE} = "validation" ]; then
python evaluator.py --predict_path ${SAVE} \
--data_path ${DATA} \
-c ${NUMCLASSES} \
--mode ${MODE}
fi