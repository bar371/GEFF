python train_ctl_model.py \
--config_file="configs/256_resnet50.yml" \
GPU_IDS [0] \
DATASETS.NAMES 'dukemtmcreid' \
DATASETS.ROOT_DIR '/data/mwieczorek/' \
SOLVER.IMS_PER_BATCH 16 \
TEST.IMS_PER_BATCH 128 \
SOLVER.BASE_LR 0.00035 \
OUTPUT_DIR './logs/dukemtmcreid/256_resnet50_base' \
DATALOADER.USE_RESAMPLING True \
USE_MIXED_PRECISION False \
MODEL.USE_CENTROIDS False \
REPRODUCIBLE_NUM_RUNS 1