# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#!/bin/bash
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --dependency=singleton
#SBATCH --job-name=gdn2_0.4B_tsz512x4k_15B_edu


NAME="512x4k_15B_swa_gdn2_1.3B"
MODEL='swa_gdn2_1.3B'
CONFIG='tsz1024x4k_100B'
EVAL_ITERS=15 # number of evaluation iterations
MICRO_BATCH_SIZE=4
IMAGE="/myroot/myimage.sqsh"
OUTPUT_ROOT="/myroot/gdn2/"
export PYTHONPATH="${OUTPUT_ROOT}":$PYTHONPATH
TRAIN_DATA=/data/fineweb-edu/data
VALIDATION_DATA=/data/val-sampled
LR=4e-4
SAVE_DIR="/myroot/save_dir"
LOGS_DIR="${SAVE_DIR}/logs/${NAME}/"
WANDB_DIR="${SAVE_DIR}/wandb/${NAME}/"
TRI_CACHE_DIR="${SAVE_DIR}/triton/${NAME}/"
export TRITON_CACHE_DIR="${SAVE_DIR}/triton/${NAME}/"

mkdir -p ${LOGS_DIR}
mkdir -p ${WANDB_DIR}
mkdir -p ${TRI_CACHE_DIR}

run_cmd="python -u ${OUTPUT_ROOT}/pretrain.py \
--train_data_dir ${TRAIN_DATA} \
--val_data_dir ${VALIDATION_DATA} \
--output_root ${SAVE_DIR} \
--exp_name ${NAME} \
--model_name ${MODEL} \
--train_config ${CONFIG} \
--eval_iters ${EVAL_ITERS} \
--learning_rate ${LR} \
--micro_batch_size ${MICRO_BATCH_SIZE} \
"

DATETIME=`date +'date_%Y-%m-%d_time_%H-%M-%S'`

srun -l \
     --container-image=${IMAGE} \
     --container-mounts=/lustre:/lustre,/home/${USER}:/home/${USER} \
     --container-workdir=${OUTPUT_ROOT} \
     --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
     sh -c "${run_cmd}"  
set +x 
