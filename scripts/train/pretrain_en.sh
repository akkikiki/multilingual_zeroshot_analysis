#!/bin/bash

TRAIN_DIR"SET_ME"
LOG_DIR="SET_ME"
OUTPUT_DIR="SET_ME"
NUM_LANGS=1

export PYTHONPATH=${PYTHONPATH}:/cliphomes/yofu1973/work/morphology_nli
python3 src/train/train_distributed.py \
    --source_train_directory /fs/clip-scratch/yofu1973/transfer_language/output/wikipedia/raw_text/downsampled_train/ \
    --source_dev_directory /fs/clip-scratch/yofu1973/transfer_language/output/wikipedia/raw_text/downsampled_dev/ \
    --source_lang en \
    --save_at_end \
    --output_directory $OUTPUT_DIR/en_pretraining_output/ \
    --experiment_name en \
    --train_sampler baseline \
    --max_seq_len 512 \
    --local_rank -1 \
    --max_steps 150001 \
    --batch_size 2 \
    --no_early_stop \
    --num_gpus 1 \
    --learning_rate 1e-4 \
    --num_langs $NUM_LANGS \
    --log_dir $LOG_DIR \
    --gradient_accumulation_steps 4
