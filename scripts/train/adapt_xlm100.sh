#!/bin/bash

TRAIN_DIR"SET_ME"
DEV_DIR"SET_ME"
LOG_DIR="SET_ME"
OUTPUT_DIR="SET_ME"
NUM_LANGS=1
ADAPT_LANG="SET_ME"

python3 src/train/train_distributed_xlm.py \
    --source_train_directory $TRAIN_DIR \
    --source_dev_directory $DEV_DIR \
    --source_lang $ADAPT_LANG \
    --save_at_end \
    --output_directory $OUTPUT_DIR \
    --load_from_checkpoint xlm-mlm-100-1280 \
    --experiment_name $ADAPT_LANG \
    --tokenizer_path xlm-mlm-100-1280 \
    --train_sampler baseline \
    --max_seq_len 200 \
    --epochs 40 \
    --batch_size 8 \
    --no_early_stop \
    --no_optimizer_scheduler \
    --save_every_10epochs \
    --num_gpus 1 \
    --seed {seed} \
    --learning_rate 2e-5 \
    --num_langs $NUM_LANGS \
    --log_dir $LOG_DIR \
    --gradient_accumulation_steps 4