#!/bin/bash

DATA_DIR="YOUR_DATA_DIR"
mkdir ${DATA_DIR}/downsampled_dev/
mkdir ${DATA_DIR}/downsampled_train/

python3 utils/downsample.py \
  --dirs $DIR/English_pretraining_catted/ $DIR/Russian_pretraining_catted/ $DIR/Chinese_pretraining/ $DIR/Arabic_pretraining/ $DIR/Hindi_pretraining/ \
  --output_dir ${DATA_DIR}/downsampled_train/ \
  --num_examples 72315 \
  --dev_output_dir ${DATA_DIR}/downsampled_dev/
