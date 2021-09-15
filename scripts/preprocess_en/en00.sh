#!/bin/bash
SPLIT=00  # Presplitted for porallel processing

DOWNLOAD_DATA_DIR="SET_ME"
OUTPUT_DIR="SET_ME"
mkdir -p $OUTPUT_DIR/English00/
mkdir -p $OUTPUT_DIR/English00_pretraining/

# Convert data
python3 src/utils/get_wiki_text.py --input_directory $DOWNLOAD_DATA_DIR/English00/ --output_directory $OUTPUT_DIR/English00/

# Preprocess
python3 src/utils/preprocess.py \
  --source_directory $OUTPUT_DIR/English00/ \
  --output_directory $OUTPUT_DIR/English00_pretraining/ \
  --tokenizer_path xlm-roberta-base \
  --max_seq_len 512 \
  --min_load_len 10 \
  --rank 0

