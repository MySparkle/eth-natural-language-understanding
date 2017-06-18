#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <triples_file>"
    exit 1
fi

export TEMP_DIR=temp
export DEV_SOURCES=${TEMP_DIR}/source.txt
export MODEL_DIR=model
PYTHON="python3"
mkdir -p ${TEMP_DIR}

# Preprocessing
${PYTHON} preprocess.py --run_test --data_dir $1 --train_dir ${TEMP_DIR} &> /dev/null

# Perplexity calculation
${PYTHON} -m bin.infer \
  --tasks "
    - class: CalculatePerplexities" \
  --model_dir $MODEL_DIR \
  --model_params "
    decoder.params:
      max_decode_length: 80" \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $DEV_SOURCES" \
  > ${TEMP_DIR}/perplexities.txt 2> /dev/null

# Format perplexity results
sed '$!N;s/\n/ /' ${TEMP_DIR}/perplexities.txt
