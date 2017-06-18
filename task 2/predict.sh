export DATA_PATH=.
export DEV_SOURCES=${DATA_PATH}/validation/source.txt
export MODEL_DIR=model

export PRED_DIR=${MODEL_DIR}/pred
mkdir -p ${PRED_DIR}

python3 -m bin.infer \
  --tasks "
    - class: DecodeText
      params:
        unk_replace: True" \
  --model_dir $MODEL_DIR \
  --model_params "
    decoder.params:
      max_decode_length: 80" \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $DEV_SOURCES" \
  >  ${PRED_DIR}/predictions.txt
