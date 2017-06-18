# Set this to where you extracted the downloaded file
export DATA_PATH=.

export VOCAB_SOURCE=${DATA_PATH}/train/vocab.txt
export VOCAB_TARGET=${DATA_PATH}/train/vocab.txt
export TRAIN_SOURCES=${DATA_PATH}/train/source.txt
export TRAIN_TARGETS=${DATA_PATH}/train/target.txt
export DEV_SOURCES=${DATA_PATH}/validation/source.txt
export DEV_TARGETS=${DATA_PATH}/validation/target.txt

export DEV_TARGETS_REF=${DATA_PATH}/validation/target.txt
export TRAIN_STEPS=40000


export MODEL_DIR=model
mkdir -p $MODEL_DIR


(python3 -m bin.train \
  --config_paths="
      ./basic_nmt_large.yml,
      ./train_seq2seq.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: ParallelTextInputPipeline
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS" \
  --batch_size 64 \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR \
  --save_checkpoints_secs 3600 \
  --keep_checkpoint_max 20) 2>&1 | tee -a outputinfo.txt
