export DATA_PATH=.
export DEV_TARGETS_REF=${DATA_PATH}/validation/target.txt
export MODEL_DIR=model
export PRED_DIR=${MODEL_DIR}/pred
export SEQ2SEQ=${HOME}/seq2seq


echo "Predictions: "

${SEQ2SEQ}/bin/tools/multi-bleu.perl ${DEV_TARGETS_REF} < ${PRED_DIR}/predictions.txt

echo "Predictions with Beam: "
${SEQ2SEQ}/bin/tools/multi-bleu.perl ${DEV_TARGETS_REF} < ${PRED_DIR}/predictionsBeam.txt




