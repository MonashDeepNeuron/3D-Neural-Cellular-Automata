EPOCHS=1
LR=1e-3
BATCH_SIZE=8

for LOSS_FN in {0..1}; do
  python3 training/cli_train.py \
    "$EPOCHS" "$LR" "$BATCH_SIZE" "$LOSS_FN" \
    > "training_output_loss_${LOSS_FN}.txt" 2>&1
done