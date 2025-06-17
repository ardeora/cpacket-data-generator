#!/bin/bash

# Fine-tune Mistral 7B with MLX-LM
# Using the correct argument format for the current version
MODEL="mistralai/Mistral-7B-Instruct-v0.2"
DATA_DIR="mlx_training_data"
ADAPTER_PATH="models/mistral-netflow"

# Create output directory
mkdir -p $ADAPTER_PATH

# Training with correct arguments
python -m mlx_lm lora \
    --model $MODEL \
    --train \
    --data $DATA_DIR \
    --adapter-path $ADAPTER_PATH \
    --batch-size 2 \
    --learning-rate 5e-5 \
    --iters 1000 \
    --val-batches 25 \
    --steps-per-report 10 \
    --steps-per-eval 100 \
    --save-every 500 \
    --max-seq-length 1024 \
    --grad-checkpoint \
    --seed 42

echo "Training complete! Adapter saved to $ADAPTER_PATH"
