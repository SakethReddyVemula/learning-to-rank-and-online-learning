#!/bin/bash

# Configuration
ITERATIONS_TRAIN=500
ITERATIONS_EVAL=100
LOG_FILE="interaction_logs.jsonl"

echo "=================================================="
echo "      Ranking System Experiment Runner"
echo "=================================================="

# 1. Collect Initial Training Data (using Baseline/Control)
# We run a short loop where everyone effectively gets baseline to populate logs
echo "[1/5] Collecting Training Data (Baseline)..."
# We can just run with any ranker but rely on the fact that we need logs.
# Let's run with XGBoost but it might fall back if model missing, or we just want logs.
# Actually, let's ensure we have some data.
export NUM_ITERATIONS=$ITERATIONS_TRAIN
export RANKER_TYPE="baseline" 
# Note: If model doesn't exist, it might error or fallback. 
# Ideally we should have a "baseline_only" mode, but "control" group gives us baseline data.
python3 -m src.main

echo "--------------------------------------------------"

# 2. Train Models
echo "[2/5] Training Models..."

echo "  > Training XGBoost..."
python3 -m src.train --model_type xgboost

echo "  > Training Matrix Factorization (MF)..."
python3 -m src.train --model_type mf

echo "  > Training BPR-MF..."
python3 -m src.train --model_type bpr

echo "--------------------------------------------------"

# Function to run evaluation
evaluate_model() {
    MODEL_NAME=$1
    echo "Evaluating $MODEL_NAME..."
    
    # Backup previous logs to keep the eval run clean
    mv $LOG_FILE "${LOG_FILE}.bak" 2>/dev/null
    
    # Run Experiment
    export RANKER_TYPE=$MODEL_NAME
    export NUM_ITERATIONS=$ITERATIONS_EVAL
    python3 -m src.main > /dev/null 2>&1
    
    # Evaluate
    echo "Results for $MODEL_NAME:"
    python3 src/evaluate.py
    
    # Restore logs (append new ones to backup if we want to keep history, or just overwrite)
    # For this script, let's keep the eval logs separate or merge them back?
    # Let's just leave the eval logs for inspection and maybe append to a master log.
    cat $LOG_FILE >> "${LOG_FILE}.master"
    rm $LOG_FILE
    mv "${LOG_FILE}.bak" $LOG_FILE 2>/dev/null
    echo "--------------------------------------------------"
}

# 3. Evaluate XGBoost
echo "[3/5] Evaluation: XGBoost"
evaluate_model "xgboost"

# 4. Evaluate MF
echo "[4/5] Evaluation: Matrix Factorization"
evaluate_model "mf"

# 5. Evaluate BPR
echo "[5/5] Evaluation: BPR-MF"
evaluate_model "bpr"

echo "Done! Master logs saved to ${LOG_FILE}.master"
