#!/bin/bash

ITERATIONS_TRAIN=1000
ITERATIONS_EVAL=200
LOG_FILE="interaction_logs.jsonl"

echo "=================================================="
echo "      Ranking System Experiment Runner"
echo "=================================================="


echo "[1/5] Collecting Training Data (Baseline)..."
export NUM_ITERATIONS=$ITERATIONS_TRAIN
export RANKER_TYPE="baseline" 

python3 -m src.main

echo "--------------------------------------------------"

echo "[2/5] Training Models..."

echo "  > Training XGBoost..."
python3 -m src.train --model_type xgboost

echo "  > Training Matrix Factorization (MF)..."
python3 -m src.train --model_type mf

echo "  > Training BPR-MF..."
python3 -m src.train --model_type bpr

echo "--------------------------------------------------"

evaluate_model() {
    MODEL_NAME=$1
    echo "Evaluating $MODEL_NAME..."
    
    mv $LOG_FILE "${LOG_FILE}.bak" 2>/dev/null
    
    export RANKER_TYPE=$MODEL_NAME
    export NUM_ITERATIONS=$ITERATIONS_EVAL
    python3 -m src.main > /dev/null 2>&1
    
    echo "Results for $MODEL_NAME:"
    python3 src/evaluate.py
    
    cat $LOG_FILE >> "${LOG_FILE}.master"
    rm $LOG_FILE
    mv "${LOG_FILE}.bak" $LOG_FILE 2>/dev/null
    echo "--------------------------------------------------"
}

# Set global evaluation iterations and extended actions flag
export NUM_ITERATIONS=$ITERATIONS_EVAL # Re-using ITERATIONS_EVAL for all evaluation runs
export USE_EXTENDED_ACTIONS=true

# 1. Baseline Data Collection (Randomized)
echo "[1/7] Baseline Data Collection (Randomized)"
export RANKER_TYPE="baseline"
export NUM_ITERATIONS=$ITERATIONS_TRAIN # Use training iterations for baseline collection
python3 -m src.main
echo "--------------------------------------------------"

# 2. Train & Evaluate XGBoost
echo "[2/7] Training XGBoost..."
python3 -m src.train --model_type xgboost --use_extended_actions
echo "Evaluation: XGBoost"
evaluate_model "xgboost"

# 3. Train & Evaluate MF
echo "[3/7] Training Matrix Factorization..."
python3 -m src.train --model_type mf --use_extended_actions
echo "Evaluation: Matrix Factorization"
evaluate_model "mf"

# 4. Train & Evaluate BPR
echo "[4/7] Training BPR-MF..."
python3 -m src.train --model_type bpr --use_extended_actions
echo "Evaluation: BPR-MF"
evaluate_model "bpr"

# 5. Train & Evaluate FM
echo "[5/7] Training Factorization Machines..."
python3 -m src.train --model_type fm --use_extended_actions
echo "Evaluation: Factorization Machines"
evaluate_model "fm"

# 6. Evaluate LinUCB
echo "[6/7] Evaluation: LinUCB (Contextual Bandits)"
evaluate_model "linucb"

# 7. Evaluate Factorization Machines (Hybrid) - This seems to be a duplicate evaluation of FM, keeping as per instruction
echo "[7/7] Evaluation: Factorization Machines (Hybrid)"
evaluate_model "fm"

echo "Done! Master logs saved to ${LOG_FILE}.master"
