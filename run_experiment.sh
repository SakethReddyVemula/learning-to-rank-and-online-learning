#!/bin/bash

ITERATIONS_TRAIN=1000
ITERATIONS_EVAL=1000
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

echo "[3/5] Evaluation: XGBoost"
evaluate_model "xgboost"

echo "[4/5] Evaluation: Matrix Factorization"
evaluate_model "mf"

echo "[5/6] Evaluation: BPR-MF"
evaluate_model "bpr"

echo "[6/7] Evaluation: LinUCB (Contextual Bandits)"
evaluate_model "linucb"

echo "[7/7] Evaluation: Factorization Machines (Hybrid)"
evaluate_model "fm"

echo "Done! Master logs saved to ${LOG_FILE}.master"
