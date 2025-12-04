# Learning to Rank & Online Learning Project

This project implements a personalized news ranking system using Elasticsearch and XGBoost. It includes a complete pipeline for data collection, model training, and online A/B testing against a user simulation platform.

## Project Structure

- `data/`: Contains the dataset (`articles.jsonl`) and Docker image.
- `src/`: Source code for the project.
    - `main.py`: Main entry point for the interaction loop (Baseline & Experiment).
    - `train.py`: Script to train the personalization model.
    - `evaluate.py`: Script to analyze interaction logs and calculate metrics.
    - `features.py`: Feature extraction logic.
    - `ranker.py`: Personalized ranker implementation.
    - `elasticsearch_client.py`: Interface for Elasticsearch.
    - `simulation_client.py`: Interface for the User Simulation API.
- `models/`: Directory where trained models are saved.
- `interaction_logs.jsonl`: Log file containing all user interactions.

## Prerequisites

- **Docker**: Required to run the user simulation and Elasticsearch.
- **Python 3.8+**: For running the client code.

## Setup

1.  **Load the Simulation Docker Image**:
    ```bash
    docker load -i data/ire_project-1.0-amd64.tar
    ```

2.  **Start Services (Elasticsearch & Simulation)**:
    ```bash
    docker compose up -d
    ```

3.  **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Run Baseline / Experiment Loop
The `main.py` script handles the interaction loop. It will run an A/B test splitting traffic between the Baseline (Elasticsearch Default) and the configured Personalized model.

**Configuration**:
To switch between ranking models, edit the `RANKER_TYPE` variable in `src/main.py`:
-   `RANKER_TYPE = "xgboost"`: Uses the XGBoost model (default).
-   `RANKER_TYPE = "mf"`: Uses the Matrix Factorization model.

```bash
python3 -m src.main
```

### 2. Train the Model
You can train different model variants using the `--model_type` argument:

**XGBoost (Feature-based)**:
```bash
python3 -m src.train --model_type xgboost
```
Saves to `models/ranker.model`.

**Matrix Factorization (Latent Factors)**:
```bash
python3 -m src.train --model_type mf
```
Saves to `models/mf_model.pkl`.

### 3. Evaluate Results
To calculate Click-Through Rate (CTR), Dwell Time, and perform statistical significance tests on the A/B experiment data:

```bash
python3 src/evaluate.py
```

### 4. Improvements
We implemented an **Epsilon-Greedy** exploration strategy and **Matrix Factorization**.
-   **Exploration**: Randomly shuffling candidates with probability `epsilon` (default 0.1).
-   **Latent Features**: Capturing hidden preferences via SVD.

## Methodology & Results
For a detailed report on the approach, architecture, and experimental results, please refer to [REPORT.md](REPORT.md).
