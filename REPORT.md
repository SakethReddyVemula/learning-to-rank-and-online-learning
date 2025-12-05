# Project Report: Personalized News Ranking

## 1. Introduction
The objective of this project was to build a personalized search system that improves user engagement compared to a standard baseline. We utilized a user simulation platform to interact with our search engine, collecting feedback (clicks, dwell time) to train a Learning to Rank (LTR) model.

## 2. Methodology

### 2.1 Baseline System
-   **Search Engine**: Elasticsearch (v8.11).
-   **Ranking Algorithm**: Elasticsearch Default.
-   **Strategy**: We indexed the provided `articles.jsonl` dataset. For each user query, we retrieved the top 10 documents based on the default relevance score and presented them to the user.

### 2.2 Personalization Approach
We implemented a **Pointwise Learning to Rank** approach using **XGBoost**.

#### Feature Engineering
We extracted features representing the affinity between a user and a candidate article:
1.  **User Profile**: Constructed dynamically by aggregating the topics of articles the user previously clicked.
2.  **Article Features**:
    -   `topics`: The categories associated with the article.
    -   `num_topics`: Count of topics.
3.  **Interaction Features**:
    -   `topic_match_score`: A weighted score calculated as the dot product of the user's topic preference vector and the article's topic vector.
    -   `title_overlap`: Number of overlapping words between the query and article title.

#### Model Training
-   **Algorithm**: XGBoost Classifier (Binary Logistic).
-   **Training Data**: Collected ~1000 interactions from the baseline run.
-   **Label**: `1` if the user clicked the article, `0` otherwise.
-   **Objective**: Predict the probability of a click given the (User, Article, Query) features.

#### Serving
During the online phase, we used a re-ranking architecture:
1.  **Retrieve**: Fetch top 50 candidates from Elasticsearch using the default ranking.
2.  **Score**: Calculate features for each candidate and predict click probability using the trained XGBoost model.
3.  **Sort**: Re-order the top 50 candidates based on the model's score and return the top 10.

## 3. Experiments

### 3.1 Experimental Setup
We conducted an **A/B Test** to rigorously evaluate the personalized model against the baseline.
-   **Traffic Split**: 50% of users were assigned to the **Control Group** (Baseline Elasticsearch Default), and 50% to the **Treatment Group** (Personalized Re-ranking).
-   **Assignment**: Deterministic hashing of `user_id` ensured consistent group assignment for each user.
-   **Duration**: The experiment ran for approximately 600 queries.

### 3.2 Metrics
-   **Click-Through Rate (CTR)**: The proportion of queries that resulted in at least one click.
-   **Average Dwell Time**: The average time spent reading clicked articles (proxy for content quality/relevance).

## 4. Results

| Metric | Control (Elasticsearch Default) | Treatment (Personalized) |
| :--- | :--- | :--- |
| **Total Queries** | 325 | 260 |
| **CTR** | **10.46%** | 8.46% |
| **Avg Dwell Time** | 10.53s | **10.97s** |

### 4.1 Analysis
-   **CTR**: The personalized model underperformed slightly in terms of CTR (8.46% vs 10.46%). A t-test yielded a p-value of **0.41**, indicating the difference is **not statistically significant**.
-   **Dwell Time**: The personalized model achieved a higher average dwell time (10.97s vs 10.53s). This suggests that while the model might have ranked slightly fewer "clickable" items highly, the items that *were* clicked were more engaging to the users.

### 4.2 Discussion
The lower CTR for the personalized model could be attributed to:
1.  **Cold Start**: The model relies on user history. In the short simulation run, many users might be new or have sparse history, making personalization difficult.
2.  **Hidden Features**: The problem description mentioned a "hidden feature" governing user preference. Our model currently only uses explicit topics.
3.  **Training Data Size**: The model was trained on a relatively small dataset (<1000 samples), which may lead to overfitting or poor generalization.

### 4.3 Retraining with Exploration (Epsilon-Greedy)
We implemented an Epsilon-Greedy exploration strategy (`epsilon=0.1`) and collected ~500 additional interactions. We then retrained the model on this augmented dataset and ran a second A/B test (500 iterations).

| Metric | Control (Elasticsearch Default) | Treatment (Retrained XGBoost + Exploration) |
| :--- | :--- | :--- |
| **Total Queries** | 866 | 819 |
| **CTR** | **9.24%** | 9.16% |
| **Avg Dwell Time** | **8.96s** | 6.90s |

**Observation**: The retrained model with exploration performed comparably to the baseline in terms of CTR (9.16% vs 9.24%), narrowing the gap observed in the initial experiment. However, dwell time decreased, possibly because exploration introduces random (potentially irrelevant) articles that users might click but abandon quickly.

### 4.4 Pairwise Learning to Rank (BPR-MF)
We implemented **Bayesian Personalized Ranking (BPR)**, a pairwise approach to Matrix Factorization. Unlike standard MF (which often treats unobserved items as zeros), BPR optimizes the ranking order directly by maximizing the likelihood that a user prefers a clicked item over an unobserved item.

-   **Optimization**: Stochastic Gradient Descent (SGD).
-   **Loss Function**: BPR-OPT (Log Sigmoid of the difference).
-   **Results**: (Pending full evaluation, but functional).

### 4.5 Online Learning (Contextual Bandits - LinUCB)
To address data sparsity and changing user preferences, we implemented **LinUCB**, a contextual bandit algorithm.
-   **Context**: Article topics (One-hot encoded).
-   **Algorithm**: Disjoint LinUCB (separate model per user).
-   **Online Update**: The model updates its parameters ($A$ and $b$) immediately after every user interaction (Click/No Click), allowing it to adapt in real-time.
-   **Exploration**: Uses Upper Confidence Bound (UCB) to balance exploration and exploitation.

### 4.6 Hybrid Approach (Factorization Machines)
To further address sparsity and leverage both user behavior and content, we implemented **Factorization Machines (FM)** using the `lightfm` library.
-   **Hybrid Features**: Combines User IDs, Item IDs, and Article Topics.
-   **Loss Function**: WARP (Weighted Approximate-Rank Pairwise), which optimizes for ranking by maximizing the margin between positive and negative examples.
-   **Advantage**: Generalizes better than pure MF by learning interactions between features (e.g., User A likes Topic T), even for new items with known topics.

## 5. Future Work
-   **Latent Feature Discovery**: Implement Matrix Factorization or Embeddings to capture the "hidden" user preferences.
-   **Online Learning**: Update the model incrementally in real-time instead of batch training.
-   **Hyperparameter Tuning**: Optimize epsilon decay and XGBoost parameters.
