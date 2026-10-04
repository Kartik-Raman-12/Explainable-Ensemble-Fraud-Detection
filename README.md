# Explainable Ensemble Learning for Imbalanced Fraud Detection

## Overview

This project develops an end-to-end machine learning pipeline for detecting fraudulent credit-card transactions under severe class imbalance. The study compares multiple tree-based classification methods, selects a final model using validation data, optimizes the decision threshold, and uses SHAP to make the final predictions interpretable.

The project focuses on **minority-class detection quality** rather than accuracy alone, since a highly imbalanced fraud dataset can produce misleadingly high accuracy.

## Objectives

- Explore the structure and class imbalance of the transaction data.
- Build a reproducible preprocessing and train/test pipeline.
- Establish a naive all-legitimate baseline and tree-based classification baselines.
- Compare multiple ensemble learning algorithms.
- Select the final model using validation performance.
- Optimize the classification threshold for fraud detection.
- Evaluate the final model on a held-out test set.
- Explain predictions using SHAP.
- Analyze false positives, false negatives, and borderline transactions.

## Dataset

The project uses the **Credit Card Fraud Detection** dataset containing anonymized credit-card transactions.

Key characteristics:

- **284,807 transactions**
- **492 fraudulent transactions**
- Fraud rate: approximately **0.17%**
- Target variable: `Class`
  - `0` → legitimate transaction
  - `1` → fraudulent transaction
- Features include anonymized PCA-transformed variables (`V1`–`V28`), `Time`, and `Amount`.

The dataset is downloaded directly in the notebook from the TensorFlow-hosted copy of the dataset.

## Methodology

### 1. Exploratory Data Analysis

The notebook examines:

- Dataset dimensions and descriptive statistics
- Missing values
- Class distribution
- Transaction amount distribution
- Transaction-time distribution
- Duplicate observations

### 2. Preprocessing

The pipeline:

1. Separates features and target.
2. Removes `Time` from the modeling features.
3. Performs a stratified 80/20 train-test split.
4. Standardizes the `Amount` feature using statistics learned only from the training data.
5. Establishes an all-legitimate baseline.

Stratification is used to preserve the rare fraud class in both training and test sets.

### 3. Model Comparison

The project evaluates several tree-based models:

- Decision Tree
- Class-weighted Decision Tree
- Random Forest
- Class-weighted Random Forest
- Extra Trees
- Class-weighted Extra Trees
- AdaBoost
- Gradient Boosting
- XGBoost

Models are evaluated using:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

**PR-AUC is particularly important in this project because of the severe class imbalance.**

### 4. Model Selection and Threshold Optimization

Candidate models are compared using validation data rather than selecting a model solely from the final test set.

After selecting the best validation model, the fraud-probability threshold is optimized to balance precision and recall according to the project's fraud-detection objective.

The final model is then evaluated once on the held-out test set.

### 5. Explainability with SHAP

SHAP is used to interpret the selected tree-based model at two levels:

- **Global explanation:** identify features that have the greatest overall influence on fraud predictions.
- **Local explanation:** examine why an individual high-confidence transaction was classified as fraudulent.

### 6. Error Analysis

The final predictions are analyzed using:

- True positives
- True negatives
- False positives
- False negatives
- False-positive rate
- False-negative rate
- Fraud-probability distributions
- Transactions close to the selected threshold

This helps identify where the model succeeds and where additional improvements may be required.

## Final Results

The final notebook selects **Extra Trees** as the final model.

| Metric | Test Result |
|---|---:|
| Precision | **0.9524** |
| Recall | **0.8163** |
| F1-score | **0.8791** |
| ROC-AUC | **0.9624** |
| PR-AUC | **0.8821** |
| False Positives | **5** |
| False Negatives | **15** |
| Selected Threshold | **0.4250** |

These results show that the final model achieves strong fraud-detection performance while maintaining a relatively low number of false positives.

## Project Structure

```text
Explainable_Ensemble_Fraud_Detection/
│
├── Explainable_Ensemble_Fraud_Detection.ipynb
├── README.md
│
├── creditcard.csv
│
├── X_train.csv
├── X_test.csv
├── y_train.csv
├── y_test.csv
│
├── final_fraud_detection_model.pkl
├── final_model_config.json
│
├── validation_model_comparison.csv
├── validation_threshold_analysis.csv
├── final_model_results.csv
├── final_metrics.csv
├── final_leaderboard.csv
│
├── shap_feature_importance.csv
├── shap_fraud_transaction_explanation.csv
│
├── error_analysis_predictions.csv
├── false_positives.csv
└── false_negatives.csv
```

## Technologies Used

- **Python**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **XGBoost**
- **SHAP**
- **Joblib**
- **Jupyter Notebook / Google Colab**

## How to Run

### 1. Clone the repository

```bash
git clone <repository-url>
cd Explainable_Ensemble_Fraud_Detection
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost shap joblib
```

### 3. Open the notebook

Open:

```text
Explainable_Ensemble_Fraud_Detection.ipynb
```

The notebook can be run in **Google Colab** or a local Jupyter environment.

### 4. Run the notebook sequentially

The notebook downloads the dataset, performs preprocessing, trains the models, evaluates them, selects the final model, performs SHAP analysis, and generates the final error-analysis outputs.

## Key Design Decisions

### Why not use accuracy as the primary metric?

With only about 0.17% fraudulent transactions, a classifier that predicts almost every transaction as legitimate can still achieve extremely high accuracy. Precision, recall, F1-score, ROC-AUC, and especially PR-AUC provide a more informative assessment.

### Why use threshold optimization?

The default classification threshold of 0.5 is not necessarily optimal for fraud detection. Adjusting the threshold allows the system to control the trade-off between catching fraudulent transactions and generating false alarms.

### Why use SHAP?

Fraud detection is a high-impact classification problem where understanding model decisions is valuable. SHAP provides both global feature-importance analysis and local explanations for individual predictions.

## Limitations

- The dataset contains anonymized PCA-derived features, limiting direct business interpretation of many variables.
- The dataset is historical and may not represent current fraud patterns.
- The selected threshold reflects the evaluation objective and may need recalibration for a different operational cost structure.
- The project is an offline machine-learning study and does not include real-time transaction ingestion or production monitoring.

## Future Improvements

Potential extensions include:

- Cost-sensitive threshold optimization using explicit fraud/false-alarm costs.
- Hyperparameter optimization using cross-validation.
- Calibration of predicted fraud probabilities.
- Temporal validation to better simulate deployment conditions.
- Comparison with additional anomaly-detection methods.
- Model monitoring and drift detection.
- Real-time inference through an API or lightweight deployment interface.

## Author

**Kartik Raman**

M.Tech Computer Science  
Indian Statistical Institute, Kolkata
