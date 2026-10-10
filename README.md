
# Telco Customer Churn — Machine Learning Lab

This repository contains my **Machine Learning Lab work** based on the **Telco Customer Churn dataset**.

The project focuses on building, evaluating, optimizing, and analyzing machine learning models for predicting customer churn.

---

## 📚 Lab Overview

| Week | Topic | Main Concepts |
|---|---|---|
| **Week 2** | Building ML Models | Data Preprocessing, Logistic Regression, Random Forest, Model Evaluation |
| **Week 3** | Model Optimization & Unsupervised Learning | Cross-Validation, Hyperparameter Tuning, XGBoost, K-Means, PCA |

---

# 📌 Week 2 — Building Machine Learning Models

## Objective

The objective of Week 2 was to build machine learning models for **Telco Customer Churn Prediction** and understand the complete machine learning workflow.

## Dataset

The **Telco Customer Churn** dataset contains information about customers, their subscribed services, account information, and whether they left the company.

## Main Steps

- Data loading and exploration
- Data cleaning
- Handling missing values
- Data type conversion
- Categorical variable encoding
- Train-test splitting
- Model training
- Model evaluation
- Comparison of machine learning models

## Models Used

### 1. Logistic Regression

Logistic Regression was used as a baseline classification model for predicting whether a customer would churn.

### 2. Random Forest

Random Forest was used as an ensemble learning method that combines multiple decision trees to improve predictive performance.

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

## Key Learning

Week 2 provided a foundation for understanding how machine learning models can be trained and evaluated for a real-world customer churn prediction problem.

---

# 📌 Week 3 — Model Optimization & Unsupervised Learning

## Objective

Week 3 focused on improving model reliability and performance through **cross-validation and hyperparameter optimization**, followed by unsupervised learning and dimensionality reduction.

---

## 1. Model Stability

Multiple train-validation splits were used to observe how model performance changes depending on the data split.

This demonstrated why relying on a single train-validation split may not provide a reliable estimate of model performance.

---

## 2. 5-Fold Cross-Validation

**Stratified 5-Fold Cross-Validation** was used to obtain a more reliable estimate of model performance.

The following models were compared:

- Logistic Regression
- Random Forest
- XGBoost

ROC-AUC was used as the primary model selection metric.

---

## 3. Hyperparameter Optimization

Two optimization techniques were investigated:

### Grid Search

Grid Search evaluated a predefined set of hyperparameter combinations.

**Random Forest Best CV ROC-AUC:** `0.8468`

### Random Search

Random Search evaluated randomly selected hyperparameter combinations.

**XGBoost Best CV ROC-AUC:** `0.8499`

Random Search provided an efficient way to explore the XGBoost hyperparameter space.

---

## 4. XGBoost

XGBoost was investigated using:

- Early stopping
- Training vs. validation loss
- Randomized hyperparameter search
- Feature importance analysis

The early-stopping experiment identified the best number of boosting trees as approximately **247**, with a validation ROC-AUC of **0.8541**.

The optimized XGBoost model achieved a mean cross-validation ROC-AUC of:

**0.8499**

---

## 5. Feature Importance

The XGBoost model identified several influential features, including:

- Internet Service — Fiber Optic
- Payment Method — Electronic Check
- Contract Type
- Tenure
- Internet Service
- Online Security
- Paperless Billing
- Total Charges
- Online Backup
- Technical Support

> **Note:** Feature importance indicates which features were useful to the model. It does not prove that these features directly cause customer churn.

---

# 📊 Final Model Comparison

| Model | Mean CV ROC-AUC |
|---|---:|
| **XGBoost — Random Search** | **0.8499** |
| Random Forest — Grid Search | 0.8468 |
| Logistic Regression | 0.8460 |

Based on the cross-validation results, **XGBoost with Random Search** was selected as the final model.

---

# 👥 K-Means Customer Segmentation

K-Means clustering was applied to customer information using:

- Tenure
- Monthly Charges
- Total Charges

The features were standardized before clustering.

## Cluster Selection

Silhouette analysis was used to determine the appropriate number of clusters.

The highest silhouette score was obtained for:

**k = 2**

**Silhouette Score:** `0.4812`

## Customer Segments

### Cluster 0 — Newer / Lower-Value Customers

- Average tenure: **19.83 months**
- Average monthly charges: **52.02**
- Average total charges: **854.47**
- Customers: **65.74%**

### Cluster 1 — Long-Term / Higher-Value Customers

- Average tenure: **56.78 months**
- Average monthly charges: **89.70**
- Average total charges: **5072.28**
- Customers: **34.26%**

> **Note:** These segments are based on customer characteristics and should not be interpreted directly as churn or loyalty groups.

---

# 📉 Principal Component Analysis (PCA)

PCA was applied for dimensionality reduction and visualization.

## Results

- **PC1:** 33.25% variance explained
- **PC2:** 11.98% variance explained
- **First 2 PCs:** 45.22% variance explained
- **15 components:** Required to explain at least 90% of the variance

PCA helped analyze the underlying structure of the high-dimensional feature space.

---

# 🏆 Final Test Performance

The final XGBoost model was evaluated on the **previously locked test set**.

| Metric | Score |
|---|---:|
| ROC-AUC | **0.8468** |
| Accuracy | **79.84%** |
| Precision | **65.62%** |
| Recall | **50.53%** |
| F1-Score | **57.10%** |

## Confusion Matrix

```text
[[936   99]
 [185  189]]

# Churn Risk Advisor — Week 4

A machine learning web application that predicts telecom customer churn risk using a trained XGBoost classifier. Built with Python and Streamlit, the application supports individual customer predictions, what-if analysis, and batch scoring through CSV uploads.

## Project Overview

Customer churn occurs when customers discontinue a service. Identifying customers who may be at risk can help telecom retention teams prioritize customers for further review.

This project packages a model developed in Week 3 into a user-facing application and addresses training-serving consistency, model metadata, deployment, and responsible model use.

## Features

- **Single-customer prediction:** Enter customer information and estimate the probability of churn.
- **Risk classification:** Display Low, Medium, or High risk categories.
- **Adjustable retention threshold:** Change the probability threshold used to flag customers for attention.
- **What-if analysis:** Compare predictions for different contract and payment-method scenarios.
- **Batch predictions:** Upload a CSV, score multiple customers, and download the results.
- **Model information:** Display evaluation metrics, model version, and limitations.
- **Training-serving parity:** Use a consistent feature-preparation approach to reduce encoding mismatches.

## Model and Dataset

- **Dataset:** IBM Telco Customer Churn
- **Dataset size:** 7,043 customers
- **Model:** XGBoost classifier
- **Input features:** 30 encoded feature columns
- **Scikit-learn version:** 1.6.1
- **XGBoost version:** 3.4.1

## Model Performance

| Metric | Result |
|---|---:|
| Five-fold cross-validation ROC-AUC | 0.849855 |
| Cross-validation ROC-AUC standard deviation | 0.012949 |
| Held-out test ROC-AUC | 0.8468 |
| Test accuracy | 0.7984 |
| Test precision | 0.6562 |
| Test recall | 0.5053 |
| Test F1-score | 0.5710 |

The cross-validation and held-out test ROC-AUC values are similar, indicating comparable discrimination performance across these evaluations. These results do not guarantee the same performance on new customers or different populations.

## Training-Serving Parity

The serving encoder converts raw customer information into the 30 feature columns expected by the trained model.

The parity test compared training-time and serving-time predicted churn probabilities for all 7,043 customers.

- **Largest absolute probability difference:** 0.0
- **Result:** Parity test passed.

This confirms matching predictions for the tested dataset and encoding workflow.

## What-If Analysis

For one selected high-risk customer, the baseline predicted churn probability was **89.6%**.

| Scenario | Predicted churn | Change from baseline |
|---|---:|---:|
| Baseline | 89.6% | — |
| One-year contract | 81.9% | -7.7 percentage points |
| Two-year contract | 64.4% | -25.1 percentage points |
| Automatic credit-card payment | 86.0% | -3.6 percentage points |

These are model-based scenario comparisons, not causal effects. Changing a customer's contract or payment method does not guarantee reduced churn.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Joblib
- Streamlit
- GitHub
- Streamlit Community Cloud

## Repository Structure

```text
churn-risk-advisor/
├── app.py
├── requirements.txt
├── churn_model.joblib
├── model_meta.json
├── sample_customers.csv
└── README.md
```

## Run Locally

Python and the required packages must be installed to run the application locally.

1. Clone this repository.
2. Install the dependencies.
3. Start the Streamlit application.

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd churn-risk-advisor
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

**Live application:** Add your public Streamlit Community Cloud URL here after deployment.

The app loads the saved model and metadata from the repository. Streamlit Community Cloud installs dependencies using `requirements.txt`.

## Intended Use

The application is intended to support telecom retention teams in prioritizing customers for further review. It is a decision-support tool, not an automated decision-maker.

## Limitations and Responsible Use

- The model learns patterns from historical data and may not generalize to every population.
- The dataset may not represent telecom customers in Pakistan or other markets.
- Customer behavior and service plans can change, causing model performance to drift.
- Predictions and what-if comparisons do not establish causation.
- Customer data should be handled responsibly, and predictions should be reviewed by humans.

## Week 4 Learning Outcomes

- Packaged a trained machine learning model for inference.
- Implemented a serving-time feature encoder.
- Tested training-serving parity on 7,043 customers.
- Explored customer scenarios through what-if analysis.
- Prepared model metadata and sample batch data.
- Built a Streamlit application for individual and batch predictions.
- Documented model performance, intended use, and limitations.

## Project Status

The application code and supporting files have been prepared. Public deployment and end-to-end testing should be marked complete only after the deployed app has been opened and tested successfully.
