
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
