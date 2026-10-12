
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

# Week 4: Churn Risk Advisor — Model Packaging, Parity & Deployment

**Name:** Anderia Nisar  
**Course:** Introduction to Applied AI  
**Project:** Telco Customer Churn Risk Prediction  
**Repository:** [telco-churn-ml-lab](https://github.com/anderianisar/telco-churn-ml-lab)

## 📌 Project Overview

Week 4 focuses on converting the customer churn prediction model into a reusable and deployment-ready application.

The goal is to ensure that the model produces consistent predictions during training and deployment, identifies customers at risk of leaving a telecom service, and provides useful insights to support customer retention decisions.

This stage extends the machine learning workflow beyond model training by focusing on model packaging, preprocessing consistency, batch predictions, what-if analysis, and deployment through Streamlit.

## 🎯 Objectives

- Package the trained machine learning model for reuse.
- Maintain consistent preprocessing during training and inference.
- Validate prediction consistency between the notebook and deployed application.
- Generate churn risk predictions for multiple customers.
- Perform what-if analysis to explore changes in predicted churn probability.
- Save model metadata and prediction results.
- Integrate the trained model with a Streamlit application.

## 🛠️ Technologies Used

- **Python** — Core programming language
- **Pandas and NumPy** — Data processing and numerical operations
- **Scikit-learn** — Machine learning and preprocessing
- **XGBoost** — Customer churn classification
- **Joblib** — Saving and loading model artifacts
- **Streamlit** — Interactive web application
- **Jupyter Notebook / Google Colab** — Model development and experimentation
- **Git and GitHub** — Version control and project management

## 📊 Previous Model Performance

The XGBoost classifier selected during the previous model experiments provides the baseline for the Week 4 deployment stage.

| Metric | Result |
|---|---:|
| Cross-validation ROC-AUC | 0.8499 |
| Test ROC-AUC | 0.8468 |
| Test Accuracy | 79.84% |
| Test Precision | 65.62% |
| Test Recall | 50.53% |
| Test F1-score | 57.10% |

These results represent the previously evaluated model. Deployment should preserve the model's learned behavior and preprocessing requirements.

## 🔍 Week 4 Tasks

### 1. Model Packaging

Save the trained model and the preprocessing components required to make predictions on new customer records.

The packaged artifacts allow the application to load the trained model without retraining it every time the application starts.

### 2. Preprocessing and Encoder Consistency

Ensure that categorical features are encoded consistently with the training workflow.

The same feature names, column order, transformations, and category handling should be maintained during inference to prevent incorrect predictions.

### 3. Prediction Parity Testing

Compare predictions generated by the original notebook workflow with predictions generated by the packaged model or application.

The purpose is to identify differences caused by preprocessing, feature ordering, encoding, or model loading.

### 4. Batch Customer Risk Prediction

Apply the trained model to multiple customer records and generate churn probabilities and risk classifications.

Batch predictions can help identify customers who may require further review and retention support.

### 5. What-If Analysis

Explore hypothetical changes in customer information, such as:

- Contract type
- Customer tenure
- Monthly charges

Observe how these changes affect the model's predicted churn probability.

These experiments demonstrate model behavior under different inputs but do not prove that changing a particular customer attribute will prevent churn.

### 6. Model Metadata

Maintain a `model_meta.json` file containing relevant model information, such as:

- Model type
- Feature configuration
- Evaluation metrics
- Model version or other relevant details

This information supports model inspection, reproducibility, and maintenance.

### 7. Streamlit Deployment

Integrate the packaged model into an interactive Streamlit application.

The interface allows users to enter customer information, generate churn predictions, and explore risk levels without running the notebook manually.

## 📁 Project Structure

The following is an example structure. Update the filenames to match the actual files in the repository.

```text
telco-churn-ml-lab/
│
├── app.py
├── requirements.txt
├── model_meta.json
├── README.md
│
├── notebooks/
│   └── Week4_Churn_Risk_Advisor_Packaging_Parity.ipynb
│
├── models/
│   └── trained_model.pkl
│
└── outputs/
    └── batch_predictions.csv
```

**Note:** The model, notebook, metadata, and output filenames above are examples. Include only the files that actually exist in the repository.

## 🚀 Running the Application Locally

### Step 1: Clone the Repository

```bash
git clone https://github.com/anderianisar/telco-churn-ml-lab.git
cd telco-churn-ml-lab
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application

```bash
streamlit run app.py
```

Ensure that all required model and preprocessing artifacts are available at the paths expected by the application.

## 🌐 Deployment

The project is intended to be deployed using Streamlit Community Cloud connected to the GitHub repository.

**Live Application:** [Churn Risk Advisor](https://anderianisar-churn-risk-advisor.streamlit.app/)

The deployment requires compatible Python dependencies, the application code, and all required model artifacts.

## 📚 Learning Outcomes

Week 4 aims to develop practical skills in:

- Packaging machine learning models for reuse.
- Maintaining preprocessing consistency during inference.
- Testing prediction consistency between development and deployment.
- Generating predictions for individual and multiple customers.
- Exploring model behavior through what-if analysis.
- Integrating machine learning models into interactive applications.
- Understanding real-world machine learning deployment challenges.

## ✅ Conclusion

Week 4 extends the Telco Customer Churn project from model development toward practical application and deployment.

By focusing on model packaging, prediction consistency, customer risk analysis, what-if experiments, and Streamlit integration, this stage demonstrates important steps in building a reusable machine learning solution.

The overall goal is to develop a Churn Risk Advisor that identifies customers who may be at risk of leaving and supports informed customer retention decisions.

---

**Author:** Anderia Nisar  
**Course:** Introduction to Applied AI  
**Project Repository:** [GitHub](https://github.com/anderianisar/telco-churn-ml-lab)
