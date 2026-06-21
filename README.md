### 1. README.md for Credit Card Fraud Detection Project

Use this as the structure for your GitHub README.

# Credit Card Fraud Detection System

## Overview

This project is an end-to-end Machine Learning application that detects fraudulent credit card transactions using highly imbalanced financial transaction data. The solution includes data preprocessing, model training, explainability, API deployment, Docker containerization, and cloud deployment on AWS.

---

## Problem Statement

Credit card fraud causes significant financial losses for banks and customers. Fraudulent transactions represent only a tiny fraction of all transactions, making the dataset highly imbalanced and challenging for traditional machine learning models.

The objective of this project is to build a robust fraud detection system that maximizes fraud detection while minimizing false alarms.

---

## Dataset

* Source: Kaggle Credit Card Fraud Detection Dataset
* Total Transactions: 284,807
* Fraudulent Transactions: 492
* Fraud Rate: ~0.17%

### Dataset Characteristics

* Highly imbalanced classification problem
* Features V1–V28 obtained using PCA transformation
* Additional features:

  * Time
  * Amount
* Target Variable:

  * 0 = Legitimate Transaction
  * 1 = Fraudulent Transaction

---

## Project Architecture

```text
creditcard_fraud/
│
├── artifacts/
├── notebooks/
├── src/
│   ├── components/
│   ├── pipeline/
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── templates/
├── static/
├── app.py
├── Dockerfile
├── requirements.txt
├── setup.py
└── .github/workflows/
```

---

## Workflow

### Data Ingestion

* Load transaction dataset
* Train-test split
* Store processed artifacts

### Data Transformation

* Feature preprocessing
* Pipeline creation
* Data preparation for training

### Model Training

Models evaluated:

* Logistic Regression
* Random Forest
* Gradient Boosting
* AdaBoost
* XGBoost
* Decision Tree
* KNN

Final Model:

* XGBoost Classifier

---

## Handling Class Imbalance

The dataset contains only 0.17% fraudulent transactions.

Techniques used:

* Stratified Train-Test Split
* Class Weight Adjustment
* Threshold Optimization
* Evaluation using Precision, Recall, F1 Score and ROC-AUC

---

## Model Evaluation Metrics

Key metrics used:

* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

Special focus was placed on maximizing Recall to reduce missed fraud cases.

---

## Deployment

### Local Deployment

* Flask Application
* Docker Containerization

### Cloud Deployment

AWS Services Used:

* Amazon EC2
* Amazon ECR
* IAM Users
* IAM Roles
* Security Groups

---

## CI/CD Pipeline

GitHub Actions pipeline implemented for automated deployment.

Workflow:

```text
Git Push
    ↓
GitHub Actions
    ↓
Docker Build
    ↓
Amazon ECR
    ↓
EC2 Deployment
```

Implemented:

* Automated Docker builds
* ECR image push
* SSH-based EC2 deployment
* AWS authentication using IAM

---

## Docker

Dockerized application using:

```bash
docker build -t fraud-app .
docker run -p 5000:5000 fraud-app
```

Image optimized from approximately:

```text
3.8 GB → 1.2 GB
```

through dependency and artifact cleanup.

---

## Key Learnings

* Working with highly imbalanced datasets
* Fraud detection model development
* XGBoost optimization
* Model explainability using SHAP
* Docker containerization
* AWS EC2 deployment
* Amazon ECR image management
* IAM authentication and authorization
* GitHub Actions CI/CD pipelines
* Production deployment troubleshooting

---

## Future Improvements

* Real-time fraud scoring API
* Monitoring and logging dashboard
* Model drift detection
* Kubernetes deployment
* Automated retraining pipeline
* A/B testing framework for model comparison

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Flask
* Docker
* AWS EC2
* Amazon ECR
* GitHub Actions

---

## Author

Abhishek Goswami

M.Sc. Mathematics | Data Science & Machine Learning Enthusiast

