# Credit Card Default Risk Modelling

## Project Overview

Credit Card Default Risk Modelling is a machine learning project that predicts whether a credit card customer is likely to default on their payment obligations.

The project uses the UCI Credit Card Clients dataset to develop and compare statistical classification models. It focuses on evaluating model performance using multiple metrics and investigating how classification thresholds affect default prediction.

The primary objective is to understand how different classification techniques perform on an imbalanced financial dataset and identify a suitable model for credit default risk prediction.

## Objectives

- Analyse the credit card dataset and understand its characteristics.
- Perform exploratory data analysis (EDA) to identify patterns and distributions.
- Prepare the data for machine learning using appropriate preprocessing techniques.
- Implement and compare Logistic Regression, Linear Discriminant Analysis (LDA), and Quadratic Discriminant Analysis (QDA).
- Evaluate the models using accuracy, precision, recall, F1-score, and ROC-AUC.
- Investigate the effect of changing the classification threshold on model performance.
- Identify the trade-offs between correctly identifying defaulters and minimising false-positive predictions.

## Dataset

**Dataset:** Default of Credit Card Clients  
**Source:** UCI Machine Learning Repository

The dataset contains information about 30,000 credit card clients and their payment behaviour.

### Dataset Details

| Attribute | Description |
|---|---|
| Total observations | 30,000 |
| Input features | 23 |
| Target variable | Y |
| Non-default cases (Y = 0) | 23,364 |
| Default cases (Y = 1) | 6,636 |

The target variable represents the customer's default payment status in the following month.

- `Y = 0`: No default
- `Y = 1`: Default

The dataset exhibits class imbalance, with approximately 77.88% non-default cases and 22.12% default cases.

This imbalance makes it important to evaluate recall and F1-score alongside accuracy.

## Methodology

The project follows a structured machine learning workflow.

### 1. Data Loading and Inspection

The dataset is loaded and examined to understand its dimensions, feature types, missing values, and target distribution.

### 2. Exploratory Data Analysis

Exploratory data analysis is performed to understand the dataset and identify relevant patterns.

The analysis includes examining the target distribution, exploring feature characteristics, and generating visualisations to support interpretation.

### 3. Data Preprocessing

The following preprocessing techniques are used:

- **Missing-value imputation:** Missing numerical values are handled using median imputation.
- **Feature scaling:** StandardScaler is used to standardise numerical features.
- **Train-test split:** The dataset is divided into training and testing sets.

The dataset is split into:

- Training set: 24,000 observations (80%)
- Testing set: 6,000 observations (20%)

Preprocessing is incorporated into a pipeline to maintain consistency between training and testing data.

### 4. Model Development

Three classification models are implemented and evaluated.

**Logistic Regression**

Used as a baseline classification model to estimate the probability of credit card default.

**Linear Discriminant Analysis (LDA)**

Uses a linear decision boundary and models class distributions to distinguish between default and non-default customers.

**Quadratic Discriminant Analysis (QDA)**

Uses a quadratic decision boundary, allowing more flexible separation between the two classes.

### 5. Model Evaluation

The models are evaluated using the following metrics:

- **Accuracy:** Proportion of all predictions that are correct.
- **Precision:** Proportion of predicted defaulters who actually defaulted.
- **Recall:** Proportion of actual defaulters correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures the model's ability to distinguish between the two classes across different classification thresholds.

### 6. Threshold Analysis

The default classification threshold is investigated to understand how changing it affects model performance.

A threshold of 0.75 is evaluated for QDA to examine the trade-off between precision and recall.

This analysis demonstrates that classification decisions depend not only on the selected model but also on the probability threshold used to classify customers.

## Model Performance

The following results were obtained on the test dataset using the evaluated models.

| Metric | Logistic Regression | LDA | QDA |
|---|---:|---:|---:|
| Accuracy | 0.8077 | 0.8087 | 0.7202 |
| Precision | 0.6868 | 0.6801 | 0.4120 |
| Recall | 0.2396 | 0.2547 | 0.6209 |
| F1-score | 0.3553 | 0.3706 | 0.4953 |
| ROC-AUC | 0.7076 | 0.7017 | 0.7256 |

### Interpretation of Results

**Logistic Regression**

Achieved an accuracy of 80.77% and a precision of 0.6868. However, its recall of 0.2396 indicates that it identified only a relatively small proportion of actual defaulters at the evaluated classification threshold.

**Linear Discriminant Analysis (LDA)**

Achieved the highest accuracy among the three models at 80.87%. Its recall was slightly higher than that of Logistic Regression, although its overall performance remained limited in identifying the default class.

**Quadratic Discriminant Analysis (QDA)**

Achieved a recall of 0.6209 and an F1-score of 0.4953. It identified a larger proportion of actual defaulters than the other two models, although this came with lower precision and accuracy.

QDA also achieved the highest ROC-AUC of 0.7256 among the three models.

These results illustrate that accuracy alone is insufficient for selecting a model for credit default prediction.

## QDA Threshold Analysis

An additional experiment evaluates QDA using a classification threshold of 0.75.

| Metric | QDA at Default Threshold | QDA at Threshold 0.75 |
|---|---:|---:|
| Accuracy | 0.7202 | 0.7625 |
| Precision | 0.4120 | 0.4687 |
| Recall | 0.6209 | 0.5524 |
| F1-score | 0.4953 | 0.5071 |
| ROC-AUC | 0.7256 | 0.7256 |

### Key Observations

- Accuracy increases from 72.02% to 76.25%.
- Precision increases from 0.4120 to 0.4687.
- Recall decreases from 0.6209 to 0.5524.
- F1-score increases from 0.4953 to 0.5071.
- ROC-AUC remains unchanged because it evaluates discrimination across thresholds rather than performance at one particular threshold.

The results demonstrate the trade-off between identifying more defaulters and improving the reliability of positive default predictions.

The choice of threshold depends on the relative costs of missed defaults and false-positive predictions.

## Key Findings

1. All three models demonstrate the ability to distinguish between default and non-default customers to some extent.
2. Logistic Regression and LDA achieve higher accuracy than QDA at the evaluated default threshold.
3. QDA achieves higher recall and F1-score at the default threshold, making it more effective at identifying the minority default class in this experiment.
4. Changing the QDA classification threshold to 0.75 improves accuracy, precision, and F1-score while reducing recall.
5. The results highlight the importance of evaluating multiple metrics when working with imbalanced financial datasets.
6. Model selection should consider the practical costs of false negatives and false positives rather than relying exclusively on accuracy.

## Technologies Used

- **Programming Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Data Visualisation:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Development Environment:** Visual Studio Code
- **Version Control:** Git and GitHub

## Project Structure

The repository is organised to separate the dataset, implementation files, generated outputs, and documentation.

```text
credit-default-risk-modelling/
│
├── data/
│   └── raw/
│       └── credit_default_raw.csv
│
├── outputs/
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   ├── final_model_comparison.csv
│   ├── final_model_comparison.png
│   ├── logistic_regression_metrics.csv
│   ├── lda_metrics.csv
│   └── qda_metrics.csv
│
├── README.md
├── requirements.txt
└── ...
