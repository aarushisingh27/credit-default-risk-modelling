# Credit Card Default Risk Modelling

## Project Overview

Credit Card Default Risk Modelling is a machine learning project developed as part of the Predictive Analytics (PA) mini-project.

The project investigates statistical classification techniques for predicting whether a credit card customer is likely to default on their payment obligations. Using the UCI Default of Credit Card Clients dataset, three classification models are implemented and compared: Logistic Regression, Linear Discriminant Analysis (LDA), and Quadratic Discriminant Analysis (QDA).

The project also examines the effect of classification threshold selection on model performance, highlighting the trade-offs between accuracy, precision, and recall.

## Objectives

- Analyse the credit card default dataset.
- Perform exploratory data analysis and data preprocessing.
- Implement Logistic Regression, LDA, and QDA classifiers.
- Compare model performance using multiple evaluation metrics.
- Investigate the effect of classification threshold selection.
- Examine the trade-offs involved in identifying customers likely to default.

## Dataset

**Dataset:** Default of Credit Card Clients  
**Source:** UCI Machine Learning Repository

**Dataset link:** https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

The dataset contains information about 30,000 credit card clients.

| Attribute | Description |
|---|---|
| Total observations | 30,000 |
| Input features | 23 |
| Target variable | Y |
| Non-default cases | 23,364 |
| Default cases | 6,636 |

The target variable represents the customer's default payment status:

- `Y = 0`: Non-default
- `Y = 1`: Default

The dataset contains approximately 77.88% non-default cases and 22.12% default cases. This class imbalance makes metrics such as recall and F1-score particularly important.

## Methodology

### 1. Data Inspection and Exploratory Data Analysis

The dataset is examined to understand its structure, feature characteristics, and target distribution. Summary statistics and visualisations are used to explore the data.

### 2. Data Preprocessing

The following preprocessing techniques are used:

- **Median imputation:** Handles missing numerical values.
- **Feature scaling:** StandardScaler standardises numerical features.
- **Train-test split:** The dataset is divided into training and testing sets.

The dataset is split into 80% training data and 20% testing data.

| Dataset | Observations |
|---|---:|
| Training set | 24,000 |
| Testing set | 6,000 |

The preprocessing operations are incorporated into a machine learning pipeline.

### 3. Classification Models

Three statistical classification techniques are implemented and evaluated.

**Logistic Regression**

Used as a baseline classification model to estimate the probability of default.

**Linear Discriminant Analysis (LDA)**

Uses a linear decision boundary to distinguish between default and non-default customers.

**Quadratic Discriminant Analysis (QDA)**

Uses a quadratic decision boundary, allowing more flexible separation between the two classes.

### 4. Model Evaluation

The models are evaluated using the following metrics:

- **Accuracy:** Proportion of predictions that are correct.
- **Precision:** Proportion of predicted defaulters who actually defaulted.
- **Recall:** Proportion of actual defaulters correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures the model's ability to distinguish between classes across classification thresholds.

### 5. Classification Threshold Analysis

An additional experiment evaluates QDA using a classification threshold of 0.75.

The analysis investigates how changing the threshold affects accuracy, precision, recall, and F1-score.

## Experimental Results

The following results were obtained on the test dataset.

| Metric | Logistic Regression | LDA | QDA |
|---|---:|---:|---:|
| Accuracy | 0.8077 | 0.8087 | 0.7202 |
| Precision | 0.6868 | 0.6801 | 0.4120 |
| Recall | 0.2396 | 0.2547 | 0.6209 |
| F1-score | 0.3553 | 0.3706 | 0.4953 |
| ROC-AUC | 0.7076 | 0.7017 | 0.7256 |

### Results Interpretation

**Logistic Regression**

Achieved an accuracy of 80.77% and a precision of 0.6868. However, its recall of 0.2396 indicates that it identified a relatively small proportion of actual defaulters.

**Linear Discriminant Analysis**

Achieved the highest accuracy among the three models at 80.87%. Its recall was slightly higher than that of Logistic Regression.

**Quadratic Discriminant Analysis**

Achieved a recall of 0.6209 and an F1-score of 0.4953 at the evaluated default threshold. It identified a larger proportion of actual defaulters, although its accuracy and precision were lower.

QDA also achieved the highest ROC-AUC of 0.7256 among the three models.

These findings demonstrate why accuracy alone may not be sufficient when evaluating models on an imbalanced dataset.

## QDA Threshold Analysis

QDA was additionally evaluated using a classification threshold of 0.75.

| Metric | Default QDA Threshold | QDA Threshold 0.75 |
|---|---:|---:|
| Accuracy | 0.7202 | 0.7625 |
| Precision | 0.4120 | 0.4687 |
| Recall | 0.6209 | 0.5524 |
| F1-score | 0.4953 | 0.5071 |
| ROC-AUC | 0.7256 | 0.7256 |

### Key Observations

- Accuracy increased from 72.02% to 76.25%.
- Precision increased from 0.4120 to 0.4687.
- Recall decreased from 0.6209 to 0.5524.
- F1-score increased from 0.4953 to 0.5071.
- ROC-AUC remained unchanged.

Increasing the classification threshold made the model more selective when predicting defaults. This improved precision but reduced recall.

The results demonstrate that threshold selection involves a trade-off between identifying more defaulters and reducing false-positive predictions.

## Technologies Used

- **Programming Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Data Visualisation:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Development Environment:** Visual Studio Code
- **Version Control:** Git and GitHub

## Repository Structure

```text
credit-default-risk-modelling/
│
├── outputs/
│   └── Experimental results, comparison tables, and visualisations
│
├── research/
│   └── Research_Article.pdf
│
├── src/
│   └── Python implementation files
│
├── .gitignore
├── README.md
└── requirements.txt
```

### Folder Description

- **`src/`**: Contains the Python implementation files.
- **`outputs/`**: Contains generated experimental results, comparison tables, and visualisations.
- **`research/`**: Contains the final research article.
- **`requirements.txt`**: Lists the Python dependencies.
- **`.gitignore`**: Specifies files and directories Git should ignore.

## Installation and Execution

### Prerequisites

- Python 3.10 or later recommended
- pip
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/aarushisingh27/credit-default-risk-modelling.git
```

### 2. Navigate to the Project Directory

```bash
cd credit-default-risk-modelling
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset

Download the Default of Credit Card Clients dataset from the UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

Prepare the dataset in the format expected by the implementation.

During development, the dataset was stored at:

`data/raw/credit_default_raw.csv`

If the dataset is not included in the repository, download and prepare it locally before running the scripts.

### 5. Run the Project

Open the repository in Visual Studio Code.

Navigate to the `src/` directory and execute the Python scripts in the order required by the project workflow.

The exact commands depend on the implementation filenames.

## Research Article

**Title:** Comparative Evaluation of Statistical Classifiers for Credit Card Default Prediction: A Threshold Analysis

The research article presents the motivation, methodology, experimental evaluation, comparison of statistical classifiers, and classification threshold analysis conducted as part of this project.

**[View the Research Article (PDF)](research/Research_Article.pdf)**

## Limitations and Future Scope

### Limitations

- The experiments use a single dataset.
- The dataset contains an imbalance between default and non-default cases.
- The evaluated models may not capture all complex relationships in customer behaviour.
- Classification performance depends on the selected decision threshold.
- Further validation is required before applying the models to real-world credit decisions.

### Future Scope

- Evaluate additional classification models.
- Investigate class-imbalance handling techniques.
- Use cross-validation for more robust performance estimates.
- Explore systematic threshold optimisation.
- Investigate model interpretability and feature importance.
- Validate performance on additional datasets.

## Conclusion

This project presents a comparative evaluation of Logistic Regression, Linear Discriminant Analysis, and Quadratic Discriminant Analysis for credit card default prediction.

The results show that the models have different strengths. Logistic Regression and LDA achieve higher accuracy at their evaluated default thresholds, whereas QDA achieves higher recall, F1-score, and ROC-AUC.

The additional threshold experiment demonstrates that adjusting the classification threshold can improve the balance between precision and recall-related objectives.

Overall, the project highlights the importance of evaluating multiple performance metrics and considering classification threshold selection when developing machine learning approaches for financial risk prediction.

## References

1. Yeh, I.-C., and Lien, C.-H., "The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients," *Expert Systems with Applications*, vol. 36, no. 2, pp. 2473–2480, 2009.

2. UCI Machine Learning Repository, "Default of Credit Card Clients Dataset." Available at: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

3. Scikit-learn Documentation. Available at: https://scikit-learn.org/

---

**Academic Project:** Predictive Analytics Mini-Project  
**Project Topic:** Credit Card Default Risk Modelling
