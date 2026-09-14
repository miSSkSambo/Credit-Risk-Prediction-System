# Credit Risk Prediction System

## Machine Learning 600 Assignment

A complete machine learning project for developing a **Credit Risk Prediction System** using a real-world Kaggle credit-risk dataset. The project follows the machine learning lifecycle from data acquisition and exploratory data analysis through preprocessing, Principal Component Analysis (PCA), classification, model evaluation and hyperparameter optimisation.

## Project Badges

![HTML5 Structure](https://img.shields.io/badge/HTML5-Structure-orange)
![CSS3 Styling](https://img.shields.io/badge/CSS3-Styling-blue)
![Responsive](https://img.shields.io/badge/Responsive-Yes-brightgreen)
![Project](https://img.shields.io/badge/Project-Active-brightgreen)
![License](https://img.shields.io/badge/License-Community-blue)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-orange)

## Table of Contents

1. Project Overview
2. Problem Statement
3. Project Objectives
4. Dataset
5. Dataset Features
6. Exploratory Data Analysis
7. Data Quality Issues
8. Data Preprocessing
9. Principal Component Analysis
10. Machine Learning Models
11. Model Evaluation Metrics
12. Results Before PCA
13. Results After PCA
14. PCA Findings
15. Hyperparameter Optimisation
16. Bias-Variance Considerations
17. Final Model Results
18. Business Interpretation
19. Key Findings
20. Technologies Used
21. Project Structure
22. How to Run the Project
23. Reproducibility
24. Limitations
25. Conclusion
26. References
27. Author

---

## 1. Project Overview

Financial institutions need reliable methods for evaluating credit risk because incorrect lending decisions can result in financial losses, while rejecting suitable applicants can result in missed business opportunities.

This project develops a machine learning-based **Credit Risk Prediction System** that uses applicant and loan information to classify loan applications according to their `loan_status`.

Three supervised classification algorithms are evaluated:

- Logistic Regression
- k-Nearest Neighbours (kNN)
- Decision Tree

The models are evaluated using the original processed feature set and a PCA-transformed feature set. The best-performing model is then selected for hyperparameter optimisation using `GridSearchCV`.

The final model selected in this experiment is a **tuned Decision Tree**.

---

## 2. Problem Statement

Credit-risk assessment is an important part of lending decisions. A financial institution needs to determine whether an applicant represents a higher or lower credit risk before approving a loan.

A model that incorrectly identifies a risky applicant as lower risk may expose the institution to potential financial losses. Conversely, incorrectly identifying a suitable applicant as risky may lead to unnecessary loan rejection and lost business opportunities.

The purpose of this project is to investigate whether machine learning classification techniques can be used to predict credit-risk status from historical applicant and loan information.

---

## 3. Project Objectives

The project aims to:

1. Develop a functional credit-risk prediction system using machine learning.
2. Acquire and analyse a real-world credit-risk dataset.
3. Explore the structure and statistical characteristics of the dataset.
4. Identify missing values, class imbalance, categorical variables and potential outliers.
5. Clean and preprocess the dataset.
6. Encode categorical variables into numerical representations.
7. Standardise numerical features.
8. Split the dataset into training and testing subsets.
9. Apply Principal Component Analysis (PCA).
10. Determine the number of components required to retain at least 90–95% of the variance.
11. Train Logistic Regression, k-Nearest Neighbours and Decision Tree models.
12. Evaluate the models using accuracy, precision, recall and F1-score.
13. Compare model performance before and after PCA.
14. Select the strongest-performing model.
15. Optimise the selected model using GridSearchCV and cross-validation.
16. Interpret the final results from a business perspective.

---

## 4. Dataset

The project uses the **Credit Risk Dataset** available through Kaggle.

**Dataset source:** https://www.kaggle.com/datasets/laotse/credit-risk-dataset

### Dataset characteristics

| Property | Value |
|---|---:|
| Dataset | Credit Risk Dataset |
| Observations | 32,581 |
| Variables | 12 |
| Target variable | `loan_status` |
| Training set | 80% |
| Testing set | 20% |

The target variable, `loan_status`, is used for binary classification.

The classes are distributed as follows:

| Loan Status | Observations | Percentage |
|---|---:|---:|
| 0 | 25,473 | 78.18% |
| 1 | 7,108 | 21.82% |

The target variable is therefore imbalanced, with class 0 representing the majority of observations.

---

## 5. Dataset Features

| Feature | Type | Description |
|---|---|---|
| `person_age` | Numerical | Applicant age |
| `person_income` | Numerical | Applicant income |
| `person_home_ownership` | Categorical | Applicant's home ownership status |
| `person_emp_length` | Numerical | Length of employment |
| `loan_intent` | Categorical | Purpose or intention of the loan |
| `loan_grade` | Categorical | Loan classification grade |
| `loan_amnt` | Numerical | Loan amount |
| `loan_int_rate` | Numerical | Loan interest rate |
| `loan_status` | Target | Credit-risk classification target |
| `loan_percent_income` | Numerical | Loan amount relative to applicant income |
| `cb_person_default_on_file` | Categorical | Previous default history |
| `cb_person_cred_hist_length` | Numerical | Length of credit history |

---

## 6. Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed before model development to understand the dataset and identify potential problems.

The analysis included:

- Dataset shape and structure
- Data types
- Missing-value counts
- Descriptive statistics
- Target-variable distribution
- Home-ownership distribution
- Loan-intent distribution
- Loan-grade distribution
- Previous default history
- Numerical feature histograms
- Correlation heatmap
- Boxplots for potential outliers

### Visualisations

The project includes visualisations for the target variable, categorical variables, numerical feature distributions, feature correlations and potential outliers.

The visualisations were used to understand the distribution and characteristics of the dataset before preprocessing.

---

## 7. Data Quality Issues

Several issues were identified during EDA.

### Missing Values

| Variable | Missing Values | Percentage |
|---|---:|---:|
| `person_emp_length` | 895 | 2.75% |
| `loan_int_rate` | 3,116 | 9.56% |

Missing numerical values were handled through median imputation.

### Target-Class Imbalance

The target distribution was:

- Class 0: 78.18%
- Class 1: 21.82%

Because the classes were not evenly distributed, accuracy was not considered sufficient as the only performance measure. Precision, recall and F1-score were also used.

### Potential Outliers

Boxplots identified potential unusual observations within numerical variables.

The observations were not automatically deleted because an extreme value may represent a legitimate applicant rather than an incorrect record.

IQR-based clipping was therefore applied during preprocessing.

### Categorical Variables

Categorical variables included:

- `person_home_ownership`
- `loan_intent`
- `loan_grade`
- `cb_person_default_on_file`

These variables were converted into numerical representations before model training.

---

## 8. Data Preprocessing

The preprocessing pipeline consisted of the following stages.

### 8.1 Duplicate Removal

Duplicate observations were identified and removed so repeated records would not unnecessarily influence model training.

### 8.2 Missing-Value Imputation

Numerical variables were processed using median imputation.

Categorical variables were handled using the most frequent category.

### 8.3 Outlier Management

Potential numerical outliers were treated using the Interquartile Range (IQR) method.

The boundaries were calculated as:

```text
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Values outside the calculated range were clipped to the respective boundary.

### 8.4 Categorical Encoding

Categorical variables were converted to numerical representations using **One-Hot Encoding**.

### 8.5 Feature Scaling

Numerical features were standardised using **StandardScaler**.

Scaling was important because features such as income, loan amount and interest rate have different numerical ranges.

### 8.6 Train-Test Split

The processed data was divided into:

- 80% training data
- 20% testing data

A stratified split was used to preserve the approximate target-class distribution between the training and testing sets.

---

## 9. Principal Component Analysis

Principal Component Analysis (PCA) was applied after preprocessing and scaling.

### Purpose of PCA

PCA is a dimensionality-reduction technique that transforms the original features into a smaller set of principal components.

The purpose of applying PCA was to investigate whether dimensionality could be reduced while retaining most of the information represented by the original features.

### PCA Result

The final PCA configuration used:

- **14 principal components**
- **95.9935% variance retained**

The cumulative explained variance was examined to select the number of components.

Although PCA retained approximately 95.99% of the variance, classification performance decreased when the PCA-transformed features were used.

---

## 10. Machine Learning Models

Three classification algorithms were evaluated.

### Logistic Regression

Logistic Regression was used as a baseline classification model for the binary target.

### k-Nearest Neighbours

kNN classifies observations according to the characteristics of nearby observations.

The project used:

```text
n_neighbors = 5
```

Feature scaling is particularly important for kNN because it relies on distances between observations.

### Decision Tree

The Decision Tree classifier was used because it can model non-linear relationships between features and the target variable.

The Decision Tree achieved the strongest F1-score before optimisation and was selected for hyperparameter tuning.

---

## 11. Model Evaluation Metrics

The models were evaluated using accuracy, precision, recall, F1-score, confusion matrices and classification reports.

### Accuracy

The proportion of all predictions classified correctly.

### Precision

The proportion of predicted positive observations that were actually positive.

### Recall

The proportion of actual positive observations that were correctly identified.

Recall is important in credit-risk classification because failing to identify a risky applicant can potentially result in financial losses.

### F1-Score

The F1-score combines precision and recall into a single measure and is useful when the target classes are imbalanced.

### Confusion Matrix

Confusion matrices were generated to show:

- True positives
- True negatives
- False positives
- False negatives

---

## 12. Results Before PCA

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 86.97% | 76.75% | 57.97% | 66.05% |
| k-Nearest Neighbours | 88.93% | 83.52% | 61.50% | 70.84% |
| Decision Tree | **89.08%** | 73.79% | **77.64%** | **75.67%** |

The Decision Tree achieved the highest F1-score and recall.

It was therefore selected as the strongest model before PCA and taken forward for optimisation.

---

## 13. Results After PCA

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 85.86% | 76.18% | 51.41% | 61.39% |
| k-Nearest Neighbours | 88.49% | 81.88% | 60.86% | 69.82% |
| Decision Tree | 83.82% | 62.47% | 65.16% | 63.79% |

All three models experienced a reduction in F1-score after PCA.

---

## 14. PCA Findings

Although 14 principal components retained approximately **95.99% of the variance**, classification performance decreased for all three algorithms.

F1-score comparison:

| Model | Before PCA | After PCA |
|---|---:|---:|
| Logistic Regression | 66.05% | 61.39% |
| kNN | 70.84% | 69.82% |
| Decision Tree | 75.67% | 63.79% |

Therefore, the original processed features were retained for the final optimisation stage.

This finding demonstrates that retaining a high percentage of statistical variance does not necessarily guarantee improved predictive classification performance.

---

## 15. Hyperparameter Optimisation

The Decision Tree was selected for optimisation because it produced the strongest F1-score before PCA.

`GridSearchCV` was used to search for a suitable combination of Decision Tree hyperparameters.

### Search Configuration

- Cross-validation: **5-fold**
- Parameter combinations: **108**
- Total model fits: **540**
- Optimisation metric: **F1-score**

F1-score was selected because the target variable was imbalanced and both precision and recall were important.

### Best Parameters

```text
criterion = entropy
max_depth = 10
min_samples_leaf = 2
min_samples_split = 5
```

Best cross-validation F1-score:

```text
0.8120
```

---

## 16. Bias-Variance Considerations

A model with high bias is generally too simple and may underfit the training data.

A model with high variance is generally too complex and may overfit the training data.

Decision Tree parameters such as:

- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

help control model complexity.

GridSearchCV evaluates multiple parameter combinations using cross-validation to identify a configuration that is more likely to generalise to unseen data.

The final tuned model was evaluated on an independent test set.

---

## 17. Final Model Results

The tuned Decision Tree achieved:

| Metric | Original Decision Tree | Tuned Decision Tree |
|---|---:|---:|
| Accuracy | 89.08% | **93.12%** |
| Precision | 73.79% | **96.37%** |
| Recall | 77.64% | 71.23% |
| F1-Score | 75.67% | **81.91%** |

### Performance Changes

- Accuracy: **89.08% → 93.12%**
- Precision: **73.79% → 96.37%**
- Recall: **77.64% → 71.23%**
- F1-score: **75.67% → 81.91%**

The tuned model improved accuracy, precision and F1-score, although recall decreased.

The tuned Decision Tree was selected as the final model based on the overall evaluation conducted in this project.

---

## 18. Business Interpretation

Credit-risk classification should not be evaluated using accuracy alone.

### False Positives

A false positive occurs when the model predicts an observation as belonging to the positive/risk class when the actual class is 0.

This may cause a potentially suitable applicant to be incorrectly classified as risky.

Possible consequences include:

- Loan rejection
- Lost business opportunities
- Reduced customer satisfaction

### False Negatives

A false negative occurs when an observation belonging to the positive/risk class is incorrectly predicted as class 0.

This can be particularly important in credit-risk assessment because a risky applicant may incorrectly be treated as lower risk.

Potential consequences include:

- Increased credit exposure
- Higher probability of loan default
- Financial losses

For this reason, precision, recall and F1-score were considered alongside accuracy.

---

## 19. Key Findings

1. The dataset contains **32,581 observations and 12 variables**.
2. Missing values were found in `person_emp_length` and `loan_int_rate`.
3. The target variable was imbalanced, with 78.18% class 0 and 21.82% class 1.
4. Potential outliers were identified and treated using IQR-based clipping.
5. Categorical variables were converted using one-hot encoding.
6. Numerical features were standardised using StandardScaler.
7. The data was divided into an 80% training set and a 20% testing set.
8. PCA retained approximately **95.99% of the variance using 14 principal components**.
9. PCA reduced classification performance for all three models.
10. The original Decision Tree produced the highest F1-score before optimisation.
11. GridSearchCV evaluated 108 parameter combinations using 5-fold cross-validation.
12. Hyperparameter optimisation improved the Decision Tree F1-score from **75.67% to 81.91%**.
13. The tuned Decision Tree achieved **93.12% accuracy and 96.37% precision**.
14. The tuned Decision Tree was selected as the final model.

---

## 20. Technologies Used

### Programming Language

- Python

### Development Environment

- Google Colaboratory
- Jupyter Notebook

### Python Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

### Machine Learning Techniques

- Exploratory Data Analysis
- Missing-value imputation
- IQR-based outlier clipping
- One-Hot Encoding
- StandardScaler
- Train-test splitting
- Principal Component Analysis
- Logistic Regression
- k-Nearest Neighbours
- Decision Tree
- GridSearchCV
- 5-fold cross-validation
- Classification evaluation

---

## 21. Project Structure

```text
Credit-Risk-Prediction-System/
│
├── 402306874_Credit_Risk_ML_Assignment.ipynb
├── credit_risk_dataset.csv
├── 402306874_Credit_Risk_Presentation.pdf
└── README.md
```

### File Descriptions

**`402306874_Credit_Risk_ML_Assignment.ipynb`**

Contains the complete Python implementation, exploratory analysis, preprocessing, PCA, model development, evaluation and hyperparameter optimisation.

**`credit_risk_dataset.csv`**

Contains the Credit Risk Dataset used for the project.

**`402306874_Credit_Risk_Presentation.pdf`**

Contains the project presentation explaining the problem, dataset, preprocessing, PCA, model comparison, optimisation and conclusions.

**`README.md`**

Provides project documentation, methodology, results, instructions and references.

---

## 22. How to Run the Project

### Google Colab

1. Open Google Colaboratory:
   https://colab.research.google.com/

2. Upload:
   `402306874_Credit_Risk_ML_Assignment.ipynb`

3. Upload:
   `credit_risk_dataset.csv`

4. Open the notebook.

5. Ensure the dataset filename matches:

```python
df = pd.read_csv("credit_risk_dataset.csv")
```

6. Run the notebook from the first cell.

7. For a clean reproducible execution, restart the runtime/session and use **Run all**.

The notebook generates the exploratory analysis, visualisations, preprocessing results, PCA analysis, model evaluations, confusion matrices, classification reports and hyperparameter optimisation results.

### Running Locally

Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Place the notebook and CSV file in the same directory, open the notebook with Jupyter Notebook or JupyterLab, and execute the cells sequentially.

---

## 23. Reproducibility

A fixed random state of:

```python
RANDOM_STATE = 42
```

was used where applicable for reproducible data splitting and model configuration.

The recommended execution sequence is:

```text
Load dataset
    ↓
Explore dataset
    ↓
Identify data issues
    ↓
Remove duplicates
    ↓
Split data
    ↓
Impute missing values
    ↓
Treat outliers
    ↓
Encode categorical variables
    ↓
Scale numerical features
    ↓
Apply PCA
    ↓
Train models
    ↓
Evaluate models
    ↓
Compare before/after PCA
    ↓
Optimise Decision Tree
    ↓
Evaluate final model
```

---

## 24. Limitations

### Class Imbalance

The target variable is imbalanced, with class 0 representing approximately 78% of observations. Therefore, accuracy alone does not fully describe minority-class performance.

### PCA Performance

PCA retained approximately 95.99% of the variance but reduced classification performance. This indicates that variance retention alone should not be used as the only criterion for selecting dimensionality reduction.

### Recall After Tuning

The tuned Decision Tree achieved higher accuracy, precision and F1-score, but recall decreased compared with the original Decision Tree. This highlights the importance of considering the business cost of false negatives.

### Dataset Scope

The model is based on the variables available in the selected Kaggle dataset. Real financial institutions may use additional information and more comprehensive credit-bureau data.

---

## 25. Conclusion

This project successfully developed a machine learning-based Credit Risk Prediction System using a real-world credit-risk dataset.

The dataset was explored and prepared through missing-value treatment, duplicate removal, IQR-based outlier clipping, categorical encoding and feature standardisation. The data was divided into training and testing subsets.

Three classification algorithms were evaluated: Logistic Regression, k-Nearest Neighbours and Decision Tree.

PCA was investigated as a dimensionality-reduction technique. Fourteen principal components retained approximately 95.99% of the variance, but the PCA-transformed data resulted in lower classification performance for all three models. The original processed features therefore provided better classification performance for this dataset.

The Decision Tree achieved the strongest performance before optimisation, with an F1-score of 75.67%. GridSearchCV was then used with five-fold cross-validation to optimise the Decision Tree.

The final tuned Decision Tree achieved:

- **93.12% Accuracy**
- **96.37% Precision**
- **71.23% Recall**
- **81.91% F1-Score**

The tuned Decision Tree was therefore selected as the final model for this experiment.

From a business perspective, the results demonstrate that credit-risk models should not be evaluated using accuracy alone. False negatives can potentially expose a financial institution to financial losses, while false positives may result in suitable applicants being rejected. A balanced consideration of precision, recall and F1-score is therefore important when evaluating credit-risk classification systems.

---

## 26. References

Kaggle (n.d.) *Credit Risk Dataset*. Available at: https://www.kaggle.com/datasets/laotse/credit-risk-dataset (Accessed: 4 September 2026).

Google (n.d.) *Google Colaboratory*. Available at: https://colab.research.google.com/ (Accessed: 4 September 2026).

scikit-learn developers (n.d.) *StandardScaler*. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html (Accessed: 4 September 2026).

scikit-learn developers (n.d.) *PCA*. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html (Accessed: 4 September 2026).

scikit-learn developers (n.d.) *GridSearchCV*. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html (Accessed: 4 September 2026).

scikit-learn developers (n.d.) *Metrics and scoring*. Available at: https://scikit-learn.org/stable/api/sklearn.metrics.html (Accessed: 4 September 2026).

scikit-learn developers (n.d.) *Decision Trees*. Available at: https://scikit-learn.org/stable/modules/tree.html (Accessed: 4 September 2026).

---

## 27. Author

**Katlego Sambo**

**Student Number:** 402306874

**Module:** Machine Learning 600

**Year:** 2026

**Semester:** Second Semester

---

## Final Model

**Tuned Decision Tree**

| Metric | Result |
|---|---:|
| Accuracy | **93.12%** |
| Precision | **96.37%** |
| Recall | **71.23%** |
| F1-Score | **81.91%** |

