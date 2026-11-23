# Bank Churn Prediction with Explainable AI (XAI)

## Overview

This project explores machine learning techniques for banking customer classification, with a focus on understanding customer-related patterns through predictive modelling and Explainable Artificial Intelligence (XAI).

The project combines data preprocessing, feature engineering, model development, and evaluation with an interactive Streamlit dashboard. Three machine learning algorithms are explored in the notebook: Logistic Regression, Random Forest, and XGBoost. The dashboard provides an interactive interface for examining the XGBoost model's predictions using SHAP and LIME, alongside additional Natural Language Processing (NLP) analysis of customer-related text data.

The objective is not only to develop a predictive model, but also to make its behaviour more interpretable by showing which features influence predictions at both the global and individual levels.

## Project Components

The project consists of two main components:

1. **Machine Learning Notebook (`EDI_Bank_Churn_Prediction XGBoost.ipynb`)**
   - Exploratory data analysis and preprocessing.
   - Feature engineering and transformation.
   - Model development using Logistic Regression, Random Forest, and XGBoost.
   - Model evaluation and hyperparameter tuning.

2. **Interactive Explainable AI Dashboard (`streamlit_xai_dashboard.py`)**
   - Dataset and model overview.
   - Global and local explanations using SHAP and LIME.
   - NLP analysis of text-based banking data.
   - Individual prediction and confidence visualisation.

## Dataset

The project uses the `Comprehensive_Banking_Database` dataset, which contains approximately 5,000 banking records and 46 columns.

The dataset includes information relating to:

- Customer demographics, such as age and gender.
- Banking account information.
- Transaction-related information.
- Customer feedback and complaint details.
- Merchant categories and transaction descriptions.
- Know Your Customer (KYC) information.
- Anti-Money Laundering (AML) indicators.
- Account freeze indicators.

The data contains both numerical and categorical variables, requiring different preprocessing techniques before model training.

**Target variable:** `aml_flag`

> **Important:** Although the project and dashboard use the name *Bank Churn Prediction*, the current notebook and dashboard implementation select `aml_flag` as the prediction target. The dashboard displays the resulting classes as "Churn" and "No Churn". These labels should not be interpreted as confirmed customer attrition outcomes unless the dataset's target definition establishes that `aml_flag` represents churn. In the current implementation, the model is trained to classify the AML flag.

The dataset is not included in this repository. Place the dataset in the appropriate project directory before running the notebook or dashboard.

## Technologies and Libraries

The project uses Python and the following libraries:

| Category | Libraries |
|---|---|
| Data manipulation | Pandas, NumPy |
| Data visualisation | Matplotlib, Seaborn, Plotly |
| Machine learning | Scikit-learn, XGBoost |
| Explainable AI | SHAP, LIME |
| Natural Language Processing | NLTK, WordCloud |
| Dashboard | Streamlit |
| Model persistence | Pickle |

## Machine Learning Workflow

### 1. Data Preprocessing

The dataset undergoes preprocessing to prepare it for machine learning.

The workflow includes:

- Handling missing values.
- Identifying numerical and categorical features.
- Encoding categorical variables.
- Scaling numerical features.
- Treating outliers.
- Preparing the target variable for binary classification.

The dashboard uses a `ColumnTransformer` with a `StandardScaler` for numerical features and `OneHotEncoder` for categorical features. The customer identifier is excluded from the numerical feature set.

### 2. Feature Engineering

Additional features are derived from existing banking information to capture customer and transaction-related patterns.

The notebook includes engineered features relating to:

- Transaction recency.
- Account age.
- KYC age.
- Signed transaction amounts.

These transformations aim to provide the models with more informative representations of the underlying banking data.

### 3. Train-Test Split

The notebook uses an 80:20 stratified train-test split.

| Dataset | Records |
|---|---:|
| Training set | 4,000 |
| Testing set | 1,000 |
| Total | 5,000 |

Stratification preserves the target-class proportions across the training and testing datasets.

### 4. Model Development

Three classification algorithms are implemented and evaluated.

#### Logistic Regression

Logistic Regression serves as a baseline classification model. It estimates the probability of a binary outcome and provides a reference point for evaluating more complex algorithms.

#### Random Forest

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees. It can capture non-linear relationships and interactions between features.

#### XGBoost

XGBoost is a gradient-boosting algorithm that builds an ensemble of decision trees sequentially. It is used as the primary model in the interactive dashboard.

The implementation incorporates class-imbalance handling through `scale_pos_weight`. The dashboard's XGBoost configuration uses 300 estimators, a learning rate of 0.1, a maximum depth of 6, and subsampling and column-sampling rates of 0.8.

The notebook also explores hyperparameter tuning using GridSearchCV.

### 5. Model Evaluation

The classification models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

The initial evaluation results saved in the notebook are:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8400 | 0.1205 | 0.1031 | 0.1111 | 0.4717 |
| Random Forest | 0.9040 | 1.0000 | 0.0103 | 0.0204 | 0.5104 |
| XGBoost | 0.9030 | 0.0000 | 0.0000 | 0.0000 | 0.4475 |

**Interpretation of the initial results**

The initial evaluation indicates that the models struggle to identify the minority class. In particular, Random Forest and XGBoost achieve relatively high accuracy while showing extremely low or zero recall.

This illustrates why accuracy alone is insufficient when evaluating imbalanced classification problems. Precision, recall, F1-score, and ROC-AUC provide additional insight into the models' ability to distinguish between the two classes.

The notebook includes class-imbalance handling and hyperparameter tuning as part of the modelling workflow. The table above represents the initial saved evaluation results, not necessarily the final tuned-model performance.

## Explainable AI Dashboard

The Streamlit dashboard provides an interactive environment for exploring the trained XGBoost model and understanding its predictions.

It contains five main sections.

### 1. Overview

The Overview page provides a summary of the dataset and model performance.

It includes:

- Number of training and testing records.
- Number of features.
- Target-class distribution.
- Dataset preview.
- Confusion matrix.
- ROC curve and AUC score.
- Classification report.

This section provides an overview of the data and the model's predictive performance.

### 2. SHAP Explanations

SHAP (SHapley Additive exPlanations) is used to explain the contribution of individual features to model predictions.

The dashboard uses `TreeExplainer`, which is designed for tree-based models such as XGBoost.

Four explanation types are available:

**Summary Plot**

Displays the overall influence and distribution of feature contributions across multiple observations. It helps identify which features have the greatest impact on model predictions.

**Waterfall Plot**

Explains an individual prediction by showing how different feature contributions move the model output away from its baseline value.

**Feature Importance**

Provides a ranking of features based on their overall SHAP contribution, helping identify the variables that influence the model most strongly.

**Dependence Plot**

Shows the relationship between a selected feature's values and its SHAP contributions, helping explore how changes in a feature affect the model output.

### 3. LIME Explanations

LIME (Local Interpretable Model-agnostic Explanations) is used to interpret individual predictions.

Unlike SHAP's global and local explanation capabilities, this dashboard section focuses on explaining a selected observation.

The LIME interface provides:

- Selection of an individual observation.
- Display of numerical and categorical feature values.
- Actual target label.
- Local feature contribution visualisation.
- Prediction probabilities for each class.
- Ranking of the most influential features for the selected prediction.

This allows users to investigate why the model produced a particular prediction for an individual record.

### 4. NLP Analysis

The NLP Analysis section explores text-based information available in the banking dataset.

The dashboard supports analysis of the following columns when present:

- `transaction_description`
- `merchant_category`
- `feedback_type`
- `resolution_status`

Four analysis options are available:

**Word Cloud**

Generates a visual representation of frequently occurring words in the selected text column.

**Sentiment Analysis**

Uses NLTK's VADER sentiment analyser to calculate positive, negative, neutral, and compound sentiment scores.

**Top Words**

Identifies and displays frequently occurring words after text processing and stopword removal.

**Text Statistics**

Provides basic text statistics, including text length and word count.

This component adds a text-analysis perspective to the project by examining patterns in transaction descriptions and customer-related information.

### 5. Single Prediction

The Single Prediction page allows users to select an individual record and examine the model's output.

It displays:

- Predicted class.
- Prediction confidence.
- Probability distribution across classes.
- Actual target label.
- Whether the prediction matches the actual label.

This section provides a straightforward way to inspect individual model predictions through the dashboard.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost streamlit shap lime wordcloud nltk scipy
```

### 4. Prepare the Dataset

Place the `Comprehensive_Banking_Database` CSV file in the project directory.

The dashboard first checks for:

```text
data_splits/
├── train_set.csv
└── test_set.csv
```

If these files are unavailable, it attempts to load the first CSV file in the project root directory.

For the notebook, update the CSV file path in the data-loading cells to match your local environment.

### 5. Run the Machine Learning Notebook

Open the notebook using Jupyter Notebook or Google Colab:

```text
EDI_Bank_Churn_Prediction XGBoost.ipynb
```

Execute the cells in sequence to perform preprocessing, feature engineering, model training, and evaluation.

### 6. Launch the Streamlit Dashboard

From the project directory, run:

```bash
streamlit run streamlit_xai_dashboard.py
```

Streamlit will start a local server and provide a URL, usually:

```text
http://localhost:8501
```

Open this address in your browser to interact with the dashboard.

The dashboard downloads the required NLTK resources when needed:

- `punkt`
- `stopwords`
- `vader_lexicon`

## Model and Preprocessor Persistence

The dashboard supports saving and loading the trained XGBoost model and preprocessing pipeline.

The following files are used:

```text
models/
├── xgb_model.pkl
└── preprocessor.pkl
```

If the saved model and preprocessor are available, the dashboard loads them. Otherwise, it trains a new XGBoost model using the available training data and saves the resulting artifacts.

Keeping the model and preprocessor together is important because new observations must undergo the same transformations used during training.

## Project Structure

A suggested repository structure is:

```text
Bank-Churn-Prediction/
│
├── EDI_Bank_Churn_Prediction XGBoost.ipynb
├── streamlit_xai_dashboard.py
├── README.md
│
├── data/
│   └── Comprehensive_Banking_Database.csv
│
├── data_splits/
│   ├── train_set.csv
│   └── test_set.csv
│
├── models/
│   ├── xgb_model.pkl
│   └── preprocessor.pkl
│
└── requirements.txt
```

The `data`, `data_splits`, and `models` directories are created or populated as required by the workflow. The dataset and generated model files are not necessarily included in the repository.

## Key Features

- Comparative implementation of Logistic Regression, Random Forest, and XGBoost.
- Data preprocessing and feature engineering for banking records.
- Stratified train-test splitting.
- Class-imbalance handling in XGBoost.
- Hyperparameter tuning using GridSearchCV in the notebook.
- Interactive Streamlit dashboard.
- Global and local model interpretation using SHAP.
- Individual prediction explanations using LIME.
- NLP-based text exploration and sentiment analysis.
- Individual prediction and confidence visualisation.
- Model and preprocessing pipeline persistence.

## Limitations and Considerations

- The initial model evaluation indicates difficulty identifying the minority target class. Further model validation and threshold optimisation may be required.
- Accuracy can be misleading when the target distribution is imbalanced.
- SHAP and LIME explanations describe the behaviour of the trained model; they do not establish causal relationships.
- Sentiment analysis is based on VADER and may not capture the full context of banking-related text.
- The dashboard currently uses `aml_flag` as its target while displaying churn-related terminology. The target definition and dashboard labels should be aligned before interpreting results as actual customer churn predictions.
- The dashboard uses XGBoost, whereas Logistic Regression and Random Forest are evaluated in the notebook.

## Future Improvements

Potential extensions include:

- Improving minority-class detection through resampling and decision-threshold optimisation.
- Comparing additional classification algorithms.
- Evaluating models with precision-recall curves and class-specific metrics.
- Improving feature selection and feature engineering.
- Expanding the dashboard with model comparison functionality.
- Adding more detailed NLP analysis of customer feedback and transaction descriptions.
- Aligning the target variable and dashboard terminology with the intended prediction objective.

## Conclusion

This project combines traditional machine learning with Explainable AI to create a more interpretable banking classification workflow.

The notebook establishes the data preparation and modelling pipeline, while the Streamlit dashboard extends the work into an interactive application. By incorporating SHAP, LIME, and NLP analysis, the project provides different perspectives on the dataset and the behaviour of the predictive model.

The overall approach demonstrates how predictive modelling and explainability techniques can be integrated into a practical data science application.
