# Diabetes Prediction Project

## Project Overview

This project focuses on predicting the likelihood of diabetes in patients based on a set of numeric and categorical variables. The dataset used for this project is sourced from Kaggle, and the analysis and machine learning models are developed using Python in a Jupyter Notebook.

The primary objective of this project is to build a predictive model that can classify whether a patient is diabetic or not based on various features, such as glucose level, blood pressure, and BMI.

---

## Dataset

The dataset is publicly available on Kaggle and can be accessed using the following link:  
[Diabetes Prediction Dataset on Kaggle](https://www.kaggle.com/code/quangnguynl/diabetes-prediction/notebook#Numeric-Variable)

The dataset contains several medical variables that may have a significant influence on the likelihood of developing diabetes. It includes the following columns:
- **Pregnancies**: Number of pregnancies the patient has had.
- **Glucose**: Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
- **BloodPressure**: Diastolic blood pressure (mm Hg).
- **SkinThickness**: Triceps skin fold thickness (mm).
- **Insulin**: 2-Hour serum insulin (mu U/ml).
- **BMI**: Body mass index (weight in kg / height in m^2).
- **DiabetesPedigreeFunction**: Diabetes pedigree function, which is a measure of diabetes in family history.
- **Age**: Age of the patient.
- **Outcome**: Whether the patient is diabetic (1) or not (0).

---

## Files in the Repository

1. **`Project Final LATEST.ipynb`**  
   This Jupyter Notebook contains the entire analysis pipeline, from data preprocessing and exploratory data analysis (EDA) to training and evaluating machine learning models. The notebook provides a step-by-step walkthrough of the process, including data cleaning, visualization, model selection, and performance evaluation.

2. **`Final Report.pdf`**  
   This document provides a comprehensive report explaining the project's methodology, results, and insights. It includes detailed explanations of the data preprocessing steps, model selection, evaluation metrics, and key findings from the analysis. 

---

## Methodology

The project follows the standard data science methodology to build a predictive model:

1. **Data Preprocessing**:
   - Missing values are handled, and numerical features are normalized.
   - Categorical features, if present, are encoded using appropriate techniques.
   - Data is split into training and testing sets to evaluate the model performance effectively.

2. **Exploratory Data Analysis (EDA)**:
   - Visualizations are created to explore the distribution of the variables and identify potential correlations between features.
   - Statistical summaries are generated to understand the key characteristics of the dataset.

3. **Model Development**:
   - Multiple machine learning models are trained to predict the outcome variable (whether the patient has diabetes or not). Common models include:
     - Logistic Regression
     - Decision Trees
     - Random Forest
   - Hyperparameter tuning is applied to optimize model performance.

4. **Model Evaluation**:
   - Models are evaluated using metrics such as accuracy, precision, recall, F1-score, and AUC-ROC to ensure that the model generalizes well to unseen data.

---

## Key Findings

1. **Data Insights**:
   - Certain features like **Glucose**, **BMI**, and **Age** have a significant impact on the prediction of diabetes. Higher glucose levels and BMI, along with increasing age, correlate with a higher probability of being diabetic.
   
2. **Model Performance**:
   - The **Random Forest** model achieved the best performance, with a high accuracy and F1-score compared to the other models. It effectively captured complex relationships between features.
   
3. **Feature Importance**:
   - The most important features influencing the diabetes prediction were found to be **Glucose**, **BMI**, and **Age**, as these had the highest feature importance in the Random Forest model.

4. **Potential for Early Diagnosis**:
   - With the developed model, early identification of patients at high risk for diabetes is possible, which can be used to take preventive measures or initiate early treatment.

---

## Requirements

This project requires the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install these libraries using pip if they are not already installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
