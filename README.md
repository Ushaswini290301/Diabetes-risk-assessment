# Diabetes-risk-assessment
Diabetes Prediction Project
Project Overview
This project focuses on predicting the likelihood of diabetes in patients based on a set of numeric and categorical variables. The dataset used for this project is sourced from Kaggle, and the analysis and machine learning models are developed using Python in a Jupyter Notebook.

The primary objective of this project is to build a predictive model that can classify whether a patient is diabetic or not based on various features, such as glucose level, blood pressure, and BMI.

Dataset
The dataset is publicly available on Kaggle and can be accessed using the following link:
Diabetes Prediction Dataset on Kaggle

The dataset contains several medical variables that may have a significant influence on the likelihood of developing diabetes. It includes the following columns:

Pregnancies: Number of pregnancies the patient has had.

Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test.

BloodPressure: Diastolic blood pressure (mm Hg).

SkinThickness: Triceps skin fold thickness (mm).

Insulin: 2-Hour serum insulin (mu U/ml).

BMI: Body mass index (weight in kg / height in m^2).

DiabetesPedigreeFunction: Diabetes pedigree function, which is a measure of diabetes in family history.

Age: Age of the patient.

Outcome: Whether the patient is diabetic (1) or not (0).

Requirements
This project requires the following Python libraries:

pandas

numpy

matplotlib

seaborn

scikit-learn

You can install these libraries using pip if they are not already installed:

bash
Copy
pip install pandas numpy matplotlib seaborn scikit-learn
Project Structure
Project Final LATEST.ipynb: This Jupyter Notebook contains the entire analysis pipeline, from data preprocessing and exploratory data analysis (EDA) to training and evaluating machine learning models.

Dataset: The dataset is expected to be downloaded manually from Kaggle, as the link provided.

Workflow
Data Preprocessing: The dataset is cleaned by handling missing values, encoding categorical variables (if any), and normalizing the numerical features.

Exploratory Data Analysis (EDA): Key insights are gained through data visualization, correlation analysis, and statistical summaries.

Model Development: Different machine learning models are evaluated, including Logistic Regression, Decision Trees, and Random Forest.

Model Evaluation: The performance of the models is assessed using metrics such as accuracy, precision, recall, F1-score, and AUC-ROC.

How to Use
Download the dataset from Kaggle using the provided link.

Run the Jupyter Notebook (Project Final LATEST.ipynb) to load and analyze the data.

Follow the steps in the notebook to train and evaluate the machine learning models.

