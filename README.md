# Task 07 - Baseline Target Feature Engineering

## 📌 Project Overview

This project focuses on Feature Engineering for an Employee Attrition Prediction system. The objective is to transform raw employee data into meaningful features that improve machine learning performance while ensuring there is no target leakage.

The project includes data preprocessing, feature engineering, model training, feature importance analysis, leakage checking, and model saving using a Random Forest Classifier.

---

## 🎯 Objectives

- Load and preprocess employee data
- Handle missing values and duplicate records
- Engineer meaningful features
- Encode categorical variables
- Train a Random Forest Classifier
- Analyze feature importance
- Perform leakage checks
- Save the trained model

---

## 📂 Project Structure

Task07_Baseline_Feature_Engineering

├── data/
│ └── employee_attrition.csv
│
├── models/
│ └── random_forest.pkl
│
├── outputs/
│ ├── feature_importance.png
│ └── leakage_report.txt
│
├── feature_engineering.py
├── main.py
├── README.md
├── requirements.txt
└── .gitignore

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

---

## ⚙️ Feature Engineering

The following engineered features were created:

- IncomePerYear
- ExperienceRatio
- IncomePerExperience
- PromotionGap

These features help improve the predictive capability of the model.

---

## 🤖 Machine Learning Model

Algorithm Used:

Random Forest Classifier

---

## 📊 Evaluation Metrics

- Accuracy Score
- Confusion Matrix
- Classification Report
- Feature Importance

---

## 🔒 Leakage Check

The project ensures:

- Target column is removed before training
- No future information is used
- Feature engineering uses only available data

Leakage Status:

PASSED

---

## ▶️ How to Run

Install dependencies

pip install -r requirements.txt

Run the project

python main.py

---

## 📁 Output Files

After execution, the following files are generated:

models/random_forest.pkl

outputs/feature_importance.png

outputs/leakage_report.txt

---

## 📈 Results

The project successfully:

- Cleaned the dataset
- Engineered new features
- Trained a Random Forest model
- Evaluated model performance
- Generated feature importance analysis
- Performed leakage checking
- Saved the trained model

---

## 🚀 Future Improvements

- Hyperparameter tuning
- Cross-validation
- Feature selection
- Advanced feature engineering
- Model deployment using Flask or FastAPI

---

## 👨‍💻 Author

Amar Jaiswal