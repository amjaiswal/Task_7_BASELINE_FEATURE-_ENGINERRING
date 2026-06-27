# Task 07 - Baseline Target Feature Engineering

## Objective

Perform baseline feature engineering on the Employee Attrition dataset to create meaningful features, prevent target leakage, and train a baseline Random Forest model.

---

## Dataset

Employee Attrition Dataset

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib

---

## Steps Performed

* Loaded the Employee Attrition dataset
* Removed duplicate records
* Handled missing values
* Engineered new features:

  * IncomePerYear
  * ExperienceRatio
  * IncomePerExperience
  * PromotionGap
* Removed unnecessary columns
* Encoded categorical features
* Split the dataset into training and testing sets
* Trained a Random Forest Classifier
* Evaluated model performance using Accuracy, Confusion Matrix, and Classification Report
* Generated Feature Importance graph
* Performed Target Leakage Check
* Saved the trained model

---

## Results

* Accuracy: **82.65%**
* Training Samples: **1176**
* Testing Samples: **294**
* Leakage Check: **PASSED**
* Model Saved: **random_forest.pkl**
* Feature Importance Graph Generated Successfully

---

## Leakage Check

The project ensures:

* Target column removed before model training
* No future information used during feature engineering
* No target leakage detected

Leakage Status: PASSED

---

## Generated Output Files

* `models/random_forest.pkl`
* `outputs/feature_importance.png`
* `outputs/leakage_report.txt`

---

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

---

## Author

Amar Jaiswal
