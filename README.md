# 🧠 Machine Learning Based Analysis of Customer Spending Patterns

This project applies machine learning techniques to analyze and predict customer spending behavior and patterns using a rich dataset sourced from Kaggle. 
By exploring customer attributes such as age, income, profession, and family size, 
the goal is to uncover actionable insights that can help businesses optimize customer engagement strategies and target marketing efforts more effectively.

---

## 📊 Dataset Overview

The dataset is publicly available on Kaggle:  
[Customer Segmentation Dataset by datascientistanna](https://www.kaggle.com/datasets/datascientistanna/customers-dataset)

It includes **2,000** entries with the following **8 features**:

| Feature              | Description |
|----------------------|-------------|
| `CustomerID`         | Unique identifier for each customer |
| `Gender`             | Customer's gender |
| `Age`                | Age of the customer |
| `Annual Income ($)`  | Customer’s yearly income |
| `Spending Score`     | A score (1–100) assigned by the mall based on customer behavior |
| `Profession`         | Customer's profession |
| `Work Experience`    | Years of work experience |
| `Family Size`        | Number of family members |

---

## 🧹 Data Preprocessing

- Checked and removed **missing values** (`Profession` column).
- Dropped **duplicates**.
- Scaled numeric features using `StandardScaler` and `MinMaxScaler`.
- Encoded categorical variables.
- Split data into training and testing sets using `train_test_split`.

---

## 📈 Exploratory Data Analysis

- Gender distribution via **pie chart**.
- Age vs. Spending Score visualization.
- Income distribution analysis.
- Spending score trends based on family size and profession.

---

## 🧠 Machine Learning Models Used

- **Regression**:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Support Vector Regression (SVR)

- **Classification** (if binning spending scores or labels used):
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier

- **Model Evaluation Metrics**:
  - R² Score
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - Accuracy
  - Confusion Matrix (for classification)
  - Classification Report

---

## 🛠 Libraries and Tools

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, classification_report,
    confusion_matrix, mean_absolute_error
)
