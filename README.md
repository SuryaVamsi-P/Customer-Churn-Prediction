# Customer Churn Prediction | Telco Dataset

**Predicting Churn to Drive Retention Strategy for Telecom Clients**


## Project Overview

This end-to-end machine learning project focuses on **predicting customer churn** using a telecom dataset. With over 7,000 customer records, the project dives deep into patterns associated with churn behavior, applying a variety of classification algorithms to uncover actionable business insights.

**Goal**: To build and evaluate multiple ML models that can identify customers likely to churn and help businesses design effective retention strategies.


## Key Features

- **Comprehensive EDA & Preprocessing**  
  → Handled missing values, indirect missingness, outliers, label encoding, and feature scaling.

- **Business Insights via Visualizations**  
  → Created advanced visualizations (donut charts, histograms, density plots, heatmaps) revealing key churn patterns like contract type, internet service, payment methods, tenure, tech support, and more.

- **Modeling Techniques**  
  → Implemented and evaluated a suite of models:
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Random Forest
  - Logistic Regression
  - Decision Tree
  - XGBoost (as AdaBoost alternative)

- **Evaluation Metrics**  
  → Accuracy, Confusion Matrix, ROC Curve, and Classification Reports were used to select the best-performing model.


## Tech Stack

- **Language**: R
- **Libraries**: `ggplot2`, `plotly`, `caret`, `randomForest`, `xgboost`, `e1071`, `glmnet`, `rpart`, `pROC`, `ROSE`
- **Visualization**: Plotly, Ggplot2, Ggcorrplot
- **Data Handling**: Dplyr, Tidyr, Data.table


## Business Insights

- Customers on **month-to-month contracts** are **5x more likely to churn**.
- Those with **fiber-optic internet** and **electronic checks** show high churn risk.
- Lack of **tech support**, **online security**, and **dependents** strongly correlates with churn.
- **New customers** (low tenure) are significantly more likely to churn.


## Impact & Applications

This project empowers telecom providers to:
- Preemptively **target at-risk customers** with retention offers.
- **Revise billing models** or offer benefits for longer contracts.
- **Improve customer service** by investing in tech support and online services.


## Repository Structure

```bash
📂 Customer-Churn-Prediction/
│
├── Group 10 Project.R                          # End-to-end R code
├── Project Outputs.docx                        # EDA results and model evaluations
├── Customer Churn Prediction (REPORT).docx     # Final project report
├── Customer_Churn_Prediction.pptx              # Presentation slide deck
├── Customer_Telecom_Data.csv                   # Input dataset
└── README.md                                   # This file
```


## Author

**Surya Vamsi Patiballa**  
M.S. in Data Science – George Washington University

Email :- svamsi2002@gmail.com

LinkedIn :- https://www.linkedin.com/in/surya-patiballa-b724851aa/
