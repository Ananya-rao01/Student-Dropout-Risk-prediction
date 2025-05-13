# Student-Dropout-Risk-prediction

## Overview
A machine learning-based web system that predicts student dropout risk on the EduLearn platform. By analyzing academic performance, socio-economic background, and engagement metrics, the model provides real-time predictions and helps educators intervene early.

## Technologies Used
- Python (Pandas, NumPy, Scikit-learn)
- Flask (Web App Integration)
- HTML, CSS (Frontend)
- MySQL
- Matplotlib, Seaborn (Data Visualization)

## Machine Learning Models
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Random Forest

## Features
- Preprocessing of student data  
- Predictive modeling for dropout risk  
- Interactive dashboard with student ID input  
- ROC, Confusion Matrix, F1-score evaluation  
- Real-time dropout alerts for stakeholders

  ## Files Included

- `README.md` – Documentation explaining the project's purpose, structure, and usage instructions.
- `Student_Dropout_risk_prediction_Analysis.ipynb` – Jupyter notebook containing data analysis, model training, and evaluation.
- `app.py` – Flask web application for deploying the dropout prediction model with a web interface.
- `classifier.joblib` – Trained machine learning model used to predict student dropout risk.
- `column_transformer.joblib` – Serialized transformer for encoding and preparing input features.
- `index.html` – Frontend interface for user input and displaying prediction results.
- `scaler.joblib` – Feature scaler used during model training to normalize data.
- `student_dropout_dataset.csv` – Dataset containing student attributes used for training and testing the model.
- `styles.css` – CSS file for styling the HTML interface.


## Objective
To empower online learning platforms with intelligent dropout risk analysis and timely intervention mechanisms to improve student retention.
