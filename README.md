# Overview
This project demonstrates how to use Support Vector Machines (SVM) with a Radial Basis Function (RBF) kernel to classify customers based on their age and transaction amount into high-risk or low-risk categories.

## It’s a simple example showing how to:
1.Preprocess data using StandardScaler  
2.Split data into training and testing sets  
3.Train an SVM classifier  
4.Evaluate performance using a confusion matrix and classification report  
5.Visualize the decision boundary of the SVM model  

## Libraries Used in Python
1.NumPy  
2.Pandas  
3.Scikit-learn  
4.Matplotlib  
5.Seaborn  

## Dataset
The dataset is a small manually defined sample of 10 customers with the following attributes:  
CustomerID — Unique customer identifier  
Age — Customer age  
Transaction_Amount — Amount spent in a transaction  
Risk_Label — Binary label (1 = High Risk, 0 = Low Risk)  

## How It Works
1.Data Preparation  
2.Features (Age, Transaction_Amount) are standardized using StandardScaler.  
3.Model Training  
4.SVM with RBF kernel is trained using sklearn.svm.SVC.  
5.Model Evaluation  
6.Prints confusion matrix and classification report.  
7.Visualization  
8.Plots the decision boundary separating the two risk classes.  
