# -*- coding: utf-8 -*-
"""
Customer Attrition Analysis Dashboard
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st

# Load data
data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Sidebar
st.sidebar.header("Customer Attrition Dashboard")
st.sidebar.markdown("---")
st.sidebar.write("Created with ❤️ by . [Abdallah Ayman](https://www.linkedin.com/in/abdallah-ayman-74a0702b4/)")

# Preprocessing
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data.dropna(inplace=True)

# Title
st.title("Customer Attrition Analysis Dashboard")
st.write("Explore customer churn patterns and predictions.")

# Data Overview
if st.sidebar.checkbox("Show Data Overview"):
    st.subheader("Data Overview")
    st.write(data.head())

# Visualization Section
st.sidebar.subheader("Visualizations")
visualization = st.sidebar.selectbox("Select Visualization", [
    "Churn by Gender",
    "Churn by Senior Citizen",
    "Tenure Distribution",
    "Monthly Charges Distribution",
    "Total Charges Distribution",
    "Churn vs Tenure",
    "Churn vs Contract Type",
    "Correlation Heatmap"
])

if visualization == "Churn by Gender":
    st.subheader("Churn by Gender")
    churn_by_gender = data.groupby('gender')['Churn'].mean() * 100
    plt.figure(figsize=(8, 5))
    sns.barplot(x='gender', y='Churn', data=data, palette='pastel', estimator=lambda x: sum(x) / len(x) * 100)
    plt.title('Churn by Gender', fontsize=14)
    plt.xlabel('Gender', fontsize=12)
    plt.ylabel('Churn Rate (%)', fontsize=12)
    male_churn_rate = churn_by_gender['Male']
    female_churn_rate = churn_by_gender['Female']
    difference = abs(male_churn_rate - female_churn_rate)
    tip = f"الذكور أكثر عرضة للخروج بنسبة {difference:.1f}% مقارنة بالإناث."
    plt.text(0.5, 0.7, tip, fontsize=10, ha='center', style='italic', bbox=dict(facecolor='white', alpha=0.5))
    st.pyplot(plt.gcf())

elif visualization == "Churn by Senior Citizen":
    st.subheader("Churn by Senior Citizen")
    plt.figure(figsize=(8, 5))
    sns.countplot(data=data, x='SeniorCitizen', hue='Churn', palette='pastel')
    plt.title('Churn by Senior Citizen', fontsize=14)
    plt.xlabel('Senior Citizen (0: No, 1: Yes)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    st.pyplot(plt.gcf())

elif visualization == "Tenure Distribution":
    st.subheader("Tenure Distribution")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='tenure', bins=30, kde=True, color='skyblue')
    plt.title('Distribution of Tenure', fontsize=14)
    plt.xlabel('Tenure (Months)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    st.pyplot(plt.gcf())

elif visualization == "Monthly Charges Distribution":
    st.subheader("Monthly Charges Distribution")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='MonthlyCharges', kde=True, bins=30, color='lightgreen')
    plt.title('Distribution of Monthly Charges', fontsize=14)
    plt.xlabel('Monthly Charges', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    st.pyplot(plt.gcf())

elif visualization == "Total Charges Distribution":
    st.subheader("Total Charges Distribution")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='TotalCharges', bins=30, kde=True, color='lightcoral')
    plt.title('Distribution of Total Charges', fontsize=14)
    plt.xlabel('Total Charges', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    st.pyplot(plt.gcf())

elif visualization == "Churn vs Tenure":
    st.subheader("Churn vs Tenure")
    churn_tenure = data.groupby('tenure')['Churn'].value_counts().unstack().fillna(0)
    plt.figure(figsize=(10, 6))
    churn_tenure.plot(kind='line', colormap='Accent')
    plt.title('Churn Rate Over Tenure', fontsize=14)
    plt.xlabel('Tenure (Months)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Churn')
    st.pyplot(plt.gcf())

elif visualization == "Churn vs Contract Type":
    st.subheader("Churn by Contract Type")
    churn_contract = data.groupby(['Contract', 'Churn']).size().unstack()
    churn_contract.plot(kind='bar', colormap='Set2')
    plt.title('Churn by Contract Type', fontsize=14)
    plt.xlabel('Contract Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Churn')
    st.pyplot(plt.gcf())

elif visualization == "Correlation Heatmap":
    st.subheader("Correlation Heatmap")
    numeric_vars = ['tenure', 'MonthlyCharges', 'TotalCharges']
    corr_matrix = data[numeric_vars].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix', fontsize=14)
    st.pyplot(plt.gcf())

# Model Building and Prediction
st.sidebar.subheader("Predict Churn")
if st.sidebar.button("Train Model"):
    # Feature Engineering
    data['Cost_Per_Tenure'] = data['MonthlyCharges'] / data['tenure']
    data['Total_Charges_Per_Month'] = data['TotalCharges'].astype(float) / data['tenure']
    service_columns = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    data['Service_Count'] = data[service_columns].apply(lambda x: sum(x == 'Yes'), axis=1)
    data['Contract_Type'] = data['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})

    # Select Features
    features = [
        'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract_Type', 'PaperlessBilling', 'MonthlyCharges',
        'Total_Charges_Per_Month', 'Cost_Per_Tenure', 'Service_Count'
    ]
    X = pd.get_dummies(data[features])
    y = data['Churn']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Classification Report
    st.subheader("Model Evaluation")
    st.text("Confusion Matrix:")
    st.write(confusion_matrix(y_test, predictions))

    st.text("Classification Report:")
    report = classification_report(y_test, predictions, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report)

# Hyperparameter Tuning
st.sidebar.subheader("Hyperparameter Tuning")
if st.sidebar.button("Optimize Model"):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    st.write(f"Best Parameters: {grid_search.best_params_}")