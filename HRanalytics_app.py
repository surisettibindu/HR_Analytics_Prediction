import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="HR Analytics Dashboard", layout="wide")
st.title("HR Analytics - Employee Attrition Dashboard")

# 1. DIRECT LOAD 

file_path = r"HR_Analytics.csv"

if not os.path.exists(file_path):
    file_path = "HR_Analytics.csv"

try:
    df = pd.read_csv(file_path)
    st.success(f"Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # 2. ENCODING 
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    # 3. GRAPHS - 8 GRAPHS
    st.subheader("Visualizations - 8 Charts")
    c1, c2 = st.columns(2)

    with c1:
        fig1, ax1 = plt.subplots(figsize=(2.5,2))
        df['Attrition'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax1)
        ax1.set_title("1. Attrition Count"); ax1.set_ylabel("")
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(2.5,2))
        pd.crosstab(df['Department'], df['Attrition']).plot(kind='bar', ax=ax2)
        ax2.set_title("2. Department vs Attrition")
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots(figsize=(2.5,2))
        ax3.hist(df['Age'], bins=20, color='orange')
        ax3.set_title("3. Age Distribution")
        st.pyplot(fig3)

        fig4, ax4 = plt.subplots(figsize=(2.5,2))
        pd.crosstab(df['OverTime'], df['Attrition']).plot(kind='bar', ax=ax4)
        ax4.set_title("4. OverTime vs Attrition")
        st.pyplot(fig4)

    with c2:
        fig5, ax5 = plt.subplots(figsize=(2.5,2))
        ax5.hist(df['MonthlyIncome'], bins=20, color='green')
        ax5.set_title("5. Monthly Income Distribution")
        st.pyplot(fig5)

        fig6, ax6 = plt.subplots(figsize=(2.5,2))
        ax6.bar(df['JobSatisfaction'].value_counts().index, df['JobSatisfaction'].value_counts().values)
        ax6.set_title("6. JobSatisfaction Count")
        st.pyplot(fig6)

        fig7, ax7 = plt.subplots(figsize=(2.5,2))
        pd.crosstab(df['Gender'], df['Attrition']).plot(kind='bar', ax=ax7)
        ax7.set_title("7. Gender vs Attrition")
        st.pyplot(fig7)

        fig8, ax8 = plt.subplots(figsize=(2.5,2))
        colors = df['Attrition'].map({'Yes':'red','No':'green'})
        ax8.scatter(df['YearsatCompany'], df['MonthlyIncome'], c=colors, alpha=0.5)
        ax8.set_title("8. YearsAtCompany vs Income (Red=Yes)")
        st.pyplot(fig8)

    # 4. MODEL
    st.subheader("Attrition Prediction Model")
    X = df_encoded.drop('Attrition', axis=1, errors='ignore')
    y = df_encoded['Attrition']
    # unwanted columns drop
    for col in ['EmployeeCount','EmployeeNumber','Over18','StandardHours']:
        if col in X.columns:
            X = X.drop(col, axis=1)
    X = X.select_dtypes(include=['int64','float64'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    st.success(f"Model Accuracy: {acc*100:.2f} %")

except FileNotFoundError:
    st.error(f"File dorakaledu Bindu! Path check chey: {file_path}")
    st.write("keep HR-Analytics.csv file in the same folder")