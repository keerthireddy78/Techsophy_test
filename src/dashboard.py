import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix

def load_data(path):
    df = pd.read_csv(path)
    severity_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df["Claims_Severity_Numeric"] = df["Claims_Severity"].map(severity_map)
    return df

def show_aggregation(df):
    st.subheader("🔹 Aggregated Summary by Policy Type")
    summary = df.groupby("Policy_Type").agg({
        "Premium_Amount": "sum",
        "Claims_Frequency": "mean",
        "Claims_Severity_Numeric": "mean"
    }).reset_index()
    st.dataframe(summary)

    fig, ax = plt.subplots()
    sns.barplot(x="Policy_Type", y="Premium_Amount", data=summary, ax=ax)
    ax.set_title("Total Premium by Policy Type")
    st.pyplot(fig)

def show_concentration(df):
    st.subheader("🔹 Premium Concentration by Policy Type")
    total_premium = df["Premium_Amount"].sum()
    concentration = df.groupby("Policy_Type")["Premium_Amount"].sum() / total_premium * 100
    concentration = concentration.reset_index().rename(columns={"Premium_Amount": "Premium_Percentage"})
    st.dataframe(concentration)

    fig, ax = plt.subplots()
    sns.barplot(x="Policy_Type", y="Premium_Percentage", data=concentration, ax=ax)
    ax.set_title("Premium Concentration (%)")
    st.pyplot(fig)

    for _, row in concentration.iterrows():
        if row["Premium_Percentage"] > 60:
            st.error(f"⚠ High concentration risk in {row['Policy_Type']}: {row['Premium_Percentage']:.2f}%")

def show_correlation(df):
    st.subheader("🔹 Correlation Matrix")
    numeric_df = df[["Premium_Amount", "Credit_Score", "Claims_Frequency", "Claims_Severity_Numeric"]]
    corr = numeric_df.corr()

    st.dataframe(corr)

    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="Blues", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

def show_stress_test(df):
    st.subheader("🔹 Stress Test by Region")
    region = st.selectbox("Select Region", df["Region"].unique())
    affected = df[df["Region"] == region]
    st.write(f"Potential Premium at Risk: {affected['Premium_Amount'].sum()}")
    st.write(f"Affected Policies: {len(affected)}")

def train_ml_model(df):
    st.subheader("🔹 ML Risk Prediction")

    df["High_Risk"] = (df["Claims_Frequency"] > 0).astype(int)
    X = df[["Age", "Credit_Score", "Premium_Amount", "Claims_Severity_Numeric"]]
    
    enc = OneHotEncoder(drop='first')
    marital_encoded = enc.fit_transform(df[["Marital_Status"]]).toarray()
    marital_cols = enc.get_feature_names_out(["Marital_Status"])
    marital_df = pd.DataFrame(marital_encoded, columns=marital_cols, index=df.index)
    X = pd.concat([X, marital_df], axis=1)

    y = df["High_Risk"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    st.text("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Visualize risk probability
    df["Predicted_Risk"] = model.predict_proba(X)[:,1]
    st.subheader("Predicted Risk Probability Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["Predicted_Risk"], bins=20, kde=True, ax=ax)
    ax.set_xlabel("Predicted Risk Probability")
    st.pyplot(fig)

def main():
    st.title("📊 Insurance Portfolio Risk Management Dashboard")

    df = load_data(r"data/sample_data.csv")  # Update the path if needed

    st.sidebar.header("📌 Dashboard Sections")
    sections = st.sidebar.multiselect("Select Sections to Display", 
                                      ["Data Preview", "Aggregation", "Concentration", "Correlation", "Stress Test", "ML Risk Prediction"],
                                      default=["Aggregation", "Concentration", "Correlation"])

    if "Data Preview" in sections:
        st.subheader("🔹 Data Preview")
        st.dataframe(df.head())

    if "Aggregation" in sections:
        show_aggregation(df)

    if "Concentration" in sections:
        show_concentration(df)

    if "Correlation" in sections:
        show_correlation(df)

    if "Stress Test" in sections:
        show_stress_test(df)

    if "ML Risk Prediction" in sections:
        train_ml_model(df)

if __name__ == "__main__":
    main()
