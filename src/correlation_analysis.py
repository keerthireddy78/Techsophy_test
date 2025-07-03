import pandas as pd

def correlation_analysis(filepath):
    df = pd.read_csv(filepath)

    # Convert Claims_Severity to numeric (Low=1, Medium=2, High=3)
    severity_map = {"Low": 1, "Medium": 2, "High": 3}
    df["Claims_Severity_Numeric"] = df["Claims_Severity"].map(severity_map)

    # Select numeric columns
    numeric_cols = ["Premium_Amount", "Credit_Score", "Claims_Frequency", "Claims_Severity_Numeric"]
    numeric_df = df[numeric_cols]

    # Compute correlation matrix
    corr_matrix = numeric_df.corr()

    print("\nCorrelation Matrix:")
    print(corr_matrix)

    return corr_matrix

if __name__ == "__main__":
    correlation_analysis(r"C:\Users\saike\OneDrive - WOXSEN UNIVERSITY\Desktop\TS_test\Techsophy_test\data\sample_data.csv")
