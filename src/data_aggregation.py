import pandas as pd

def load_and_aggregate_data(filepath):
    df = pd.read_csv(filepath)

    df = df.rename(columns={
        "Premium_Amount": "premium",
        "Policy_Type": "policy_type",
        "Region": "region",
    })

    # Convert Claims_Frequency & Claims_Severity to numeric
    df["Claims_Frequency"] = pd.to_numeric(df["Claims_Frequency"], errors="coerce").fillna(0)
    df["Claims_Severity"] = pd.to_numeric(df["Claims_Severity"], errors="coerce").fillna(0)

    # Compute risk_score
    df["risk_score"] = (df["Claims_Frequency"] + df["Claims_Severity"]) / 2
    df["risk_score"] = df["risk_score"] / df["risk_score"].max()

    # Estimate coverage
    df["coverage"] = df["premium"] * 20

    summary = df.groupby("policy_type").agg({
        "premium": "sum",
        "coverage": "sum",
        "risk_score": "mean"
    }).reset_index()

    print("Aggregated Summary:")
    print(summary)

    return df, summary

if __name__ == "__main__":
    load_and_aggregate_data(r"C:\Users\saike\OneDrive - WOXSEN UNIVERSITY\Desktop\TS_test\Techsophy_test\data\sample_data.csv")
