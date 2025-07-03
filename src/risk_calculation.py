import pandas as pd

def calculate_risk(filepath):
    df = pd.read_csv(filepath)

    # Total portfolio stats
    total_premium = df["Premium_Amount"].sum()
    total_coverage = df["coverage"].sum() if "coverage" in df.columns else 0
    print(f"Total Premium: {total_premium}")
    print(f"Total Coverage: {total_coverage}")

    # Premium concentration
    premium_by_type = df.groupby("Policy_Type")["Premium_Amount"].sum()
    concentration = (premium_by_type / total_premium * 100).reset_index()
    concentration.columns = ["Policy_Type", "Premium_Percentage"]

    print("\nPremium Concentration by Policy Type:")
    print(concentration)

    return concentration

if __name__ == "__main__":
    calculate_risk(r"C:\Users\saike\OneDrive - WOXSEN UNIVERSITY\Desktop\TS_test\Techsophy_test\data\sample_data.csv")
