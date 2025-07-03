import pandas as pd

def stress_test(filepath, region):
    df = pd.read_csv(filepath)

    affected = df[df["Region"] == region]

    total_premium = affected["Premium_Amount"].sum()
    total_coverage = affected["coverage"].sum() if "coverage" in df.columns else 0

    print(f"\nStress Test Result: {region}")
    print(f"Number of policies affected: {len(affected)}")
    print(f"Potential premium at risk: {total_premium}")
    print(f"Potential coverage exposure: {total_coverage}")

if __name__ == "__main__":
    # Example: test Urban region
    stress_test(r"C:\Users\saike\OneDrive - WOXSEN UNIVERSITY\Desktop\TS_test\Techsophy_test\data\sample_data.csv", "Urban")
