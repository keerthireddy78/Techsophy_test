# Insurance Portfolio Risk Management Dashboard

## Overview

This project implements a **risk management dashboard** that monitors and analyzes the overall risk exposure across an insurance company's policy portfolio.
It identifies risk concentrations, detects correlations between key metrics, performs stress testing, and applies machine learning for risk prediction.

## About the Dataset (from Kaggle)

The dataset I took is a synthetic but realistic representation of personal auto insurance data, generated using real-world statistics.
It is safe for public use and ideal for data science projects in insurance risk management.
The dataset used contains **10,000 rows** of policy data.

## Features

- Data aggregation by policy type
- Premium concentration analysis with alert system for concentration risks
- Correlation matrix of key risk factors
- Stress testing by region
- Machine Learning model (Logistic Regression) for risk prediction
- Interactive visualizations (bar charts, pie charts, heatmaps)

## Architecture

1. **Data Loading:** Load policy data from CSV (10,000 rows).
2. **Aggregation:** Summarize premiums, claims frequency, and severity by policy type.
3. **Concentration Risk:** Identify and alert if premium concentration exceeds 60%.
4. **Correlation Analysis:** Analyze relationships between numeric features.
5. **Stress Testing:** Simulate exposure by region.
6. **ML Risk Prediction:** Predict high-risk policies using Logistic Regression.
7. **Dashboard:** Streamlit app displaying interactive tables and visualizations.

## Project Structure

```
Techsophy_test/
├── data/
│   └── sample_data.csv
├── src/
│   ├── data_aggregation.py
│   ├── risk_calculation.py
│   ├── correlation_analysis.py
│   ├── stress_testing.py
│   ├── risk_ML_model.py
│   └── dashboard.py
├── requirements.txt
└── README.md

```

## How to Run

1. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

2. Launch the dashboard:

   ```
   streamlit run src/dashboard.py
   ```

## Outputs

- Aggregated summary of key metrics by policy type
- Premium concentration table and pie chart with risk alerts
- Correlation matrix of numeric features
- Stress test exposure by region
- High-risk policy prediction (ML-based)
- in streamlit interface make sure to select required dashboard sections(Select Sections to Display)

## Notes

- The default risk alert threshold is set at **60% premium concentration**.
- You can update `sample_data.csv` or point to another dataset in the code.
- The system is modular: separate scripts for aggregation, risk calculation, correlation analysis, stress testing, and visualization.
