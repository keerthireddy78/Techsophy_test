import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load data
df = pd.read_csv(r"C:\Users\saike\OneDrive - WOXSEN UNIVERSITY\Desktop\TS_test\Techsophy_test\data\sample_data.csv")

# Encode categorical variables
categorical_cols = ["Marital_Status", "Policy_Type", "Region"]
encoder = OneHotEncoder(drop="first")
encoded = encoder.fit_transform(df[categorical_cols]).toarray()
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

# Combine with numeric
numeric_cols = ["Age", "Premium_Amount", "Credit_Score", "Claims_Frequency"]
X = pd.concat([df[numeric_cols].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

# Target (you can adjust depending on your risk definition)
y = (df["Claims_Frequency"] > 0).astype(int)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model with higher iterations + saga solver
model = LogisticRegression(max_iter=5000, solver='saga')
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
