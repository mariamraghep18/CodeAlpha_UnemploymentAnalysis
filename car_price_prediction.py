"""Task 3: Car Price Prediction with Machine Learning

Author: Mariam Hossam Raghep
Student ID: CA/DF1/224387
Domain: Data Science (CodeAlpha Internship)
"""

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

print("=" * 60)
print("TASK 3: CAR PRICE PREDICTION WITH MACHINE LEARNING")
print("=" * 60)

data_url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/car%20data.csv"

try:
  df = pd.read_csv(data_url)
  print("[INFO] Online car dataset loaded successfully.")
except Exception:
  print("[WARN] Generating synthetic dataset...")
  np.random.seed(42)
  n = 300
  years = np.random.randint(2008, 2022, n)
  present_prices = np.random.uniform(3.0, 35.0, n)
  kms = np.random.randint(5000, 120000, n)
  fuel = np.random.choice(["Petrol", "Diesel", "CNG"], n)
  seller = np.random.choice(["Dealer", "Individual"], n)
  trans = np.random.choice(["Manual", "Automatic"], n)
  owners = np.random.randint(0, 3, n)
  selling_prices = (
      present_prices * 0.65
      - (2024 - years) * 0.2
      - (kms / 100000) * 0.5
      + np.random.normal(0, 0.8, n)
  )
  selling_prices = np.clip(selling_prices, 0.5, 35.0)
  df = pd.DataFrame({
      "Car_Name": ["car_" + str(i) for i in range(n)],
      "Year": years,
      "Selling_Price": np.round(selling_prices, 2),
      "Present_Price": np.round(present_prices, 2),
      "Kms_Driven": kms,
      "Fuel_Type": fuel,
      "Seller_Type": seller,
      "Transmission": trans,
      "Owner": owners,
  })

print("\n--- First 5 Rows of Dataset ---")
print(df.head())

print("\n--- Missing Values Check ---")
print(df.isnull().sum())

# Feature Engineering
current_year = datetime.now().year
df["Car_Age"] = current_year - df["Year"]

if "Car_Name" in df.columns:
  df.drop(columns=["Car_Name"], inplace=True)
df.drop(columns=["Year"], inplace=True)

df = pd.get_dummies(df, drop_first=True)

print("\n--- Processed Features Sample ---")
print(df.head())

print("\n[INFO] Generating correlation heatmap...")
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Car Features Correlation Matrix", fontsize=14)
plt.tight_layout()
plt.savefig("car_correlation_matrix.png", dpi=300)
plt.close()
print("[SUCCESS] Saved: car_correlation_matrix.png")

X = df.drop(columns=["Selling_Price"])
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

regressors = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=100, random_state=42
    ),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
}

print("\n" + "=" * 60)
print("REGRESSION MODEL PERFORMANCE:")
print("=" * 60)

best_r2 = -float("inf")
best_rf = None

for name, model in regressors.items():
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)

  mae = mean_absolute_error(y_test, y_pred)
  rmse = np.sqrt(mean_squared_error(y_test, y_pred))
  r2 = r2_score(y_test, y_pred)

  print(f"\n[MODEL] {name}")
  print(f"  Mean Absolute Error (MAE)     : {mae:.3f}")
  print(f"  Root Mean Squared Error (RMSE): {rmse:.3f}")
  print(f"  R-Squared Score (R2)          : {r2:.3f} ({r2*100:.1f}%)")

  if r2 > best_r2:
    best_r2 = r2
    if name == "Random Forest Regressor":
      best_rf = model

if best_rf is not None:
  print("\n[INFO] Generating feature importance plot...")
  importances = pd.Series(best_rf.feature_importances_, index=X.columns)
  plt.figure(figsize=(8, 5))
  importances.sort_values().plot(kind="barh", color="#e6550d")
  plt.title("Feature Importance - Random Forest Regressor", fontsize=14)
  plt.xlabel("Importance Score")
  plt.tight_layout()
  plt.savefig("car_feature_importance.png", dpi=300)
  plt.close()
  print("[SUCCESS] Saved: car_feature_importance.png")

print("\n[COMPLETED] Task 3 executed successfully!")