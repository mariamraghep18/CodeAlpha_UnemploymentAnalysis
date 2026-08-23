"""Task 2: Unemployment Analysis with Python

Author: Mariam Hossam Raghep
Student ID: CA/DF1/224387
Domain: Data Science (CodeAlpha Internship)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

print("=" * 60)
print("TASK 2: UNEMPLOYMENT RATE ANALYSIS WITH PYTHON")
print("=" * 60)

data_url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/unemployment.csv"

try:
  df = pd.read_csv(data_url)
  print("[INFO] Online dataset loaded successfully.")
except Exception:
  print("[WARN] Loading fallback synthetic dataset...")
  dates = pd.date_range(start="2019-01-01", end="2020-10-01", freq="M")
  regions = ["North", "South", "East", "West", "Central"]
  areas = ["Urban", "Rural"]
  records = []
  for d in dates:
    for r in regions:
      for a in areas:
        base = 8.5 if d < pd.to_datetime("2020-03-01") else 19.5
        rate = base + np.random.uniform(-3, 6)
        records.append({
            "Region": r,
            "Date": d.strftime("%d-%m-%Y"),
            "Estimated Unemployment Rate (%)": round(rate, 2),
            "Estimated Employed": np.random.randint(100000, 500000),
            "Estimated Labour Participation Rate (%)": round(
                np.random.uniform(35, 50), 2
            ),
            "Area": a,
        })
  df = pd.DataFrame(records)

df.columns = df.columns.str.strip()
print("\n--- Dataset Information ---")
print(df.info())

df["Date"] = pd.to_datetime(df["Date"].str.strip(), dayfirst=True)
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month_name()

df["Covid_Period"] = np.where(df["Date"] >= "2020-03-01", "Covid", "Pre-Covid")

print("\n" + "=" * 60)
print("COVID-19 IMPACT ANALYSIS (PRE-COVID VS. COVID PERIOD):")
print("=" * 60)
covid_comparison = (
    df.groupby("Covid_Period")["Estimated Unemployment Rate (%)"]
    .agg(["mean", "median", "std", "max"])
    .reset_index()
)
print(covid_comparison)

sns.set_theme(style="whitegrid")

# 1. Distribution Plot
plt.figure(figsize=(9, 5))
sns.histplot(
    df["Estimated Unemployment Rate (%)"], kde=True, color="darkcyan", bins=20
)
plt.title("Distribution of Unemployment Rate (%)", fontsize=14)
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("unemployment_distribution.png", dpi=300)
plt.close()

# 2. Covid Comparison Barplot
plt.figure(figsize=(7, 5))
sns.barplot(
    data=df,
    x="Covid_Period",
    y="Estimated Unemployment Rate (%)",
    hue="Covid_Period",
    palette=["#2b5c8f", "#d95f02"],
    legend=False,
)
plt.title("Mean Unemployment Rate: Pre-Covid vs During Covid", fontsize=14)
plt.ylabel("Average Unemployment Rate (%)")
plt.tight_layout()
plt.savefig("covid_impact_comparison.png", dpi=300)
plt.close()

# 3. Time Series Trend
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=df,
    x="Date",
    y="Estimated Unemployment Rate (%)",
    hue="Area" if "Area" in df.columns else None,
    marker="o",
)
plt.axvline(
    pd.to_datetime("2020-03-01"),
    color="red",
    linestyle="--",
    label="Covid-19 Outbreak (March 2020)",
)
plt.title("Unemployment Rate Trend (2019 - 2020)", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.legend()
plt.tight_layout()
plt.savefig("unemployment_trend.png", dpi=300)
plt.close()

print("\n[SUCCESS] Generated and saved all visual charts:")
print("  - unemployment_distribution.png")
print("  - covid_impact_comparison.png")
print("  - unemployment_trend.png")
print("\n[COMPLETED] Task 2 executed successfully!")