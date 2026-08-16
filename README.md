# 🚗 Car Price Prediction with Machine Learning - CodeAlpha Task 3

## 📌 Project Overview
This project applies supervised machine learning regression techniques to predict used car market selling prices based on vehicular features such as current showroom price, vehicle age, mileage driven, fuel type, transmission, and ownership history.

---

## 👩‍💻 Student Information
- **Student Name**: Mariam Hossam Raghep
- **Student ID**: CA/DF1/224387
- **Internship Domain**: Data Science
- **Organization**: CodeAlpha

---

## ⚙️ Methodology & Implementation
1. **Feature Engineering**:
   - Converted release year to vehicle age (`Car_Age = Current_Year - Year`).
   - Applied One-Hot Encoding to handle categorical variables (`Fuel_Type`, `Seller_Type`, `Transmission`).
2. **Exploratory Analysis**: Generated feature correlation heatmaps to assess strong predictors of price.
3. **Machine Learning Algorithms**:
   - Linear Regression
   - Random Forest Regressor
   - Gradient Boosting Regressor
4. **Evaluation Metrics**: Evaluated models using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score.

---

## 🚀 How to Run
```bash
pip install -r requirements.txt
python car_price_prediction.py