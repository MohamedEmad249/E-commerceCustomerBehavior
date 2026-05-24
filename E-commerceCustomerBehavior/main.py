#main

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from models.traditional_ml import train_random_forest, predict, get_all_models
from utils.data_preprocessing import handle_missing_values, encode_categorical_columns, scale_data
from evaluation.evaluation import calculate_mse, calculate_mae, calculate_r2
from plots.plots import (
    plot_actual_vs_predicted,
    plot_model_comparison,
    plot_age_group_distribution,
    plot_city_aggregates,
    plot_items_per_spend,
    plot_recency_distribution,
)

# ──────────────────────────────────────────────
# Load & preprocess
# ──────────────────────────────────────────────
file_path = "data/Customer_Dataset_US.csv"
data = pd.read_csv(file_path)
raw_data = data.copy()

data = handle_missing_values(data)
categorical_columns = ['Gender', 'City', 'Membership Type', 'Satisfaction Level']
data, label_encoders = encode_categorical_columns(data, categorical_columns)

# ──────────────────────────────────────────────
# ML pipeline
# ──────────────────────────────────────────────
X = data.drop(columns=['Total Spend'])
y = data['Total Spend']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

model = train_random_forest(X_train_scaled, y_train)
y_pred = predict(model, X_test_scaled)

mse  = calculate_mse(y_test, y_pred)
mae  = calculate_mae(y_test, y_pred)
r2   = calculate_r2(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n\n")
print("="*35)
print("Random Forest Regressor Performance:")
print(f"Mean Squared Error : {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"RMSE               : {rmse:.2f}")
print(f"R² Score           : {r2:.4f}")
print("="*35)

# ──────────────────────────────────────────────
# Model comparison
# ──────────────────────────────────────────────
print("\n\nModel Comparison:")
print("-" * 67)
print(f"{'Model':<25} {'MSE':>10} {'RMSE':>10} {'MAE':>10} {'R2':>8}")
print("-" * 67)

results = {}
for name, comp_model in get_all_models().items():
    comp_model.fit(X_train_scaled, y_train)
    yp     = comp_model.predict(X_test_scaled)
    mse  = calculate_mse(y_test, yp)
    rmse = np.sqrt(mse)
    mae  = calculate_mae(y_test, yp)
    r2   = calculate_r2(y_test, yp)
    results[name] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}
    print(f"{name:<25} {mse:>10.2f} {rmse:>10.2f} {mae:>10.2f} {r2:>8.4f}")

print("-" * 67)
print("best model is: ", max(results, key=lambda k: results[k]['R2']))
print("-" * 67)

# ──────────────────────────────────────────────
# Plots (skips entirely if file already exists)
# ──────────────────────────────────────────────
print("\nGenerating plots...")
plot_model_comparison(results)
plot_actual_vs_predicted(y_test, y_pred, mse, mae, r2)
plot_age_group_distribution(raw_data)
plot_city_aggregates(raw_data)
plot_items_per_spend(raw_data)
plot_recency_distribution(raw_data)
print("Plots saved to 'plots/' directory.")
print("\n")
# ──────────────────────────────────────────────
# New customer — terminal input
# ──────────────────────────────────────────────
print("\n" + "=" * 50)
print("                 NEW CUSTOMER SPEND PREDICTION")
print("=" * 50)

valid_genders     = sorted(label_encoders['Gender'].classes_)
valid_cities      = sorted(label_encoders['City'].classes_)
valid_memberships = sorted(label_encoders['Membership Type'].classes_)
valid_sats        = sorted(label_encoders['Satisfaction Level'].classes_)


def prompt_choice(field, options, cols=4):
    """Numbered menu printed in a horizontal grid."""
    print(f"\n{field}:")
    # Build fixed-width columns so options wrap neatly
    col_width = max(len(f"{i+1}. {opt}") for i, opt in enumerate(options)) + 3
    for i, opt in enumerate(options):
        label = f"{i+1}. {opt}"
        end   = '\n' if (i + 1) % cols == 0 or i == len(options) - 1 else ''
        print(f"  {label:<{col_width}}", end=end)
    while True:
        try:
            idx = int(input("Enter number: ").strip())
            if 1 <= idx <= len(options):
                return options[idx - 1]
        except ValueError:
            pass
        print(f"  Invalid — enter a number between 1 and {len(options)}.")


def prompt_int(field, lo, hi):
    while True:
        try:
            val = int(input(f"{field} ({lo}–{hi}): ").strip())
            if lo <= val <= hi:
                return val
        except ValueError:
            pass
        print(f"  Invalid — enter an integer between {lo} and {hi}.")


def prompt_float(field, lo, hi):
    while True:
        try:
            val = float(input(f"{field} ({lo}–{hi}): ").strip())
            if lo <= val <= hi:
                return val
        except ValueError:
            pass
        print(f"  Invalid — enter a number between {lo} and {hi}.")


def prompt_bool(field):
    while True:
        val = input(f"{field} (y/n): ").strip().lower()
        if val in ('y', 'yes', '1', 'true'):  return 1
        if val in ('n', 'no',  '0', 'false'): return 0
        print("  Invalid — enter y or n.")


gender       = prompt_choice("Gender",           valid_genders,     cols=4)
age          = prompt_int   ("Age",              18, 100)
city         = prompt_choice("City",             valid_cities,      cols=4)
membership   = prompt_choice("Membership Type",  valid_memberships, cols=4)
items        = prompt_int   ("Items Purchased",  1, 100)
rating       = prompt_float ("Average Rating",   1.0, 5.0)
discount     = prompt_bool  ("Discount Applied")
days         = prompt_int   ("Days Since Last Purchase", 0, 365)
satisfaction = prompt_choice("Satisfaction Level", valid_sats,      cols=4)

new_customer = pd.DataFrame([{
    'Customer ID':              9999,
    'Gender':                   label_encoders['Gender'].transform([gender])[0],
    'Age':                      age,
    'City':                     label_encoders['City'].transform([city])[0],
    'Membership Type':          label_encoders['Membership Type'].transform([membership])[0],
    'Items Purchased':          items,
    'Average Rating':           rating,
    'Discount Applied':         discount,
    'Days Since Last Purchase': days,
    'Satisfaction Level':       label_encoders['Satisfaction Level'].transform([satisfaction])[0],
}])[X.columns]

new_customer_scaled = scaler.transform(new_customer)
predicted_spend = model.predict(new_customer_scaled)

print("\n" + "-" * 50)
print(f"Predicted Total Spend: ${predicted_spend[0]:.2f}")
print("-" * 50)
