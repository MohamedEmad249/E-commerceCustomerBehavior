#main

import pandas as pd
import matplotlib.pyplot as plt
from models.traditional_ml import train_random_forest, predict, get_all_models
from utils.data_preprocessing import handle_missing_values, encode_categorical_columns, scale_data
from evaluation.evaluation import calculate_mse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np

# Load data
file_path = "data/Customer_Dataset.csv"
data = pd.read_csv(file_path)

# Step 1: Data Preprocessing
data = handle_missing_values(data)

# Categorical columns to encode
categorical_columns = ['Gender', 'City', 'Membership Type', 'Satisfaction Level']
data, label_encoders = encode_categorical_columns(data, categorical_columns)

# Step 2: Prepare data for training
X = data.drop(columns=['Total Spend'])
y = data['Total Spend']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Scale data
X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

# Step 4: Train Random Forest model
model = train_random_forest(X_train_scaled, y_train)

# Step 5: Make predictions
y_pred = predict(model, X_test_scaled)

# Step 6: Evaluate the model
mse = calculate_mse(y_test, y_pred)
print("Mean Squared Error: ", mse)
print("Root Mean Squared Error: ", np.sqrt(mse))

print("Mean Absolute Error: ", mean_absolute_error(y_test, y_pred))

# Step 7: Model Comparison
print("\nModel Comparison:")
print("-" * 58)
print(f"{'Model':<25} {'MSE':>10} {'RMSE':>10} {'MAE':>10}")
print("-" * 58)

models = get_all_models()
for name, comp_model in models.items():
    comp_model.fit(X_train_scaled, y_train)
    y_pred_comp = comp_model.predict(X_test_scaled)
    mse_comp = mean_squared_error(y_test, y_pred_comp)
    rmse_comp = np.sqrt(mse_comp)
    mae_comp = mean_absolute_error(y_test, y_pred_comp)
    print(f"{name:<25} {mse_comp:>10.2f} {rmse_comp:>10.2f} {mae_comp:>10.2f}")

print("-" * 58)

#Plot Actual vs Predicted
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Total Spend')
plt.ylabel('Predicted Total Spend')
plt.title('Actual vs Predicted Total Spend')
file_name = 'prediction_total_spend.png'
plt.savefig(file_name, format='png', dpi=300)
print(f"Chart saved as {file_name}")
plt.show()

# Step 8: Categorize Age Groups
print("\nCategorizing Age Groups...")
data['Age Group'] = pd.cut(
    data['Age'],
    bins=[0, 18, 24, 34, 44, 54, 64, 100],
    labels=['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
)

age_group_counts = data['Age'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(age_group_counts, labels=age_group_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Customers by Age Group')
plt.axis('equal')
plt.tight_layout()
file_name = 'age_group_distribution.png'
plt.savefig(file_name, format='png', dpi=300)
print(f"Chart saved as {file_name}")
plt.show()

# Step 9: Calculate City-Based Aggregates
print("\nCalculating City-Based Aggregates...")
city_agg = data.groupby('City').agg({
    'Total Spend': 'mean',
    'Items Purchased': 'mean',
    'Satisfaction Level': 'mean'
}).reset_index()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].bar(city_agg['City'], city_agg['Total Spend'], color='skyblue')
axes[0].set_title('Average Total Spend by City')
axes[0].set_xlabel('City')
axes[0].set_ylabel('Average Total Spend')
axes[0].tick_params(axis='x', rotation=45)

axes[1].bar(city_agg['City'], city_agg['Items Purchased'], color='lightgreen')
axes[1].set_title('Average Items Purchased by City')
axes[1].set_xlabel('City')
axes[1].set_ylabel('Average Items Purchased')
axes[1].tick_params(axis='x', rotation=45)

axes[2].bar(city_agg['City'], city_agg['Satisfaction Level'], color='salmon')
axes[2].set_title('Average Satisfaction Level by City')
axes[2].set_xlabel('City')
axes[2].set_ylabel('Average Satisfaction Level')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
file_name = 'Calculating_City_based_aggregates.png'
plt.savefig(file_name, format='png', dpi=300)
print(f"Chart saved as {file_name}")
plt.show()

# Step 10: Calculate Items per Spend
print("\nCalculating Items per Spend...")
data['Items per Spend'] = data['Items Purchased'] / data['Total Spend']

plt.figure(figsize=(12, 8))
plt.hist(data['Items per Spend'], bins=40, color='lightseagreen', edgecolor='black', alpha=0.7)
plt.title('Distribution of Items per Spend', fontsize=16, weight='bold')
plt.xlabel('Frequency', fontsize=14)
plt.ylabel('Items per Spend', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

mean_items_per_spend = data['Items per Spend'].mean()
median_items_per_spend = data['Items per Spend'].median()

plt.axvline(mean_items_per_spend, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_items_per_spend:.2f}')
plt.axvline(median_items_per_spend, color='blue', linestyle='dashed', linewidth=2, label=f'Median: {median_items_per_spend:.2f}')
plt.legend(loc='upper right', fontsize=12)
plt.tight_layout()

file_name = 'frequency_diagram_items_per_spend_improved.png'
plt.savefig(file_name, format='png', dpi=300)
print(f"Chart saved as {file_name}")
plt.show()

# Step 11: Recency Analysis
def categorize_recency(days):
    if days <= 30:
        return 'Recent'
    elif days <= 60:
        return 'Lapsed'
    else:
        return 'Dormant'

data['Recency Category'] = data['Days Since Last Purchase'].apply(categorize_recency)
recency_counts = data['Recency Category'].value_counts()

plt.figure(figsize=(8, 5))
bars = plt.bar(recency_counts.index, recency_counts.values, color=['#2ecc71', '#e67e22', '#e74c3c'])
plt.title('Distribution of Customers by Recency Category', fontweight='bold')
plt.xlabel('Recency Category')
plt.ylabel('Number of Customers')

for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             str(int(bar.get_height())), ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('recency_distribution.png')
plt.show()

print("Recency Category")
print(data['Recency Category'].value_counts())
