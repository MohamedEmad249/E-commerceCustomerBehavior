#main

import pandas as pd
import matplotlib.pyplot as plt
from models.traditional_ml import train_random_forest, predict
from utils.data_preprocessing import handle_missing_values, encode_categorical_columns, scale_data
from evaluation.evaluation import calculate_mse
from sklearn.model_selection import train_test_split

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

# Step 4: Train model
model = train_random_forest(X_train_scaled, y_train)

# Step 5: Make predictions
y_pred = predict(model, X_test_scaled)

# Step 6: Evaluate the model
mse = calculate_mse(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Optional: Plot Actual vs Predicted
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Total Spend')
plt.ylabel('Predicted Total Spend')
plt.title('Actual vs Predicted Total Spend')

# Save the chart as a PNG file
file_name = 'prediction_total_spend.png'
plt.savefig(file_name, format='png', dpi=300)
print(f"Chart saved as {file_name}")

plt.show()

# Step 7: Categorize Age Groups
print("\nCategorizing Age Groups...")
data['Age Group'] = pd.cut(
    data['Age'],
    bins=[0, 18, 24, 34, 44, 54, 64, 100],
    labels=['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
)

# Pie chart for Categorizing Age Groups
age_group_counts = data['Age'].value_counts()

# Plotting the pie chart
plt.figure(figsize=(8, 8))
plt.pie(age_group_counts, labels=age_group_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Customers by Age Group')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.tight_layout()

# Save the chart as a PNG file
file_name = 'age_group_distribution.png'
plt.savefig(file_name, format='png', dpi=300)
print(f"Chart saved as {file_name}")

plt.show()

# Step 8: Calculate City-Based Aggregates
print("\nCalculating City-Based Aggregates...")
city_agg = data.groupby('City').agg({
    'Total Spend': 'mean',
    'Items Purchased': 'mean',
    'Satisfaction Level': 'mean'
}).reset_index()

# Plot City-Based Aggregates
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Total Spend by City
axes[0].bar(city_agg['City'], city_agg['Total Spend'], color='skyblue')
axes[0].set_title('Average Total Spend by City')
axes[0].set_xlabel('City')
axes[0].set_ylabel('Average Total Spend')
axes[0].tick_params(axis='x', rotation=45)

# Items Purchased by City
axes[1].bar(city_agg['City'], city_agg['Items Purchased'], color='lightgreen')
axes[1].set_title('Average Items Purchased by City')
axes[1].set_xlabel('City')
axes[1].set_ylabel('Average Items Purchased')
axes[1].tick_params(axis='x', rotation=45)

# Satisfaction Level by City
axes[2].bar(city_agg['City'], city_agg['Satisfaction Level'], color='salmon')
axes[2].set_title('Average Satisfaction Level by City')
axes[2].set_xlabel('City')
axes[2].set_ylabel('Average Satisfaction Level')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()

# Save the chart as a PNG file
file_name = 'Calculating_City_based_aggregates.png'
plt.savefig(file_name, format='png', dpi=300)
print(f"Chart saved as {file_name}")

plt.show()

# Step 9: Calculate Items per Spend
print("\nCalculating Items per Spend...")
data['Items per Spend'] = data['Items Purchased'] / data['Total Spend']

# Plotting Frequency Diagram (Histogram) for Items per Spend
plt.figure(figsize=(12, 8))

# Plot the histogram with more bins for better resolution
plt.hist(data['Items per Spend'], bins=40, color='lightseagreen', edgecolor='black', alpha=0.7)

# Adding a title and labels with improved readability
plt.title('Distribution of Items per Spend', fontsize=16, weight='bold')
plt.xlabel('Frequency', fontsize=14)
plt.ylabel('Items per Spend', fontsize=14)

# Add gridlines (lighter and more subtle)
plt.grid(True, linestyle='--', alpha=0.7)

# Optional: Add lines for the mean and median
mean_items_per_spend = data['Items per Spend'].mean()
median_items_per_spend = data['Items per Spend'].median()

plt.axvline(mean_items_per_spend, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_items_per_spend:.2f}')
plt.axvline(median_items_per_spend, color='blue', linestyle='dashed', linewidth=2, label=f'Median: {median_items_per_spend:.2f}')

# Adding a legend to the plot
plt.legend(loc='upper right', fontsize=12)

# Show the plot
plt.tight_layout()

# Save the chart as a PNG file
file_name = 'frequency_diagram_items_per_spend_improved.png'
plt.savefig(file_name, format='png', dpi=300)
print(f"Chart saved as {file_name}")

plt.show()