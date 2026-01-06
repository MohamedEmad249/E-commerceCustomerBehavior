# E-commerce Customer Behavior Analysis

This project involves analyzing customer behavior data in an e-commerce setting, with a focus on predicting customer spending, categorizing customers based on their attributes, and visualizing key metrics. The analysis leverages machine learning techniques and data visualization for a comprehensive understanding of customer behavior.

## Project Overview

The script performs the following tasks:

1. **Data Loading and Preprocessing**:
   - Loads a customer dataset from a CSV file.
   - Handles missing values by filling them with the mean for numeric columns and the mode for categorical columns.
   - Encodes categorical variables using `LabelEncoder`.

2. **Prediction of Total Spend**:
   - Uses a Random Forest Regressor to predict the target variable `Total Spend` based on other features.
   - Evaluates the model's performance using Mean Squared Error (MSE).
   - Visualizes the comparison between actual and predicted `Total Spend`.

3. **Customer Demographics Analysis**:
   - Categorizes customers into age groups and performs one-hot encoding.
   - Generates a pie chart visualizing the distribution of customers across age groups.

4. **Items per Spend Analysis**:
   - Calculates the ratio of `Items Purchased` to `Total Spend` and visualizes the distribution using a histogram.

5. **City-Based Aggregates**:
   - Calculates city-wise aggregates for `Total Spend`, `Items Purchased`, and `Satisfaction Level`.
   - Visualizes these aggregates using bar charts.

6. **Recency Analysis**:
   - Categorizes customers based on the recency of their last purchase.
   - Visualizes the distribution of recency categories using bar charts.

## Requirements

Make sure you have the following Python libraries installed:

- pandas
- numpy
- matplotlib
- scikit-learn

You can install them using pip:

```bash
pip install pandas numpy matplotlib scikit-learn