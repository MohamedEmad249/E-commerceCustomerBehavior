# E-commerce Customer Behavior Analysis

A machine learning pipeline for analyzing US e-commerce customer behavior — predicting total spend, comparing models, and visualizing key metrics across 55 US cities.

## Project Structure

```bash
├── main.py                     # Entry point — runs the full pipeline
├── plots.py                    # All plot functions (auto-skips if already saved)
├── data/
│   └── Customer_Dataset_US.csv # 1,100 customers across 55 US cities
├── models/
│   └── traditional_ml.py       # Model definitions (RF, GB, LR, DT, KNN)
├── utils/
│   └── data_preprocessing.py   # Missing value handling, encoding, scaling
├── evaluation/
│   └── evaluation.py           # MSE calculation
└── plots/                      # Auto-created — saved plot images go here
    └──plots.py                    # All plot functions (auto-skips if already saved)
```

## What It Does

### 1. Data Loading & Preprocessing
- Loads `Customer_Dataset_US.csv` — 1,100 customers across 55 major US cities, grouped into 3 economic tiers
- Handles missing values (mean for numeric, mode for categorical)
- Label-encodes categorical columns: `Gender`, `City`, `Membership Type`, `Satisfaction Level`
- Scales features with `StandardScaler` — scaler is persisted for reuse on new customers

### 2. Model Training & Evaluation
- Trains a **Random Forest Regressor** to predict `Total Spend`
- Evaluates with **MSE, RMSE, MAE, and R²**
- Compares 5 models side-by-side:

| Model | Description |
|---|---|
| Linear Regression | Baseline linear model |
| Decision Tree | Single tree, interpretable |
| Random Forest | Ensemble of trees (primary model) |
| Gradient Boosting | Sequential boosting ensemble |
| KNN | K-nearest neighbors (k=5) |

### 3. Visualizations
Plots are saved to the `plots/` folder. **On re-runs, existing plots are skipped — no pop-up windows.**

| File | Description |
|---|---|
| `prediction_total_spend.png` | Actual vs Predicted scatter with MAE, RMSE, R² |
| `model_comparison.png` | Horizontal bar chart — RMSE + R² per model |
| `age_group_distribution.png` | Pie chart of customers by age group |
| `city_aggregates.png` | Top 20 cities — avg spend, items, satisfaction |
| `items_per_spend.png` | Distribution histogram with mean/median lines |
| `recency_distribution.png` | Recent / Lapsed / Dormant customer breakdown |

### 4. New Customer Prediction (Interactive)
At the end of each run, the terminal prompts for a new customer's details and predicts their total spend. All categorical fields are shown as **numbered horizontal menus** — no need to type raw values.

```
City:
  1. Albuquerque      2. Anaheim          3. Anchorage        4. Atlanta
  5. Aurora           6. Austin           7. Baltimore        8. Bakersfield
  ...
Enter number:
```

## Dataset

`Customer_Dataset_US.csv` contains **1,100 synthetic customers** across **55 US cities** in 3 economic tiers:

- **Tier 1** — High cost cities (NY, SF, LA, Seattle…): Gold membership heavy, spend $800–$2000
- **Tier 2** — Mid-tier cities (Houston, Dallas, Phoenix…): Silver heavy, spend $400–$1200
- **Tier 3** — Lower cost cities (Memphis, Fresno, Omaha…): Bronze heavy, spend $200–$800

### Columns

| Column | Type | Description |
|---|---|---|
| Customer ID | Integer | Unique identifier |
| Gender | String | Male / Female |
| Age | Integer | 18–70 |
| City | String | One of 55 US cities |
| Membership Type | String | Gold / Silver / Bronze |
| Total Spend | Float | Target variable ($) |
| Items Purchased | Integer | Number of items bought |
| Average Rating | Float | 1.0–5.0 |
| Discount Applied | Boolean | Whether a discount was used |
| Days Since Last Purchase | Integer | Recency indicator |
| Satisfaction Level | String | Satisfied / Neutral / Unsatisfied |

## Requirements

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Usage

```bash
python main.py
```

The pipeline runs fully automatically. At the end, follow the terminal prompts to predict spend for a new customer.
