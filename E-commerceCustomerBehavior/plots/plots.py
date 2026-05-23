#plots

import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

PLOT_STYLE = {
    'figure.facecolor': '#f9f9f9',
    'axes.facecolor':   '#f0f0f0',
    'axes.grid':        True,
    'grid.color':       'white',
    'grid.linewidth':   1.2,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'font.family':      'sans-serif',
}


def _plot(filename, draw_fn):
    """
    Render and save a plot.
    - First run  → saves + shows the window.
    - Later runs → skips rendering entirely (no window, no overhead).
    """
    filepath = os.path.join(PLOTS_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipped (already saved): {filepath}")
        return

    # Only switch to interactive backend when we actually need to show
    matplotlib.use('TkAgg')   # change to 'Qt5Agg' if Tk isn't installed
    plt.rcParams.update(PLOT_STYLE)

    draw_fn(filepath)


def plot_actual_vs_predicted(y_test, y_pred, mse, mae, r2):
    def draw(filepath):
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(y_test, y_pred, alpha=0.55, edgecolors='white',
                   linewidths=0.4, s=60, color='steelblue', label='Predictions')
        lims = [min(y_test.min(), y_pred.min()) - 50,
                max(y_test.max(), y_pred.max()) + 50]
        ax.plot(lims, lims, '--', color='#e74c3c', linewidth=1.8, label='Perfect fit')
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('Actual Total Spend ($)', fontsize=12)
        ax.set_ylabel('Predicted Total Spend ($)', fontsize=12)
        ax.set_title('Actual vs Predicted Total Spend\n(Random Forest)',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.text(0.04, 0.89,
                f'MAE  = ${mae:.1f}\nRMSE = ${np.sqrt(mse):.1f}\nR²   = {r2:.4f}',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.show()
        plt.close()
    _plot('prediction_total_spend.png', draw)


def plot_model_comparison(results):
    def draw(filepath):
        fig, ax = plt.subplots(figsize=(9, 5))
        names  = list(results.keys())
        rmses  = [results[n]['RMSE'] for n in names]
        r2s    = [results[n]['R2']   for n in names]
        colors = ['#2ecc71' if r == min(rmses) else '#3498db' for r in rmses]
        bars   = ax.barh(names, rmses, color=colors, edgecolor='white', height=0.55)
        for bar, val, r2 in zip(bars, rmses, r2s):
            ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}  (R²={r2:.3f})', va='center', fontsize=9)
        ax.set_xlabel('RMSE (lower is better)', fontsize=12)
        ax.set_title('Model Comparison — RMSE & R²', fontsize=13, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.show()
        plt.close()
    _plot('model_comparison.png', draw)


def plot_age_group_distribution(raw_data):
    def draw(filepath):
        raw = raw_data.copy()
        raw['Age Group'] = pd.cut(
            raw['Age'],
            bins=[0, 18, 24, 34, 44, 54, 64, 100],
            labels=['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        )
        age_group_counts = raw['Age Group'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(9, 9))
        wedge_colors = ['#3498db','#2ecc71','#e67e22','#9b59b6','#e74c3c','#1abc9c','#f39c12']
        wedges, texts, autotexts = ax.pie(
            age_group_counts, labels=age_group_counts.index,
            autopct='%1.1f%%', startangle=140, colors=wedge_colors,
            pctdistance=0.82, wedgeprops=dict(edgecolor='white', linewidth=1.5)
        )
        for t in autotexts:
            t.set_fontsize(10); t.set_fontweight('bold')
        ax.set_title('Customer Distribution by Age Group', fontsize=14, fontweight='bold', pad=20)
        legend_labels = [f"{g}  ({c} customers)"
                         for g, c in zip(age_group_counts.index, age_group_counts.values)]
        ax.legend(wedges, legend_labels, loc='lower center',
                  bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=9, frameon=False)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.show()
        plt.close()
    _plot('age_group_distribution.png', draw)


def plot_city_aggregates(raw_data):
    def draw(filepath):
        sat_map = {'Satisfied': 3, 'Neutral': 2, 'Unsatisfied': 1}
        raw = raw_data.copy()
        raw['Sat Score'] = raw['Satisfaction Level'].map(sat_map)
        city_agg = raw.groupby('City').agg(
            Total_Spend=('Total Spend', 'mean'),
            Items_Purchased=('Items Purchased', 'mean'),
            Satisfaction=('Sat Score', 'mean')
        ).reset_index().sort_values('Total_Spend', ascending=False).head(20)

        fig, axes = plt.subplots(3, 1, figsize=(14, 16))
        fig.suptitle('City-Based Aggregates — Top 20 Cities by Avg Spend',
                     fontsize=15, fontweight='bold', y=1.01)
        metrics = [
            ('Total_Spend',     'Avg Total Spend ($)',          '#3498db'),
            ('Items_Purchased', 'Avg Items Purchased',          '#2ecc71'),
            ('Satisfaction',    'Avg Satisfaction (1–3 scale)', '#e67e22'),
        ]
        for ax, (col, ylabel, color) in zip(axes, metrics):
            bars = ax.bar(city_agg['City'], city_agg[col], color=color,
                          edgecolor='white', width=0.65)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_xticks(range(len(city_agg['City'])))
            ax.set_xticklabels(city_agg['City'], rotation=40, ha='right', fontsize=9)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(
                lambda x, _, c=col: f'${x:,.0f}' if c == 'Total_Spend' else f'{x:.1f}'
            ))
            for bar in bars:
                h = bar.get_height()
                label = f'${h:,.0f}' if col == 'Total_Spend' else f'{h:.1f}'
                ax.text(bar.get_x() + bar.get_width() / 2, h + h * 0.01,
                        label, ha='center', va='bottom', fontsize=7.5)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.show()
        plt.close()
    _plot('city_aggregates.png', draw)


def plot_items_per_spend(raw_data):
    def draw(filepath):
        raw = raw_data.copy()
        raw['Items per Spend'] = raw['Items Purchased'] / raw['Total Spend']
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.hist(raw['Items per Spend'], bins=45, color='#3498db',
                edgecolor='white', alpha=0.85)
        mean_ips   = raw['Items per Spend'].mean()
        median_ips = raw['Items per Spend'].median()
        ax.axvline(mean_ips,   color='#e74c3c', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_ips:.4f}')
        ax.axvline(median_ips, color='#2ecc71', linestyle='--', linewidth=2,
                   label=f'Median: {median_ips:.4f}')
        ax.set_xlabel('Items per Dollar Spent', fontsize=12)
        ax.set_ylabel('Number of Customers', fontsize=12)
        ax.set_title('Distribution of Items per Spend', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.show()
        plt.close()
    _plot('items_per_spend.png', draw)


def plot_recency_distribution(raw_data):
    def draw(filepath):
        def categorize_recency(days):
            if days <= 30:   return 'Recent'
            elif days <= 60: return 'Lapsed'
            else:            return 'Dormant'
        raw = raw_data.copy()
        raw['Recency Category'] = raw['Days Since Last Purchase'].apply(categorize_recency)
        recency_counts = raw['Recency Category'].value_counts().reindex(['Recent', 'Lapsed', 'Dormant'])
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = ['#2ecc71', '#e67e22', '#e74c3c']
        bars   = ax.bar(recency_counts.index, recency_counts.values,
                        color=colors, edgecolor='white', width=0.5)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 5,
                    str(int(h)), ha='center', fontsize=12, fontweight='bold')
        ax.set_xlabel('Recency Category', fontsize=12)
        ax.set_ylabel('Number of Customers', fontsize=12)
        ax.set_title('Customer Distribution by Recency\n(Recent ≤30d | Lapsed ≤60d | Dormant >60d)',
                     fontsize=13, fontweight='bold')
        ax.set_ylim(0, recency_counts.max() * 1.15)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.show()
        plt.close()
    _plot('recency_distribution.png', draw)
