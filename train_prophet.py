import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MAE

sns.set_palette("husl")

# ========== CONFIGURATION ==========
DATABASE_URL = "postgresql://postgres:%40Dheeraj123@localhost:5432/supply_chain_db"
PRODUCT_ID_TO_FORECAST = 1073

SEQ_LEN = 56
PRED_LEN = 14
TRAIN_RATIO = 0.7

print("="*70)
print("N-HiTS: Neural Hierarchical Interpolation for Time Series")
print("="*70)

# Connect to database
try:
    engine = create_engine(DATABASE_URL)
    print("\n✓ Connected to database")
except Exception as e:
    print(f"✗ Database connection failed: {e}")
    sys.exit(1)

# Fetch data
print(f"✓ Fetching sales data for Product ID: {PRODUCT_ID_TO_FORECAST}...")
sql_query = text("""
    SELECT
        DATE_TRUNC('day', order_date) AS ds,
        SUM(order_item_quantity) AS y
    FROM order_items
    WHERE product_card_id = :product_id
    GROUP BY ds
    ORDER BY ds;
""")

try:
    df = pd.read_sql(sql_query, engine, params={'product_id': PRODUCT_ID_TO_FORECAST})
    if len(df) < SEQ_LEN + PRED_LEN:
        print(f"✗ Not enough data. Need at least {SEQ_LEN + PRED_LEN} days.")
        sys.exit(1)
    print(f"✓ Successfully fetched {len(df)} daily observations")
except Exception as e:
    print(f"✗ Error fetching data: {e}")
    sys.exit(1)

# Prepare data
df['ds'] = pd.to_datetime(df['ds'])
df = df.sort_values('ds').reset_index(drop=True)

date_range = pd.date_range(df['ds'].min(), df['ds'].max(), freq='D')
full_df = pd.DataFrame({'ds': date_range})
df = full_df.merge(df, on='ds', how='left').fillna(0)

df['unique_id'] = 'product_1073'
df = df[['unique_id', 'ds', 'y']]

print(f"✓ Date range: {df['ds'].min().date()} to {df['ds'].max().date()}")
print(f"✓ Total days: {len(df)}")

# ========== USER INPUT ==========
print(f"\n{'='*70}")
print("FUTURE FORECAST CONFIGURATION")
print(f"{'='*70}")
print(f"Available forecast horizons: 7, 14, 21, 28, 30, 60, 90 days")

while True:
    try:
        FUTURE_DAYS = int(input(f"\nHow many days to forecast into the future? [default: 14]: ") or "14")
        valid_horizons = [7, 14, 21, 28, 30, 60, 90]
        if FUTURE_DAYS not in valid_horizons:
            closest = min(valid_horizons, key=lambda x: abs(x - FUTURE_DAYS))
            print(f"⚠ {FUTURE_DAYS} days not optimal. Using closest: {closest} days")
            FUTURE_DAYS = closest
        if FUTURE_DAYS < 1 or FUTURE_DAYS > 180:
            print("⚠ Please enter a value between 1 and 180 days")
            continue
        break
    except ValueError:
        print("⚠ Please enter a valid number")
        continue

print(f"✓ Will forecast {FUTURE_DAYS} days into the future")

# ========== TRAIN/TEST VALIDATION ==========
print(f"\n{'='*70}")
print("PART 1: MODEL VALIDATION (70/30 SPLIT)")
print(f"{'='*70}")

train_size = int(len(df) * TRAIN_RATIO)
train_df = df[:train_size].copy()
test_df = df[train_size:].copy()

print(f"Train: {len(train_df)} days ({train_df['ds'].min().date()} to {train_df['ds'].max().date()})")
print(f"Test:  {len(test_df)} days ({test_df['ds'].min().date()} to {test_df['ds'].max().date()})")

print("\nTraining N-HiTS model...")
validation_model = NHITS(
    h=PRED_LEN, input_size=SEQ_LEN, loss=MAE(), max_steps=300,
    val_check_steps=50, early_stop_patience_steps=3, learning_rate=1e-3,
    stack_types=['identity', 'identity', 'identity'], n_blocks=[1, 1, 1],
    mlp_units=[[256, 256], [256, 256], [256, 256]],
    n_pool_kernel_size=[2, 2, 1], n_freq_downsample=[4, 2, 1],
    pooling_mode='MaxPool1d', interpolation_mode='linear',
    batch_size=32, random_seed=42, scaler_type='robust',
)

nf_validation = NeuralForecast(models=[validation_model], freq='D')
nf_validation.fit(df=train_df, val_size=56)
print("✓ Validation model trained")

# Evaluate
print("\nEvaluating on test set...")
predictions_list, actuals_list, dates_list = [], [], []

step_size = PRED_LEN
num_windows = (len(test_df) - PRED_LEN) // step_size + 1

for window_idx in range(num_windows):
    start_idx = window_idx * step_size
    if start_idx + PRED_LEN > len(test_df):
        break
    cutoff_point = train_size + start_idx
    input_df = df[:cutoff_point].copy()
    if len(input_df) < SEQ_LEN:
        continue
    try:
        forecast = nf_validation.predict(df=input_df)
        pred_values = forecast['NHITS'].values[:PRED_LEN]
        actual_values = test_df.iloc[start_idx:start_idx + PRED_LEN]['y'].values
        actual_dates = test_df.iloc[start_idx:start_idx + PRED_LEN]['ds'].values
        predictions_list.extend(pred_values)
        actuals_list.extend(actual_values)
        dates_list.extend(actual_dates)
        print(f"  ✓ Window {window_idx + 1}/{num_windows}")
    except Exception as e:
        print(f"  ✗ Window {window_idx + 1} failed: {e}")
        continue

predictions = np.array(predictions_list)
actuals = np.array(actuals_list)
dates = pd.DatetimeIndex(dates_list)

mae = mean_absolute_error(actuals, predictions)
rmse = np.sqrt(mean_squared_error(actuals, predictions))
non_zero_mask = actuals != 0
if non_zero_mask.sum() > 0:
    mape = np.mean(np.abs((actuals[non_zero_mask] - predictions[non_zero_mask]) / actuals[non_zero_mask])) * 100
else:
    mape = 0.0

print(f"\n{'='*70}")
print("VALIDATION METRICS")
print(f"{'='*70}")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"{'='*70}")

# ========== FUTURE FORECASTING ==========
print(f"\n{'='*70}")
print(f"PART 2: FUTURE FORECAST ({FUTURE_DAYS} DAYS)")
print(f"{'='*70}")

if FUTURE_DAYS != PRED_LEN:
    print(f"\nRetraining model with h={FUTURE_DAYS}...")
    future_model = NHITS(
        h=FUTURE_DAYS, input_size=SEQ_LEN, loss=MAE(), max_steps=300,
        val_check_steps=50, early_stop_patience_steps=3, learning_rate=1e-3,
        stack_types=['identity', 'identity', 'identity'], n_blocks=[1, 1, 1],
        mlp_units=[[256, 256], [256, 256], [256, 256]],
        n_pool_kernel_size=[2, 2, 1], n_freq_downsample=[4, 2, 1],
        pooling_mode='MaxPool1d', interpolation_mode='linear',
        batch_size=32, random_seed=42, scaler_type='robust',
    )
    nf_future = NeuralForecast(models=[future_model], freq='D')
    nf_future.fit(df=df, val_size=56)
    print("✓ Model retrained")
else:
    nf_future = nf_validation

print(f"\nGenerating {FUTURE_DAYS}-day forecast...")
future_forecast = nf_future.predict(df=df)
future_predictions = future_forecast['NHITS'].values

last_date = df['ds'].max()
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=FUTURE_DAYS, freq='D')

print(f"✓ Forecast period: {future_dates[0].date()} to {future_dates[-1].date()}")

# ========== EDA FOR FORECASTED DATA ==========
print(f"\n{'='*70}")
print("EXPLORATORY DATA ANALYSIS: FORECASTED DATA")
print(f"{'='*70}")

# Create forecast dataframe with integer index
forecast_df = pd.DataFrame({
    'date': future_dates,
    'demand': future_predictions,
    'day_of_week': future_dates.day_name(),
    'week_number': future_dates.isocalendar().week,
    'month': future_dates.month_name(),
    'is_weekend': future_dates.dayofweek >= 5,
    'day_of_month': future_dates.day
})
forecast_df = forecast_df.reset_index(drop=True)  # Ensure integer index

# Historical data analysis
historical_last_90 = df.tail(90).copy()
historical_last_90['day_of_week'] = pd.to_datetime(historical_last_90['ds']).dt.day_name()
historical_last_90['is_weekend'] = pd.to_datetime(historical_last_90['ds']).dt.dayofweek >= 5

# Statistical tests
print("\n1. STATISTICAL SUMMARY")
print("-" * 70)
print(f"Mean:                    {forecast_df['demand'].mean():.2f}")
print(f"Median:                  {forecast_df['demand'].median():.2f}")
print(f"Std Deviation:           {forecast_df['demand'].std():.2f}")
print(f"Coefficient of Variation: {(forecast_df['demand'].std() / forecast_df['demand'].mean() * 100):.2f}%")
print(f"Skewness:                {stats.skew(forecast_df['demand']):.3f}")
print(f"Kurtosis:                {stats.kurtosis(forecast_df['demand']):.3f}")

# Fix for min/max day reporting
min_idx = forecast_df['demand'].idxmin()
max_idx = forecast_df['demand'].idxmax()
print(f"Min:                     {forecast_df['demand'].min():.2f} (Day {min_idx + 1}, {forecast_df.loc[min_idx, 'date'].strftime('%Y-%m-%d')})")
print(f"Max:                     {forecast_df['demand'].max():.2f} (Day {max_idx + 1}, {forecast_df.loc[max_idx, 'date'].strftime('%Y-%m-%d')})")
print(f"Range:                   {forecast_df['demand'].max() - forecast_df['demand'].min():.2f}")
print(f"IQR:                     {forecast_df['demand'].quantile(0.75) - forecast_df['demand'].quantile(0.25):.2f}")

# Day of week analysis
print(f"\n2. DAY-OF-WEEK ANALYSIS")
print("-" * 70)
dow_forecast = forecast_df.groupby('day_of_week')['demand'].agg(['mean', 'sum', 'count'])
dow_historical = historical_last_90.groupby('day_of_week')['y'].mean()

print(f"{'Day':<12} {'Forecast Avg':<15} {'Historical Avg':<15} {'Difference':<12}")
print("-" * 70)
for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
    if day in dow_forecast.index:
        f_avg = dow_forecast.loc[day, 'mean']
        h_avg = dow_historical.get(day, 0)
        diff = f_avg - h_avg
        print(f"{day:<12} {f_avg:<15.2f} {h_avg:<15.2f} {diff:+.2f}")

# Weekend vs Weekday
print(f"\n3. WEEKEND VS WEEKDAY COMPARISON")
print("-" * 70)
forecast_weekend = forecast_df[forecast_df['is_weekend']]['demand'].mean()
forecast_weekday = forecast_df[~forecast_df['is_weekend']]['demand'].mean()
historical_weekend = historical_last_90[historical_last_90['is_weekend']]['y'].mean()
historical_weekday = historical_last_90[~historical_last_90['is_weekend']]['y'].mean()

print(f"{'Period':<15} {'Forecast':<15} {'Historical':<15} {'Diff %':<12}")
print("-" * 70)
if historical_weekday > 0:
    print(f"{'Weekday':<15} {forecast_weekday:<15.2f} {historical_weekday:<15.2f} {((forecast_weekday/historical_weekday - 1) * 100):+.1f}%")
if historical_weekend > 0:
    print(f"{'Weekend':<15} {forecast_weekend:<15.2f} {historical_weekend:<15.2f} {((forecast_weekend/historical_weekend - 1) * 100):+.1f}%")

# Trend analysis
print(f"\n4. TREND ANALYSIS")
print("-" * 70)
days = np.arange(len(future_predictions))
slope, intercept, r_value, p_value, std_err = stats.linregress(days, future_predictions)
print(f"Linear Trend Slope:      {slope:.4f} units/day")
print(f"Trend Direction:         {'Increasing ↑' if slope > 0 else 'Decreasing ↓' if slope < 0 else 'Flat →'}")
print(f"R-squared:               {r_value**2:.4f}")
print(f"P-value:                 {p_value:.4f} {'(Significant)' if p_value < 0.05 else '(Not Significant)'}")
print(f"Expected change:         {slope * FUTURE_DAYS:.2f} units over {FUTURE_DAYS} days")

# Volatility analysis
print(f"\n5. VOLATILITY METRICS")
print("-" * 70)
rolling_window = min(7, len(future_predictions))
rolling_std = pd.Series(future_predictions).rolling(window=rolling_window).std()
print(f"Average volatility ({rolling_window}-day): {rolling_std.mean():.2f}")
print(f"Max volatility:             {rolling_std.max():.2f}")
daily_changes = np.diff(future_predictions)
print(f"Daily changes mean:         {daily_changes.mean():.2f}")
print(f"Daily changes std:          {daily_changes.std():.2f}")
print(f"Max daily increase:         {daily_changes.max():.2f}")
print(f"Max daily decrease:         {daily_changes.min():.2f}")

# Percentile analysis
print(f"\n6. PERCENTILE ANALYSIS")
print("-" * 70)
percentiles = [10, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    val = np.percentile(future_predictions, p)
    print(f"P{p:2d}:  {val:7.2f} units")

# Risk analysis
print(f"\n7. INVENTORY PLANNING RECOMMENDATIONS")
print("-" * 70)
safety_stock = forecast_df['demand'].mean() + 1.96 * forecast_df['demand'].std()
reorder_point = forecast_df['demand'].mean() * 7
max_demand = forecast_df['demand'].max()
avg_demand = forecast_df['demand'].mean()

print(f"Recommended Safety Stock:    {safety_stock:.0f} units (95% service level)")
print(f"Reorder Point (7-day supply): {reorder_point:.0f} units")
print(f"Expected Total Demand:       {forecast_df['demand'].sum():.0f} units")
print(f"Peak Day Demand:             {max_demand:.0f} units")
print(f"Average Daily Demand:        {avg_demand:.0f} units")
print(f"Days above average:          {(forecast_df['demand'] > avg_demand).sum()} days ({(forecast_df['demand'] > avg_demand).sum() / len(forecast_df) * 100:.1f}%)")
print(f"Days below average:          {(forecast_df['demand'] < avg_demand).sum()} days ({(forecast_df['demand'] < avg_demand).sum() / len(forecast_df) * 100:.1f}%)")

# Weekly breakdown
if FUTURE_DAYS >= 7:
    print(f"\n8. WEEKLY BREAKDOWN")
    print("-" * 70)
    forecast_df['week'] = (forecast_df.index // 7) + 1
    weekly_summary = forecast_df.groupby('week')['demand'].agg(['sum', 'mean', 'min', 'max'])
    print(f"{'Week':<8} {'Total':<12} {'Average':<12} {'Min':<10} {'Max':<10}")
    print("-" * 70)
    for week in weekly_summary.index:
        print(f"Week {int(week):<3} {weekly_summary.loc[week, 'sum']:<12.1f} {weekly_summary.loc[week, 'mean']:<12.1f} {weekly_summary.loc[week, 'min']:<10.1f} {weekly_summary.loc[week, 'max']:<10.1f}")

# ========== ENHANCED VISUALIZATION ==========
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(5, 3, hspace=0.35, wspace=0.3)

# Plot 1: Validation comparison
ax1 = fig.add_subplot(gs[0, :2])
correlation = np.corrcoef(actuals, predictions)[0, 1]
ax1.plot(dates, actuals, label='Actual', marker='o', alpha=0.7, linewidth=2, markersize=3, color='#2E86AB')
ax1.plot(dates, predictions, label='Predicted', marker='s', alpha=0.7, linewidth=2, markersize=3, color='#A23B72')
ax1.fill_between(dates, actuals, predictions, alpha=0.2, color='gray')
ax1.set_title(f"Validation: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%, R={correlation:.3f}", 
              fontsize=12, fontweight='bold')
ax1.set_xlabel("Date", fontsize=10)
ax1.set_ylabel("Demand", fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot 2: Scatter plot
ax2 = fig.add_subplot(gs[0, 2])
ax2.scatter(actuals, predictions, alpha=0.6, s=30, color='#2E86AB', edgecolors='white', linewidth=0.5)
max_val = max(actuals.max(), predictions.max())
ax2.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
ax2.text(0.05, 0.95, f'R² = {correlation**2:.3f}', transform=ax2.transAxes, 
         fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax2.set_xlabel("Actual", fontsize=10)
ax2.set_ylabel("Predicted", fontsize=10)
ax2.set_title("Validation Accuracy", fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Future forecast with history
ax3 = fig.add_subplot(gs[1, :])
history_days = min(90, len(df))
ax3.plot(df['ds'].values[-history_days:], df['y'].values[-history_days:], 
         label='Historical', marker='o', alpha=0.7, linewidth=2, markersize=3, color='#2E86AB')
ax3.plot(future_dates, future_predictions, label=f'{FUTURE_DAYS}-Day Forecast', 
         marker='s', alpha=0.9, linewidth=2.5, markersize=4, color='#F18F01', linestyle='--')
ax3.axvline(x=last_date, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Forecast Start')
ax3.axhline(y=future_predictions.mean(), color='green', linestyle='--', linewidth=1.5, alpha=0.7, 
            label=f'Forecast Mean: {future_predictions.mean():.1f}')
ax3.fill_between(future_dates, 0, future_predictions, alpha=0.2, color='#F18F01')
ax3.set_title(f"Forecast: {future_dates[0].date()} to {future_dates[-1].date()}", fontsize=13, fontweight='bold')
ax3.set_xlabel("Date", fontsize=10)
ax3.set_ylabel("Demand", fontsize=10)
ax3.legend(fontsize=9, loc='best')
ax3.grid(True, alpha=0.3)

# Plot 4: Day of week analysis
ax4 = fig.add_subplot(gs[2, 0])
dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_data = forecast_df.groupby('day_of_week')['demand'].mean().reindex(dow_order)
colors = ['#3498db' if day not in ['Saturday', 'Sunday'] else '#e74c3c' for day in dow_order]
bars = ax4.bar(range(len(dow_data)), dow_data.values, color=colors, alpha=0.7, edgecolor='black')
ax4.set_xticks(range(len(dow_data)))
ax4.set_xticklabels([d[:3] for d in dow_order], fontsize=9)
ax4.set_title("Average Demand by Day of Week", fontsize=11, fontweight='bold')
ax4.set_ylabel("Avg Demand", fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')

# Add values on bars
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}', ha='center', va='bottom', fontsize=8)

# Plot 5: Distribution histogram
ax5 = fig.add_subplot(gs[2, 1])
ax5.hist(future_predictions, bins=15, color='#F18F01', alpha=0.7, edgecolor='black')
mean_val = np.mean(future_predictions)
median_val = np.median(future_predictions)
ax5.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
ax5.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
ax5.set_title("Forecast Distribution", fontsize=11, fontweight='bold')
ax5.set_xlabel("Demand", fontsize=10)
ax5.set_ylabel("Frequency", fontsize=10)
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3, axis='y')


# Plot 6: Box plot
ax6 = fig.add_subplot(gs[2, 2])
bp = ax6.boxplot(future_predictions, vert=True, patch_artist=True,
                 boxprops=dict(facecolor='#F18F01', alpha=0.7),
                 medianprops=dict(color='red', linewidth=2),
                 whiskerprops=dict(linewidth=1.5),
                 capprops=dict(linewidth=1.5))
ax6.set_title("Forecast Variability", fontsize=11, fontweight='bold')
ax6.set_ylabel("Demand", fontsize=10)
ax6.set_xticklabels(['Forecast'])
ax6.grid(True, alpha=0.3, axis='y')

# Add statistics to box plot
q1, median, q3 = np.percentile(future_predictions, [25, 50, 75])
ax6.text(1.15, median, f'Median: {median:.1f}', va='center', fontsize=8)
ax6.text(1.15, q3, f'Q3: {q3:.1f}', va='center', fontsize=8)
ax6.text(1.15, q1, f'Q1: {q1:.1f}', va='center', fontsize=8)

# Plot 7: Cumulative demand
ax7 = fig.add_subplot(gs[3, 0])
cumulative = np.cumsum(future_predictions)
ax7.plot(range(1, len(cumulative) + 1), cumulative, marker='o', color='#9B59B6', linewidth=2, markersize=4)
ax7.fill_between(range(1, len(cumulative) + 1), 0, cumulative, alpha=0.3, color='#9B59B6')
ax7.set_title("Cumulative Demand Forecast", fontsize=11, fontweight='bold')
ax7.set_xlabel("Day", fontsize=10)
ax7.set_ylabel("Cumulative Demand", fontsize=10)
ax7.grid(True, alpha=0.3)

# Plot 8: Trend analysis
ax8 = fig.add_subplot(gs[3, 1])
ax8.scatter(range(len(future_predictions)), future_predictions, alpha=0.6, s=40, color='#F18F01')
trend_line = slope * days + intercept
ax8.plot(days, trend_line, 'r--', linewidth=2, label=f'Trend: {slope:.3f} units/day')
ax8.set_title("Trend Analysis", fontsize=11, fontweight='bold')
ax8.set_xlabel("Day", fontsize=10)
ax8.set_ylabel("Demand", fontsize=10)
ax8.legend(fontsize=8)
ax8.grid(True, alpha=0.3)

# Plot 9: Comparison with historical patterns
ax9 = fig.add_subplot(gs[3, 2])
hist_dow = historical_last_90.groupby('day_of_week')['y'].mean().reindex(dow_order)
fore_dow = forecast_df.groupby('day_of_week')['demand'].mean().reindex(dow_order)
x = np.arange(len(dow_order))
width = 0.35
ax9.bar(x - width/2, hist_dow.values, width, label='Historical', color='#2E86AB', alpha=0.7)
ax9.bar(x + width/2, fore_dow.values, width, label='Forecast', color='#F18F01', alpha=0.7)
ax9.set_xticks(x)
ax9.set_xticklabels([d[:3] for d in dow_order], fontsize=8)
ax9.set_title("Historical vs Forecast by Day", fontsize=11, fontweight='bold')
ax9.set_ylabel("Avg Demand", fontsize=10)
ax9.legend(fontsize=8)
ax9.grid(True, alpha=0.3, axis='y')

# Plot 10: Daily forecast bar chart
ax10 = fig.add_subplot(gs[4, :])
colors_daily = ['#e74c3c' if is_weekend else '#3498db' for is_weekend in forecast_df['is_weekend']]
bars = ax10.bar(range(len(future_predictions)), future_predictions, color=colors_daily, alpha=0.7, edgecolor='black')
ax10.axhline(y=future_predictions.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {future_predictions.mean():.1f}')
ax10.set_title(f"Daily Forecast Breakdown ({FUTURE_DAYS} Days)", fontsize=12, fontweight='bold')
ax10.set_xlabel("Day", fontsize=10)
ax10.set_ylabel("Demand", fontsize=10)
ax10.legend(['Mean', 'Weekday', 'Weekend'], fontsize=9)
ax10.grid(True, alpha=0.3, axis='y')
ax10.set_xticks(range(0, len(future_predictions), max(1, len(future_predictions)//10)))

plt.savefig('nhits_complete_eda_forecast.png', dpi=300, bbox_inches='tight')
print(f"\n✓ EDA visualization saved as 'nhits_complete_eda_forecast.png'")
plt.show()

# ========== DETAILED FORECAST TABLE ==========
print(f"\n{'='*100}")
print(f"DETAILED FORECAST TABLE")
print(f"{'='*100}")
print(f"{'Date':<12} {'Day':<10} {'Demand':<10} {'Cumulative':<12} {'% of Total':<12} {'Trend':<10}")
print(f"{'-'*100}")

cumulative = 0
total_demand = forecast_df['demand'].sum()
for idx, row in forecast_df.iterrows():
    cumulative += row['demand']
    pct = (row['demand'] / total_demand * 100)
    trend_val = slope * idx + intercept
    day_short = row['day_of_week'][:3]
    print(f"{str(row['date'].date()):<12} {day_short:<10} {row['demand']:<10.2f} {cumulative:<12.2f} {pct:<12.1f}% {trend_val:<10.2f}")

print(f"\n{'='*100}")
print("✓ N-HiTS forecasting with comprehensive EDA complete!")
print(f"{'='*100}")
