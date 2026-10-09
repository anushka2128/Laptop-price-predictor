# ==============================================================================
# STEP 1: Libraries & Dataset Loading
# ==============================================================================
import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Upload laptop_data.csv directly in Colab or specify your path
CSV_PATH = "laptop_data.csv"

if not os.path.exists(CSV_PATH):
    try:
        from google.colab import files
        print("Please upload your laptop_data.csv:")
        uploaded = files.upload()
        CSV_PATH = list(uploaded.keys())[0]
    except Exception as e:
        raise FileNotFoundError(f"Dataset file '{CSV_PATH}' not found: {e}")

df = pd.read_csv(CSV_PATH)
print("Dataset loaded successfully. Shape:", df.shape)
print(df.head())

# ==============================================================================
# STEP 2: Feature Engineering & Preprocessing
# ==============================================================================
# Drop redundant index column if present
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# 1. Clean 'Ram' and 'Weight'
df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)

# 2. Extract Touchscreen and IPS flags from ScreenResolution
df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
df['IPS'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)

# 3. Calculate Pixels Per Inch (PPI)
def extract_resolution(res_str):
    match = re.findall(r'(\d{3,4})x(\d{3,4})', res_str)
    if match:
        x_res, y_res = map(int, match[0])
        return x_res, y_res
    return np.nan, np.nan

df[['X_res', 'Y_res']] = df['ScreenResolution'].apply(extract_resolution).tolist()
df['PPI'] = (((df['X_res']**2) + (df['Y_res']**2))**0.5 / df['Inches']).astype(float)
df.drop(columns=['ScreenResolution', 'Inches', 'X_res', 'Y_res'], inplace=True)

# 4. CPU Brand Categorization (Matches UI options)
def fetch_cpu_brand(text):
    if text.startswith('Intel Core i7'):
        return 'Intel Core i7'
    elif text.startswith('Intel Core i5'):
        return 'Intel Core i5'
    elif text.startswith('Intel Core i3'):
        return 'Intel Core i3'
    elif 'Intel' in text:
        return 'Other Intel Processor'
    elif 'AMD' in text:
        return 'AMD Processor'
    return 'Other'

df['Cpu_Brand'] = df['Cpu'].apply(fetch_cpu_brand)
df.drop(columns=['Cpu'], inplace=True)

# 5. Extract Storage (HDD & SSD in GB)
df['Memory'] = df['Memory'].astype(str).replace(r'\.0', '', regex=True)
df['Memory'] = df['Memory'].str.replace('GB', '')
df['Memory'] = df['Memory'].str.replace('TB', '000')

split_mem = df['Memory'].str.split("+", n=1, expand=True)
df['first'] = split_mem[0].str.strip()
df['second'] = split_mem[1].str.strip().fillna("0")

for prefix in ['first', 'second']:
    df[f'Layer1HDD_{prefix}'] = df[prefix].apply(lambda x: 1 if "HDD" in x else 0)
    df[f'Layer1SSD_{prefix}'] = df[prefix].apply(lambda x: 1 if "SSD" in x else 0)
    df[f'Layer1Hybrid_{prefix}'] = df[prefix].apply(lambda x: 1 if "Hybrid" in x else 0)
    df[f'Layer1Flash_{prefix}'] = df[prefix].apply(lambda x: 1 if "Flash Storage" in x else 0)
    df[prefix] = df[prefix].str.replace(r'\D', '', regex=True).astype(int)

df['HDD'] = (df['first'] * df['Layer1HDD_first'] + df['second'] * df['Layer1HDD_second'])
df['SSD'] = (df['first'] * df['Layer1SSD_first'] + df['second'] * df['Layer1SSD_second'])
df.drop(columns=['first', 'second', 'Layer1HDD_first', 'Layer1HDD_second',
                 'Layer1SSD_first', 'Layer1SSD_second', 'Layer1Hybrid_first',
                 'Layer1Hybrid_second', 'Layer1Flash_first', 'Layer1Flash_second', 'Memory'], inplace=True)

# 6. Simplify GPU & OS
df = df[df['Gpu'].apply(lambda x: x.split()[0]) != 'ARM'] # Remove rare ARM edge case
df['Gpu_Brand'] = df['Gpu'].apply(lambda x: x.split()[0])
df.drop(columns=['Gpu'], inplace=True)

def categorize_os(os_name):
    if 'Windows' in os_name:
        return 'Windows'
    elif 'Mac' in os_name or 'macOS' in os_name:
        return 'Mac'
    else:
        return 'Others/No OS/Linux'

df['OpSys'] = df['OpSys'].apply(categorize_os)

# Handle duplicate entries if any
duplicates = df.duplicated().sum()
if duplicates > 0:
    df.drop_duplicates(inplace=True)
print(f"Duplicates removed: {duplicates}. Final Cleaned Shape: {df.shape}")
print(df.head())

# ==============================================================================
# STEP 3: Publication-Grade Visual EDA
# ==============================================================================
# Theme settings matching sample document
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams['figure.facecolor'] = '#FAFAFA'
plt.rcParams['axes.facecolor'] = '#FAFAFA'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.labelcolor'] = '#222222'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['grid.color'] = '#DDDDDD'
plt.rcParams['grid.linewidth'] = 0.6

PRIMARY, ACCENT = "#D1495B", "#3E8FC1"

def annotate(ax, text, xy=(0.98, 0.95)):
    ax.text(*xy, text, transform=ax.transAxes, fontsize=9.5, ha='right', va='top',
            bbox=dict(boxstyle='round, pad=0.5', facecolor='white', edgecolor='#999999', alpha=0.9))

# Chart 1: Target Variable Distribution (Price & Log-Price)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df['Price'], kde=True, color=PRIMARY, ax=axes[0])
axes[0].set_title("Original Price Distribution (Right Skewed)")
annotate(axes[0], "High right skew: Log-transformation\nis required for linear models.", xy=(0.95, 0.90))

sns.histplot(np.log(df['Price']), kde=True, color=ACCENT, ax=axes[1])
axes[1].set_title("Log-Transformed Price (Normal Shape)")
annotate(axes[1], "Bell curve restored:\nStabilizes residual variance.", xy=(0.95, 0.90))
plt.tight_layout()
plt.savefig('price_distribution.png', dpi=300)
plt.close()

# Chart 2: Ranked Feature Correlation with Price
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_with_price = df[numeric_cols].corr()['Price'].drop('Price').sort_values()

fig, ax = plt.subplots(figsize=(8, 5))
colors = [PRIMARY if v > 0.3 else ACCENT for v in corr_with_price.values]
ax.barh(corr_with_price.index, corr_with_price.values, color=colors)
ax.axvline(0, color='#333333', linewidth=1)
ax.set_title("Correlation of Numerical Specs with Price")
ax.set_xlabel("Pearson Correlation Coefficient")
ax.spines[['top', 'right']].set_visible(False)
annotate(ax, "RAM & SSD display the\nhighest price pull.", xy=(0.95, 0.20))
plt.tight_layout()
plt.savefig('correlation_price.png', dpi=300)
plt.close()

# Chart 3: Median Price by Laptop Brand
fig, ax = plt.subplots(figsize=(10, 5))
brand_order = df.groupby('Company')['Price'].median().sort_values(ascending=False).index
sns.barplot(data=df, x='Company', y='Price', order=brand_order, errorbar=None, color=ACCENT, ax=ax)
plt.xticks(rotation=45, ha='right')
ax.set_title("Median Price Across Laptop Brands")
ax.set_ylabel("Price (INR)")
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('brand_median_price.png', dpi=300)
plt.close()

# ==============================================================================
# STEP 4: IQR Outlier Inspection & Train-Test Pipeline Split
# ==============================================================================
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# IQR check on numerical columns
def count_outliers_iqr(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return ((data[col] < lower) | (data[col] > upper)).sum()

print("\nOutlier counts (IQR method) per numeric column:")
for col in ['Ram', 'Weight', 'PPI', 'HDD', 'SSD']:
    print(f"{col}: {count_outliers_iqr(df, col)}")

# Separate features & log-transformed target
X = df.drop(columns=['Price'])
y = np.log(df['Price'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)
print(f"\nTrain shape: {X_train.shape} | Test shape: {X_test.shape}")

# Preprocessor Definition: Encodes categorical variables & scales numeric
cat_cols = ['Company', 'TypeName', 'Cpu_Brand', 'Gpu_Brand', 'OpSys']
num_cols = ['Ram', 'Weight', 'Touchscreen', 'IPS', 'PPI', 'HDD', 'SSD']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ],
    remainder='passthrough'
)

# ==============================================================================
# STEP 5: Regression Models with Cross-Validation & GridSearch
# ==============================================================================
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV

cv = KFold(n_splits=5, shuffle=True, random_state=42)

# 1. Ridge Regression Pipeline
ridge_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=10.0))
])
ridge_pipe.fit(X_train, y_train)

# 2. Random Forest Regressor with GridSearchCV
rf_base = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])
rf_param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [15, 20],
    'regressor__min_samples_split': [2, 5]
}
rf_grid = GridSearchCV(rf_base, rf_param_grid, scoring='r2', cv=cv, n_jobs=-1)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_

# 3. XGBoost Regressor (optional / conditional import)
models = {
    'Ridge Regression': ridge_pipe,
    'Random Forest Regressor': rf_best
}

try:
    from xgboost import XGBRegressor
    xgb_base = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(random_state=42, objective='reg:squarederror'))
    ])
    xgb_param_grid = {
        'regressor__n_estimators': [150, 300],
        'regressor__max_depth': [4, 6],
        'regressor__learning_rate': [0.05, 0.1]
    }
    xgb_grid = GridSearchCV(xgb_base, xgb_param_grid, scoring='r2', cv=cv, n_jobs=-1)
    xgb_grid.fit(X_train, y_train)
    models['XGBoost Regressor'] = xgb_grid.best_estimator_
    print("XGBoost Regressor tuned and trained.")
except ImportError:
    print("XGBoost not installed; continuing with Ridge & Random Forest.")

print("All models successfully tuned and trained.")

# ==============================================================================
# STEP 6: Multi-Model Evaluation Table & Residuals
# ==============================================================================
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

results = []
predictions = {}

for name, model in models.items():
    # Convert predictions back to INR using np.exp
    y_pred_log = model.predict(X_test)
    y_pred = np.exp(y_pred_log)
    y_true = np.exp(y_test)

    predictions[name] = (y_true, y_pred)

    results.append({
        'Model': name,
        'R2 Score': r2_score(y_test, y_pred_log),
        'MAE (INR)': mean_absolute_error(y_true, y_pred),
        'RMSE (INR)': np.sqrt(mean_squared_error(y_true, y_pred))
    })

results_df = pd.DataFrame(results).sort_values('R2 Score', ascending=False).reset_index(drop=True)
print("\n" + "="*60)
print("MODEL REGRESSION COMPARISON TABLE (Sorted by R2)")
print("="*60)
print(results_df.to_string(index=False))

# Winner selection
winner_name = results_df.iloc[0]['Model']
winner_model = models[winner_name]
print(f"\nWINNER: {winner_name} with R2 = {results_df.iloc[0]['R2 Score']:.4f}")

# Actual vs Predicted Plot
num_models = len(models)
fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5))
if num_models == 1:
    axes = [axes]

for i, (name, (actual, pred)) in enumerate(predictions.items()):
    axes[i].scatter(actual, pred, alpha=0.4, color='#3E8FC1')
    axes[i].plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=1.5)
    axes[i].set_title(name)
    axes[i].set_xlabel("Actual Price (INR)")
    axes[i].set_ylabel("Predicted Price (INR)")
    axes[i].spines[['top', 'right']].set_visible(False)
plt.suptitle("Actual vs. Predicted Laptop Prices", fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300)
plt.close()

# ==============================================================================
# STEP 7: Export Winning Model & Model Configuration JSON
# ==============================================================================
# 1. Save winning pipeline (includes preprocessor + model)
joblib.dump(winner_model, 'laptop_price_model.pkl')
print("Saved pipeline: laptop_price_model.pkl")

# 2. Extract feature names & UI defaults
cat_features = list(winner_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_cols))
all_features = cat_features + num_cols

config = {
    "winner_model": winner_name,
    "input_features": list(X_train.columns),
    "companies": sorted(df['Company'].unique().tolist()),
    "types": sorted(df['TypeName'].unique().tolist()),
    "ram_options": sorted(df['Ram'].unique().tolist()),
    "os_options": sorted(df['OpSys'].unique().tolist()),
    "cpu_brands": sorted(df['Cpu_Brand'].unique().tolist()),
    "gpu_brands": sorted(df['Gpu_Brand'].unique().tolist()),
    "defaults": {
        "Ram": int(df['Ram'].median()),
        "Weight": float(df['Weight'].median()),
        "Touchscreen": 0,
        "IPS": 0,
        "PPI": float(df['PPI'].median()),
        "HDD": int(df['HDD'].median()),
        "SSD": int(df['SSD'].median())
    }
}

with open('model_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("Saved configuration: model_config.json")

# Style setup matching sample document
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams['figure.facecolor'] = '#FAFAFA'
plt.rcParams['axes.facecolor'] = '#FAFAFA'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.labelcolor'] = '#222222'

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Price vs Ram
sns.boxplot(data=df, x='Ram', y='Price', hue='Ram', legend=False, ax=axes[0, 0], palette='Blues_r')
axes[0, 0].set_title("Price Distribution by RAM (GB)")
axes[0, 0].set_ylabel("Price (INR)")
axes[0, 0].set_xlabel("RAM (GB)")
axes[0, 0].spines[['top', 'right']].set_visible(False)

# 2. Price vs TypeName
type_order = df.groupby('TypeName')['Price'].median().sort_values(ascending=False).index
sns.boxplot(data=df, x='TypeName', y='Price', hue='TypeName', order=type_order, legend=False, ax=axes[0, 1], palette='flare')
axes[0, 1].set_title("Price Distribution by Laptop Type")
axes[0, 1].set_ylabel("Price (INR)")
axes[0, 1].set_xlabel("Laptop Type")
axes[0, 1].tick_params(axis='x', rotation=30)
axes[0, 1].spines[['top', 'right']].set_visible(False)

# 3. Price vs Top 8 Brands
top_brands = df['Company'].value_counts().head(8).index
df_top_brands = df[df['Company'].isin(top_brands)]
brand_order = df_top_brands.groupby('Company')['Price'].median().sort_values(ascending=False).index
sns.boxplot(data=df_top_brands, x='Company', y='Price', hue='Company', order=brand_order, legend=False, ax=axes[1, 0], palette='crest')
axes[1, 0].set_title("Price Distribution by Top Brands")
axes[1, 0].set_ylabel("Price (INR)")
axes[1, 0].set_xlabel("Brand")
axes[1, 0].tick_params(axis='x', rotation=30)
axes[1, 0].spines[['top', 'right']].set_visible(False)

# 4. Outlier Inspection for Numeric Features
sns.boxplot(data=df[['Weight']], ax=axes[1, 1], color='#E8A33D', width=0.3)
axes[1, 1].set_title("Weight (kg) Outlier Spread")
axes[1, 1].set_ylabel("Weight (kg)")
axes[1, 1].spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('laptop_boxplots.png', dpi=300)
plt.close()
print("Plots generated successfully without warnings.")
