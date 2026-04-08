import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

import pandas as pd

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv('Gold Price (2013-2023).csv')

print("=== DATASET (5 BARIS PERTAMA) ===")
print(df.head())


# =========================
# 2. INFORMASI DATASET
# =========================
print("\n=== INFORMASI DATASET ===")
print(df.info())


# =========================
# 3. DESKRIPSI FITUR
# =========================
print("\n=== NAMA FITUR / KOLOM ===")
print(df.columns)


# =========================
# 4. STATISTIK DESKRIPTIF
# =========================
print("\n=== STATISTIK DESKRIPTIF ===")
print(df.describe(include='all'))

def convert_volume(val):
    try:
        if isinstance(val, str):
            val = val.strip()

            if val == '-' or val == '':
                return 0

            if val.endswith('K'):
                return float(val[:-1]) * 1_000
            elif val.endswith('M'):
                return float(val[:-1]) * 1_000_000
            elif val.endswith('B'):
                return float(val[:-1]) * 1_000_000_000
            else:
                return float(val)
        return val
    except:
        return 0


# =========================
# CLEANING
# =========================
for col in ['Price', 'Open', 'High', 'Low']:
    df[col] = df[col].str.replace(',', '').astype(float)

df['Change %'] = df['Change %'].str.replace('%', '').astype(float)
df['Vol.'] = df['Vol.'].apply(convert_volume)

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

# =========================
# MISSING VALUE
# =========================
print(df.isnull().sum())
df = df.dropna()

# =========================
# OUTLIER (IQR)
# =========================
Q1 = df[['Price', 'Open', 'High', 'Low']].quantile(0.25)
Q3 = df[['Price', 'Open', 'High', 'Low']].quantile(0.75)
IQR = Q3 - Q1

df = df[~((df[['Price', 'Open', 'High', 'Low']] < (Q1 - 1.5 * IQR)) |
          (df[['Price', 'Open', 'High', 'Low']] > (Q3 + 1.5 * IQR))).any(axis=1)]

df['lag_1'] = df['Price'].shift(1)
df['lag_2'] = df['Price'].shift(2)

df = df.dropna()


X = df[['Open', 'High', 'Low', 'Vol.', 'Change %',
        'lag_1', 'lag_2']]

y = df['Price']


import seaborn as sns

# =========================
# SET STYLE (BIAR BAGUS)
# =========================
sns.set_style('whitegrid')

# =========================
# 1. DISTRIBUSI DATA
# =========================
plt.figure(figsize=(10,8))
df[['Price', 'Open', 'High', 'Low']].hist(bins=30, figsize=(10,8))
plt.suptitle('Distribusi Data Harga Emas', fontsize=14)
plt.tight_layout()
plt.show()

# =========================
# 2. DISTRIBUSI KHUSUS PRICE
# =========================
plt.figure(figsize=(8,5))
sns.histplot(df['Price'], kde=True)
plt.title('Distribusi Harga Emas')
plt.xlabel('Price')
plt.ylabel('Frekuensi')
plt.tight_layout()
plt.show()

# =========================
# 3. KORELASI (HANYA NUMERIK)
# =========================
num_cols = df.select_dtypes(include=['float64', 'int64'])

plt.figure(figsize=(10,7))
sns.heatmap(num_cols.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# =========================
# 4. TIME SERIES
# =========================
plt.figure(figsize=(12,5))
plt.plot(df['Date'], df['Price'])
plt.title('Pergerakan Harga Emas (2013-2023)')
plt.xlabel('Tahun')
plt.ylabel('Harga')
plt.grid(True)
plt.tight_layout()
plt.show()


def evaluate_split(split_ratio):

    # ================================
    # Data Splitting (Time Series)
    # ================================
    train_size = int(len(df) * split_ratio)

    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    print("\n" + "="*50)
    print(f"         SPLIT {round(split_ratio*100)}:{round((1-split_ratio)*100)}")
    print("="*50)

    # ================================
    # NORMALISASI (ANTI DATA LEAKAGE)
    # ================================
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    X_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X.columns,
        index=X_train.index
    )

    X_test = pd.DataFrame(
        scaler.transform(X_test),
        columns=X.columns,
        index=X_test.index
    )

    # ================================
    # Cross Validation (Time Series)
    # ================================
    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV

    tscv = TimeSeriesSplit(n_splits=5)

    # ================================
    # Linear Regression
    # ================================
    from sklearn.linear_model import LinearRegression

    lr = LinearRegression()
    lr_param_grid = {
        'fit_intercept': [True, False],
        'positive': [True, False]
    }

    lr_grid = GridSearchCV(lr, lr_param_grid, cv=tscv, n_jobs=-1)
    lr_grid.fit(X_train, y_train)
    best_lr = lr_grid.best_estimator_

    # ================================
    # Random Forest
    # ================================
    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(random_state=42)

    rf_param_grid = {
        'n_estimators': [100, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2, 4]
    }

    rf_grid = GridSearchCV(rf, rf_param_grid, cv=tscv, n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_

    # ================================
    # XGBoost
    # ================================
    from xgboost import XGBRegressor

    xgb = XGBRegressor(random_state=42)

    xgb_param_dist = {
        'n_estimators': [100, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1]
    }

    xgb_random = RandomizedSearchCV(
        xgb,
        xgb_param_dist,
        n_iter=10,
        cv=tscv,
        random_state=42,
        n_jobs=-1
    )

    xgb_random.fit(X_train, y_train)
    best_xgb = xgb_random.best_estimator_

    # ================================
    # TAMPILKAN HASIL TUNING
    # ================================
    print("\n🎯 BEST HYPERPARAMETERS")
    print("-"*50)
    print("Best LR Params  :", lr_grid.best_params_)
    print("Best RF Params  :", rf_grid.best_params_)
    print("Best XGB Params :", xgb_random.best_params_)

    # ================================
    # Model
    # ================================
    models = {
        "Linear Regression": best_lr,
        "Random Forest": best_rf,
        "XGBoost": best_xgb
    }

    predictions = {}

    print("\n📊 HASIL EVALUASI MODEL")
    print("-"*50)

    # ================================
    # Training & Evaluasi
    # ================================
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.metrics import mean_absolute_percentage_error
    import numpy as np

    for name, model in models.items():

        if name == "XGBoost":
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        predictions[name] = y_pred

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        train_r2 = model.score(X_train, y_train)
        test_r2 = model.score(X_test, y_test)
        gap = train_r2 - test_r2

        print(f"\n🔹 {name}")
        print(f"{'-'*30}")
        print(f"MAE              : {mae:.4f}")
        print(f"MSE              : {mse:.4f}")
        print(f"RMSE             : {rmse:.4f}")
        print(f"MAPE             : {mape*100:.2f}%")
        print(f"R² Score         : {r2:.4f}")
        print(f"Train R²         : {train_r2:.4f}")
        print(f"Test R²          : {test_r2:.4f}")
        print(f"Overfitting Gap  : {gap:.4f}")

    # ================================
    # Visualisasi
    # ================================
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        ax.plot(y_test.reset_index(drop=True), label='Actual')
        ax.plot(pd.Series(y_pred), label='Predicted')
        ax.set_title(name)
        ax.legend()

    fig.suptitle(f'Actual vs Predicted - Split {round(split_ratio*100)}:{round((1-split_ratio)*100)}')
    plt.tight_layout()
    plt.show()
    
    
evaluate_split(0.8)   # 80:20
evaluate_split(0.85)  # 85:15
evaluate_split(0.9)   # 90:10

# contoh training final (pakai seluruh data)
lr = LinearRegression().fit(X, y)
rf = RandomForestRegressor().fit(X, y)
xgb = XGBRegressor().fit(X, y)

joblib.dump(lr, 'lr.pkl')
joblib.dump(rf, 'rf.pkl')
joblib.dump(xgb, 'xgb.pkl')