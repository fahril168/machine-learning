import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Prediksi Harga Emas", layout="wide")

st.title("📈 Prediksi Harga Emas (2013-2023)")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv('Gold Price (2013-2023).csv')

    for col in ['Price', 'Open', 'High', 'Low']:
        df[col] = df[col].str.replace(',', '').astype(float)

    df['Change %'] = df['Change %'].str.replace('%', '').astype(float)

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

    df['Vol.'] = df['Vol.'].apply(convert_volume)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')

    df['lag_1'] = df['Price'].shift(1)
    df['lag_2'] = df['Price'].shift(2)

    df = df.dropna()

    return df


df = load_data()

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Pengaturan Model")

split_ratio = st.sidebar.selectbox(
    "Pilih Split Data",
    [0.8, 0.85, 0.9]
)

model_choice = st.sidebar.selectbox(
    "Pilih Algoritma",
    ["Linear Regression", "Random Forest", "XGBoost"]
)

# =========================
# FEATURE & TARGET
# =========================
X = df[['Open', 'High', 'Low', 'Vol.', 'Change %', 'lag_1', 'lag_2']]
y = df['Price']

train_size = int(len(df) * split_ratio)

X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

# =========================
# SCALING
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# MODEL SELECTION
# =========================
if model_choice == "Linear Regression":
    model = LinearRegression()
elif model_choice == "Random Forest":
    model = RandomForestRegressor(random_state=42)
else:
    model = XGBRegressor(random_state=42)

# =========================
# TRAIN
# =========================
model.fit(X_train, y_train)

# =========================
# PREDIKSI
# =========================
y_pred = model.predict(X_test)

# =========================
# METRIK
# =========================
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Hasil Evaluasi")
st.write(f"Split Data: {round(split_ratio*100)}:{round((1-split_ratio)*100)}")
st.write(f"Model: {model_choice}")

st.metric("MAE", f"{mae:.2f}")
st.metric("R2 Score", f"{r2:.4f}")

# =========================
# VISUALISASI
# =========================
st.subheader("📉 Actual vs Predicted")

chart_data = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})

st.line_chart(chart_data)

# =========================
# PREDIKSI MANUAL
# =========================
st.subheader("🔮 Prediksi Manual")

open_val = st.number_input("Open")
high_val = st.number_input("High")
low_val = st.number_input("Low")
vol_val = st.number_input("Volume")
change_val = st.number_input("Change %")
lag1_val = st.number_input("Harga Emas Kemarin (USD)")
lag2_val = st.number_input("Harga Emas 2 Hari Lalu (USD)")

if st.button("Prediksi Harga"):
    input_data = np.array([[open_val, high_val, low_val, vol_val, change_val, lag1_val, lag2_val]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.success(f"Prediksi Harga Emas: {prediction[0]:.2f} USD")
