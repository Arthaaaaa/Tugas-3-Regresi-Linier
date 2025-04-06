import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("housing.csv")

# Encoding lokasi
le = LabelEncoder()
df["Lokasi"] = le.fit_transform(df["Lokasi"])

# Feature dan label
X = df[["LuasTanah", "Lokasi"]]
y = df["Harga"]

# Split dan latih model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Simpan model dan encoder
joblib.dump((model, le), "model.pkl")

print("✅ Model baru berhasil disimpan!")
