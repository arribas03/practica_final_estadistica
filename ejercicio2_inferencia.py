import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ================================
# 1. CONFIGURACIÓN
# ================================

np.random.seed(42)
os.makedirs("data/output", exist_ok=True)

df = pd.read_csv("data/car_price.csv")

# ================================
# 2. PREPROCESAMIENTO
# ================================

if "CarName" in df.columns:
    df = df.drop(columns=["CarName"])

target = "price"
X = df.drop(columns=[target])
y = df[target]

# Encoding categóricas
X = pd.get_dummies(X, drop_first=True)

# ================================
# 3. TRAIN / TEST
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# 4. MODELO
# ================================

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ================================
# 5. MÉTRICAS MODELO
# ================================

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# ================================
# 6. BASELINE (MUY IMPORTANTE)
# ================================

baseline_pred = np.full_like(y_test, y_train.mean())

mae_baseline = mean_absolute_error(y_test, baseline_pred)
rmse_baseline = np.sqrt(mean_squared_error(y_test, baseline_pred))

# ================================
# 7. ANÁLISIS DE RESIDUOS
# ================================

residuos = y_test - y_pred

print("\n=== RESULTADOS MODELO ===")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2:.4f}")

print("\n=== BASELINE ===")
print(f"MAE baseline: {mae_baseline:.2f}")
print(f"RMSE baseline: {rmse_baseline:.2f}")

print("\n=== RESIDUOS ===")
print("Media residuos:", np.mean(residuos))
print("Std residuos:", np.std(residuos))

# ================================
# 8. INTERPRETACIÓN AUTOMÁTICA
# ================================

if r2 > 0.7:
    calidad = "bueno"
elif r2 > 0.4:
    calidad = "aceptable"
else:
    calidad = "mejorable"

mejora_mae = mae_baseline - mae
mejora_rmse = rmse_baseline - rmse

print("\n=== INTERPRETACIÓN ===")
print(f"El modelo tiene un rendimiento {calidad}.")
print(f"Mejora respecto al baseline en MAE: {mejora_mae:.2f}")
print(f"Mejora respecto al baseline en RMSE: {mejora_rmse:.2f}")

# ================================
# 9. GUARDAR RESULTADOS
# ================================

with open("data/output/ej2_metricas.txt", "w") as f:
    f.write(f"MAE: {mae}\n")
    f.write(f"RMSE: {rmse}\n")
    f.write(f"R2: {r2}\n")
    f.write(f"MAE baseline: {mae_baseline}\n")
    f.write(f"RMSE baseline: {rmse_baseline}\n")