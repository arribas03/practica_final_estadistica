import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from scipy.stats import normaltest

# ================================
# 1. CONFIGURACIÓN
# ================================

np.random.seed(42)
os.makedirs("data/output", exist_ok=True)

# ================================
# 2. GENERAR SERIE TEMPORAL
# ================================

n = 365 * 2  # 2 años
t = np.arange(n)

# Tendencia
trend = 0.05 * t

# Estacionalidad
seasonal = 10 * np.sin(2 * np.pi * t / 365)

# Ruido
noise = np.random.normal(0, 2, n)

# Serie final
y = trend + seasonal + noise

dates = pd.date_range(start="2020-01-01", periods=n)
df = pd.DataFrame({"date": dates, "value": y})
df.set_index("date", inplace=True)

# ================================
# 3. DESCOMPOSICIÓN
# ================================

decomposition = seasonal_decompose(df["value"], model="additive", period=365)

trend_comp = decomposition.trend.dropna()
seasonal_comp = decomposition.seasonal
resid = decomposition.resid.dropna()

# ================================
# 4. ANÁLISIS DE TENDENCIA
# ================================

# pendiente aproximada
coef = np.polyfit(np.arange(len(trend_comp)), trend_comp, 1)
pendiente = coef[0]

print("\n=== TENDENCIA ===")
print("Pendiente aproximada:", pendiente)

# ================================
# 5. ESTACIONALIDAD
# ================================

amplitud = seasonal_comp.max() - seasonal_comp.min()

print("\n=== ESTACIONALIDAD ===")
print("Periodo aproximado: 365 días")
print("Amplitud:", amplitud)

# ================================
# 6. CICLOS (AUTOCORRELACIÓN)
# ================================

acf_vals = acf(df["value"], nlags=400)

# detectar picos importantes
lags_importantes = np.where(acf_vals > 0.5)[0]

print("\n=== CICLOS ===")
print("Lags con autocorrelación alta:", lags_importantes[:10])

# ================================
# 7. ANÁLISIS DE RESIDUO
# ================================

media_resid = np.mean(resid)
std_resid = np.std(resid)

stat, p_value = normaltest(resid)

print("\n=== RESIDUOS ===")
print("Media:", media_resid)
print("Desviación típica:", std_resid)
print("p-value:", p_value)

# ================================
# 8. GRÁFICAS
# ================================

decomposition.plot()
plt.tight_layout()
plt.savefig("data/output/ej4_descomposicion.png")
plt.close()

# ================================
# 9. GUARDAR RESULTADOS
# ================================

with open("data/output/ej4_resultados.txt", "w") as f:
    f.write("=== TENDENCIA ===\n")
    f.write(f"Pendiente: {pendiente}\n\n")

    f.write("=== ESTACIONALIDAD ===\n")
    f.write(f"Periodo: 365 días\n")
    f.write(f"Amplitud: {amplitud}\n\n")

    f.write("=== RESIDUOS ===\n")
    f.write(f"Media: {media_resid}\n")
    f.write(f"Std: {std_resid}\n")
    f.write(f"p-value: {p_value}\n")