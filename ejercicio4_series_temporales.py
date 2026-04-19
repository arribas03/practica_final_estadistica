import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from scipy.stats import normaltest, norm

# ================================
# 1. CONFIGURACIÓN
# ================================

np.random.seed(42)
os.makedirs("data/output", exist_ok=True)

# ================================
# 2. GENERAR SERIE TEMPORAL
# ================================

n = 365 * 2
t = np.arange(n)

trend = 0.05 * t
seasonal = 10 * np.sin(2 * np.pi * t / 365)
noise = np.random.normal(0, 2, n)

y = trend + seasonal + noise

dates = pd.date_range(start="2020-01-01", periods=n)
df = pd.DataFrame({"date": dates, "value": y})
df.set_index("date", inplace=True)

# ================================
# 3. GRÁFICO SERIE ORIGINAL
# ================================

plt.figure(figsize=(10,5))
plt.plot(df.index, df["value"])
plt.title("Serie temporal original")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.tight_layout()
plt.savefig("data/output/ej4_serie_original.png")
plt.close()

# ================================
# 4. DESCOMPOSICIÓN
# ================================

decomposition = seasonal_decompose(df["value"], model="additive", period=365)

trend_comp = decomposition.trend.dropna()
seasonal_comp = decomposition.seasonal
resid = decomposition.resid.dropna()

# Guardar descomposición
decomposition.plot()
plt.tight_layout()
plt.savefig("data/output/ej4_descomposicion.png")
plt.close()

# ================================
# 5. ANÁLISIS DE TENDENCIA
# ================================

coef = np.polyfit(np.arange(len(trend_comp)), trend_comp, 1)
pendiente = coef[0]

print("\n=== TENDENCIA ===")
print("Pendiente:", pendiente)

# ================================
# 6. ESTACIONALIDAD
# ================================

amplitud = seasonal_comp.max() - seasonal_comp.min()

print("\n=== ESTACIONALIDAD ===")
print("Periodo: 365 días")
print("Amplitud:", amplitud)

# ================================
# 7. ACF y PACF DEL RESIDUO
# ================================

acf_vals = acf(resid, nlags=50)
pacf_vals = pacf(resid, nlags=50)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.stem(acf_vals)
plt.title("ACF")

plt.subplot(1,2,2)
plt.stem(pacf_vals)
plt.title("PACF")

plt.tight_layout()
plt.savefig("data/output/ej4_acf_pacf.png")
plt.close()

# ================================
# 8. HISTOGRAMA DEL RUIDO
# ================================

media_resid = np.mean(resid)
std_resid = np.std(resid)

plt.figure(figsize=(6,4))
plt.hist(resid, bins=30, density=True)

# curva normal
x = np.linspace(resid.min(), resid.max(), 100)
plt.plot(x, norm.pdf(x, media_resid, std_resid))

plt.title("Histograma del residuo")
plt.tight_layout()
plt.savefig("data/output/ej4_histograma_ruido.png")
plt.close()

# ================================
# 9. TEST DE NORMALIDAD
# ================================

stat, p_value = normaltest(resid)

print("\n=== RESIDUOS ===")
print("Media:", media_resid)
print("Std:", std_resid)
print("p-value:", p_value)

# ================================
# 10. GUARDAR RESULTADOS
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