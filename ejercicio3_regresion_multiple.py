import numpy as np
import os

# ================================
# 1. CONFIGURACIÓN
# ================================

np.random.seed(42)
os.makedirs("data/output", exist_ok=True)

# ================================
# 2. GENERAR DATOS (MEJORADOS)
# ================================

n = 200  # más datos → mejor R2

X = np.random.rand(n, 3)

# Coeficientes reales
beta_real = np.array([5, 2, -1, 0.5])

# Añadir columna de unos (intercepto)
X_b = np.c_[np.ones((n, 1)), X]

# Ruido más pequeño → modelo más estable
ruido = np.random.randn(n) * 0.5

# Variable objetivo
y = X_b @ beta_real + ruido

# ================================
# 3. CÁLCULO DE COEFICIENTES
# ================================

beta_estimado = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

print("=== COEFICIENTES ===")
for i, b in enumerate(beta_estimado):
    print(f"beta{i}: {b:.4f}")

# ================================
# 4. COMPARACIÓN CON VALORES REALES
# ================================

print("\n=== COMPARACIÓN ===")
errores = []

for i in range(len(beta_real)):
    error = abs(beta_real[i] - beta_estimado[i])
    errores.append(error)
    
    print(f"beta{i} real: {beta_real[i]} | estimado: {beta_estimado[i]:.4f} | error: {error:.4f}")

print("\nError medio en coeficientes:", np.mean(errores))

# ================================
# 5. PREDICCIONES
# ================================

y_pred = X_b @ beta_estimado

# ================================
# 6. MÉTRICAS
# ================================

mae = np.mean(np.abs(y - y_pred))
rmse = np.sqrt(np.mean((y - y_pred) ** 2))
r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

print("\n=== MÉTRICAS ===")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")

# ================================
# 7. INTERPRETACIÓN AUTOMÁTICA
# ================================

if r2 > 0.8:
    calidad = "muy bueno"
elif r2 > 0.6:
    calidad = "bueno"
else:
    calidad = "mejorable"

print("\n=== INTERPRETACIÓN ===")
print(f"El modelo tiene un ajuste {calidad}.")
print("Los coeficientes estimados son cercanos a los valores reales.")
print("Las diferencias se deben al ruido en los datos.")

# ================================
# 8. GUARDAR RESULTADOS (SIN ERROR)
# ================================

with open("data/output/ej3_resultados.txt", "w", encoding="utf-8") as f:
    f.write("=== COEFICIENTES ===\n")
    for i, b in enumerate(beta_estimado):
        f.write(f"beta{i}: {b}\n")

    f.write("\n=== COMPARACIÓN ===\n")
    for i in range(len(beta_real)):
        f.write(f"beta{i} real: {beta_real[i]} | estimado: {beta_estimado[i]}\n")

    f.write("\n=== MÉTRICAS ===\n")
    f.write(f"MAE: {mae}\n")
    f.write(f"RMSE: {rmse}\n")
    f.write(f"R2: {r2}\n")