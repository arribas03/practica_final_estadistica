import numpy as np
import matplotlib.pyplot as plt
import os

# ================================
# 1. CONFIGURACIÓN
# ================================

np.random.seed(42)
os.makedirs("data/output", exist_ok=True)

# ================================
# 2. GENERAR DATOS
# ================================

n = 200

X = np.random.rand(n, 3)

beta_real = np.array([5, 2, -1, 0.5])

X_b = np.c_[np.ones((n, 1)), X]

ruido = np.random.randn(n) * 0.5

y = X_b @ beta_real + ruido

# ================================
# 3. CÁLCULO DE COEFICIENTES
# ================================

beta_estimado = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

print("=== COEFICIENTES ===")
for i, b in enumerate(beta_estimado):
    print(f"beta{i}: {b:.4f}")

# ================================
# 4. COMPARACIÓN
# ================================

errores = []

print("\n=== COMPARACIÓN ===")
for i in range(len(beta_real)):
    error = abs(beta_real[i] - beta_estimado[i])
    errores.append(error)
    print(f"beta{i} real: {beta_real[i]} | estimado: {beta_estimado[i]:.4f} | error: {error:.4f}")

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
# 7. SCATTER PLOT (OBLIGATORIO)
# ================================

plt.figure(figsize=(6,6))
plt.scatter(y, y_pred)
plt.xlabel("Valores reales")
plt.ylabel("Valores predichos")
plt.title("Predicciones vs Reales")

# línea perfecta
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')

plt.tight_layout()
plt.savefig("data/output/ej3_predicciones.png")
plt.close()

# ================================
# 8. GUARDAR COEFICIENTES
# ================================

with open("data/output/ej3_coeficientes.txt", "w") as f:
    f.write("Coeficientes reales vs estimados\n\n")
    for i in range(len(beta_real)):
        f.write(f"beta{i} real: {beta_real[i]} | estimado: {beta_estimado[i]}\n")

# ================================
# 9. GUARDAR MÉTRICAS
# ================================

with open("data/output/ej3_metricas.txt", "w") as f:
    f.write(f"MAE: {mae}\n")
    f.write(f"RMSE: {rmse}\n")
    f.write(f"R2: {r2}\n")