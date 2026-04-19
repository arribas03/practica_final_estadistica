import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================================
# 1. CONFIGURACIÓN
# ================================

os.makedirs("data/output", exist_ok=True)

df = pd.read_csv("data/car_price.csv")

# ================================
# 2. RESUMEN ESTRUCTURAL
# ================================

print("=== INFORMACIÓN GENERAL ===")
print("Shape:", df.shape)
print("\nTipos de datos:\n", df.dtypes)
print("\nValores nulos:\n", df.isnull().sum())

# ================================
# 3. ESTADÍSTICOS DESCRIPTIVOS
# ================================

desc = df.describe()
desc.to_csv("data/output/ej1_descriptivo.csv")

# ================================
# 4. HISTOGRAMAS (UNO SOLO)
# ================================

df.hist(figsize=(15,10))
plt.tight_layout()
plt.savefig("data/output/ej1_histogramas.png")
plt.close()

# ================================
# 5. VARIABLES CATEGÓRICAS (UNA FIGURA)
# ================================

categorical_cols = df.select_dtypes(include=['object']).columns

n_cols = 2
n_rows = int(np.ceil(len(categorical_cols) / n_cols))

plt.figure(figsize=(12, n_rows * 4))

for i, col in enumerate(categorical_cols):
    plt.subplot(n_rows, n_cols, i + 1)
    df[col].value_counts().plot(kind='bar')
    plt.title(col)

plt.tight_layout()
plt.savefig("data/output/ej1_categoricas.png")
plt.close()

# ================================
# 6. BOXPLOTS (UNA FIGURA)
# ================================

target = "price"

plt.figure(figsize=(12, n_rows * 4))

for i, col in enumerate(categorical_cols):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.boxplot(x=df[col], y=df[target])
    plt.xticks(rotation=45)
    plt.title(col)

plt.tight_layout()
plt.savefig("data/output/ej1_boxplots.png")
plt.close()

# ================================
# 7. CORRELACIONES
# ================================

corr = df.corr(numeric_only=True)

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.tight_layout()
plt.savefig("data/output/ej1_heatmap_correlacion.png")
plt.close()

# ================================
# 8. TOP CORRELACIONES
# ================================

corr_target = corr[target].abs().sort_values(ascending=False)
print("\n=== CORRELACIÓN CON TARGET ===")
print(corr_target)

# ================================
# 9. OUTLIERS (TARGET)
# ================================

Q1 = df[target].quantile(0.25)
Q3 = df[target].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df[target] < lower) | (df[target] > upper)]

print("\nNúmero de outliers en price:", len(outliers))

# ================================
# 10. OUTLIERS POR VARIABLE
# ================================

for col in df.select_dtypes(include=np.number).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers_col = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
    
    print(f"{col}: {len(outliers_col)} outliers")

# ================================
# 11. NULOS
# ================================

null_counts = df.isnull().sum()
total_nulls = null_counts.sum()
null_percentage = (null_counts / len(df)) * 100

print("\n=== ANÁLISIS DE NULOS ===")
print("\nNulos por columna:\n", null_counts)
print("\nPorcentaje (%):\n", null_percentage)
print("\nTotal nulos:", total_nulls)
print("Porcentaje total:", (total_nulls / (df.shape[0] * df.shape[1])) * 100)