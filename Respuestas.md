# Respuestas — Práctica Final: Análisis y Modelado de Datos

---

# 🔵 Ejercicio 1 — Análisis Estadístico Descriptivo

## Pregunta 1.1 — ¿De qué fuente proviene el dataset y cuál es la variable objetivo (target)? ¿Por qué tiene sentido hacer regresión sobre ella?

El dataset utilizado proviene de una fuente pública (Kaggle) y contiene información sobre características técnicas de vehículos.

La variable objetivo seleccionada es **price**, que representa el precio del vehículo.

Tiene sentido aplicar regresión sobre esta variable ya que es una variable **numérica continua**, y su valor depende de múltiples factores como la potencia del motor, el tamaño, el peso o el tipo de vehículo. Por tanto, es adecuado modelar su comportamiento mediante técnicas de regresión.

---

## Pregunta 1.2 — ¿Qué distribución tienen las principales variables numéricas y has encontrado outliers? Indica en qué variables y qué has decidido hacer con ellos.

Las variables numéricas presentan distribuciones variadas, en general con cierta asimetría hacia la derecha, especialmente en variables relacionadas con el precio o la potencia.

Se han detectado outliers utilizando el método del rango intercuartílico (IQR), especialmente en variables como **price, horsepower y enginesize**.

Estos outliers corresponden a vehículos de gama alta o características extremas, por lo que se ha decidido **no eliminarlos**, ya que representan valores reales del mercado y pueden aportar información relevante al modelo.

---

## Pregunta 1.3 — ¿Qué tres variables numéricas tienen mayor correlación (en valor absoluto) con la variable objetivo? Indica los coeficientes.

Las variables con mayor correlación con la variable objetivo **price** son:

* **enginesize** → alta correlación positiva
* **curbweight** → alta correlación positiva
* **horsepower** → alta correlación positiva

Estas variables tienen una relación directa con el precio del vehículo, ya que reflejan características clave como la potencia, el tamaño y el peso.

---

## Pregunta 1.4 — ¿Hay valores nulos en el dataset? ¿Qué porcentaje representan y cómo los has tratado?

Tras analizar el dataset, no se han encontrado valores nulos.

* Total de valores nulos: 0
* Porcentaje de valores nulos: 0%

Por tanto, no ha sido necesario aplicar ningún tratamiento adicional sobre los datos.

---

# 🟢 Ejercicio 2 — Inferencia con Scikit-Learn

## Pregunta 2.1 — Indica los valores de MAE, RMSE y R² de la regresión lineal sobre el test set. ¿El modelo funciona bien? ¿Por qué?

El modelo de regresión lineal ha obtenido los siguientes resultados sobre el conjunto de test:

* MAE: 2222.58
* RMSE: 3142.49
* R²: 0.8689

El modelo presenta un rendimiento **bueno**, ya que el valor de R² indica que explica aproximadamente el 86.89% de la variabilidad del precio.

Además, al compararlo con un modelo baseline basado en la media, se observa una mejora significativa en los errores:

* MAE baseline: 6103.92
* RMSE baseline: 8952.05

Esto indica que el modelo es capaz de capturar relaciones relevantes entre las variables.

Por otro lado, los residuos presentan una media cercana a cero, lo que sugiere que no existe un sesgo sistemático en las predicciones.

---

# 🔴 Ejercicio 3 — Regresión Lineal Múltiple en NumPy

## Pregunta 3.1 — Explica en tus propias palabras qué hace la fórmula β = (XᵀX)⁻¹ Xᵀy y por qué es necesario añadir una columna de unos a la matriz X.

La fórmula β = (XᵀX)⁻¹ Xᵀy permite calcular los coeficientes óptimos de una regresión lineal minimizando el error cuadrático entre las predicciones y los valores reales.

En esencia, esta expresión encuentra los parámetros que mejor ajustan un modelo lineal a los datos disponibles.

Es necesario añadir una columna de unos a la matriz X para poder estimar el término independiente del modelo (β₀), que representa el valor base de la variable objetivo cuando el resto de variables son cero.

---

## Pregunta 3.2 — Copia aquí los cuatro coeficientes ajustados por tu función y compáralos con los valores de referencia del enunciado.

Los coeficientes obtenidos son:

* β₀ ≈ 5.0579 (real: 5.0)
* β₁ ≈ 2.0177 (real: 2.0)
* β₂ ≈ -1.1490 (real: -1.0)
* β₃ ≈ 0.4607 (real: 0.5)

El error medio en los coeficientes es aproximadamente 0.066, lo que indica que el modelo ha sido capaz de estimar correctamente los valores reales.

Las pequeñas diferencias se deben al ruido introducido en los datos.

---

## Pregunta 3.3 — ¿Qué valores de MAE, RMSE y R² has obtenido? ¿Se aproximan a los de referencia?

El modelo ha obtenido:

* MAE: 0.4027
* RMSE: 0.5058
* R²: 0.6387

Los errores MAE y RMSE son bajos, lo que indica que las predicciones son precisas.

El valor de R² es moderadamente alto, lo que indica que el modelo explica una parte significativa de la variabilidad de los datos.

En conjunto, los resultados son buenos y se aproximan a los valores esperados, teniendo en cuenta la presencia de ruido.

---

# Ejercicio 4 — Series Temporales

## ¿La serie presenta tendencia? ¿De qué tipo?

La serie presenta una **tendencia creciente de tipo aproximadamente lineal**.

Esto se observa en la pendiente estimada de la componente de tendencia, que es aproximadamente **0.0496**, lo que indica un incremento progresivo y constante de los valores a lo largo del tiempo.

---

## ¿Hay estacionalidad? ¿Cuál es el periodo aproximado y la amplitud?

Sí, la serie presenta una **estacionalidad clara**.

* **Periodo aproximado:** 365 días (patrón anual)
* **Amplitud:** aproximadamente **31.83**

Esto indica que la serie experimenta fluctuaciones periódicas regulares a lo largo del año, con variaciones significativas alrededor de la tendencia.

---

## ¿Se aprecian ciclos de largo plazo? ¿Cómo los distingues de la tendencia?

No se observan ciclos de largo plazo claramente diferenciados.

El análisis de autocorrelación muestra valores altos únicamente en lags pequeños (corto plazo), lo que indica dependencia temporal inmediata, pero no la existencia de patrones cíclicos prolongados.

Los ciclos se diferencian de la tendencia en que:

* la **tendencia** representa un cambio sistemático y continuo (en este caso creciente)
* los **ciclos** serían fluctuaciones irregulares de largo plazo sin periodicidad fija

En esta serie, la variación se explica principalmente por la tendencia y la estacionalidad, sin evidencia de ciclos adicionales.

---

## ¿Hay ruido? ¿Cuánto (en términos de desviación típica del residuo)?

Sí, la serie presenta un componente de ruido.

* **Media del residuo:** -0.045 (aproximadamente 0)
* **Desviación típica:** 0.068

La baja desviación típica indica que el ruido tiene una variabilidad reducida en comparación con la señal principal, por lo que no afecta significativamente a la estructura de la serie.

---

## ¿El ruido se ajusta a un ruido ideal (gaussiano, media ≈ 0, sin autocorrelación)? Justifica con los resultados de los tests.

El residuo presenta características cercanas a un ruido ideal, aunque no cumple completamente todas las condiciones.

* La **media es cercana a cero**, lo que indica ausencia de sesgo
* La **desviación típica es baja**, lo que indica baja dispersión
* Sin embargo, el test de normalidad arroja un **p-value ≈ 1.84e-42**, lo que indica que **no se puede asumir normalidad estricta**

A pesar de ello, el comportamiento general del residuo es consistente con un ruido aleatorio, por lo que se considera adecuado para el análisis de la serie.

---

