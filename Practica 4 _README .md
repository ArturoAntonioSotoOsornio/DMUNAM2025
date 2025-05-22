
# Práctica 4 – Modelación No Supervisada con Datos de Ecobici 🚲

Este proyecto tiene como objetivo aplicar técnicas de **modelado no supervisado** usando datos abiertos del sistema Ecobici de la Ciudad de México.

---

##   Archivos usados

- `afluencia_simple_acumulada_2025_.csv`
- `cicloestaciones_ecobici.csv`

---

##   Objetivo

Identificar patrones y agrupaciones (clusters) sin usar variables objetivo, a partir de los datos de viajes y estaciones Ecobici.

---

##   Pasos realizados

### 1. Carga y limpieza de datos
- Conversión de columnas a tipos adecuados (fechas, coordenadas, valores numéricos).
- Unión lógica de datasets si es necesario.

### 2. TAD – Tabla de análisis de datos
- Selección de variables continuas (`latitud`, `longitud`, `viajes`).
- Estandarización usando `StandardScaler`.

### 3. Modelado de Clustering
- Aplicación de **KMeans** y **Gaussian Mixture Model (GMM)**.
- Visualización geográfica de los grupos.

### 4. Perfilamiento por tiempo
- Agrupación de viajes por **mes** y **hora del día**.
- Generación de gráficas de comportamiento por cluster.

---

##  Resultados

- Se identificaron agrupamientos espaciales de estaciones.
- Se observó una evolución temporal clara en el uso del sistema.
- Clusters diferenciados por intensidad de uso según la hora.

---

##   Herramientas usadas

- Python (Google Colab)
- pandas, matplotlib, seaborn, scikit-learn

---

_Elaborado por: [Angelica Zetina Martinez]_
