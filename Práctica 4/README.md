# Práctica 4 

## Descripción

Esta práctica tiene como objetivo aplicar técnicas de **aprendizaje no supervisado** para analizar datos del sistema Ecobici de la CDMX. Se realiza el preprocesamiento de los datos, reducción de dimensionalidad y entrenamiento de modelos de clustering para agrupar patrones de uso de bicicletas.

---

## Datos

- **Fuente:** Datos abiertos de Ecobici (CDMX)
- **Periodo:** 2010 - 2025


---

## Proceso

### Paso 1: Recolección
- Lectura de datos crudos desde archivo CSV.
- Selección de 100,000 registros para evitar saturación de memoria.

### Paso 2: Preprocesamiento
- Conversión de variables categóricas (`Género_Usuario`) con `LabelEncoder`.
- Escalamiento de variables con `MinMaxScaler`.
- Conversión de fechas y horas a valores numéricos continuos.
- Eliminación de registros inválidos.

### Paso 3: Modelado (Entrenamiento)
Se aplicaron tres métodos de clustering:
1. **Clustering Jerárquico (Aglomerativo)**
2. **K-Means**
3. **Gaussian Mixture Models (GMM)**

### Paso 4: Perfilamiento
- Se analizó el comportamiento de los clusters en función de la **fecha**, identificando posibles patrones estacionales o por día de la semana.
- Los grupos fueron visualizados en espacios 3D generados mediante **PCA**, **MDS** y **TSNE**.

---

## Resultados de los Clusters

- Se identificaron **3 a 5 clusters** principales, dependiendo del modelo.
- Cada cluster representa un **perfil distinto de uso**:
  - Usuarios frecuentes con trayectos cortos en días laborales.
  - Usuarios esporádicos que viajan en fines de semana.
  - Viajes turísticos entre estaciones céntricas.
  
- Las gráficas mostraron:
  - Distribuciones de cada cluster por **día del año**.
  - Dispersión 3D de los datos con colores según cluster (`plotly.scatter_3d`).
  - Tendencias de estaciones de origen/destino por grupo.

---

## Archivos entregados

- `proceso_ML_v_2.0_cluster.ipynb`: Notebook con todo el proceso.
- `proceso_ML_v_2.0_cluster combinar datos.ipynb`: Notebook donde se unieron todos los arhivos de 2010 - 2025.
- `ecobicidatos.csv`: CSV de todos los arhivos de 2010 - 2025 unidos.
- `README.md`: Este archivo con la descripción de la práctica.

---

## Herramientas y librerías

- `pandas`, `numpy`
- `scikit-learn`: para escalado, PCA, clustering
- `plotly`, `cufflinks`: para visualización interactiva
- `matplotlib`, `seaborn`: para gráficos adicionales
