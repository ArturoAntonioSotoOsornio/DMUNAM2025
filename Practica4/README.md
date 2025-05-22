# Datos Abiertos de ECOBICI (2010-2025)

## Descripción

Este proyecto tiene como objetivo analizar y segmentar el comportamiento de usuarios del sistema ECOBICI de la Ciudad de México, utilizando datos abiertos disponibles desde 2010 hasta 2025. Se aplican técnicas de limpieza masiva, construcción de una Tabla Analítica de Datos (TAD), y métodos de clustering no supervisado para identificar perfiles de usuarios y patrones temporales.

## Paso 1: Extracción y Limpieza de Datos

- Se descargaron aproximadamente 180 archivos CSV correspondientes a viajes mensuales de ECOBICI desde 2010 hasta 2025.
- Se implementó un proceso automatizado para:
  - Eliminar columnas de fecha y hora para optimizar espacio.
  - Unir todos los archivos en un solo archivo CSV limpio para facilitar el análisis posterior.

---

## Paso 2: Construcción de la Tabla Analítica de Datos (TAD)

- Se extrajo una muestra representativa de 1 millón de registros para manejar el volumen y evitar saturar recursos computacionales.
- Se seleccionaron variables continuas relevantes para el análisis:
  - Edad del usuario (`Edad_Usuario`)
  - Estaciones de retiro y arribo (`Ciclo_Estacion_Retiro` y `Ciclo_Estacion_Arribo`)
  - Género codificado (`Genero_Usuario`)
- Se eliminaron outliers en las estaciones con valores superiores a 500, ya que representan un porcentaje mínimo y podrían distorsionar el análisis.

---

## Paso 3: Entrenamiento de Modelos de Clustering

Se aplicaron tres técnicas de clustering para segmentar a los usuarios:

- **K-Means**: Segmenta en 5 clusters, separando usuarios según edad, género y estaciones más usadas.
- **Gaussian Mixture Model (GMM)**: Modelo probabilístico que detecta clusters elípticos y mejor separa perfiles femeninos y masculinos.

Los modelos fueron evaluados y visualizados con reducción dimensional PCA para facilitar la interpretación.

---

## Paso 4: Perfilamiento Temporal

- Se realizó un análisis temporal mensual para observar cómo evolucionan los patrones de uso y los perfiles de usuario a lo largo del tiempo.
- Se encontró que ciertos clusters tienen comportamiento estacional o cambios en la proporción de usuarios según género y edad.

---

## Resultados

- Se identificaron perfiles claros de usuarios basados en edad, género y estaciones de uso.
- GMM mostró mejor capacidad para separar grupos femeninos y masculinos comparado con K-Means.
- El análisis temporal reveló patrones estacionales y cambios de comportamiento a lo largo de los años.
