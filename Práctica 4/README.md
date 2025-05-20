# Práctica 4
## Procesamiento de Datos Ecobici

### Paso 1: Ingreso y Limpieza de Datos

**Objetivo**

Preparar, estandarizar y consolidar los archivos de datos crudos de Ecobici, descargados en formato CSV comprimido, para su uso posterior en análisis exploratorio y modelado. Se maneja un gran volumen de archivos distribuidos por año, por lo que el procesamiento se realiza en bloques (chunks) y se automatiza con registros de log.

**Flujo de trabajo**

    Conexión con Google Drive:
    Se monta el entorno de Google Drive para acceder y guardar archivos desde Colab.

    Descompresión de datos:
    El archivo ecobici.zip se descomprime automáticamente en un directorio temporal (/content/ecobici).

    Procesamiento por año:

        Lectura en bloques (CHUNK_SIZE = 100,000) para optimizar memoria.

        Limpieza y estandarización de formatos:

            Fechas (Fecha_Retiro, Fecha_Arribo) convertidas a YYYY-MM-DD.

            Horas (Hora_Retiro, Hora_Arribo) normalizadas a HH:MM:SS, eliminando milisegundos y entradas corruptas.

        Escritura intermedia por año en archivos tipo merged_2010.csv, merged_2011.csv, etc.

    Consolidación Final:

        Se fusionan todos los archivos anuales limpios en un único archivo: ecobici_merged_clean.csv.

        Se crea un log detallado de ejecución (ecobici_merge_clean_log.txt) y un archivo de errores si los hay (merge_errors.log).

        Todo se guarda automáticamente en Google Drive para respaldo.

**Funciones clave**

    standardize_date_format(df): Estandariza columnas de fecha a un formato uniforme.

    standardize_time_format(df): Limpia y normaliza los formatos de hora.

    process_year(year): Procesa todos los archivos de un año en particular.

    merge_year_files(): Fusiona todos los archivos anuales en un solo CSV.

    validate_and_save(): Valida el archivo final y lo guarda en Drive.

**Archivos generados**

    ecobici_merged_clean.csv: Base de datos limpia y consolidada.

    ecobici_merge_clean_log.txt: Registro detallado de procesamiento.

    merge_errors.log: (si aplica) Registro de errores por archivo.

    Copias de respaldo automáticas en Google Drive.

**Requerimientos**

Google Colab
pandas >= 1.3


### Paso 2: Modelación No Supervisada (TAD Continuas)

**Objetivo**

Transformar y analizar los datos de uso de Ecobici para identificar patrones temporales y demográficos utilizando técnicas de modelado no supervisado. Este paso incluye limpieza adicional, transformación de variables, reducción de dimensionalidad y visualización avanzada.

**Flujo de trabajo**

    Preparación del entorno:

        Instalación y actualización automática de librerías (scikit-learn, plotly, tslearn).

        Montaje de Google Drive y descompresión del archivo ecobici_merged_clean.csv.zip.

    Análisis exploratorio de datos:

        Muestreo inicial de hasta 500,000 registros.

        Estadísticas descriptivas de variables numéricas y categóricas.

        **Visualización** inicial: distribución de edad, género, hora del día y estaciones más usadas.

    Preprocesamiento:

        Filtrado de edad: Se conservan registros con edad entre 15 y 80 años.

        Codificación de género: Se estandariza y codifica como variables dummy (Genero_M, Genero_F).

        Transformación horaria: Se aplica codificación circular (Hora_Seno, Hora_Coseno).

        Segmentación temporal: Se asigna a cada viaje un Periodo_Dia (Mañana, Tarde, Tarde-Noche, Noche).

        Normalización de edad: Se utiliza StandardScaler para estandarizar la variable.

    Transformación por bloques (chunks):

        El archivo completo se procesa en fragmentos de 100,000 registros.

        Cada bloque se transforma individualmente y se guarda en ecobici_transformed.csv.

    Reducción de dimensionalidad con PCA:

        Se aplican componentes principales para visualizar los patrones latentes.

        Se imprimen las proporciones de varianza explicada por cada componente.

    **Visualización** avanzada:

        Plotly: Gráficos interactivos 3D de edad vs hora vs periodo, sunburst por género y periodo.

        Matplotlib/Seaborn: Heatmaps de correlación, histogramas horarios, boxplots de edad por periodo, gráficos circulares.

**Transformaciones aplicadas**

Variable original	Transformación	Salida
Edad_Usuario	Estandarización (StandardScaler)	Edad_Estandarizada (float)
Hora_Retiro	Codificación circular	Hora_Seno, Hora_Coseno
Genero_Usuario	One-hot encoding	Genero_M, Genero_F (binario)
-	Segmentación por horario	Periodo_Dia → variables dummy

**Resultados del PCA**

    Se extraen los principales componentes que resumen la información contenida en las variables transformadas.

    Se muestra la varianza explicada individual y acumulada, facilitando la selección de componentes relevantes.

**Archivos generados**

    ecobici_transformed.csv: Archivo con los datos transformados y listos para clustering.

    Gráficos interactivos (Plotly) y estáticos (Matplotlib, Seaborn).

**Requisitos técnicos**

pip install scikit-learn plotly seaborn tqdm tslearn



### Paso 3: Entrenamiento de Modelos

**Objetivo**

Aplicar técnicas de clustering no supervisado para identificar patrones de uso del sistema Ecobici, comparando el rendimiento de distintos algoritmos sobre los datos transformados. Se busca segmentar usuarios y viajes de acuerdo a características horarias, demográficas y de uso.

**Modelos Entrenados**

Se entrenaron tres modelos de clustering utilizando datos escalados y procesados por bloques (chunks) para asegurar eficiencia y compatibilidad con grandes volúmenes de datos:
Modelo	            Tipo	                Ventajas	                                                        Desventajas
K-Means	            Particional	            Rápido, escalable, ideal para datos en tiempo real	                Asume clusters esféricos
BIRCH	            Jerárquico	            Eficiente en memoria, adecuado para grandes datos	                Menor precisión si los datos son dispersos
GMM (Bayesiano)	    Probabilístico	        Capta clusters con forma libre y densidades variadas	            Más costoso computacionalmente

    Nota: Se utilizó BIRCH en lugar de Clustering Aglomerativo tradicional, ya que este último no era viable para procesar más de 100 millones de registros por limitaciones de memoria.

**Flujo de Entrenamiento**

    Descompresión y carga de datos desde Google Drive.

    Limpieza y filtrado de columnas problemáticas.

    Escalado de variables numéricas mediante StandardScaler.

    Entrenamiento por bloques o muestra:

        K-Means: Entrenamiento incremental (partial_fit) por chunks.

        BIRCH: Entrenamiento sobre muestra aleatoria de 10,000 registros.

        GMM: Entrenamiento iterativo con warm_start y mezcla bayesiana.

    Guardado de modelos en .joblib.

**Evaluación de Desempeño**

Se utilizaron dos métricas estándar para comparar los modelos:

    Silhouette Score: Mide qué tan bien definidos están los clusters (0 a 1).

    Calinski-Harabasz Index: Evalúa la separación entre clusters (mayor es mejor).


| Modelo   | Silhouette Score | Calinski-Harabasz | Clusters Efectivos  |
|----------|------------------|-------------------|---------------------|
    BIRCH	    0.524	            7,985	                5 
    K-Means	    0.430	            4,941	                5
    GMM	        0.260	            2,599	                3

**Visualización**

Se aplicó PCA para reducir la dimensionalidad y visualizar los clusters en 2D:

    Se generaron gráficos comparativos de clusters para los tres modelos usando matplotlib y seaborn.

    También se creó un gráfico de barras para comparar visualmente las métricas de cada modelo.

Ejemplo de visualización de clusters:

plot_clusters(kmeans_model, X_sample_scaled, "K-Means Clustering")

**Análisis por Cluster:**

    Distribuciones de usuarios por edad, género, estación y hora del día.

    **Visualización** con heatmaps para interpretar las medias de cada grupo.

    Funciones avanzadas de interpretación y debug para validar resultados.

**Archivos Generados**

    kmeans_model.joblib

    birch_model.joblib

    gmm_model.joblib

    scaler.joblib

    Gráficos de evaluación (Silhouette, Calinski-Harabasz)

    Visualizaciones por cluster en mapas de calor


### Paso 4: Perfilamiento Temporal por Cluster

**Objetivo**

Unir los datos crudos (fecha y hora originales) con los datos transformados y etiquetados por modelo, para generar un perfil temporal detallado por cluster. El objetivo es entender patrones de uso según hora del día, día de la semana y otras variables demográficas.

**Flujo de trabajo**

    Unificación de datos:

        Se combinan los archivos ecobici_merged_clean.csv (fechas y horas) con ecobici_transformed.csv (datos listos para clustering).

        El resultado es un archivo unificado: ecobici_con_clusters.csv.

    Asignación de clusters:

        Se carga el modelo entrenado (kmeans_model.joblib) y el escalador.

        Por bloques (chunks), se limpia, escala y predice el cluster de cada registro.

        Se guarda el resultado en ecobici_con_clusters_final.csv.

    Generación de variables temporales:

        Se derivan variables como:

            Hora_Retiro: convertida a número entero (hora).

            Dia_Semana: día de la semana en texto.

            Es_FinDeSemana: indicador binario.

            Mes: mes del año.

        Archivo de salida: ecobici_perfilamiento_final.csv.

    Análisis por cluster:

        Para cada cluster se calculan:

            Horario pico de uso.

            Día más activo.

            Edad promedio (desescalada).

            Porcentaje de usuarios masculinos.

        Los resultados se resumen en resumen_clusters.csv.

    **Visualización**:

        Histogramas por hora y por día de la semana para cada cluster.

        Heatmap combinado: distribución de viajes por hora y cluster.

**Resultados resumidos (ejemplo):**

| Cluster | Horario Pico | Día Más Activo | Edad Promedio  | % Masculino  |
|---------|--------------|----------------|----------------|--------------|
    0	        8:00	    Wednesday	    32.5 años	        68%
    1	        6:00 PM	    Friday	        28.7 años	        72%
    2	        12:00 PM	Saturday	    35.2 años	        54%
    3	        7:30 AM	    Monday	        41.8 años	        63%
    4	        7:00 PM	    Thursday	    27.3 años	        59%

    Las edades fueron desescaladas a partir del modelo original utilizando la media y desviación estándar.

**Archivos generados**

    ecobici_con_clusters.csv: Datos crudos + transformados

    ecobici_con_clusters_final.csv: Con cluster asignado

    ecobici_perfilamiento_final.csv: Con variables temporales

    resumen_clusters.csv: Resumen estadístico por cluster

    Gráficos:

        Histogramas de uso por hora

        Barras por día de la semana

        Heatmap de uso por hora y cluster
