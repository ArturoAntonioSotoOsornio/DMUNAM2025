Proyecto de análisis de datos de ECOBICI (2013–2025)
Autora: Brenda Guadalupe Guerrero Sánchez
Asistencia técnica: ChatGPT, modelo GPT-4.5 de OpenAI

---

Resumen del trabajo realizado:

1. Consolidación de datos
   Se recopilaron y unificaron los archivos mensuales de viajes en bicicleta del sistema ECOBICI desde enero de 2013 hasta abril de 2025. Los archivos en formato CSV fueron convertidos a un único archivo en formato Parquet llamado ecobici_2013_2025.parquet, lo que mejora la eficiencia de lectura y almacenamiento.

2. Transformación de variables
   Se extrajeron y transformaron variables relevantes:
   - Edad del usuario.
   - Hora del viaje.
   - Día de la semana.
   - Mes del año.

   Estas variables se almacenaron en archivos tipo chunk en la carpeta chunks_transformados para facilitar el procesamiento por lotes y evitar saturación de memoria.

3. Modelado de clustering
   Se aplicó un modelo de agrupamiento Gaussian Mixture Model (GMM) sobre una muestra de 2 millones de registros transformados, previamente estandarizados.
   Este análisis ayuda a identificar patrones de comportamiento de los usuarios y agruparlos en perfiles según características de tiempo y edad.

4. Perfilamiento por fecha
   Se realizó un perfilamiento adicional por clúster, tomando en cuenta la fecha y hora de retiro, para analizar el comportamiento temporal de cada grupo.

---

Notas técnicas:

- El procesamiento fue realizado en Google Colab, aprovechando procesamiento por chunks para evitar problemas de memoria RAM.
- El uso de modelos de clustering como GMM permite identificar patrones sin etiquetas previas (aprendizaje no supervisado).
- El código y las transformaciones fueron desarrollados en colaboración con ChatGPT, modelo GPT-4.5 de OpenAI.