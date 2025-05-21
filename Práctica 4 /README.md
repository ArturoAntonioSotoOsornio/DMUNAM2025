
# Práctica 4: Aprendizaje no supervisado

## Descripción General

Este proyecto tiene como objetivo aplicar técnicas de aprendizaje no supervisado, en particular métodos de agrupamiento (clustering), para descubrir patrones ocultos en conjuntos de datos sin etiquetar. A través de este notebook se lleva a cabo un flujo completo de análisis de datos, desde la carga y limpieza, hasta la visualización y evaluación de resultados de agrupamiento usando distintos algoritmos y técnicas de reducción de dimensionalidad.

## Objetivos Específicos

- Implementar distintos métodos de clustering y comparar su rendimiento.
- Visualizar las estructuras formadas por los clústeres en espacios reducidos.
- Evaluar los modelos mediante métricas internas de validación.
- Determinar la relevancia de variables en la formación de clústeres.
- Proporcionar un análisis reproducible y modular.

---

## Contenido del Notebook

El flujo de trabajo cubierto por este notebook incluye:

### 1. Carga y Exploración de Datos
- Carga de archivos CSV desde el sistema de archivos o Google Drive.
- Exploración inicial de los datos: dimensiones, tipos, valores nulos.

### 2. Preprocesamiento
- Imputación de valores faltantes.
- Escalamiento de características usando `StandardScaler`.
- Posible balanceo de clases si se trabaja con subconjuntos.

### 3. Reducción de Dimensionalidad
Se utilizan técnicas de reducción para facilitar la visualización y mejorar el rendimiento de los algoritmos de clustering:

- **PCA (Principal Component Analysis)**
- **Incremental PCA**
- **t-SNE (t-distributed Stochastic Neighbor Embedding)**

### 4. Algoritmos de Clustering Aplicados

| Algoritmo               | Descripción breve                                      |
|-------------------------|--------------------------------------------------------|
| `KMeans`                | Agrupamiento basado en centroides, requiere número de clústeres. |
| `MiniBatchKMeans`       | Variante optimizada para grandes volúmenes de datos.  |
| `Gaussian Mixture`      | Modelado probabilístico usando mezclas de distribuciones normales. |
| `Agglomerative Clustering` | Método jerárquico de agrupamiento ascendente.       |

### 5. Evaluación de Clustering

Para evaluar la calidad de los clústeres generados, se emplean varias métricas internas:

- **Silhouette Score**
- **Índice de Calinski-Harabasz**
- **Índice de Davies-Bouldin**

Estas métricas no requieren etiquetas verdaderas y permiten cuantificar qué tan bien separados y compactos están los clústeres.

### 6. Visualización
- Gráficas en 2D de los clústeres proyectados con PCA o t-SNE.
- Análisis de las características dominantes por clúster.
- Uso de `matplotlib` y `seaborn` para mostrar resultados.

---

## Tecnologías y Librerías Utilizadas

El notebook utiliza las siguientes bibliotecas:

- `pandas`, `numpy`: Manipulación y análisis de datos.
- `matplotlib`, `seaborn`: Visualización.
- `scikit-learn`: Algoritmos de clustering, reducción de dimensionalidad y métricas.
- `tqdm`: Barra de progreso.
- `os`, `glob`, `shutil`: Manejo de archivos.
- `warnings`, `datetime`: Gestión de advertencias y tiempos de ejecución.
- `google.colab`: Montaje de Google Drive (en Colab).

---

## Requisitos de Entorno

Este proyecto está diseñado para ejecutarse en **Google Colab**, aunque también puede funcionar localmente si los archivos necesarios están disponibles.

Para correrlo localmente, asegúrate de instalar los siguientes paquetes:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tqdm
