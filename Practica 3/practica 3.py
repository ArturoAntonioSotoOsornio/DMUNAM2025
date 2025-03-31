import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report

# Cargar datos
df = pd.read_csv("titanic.csv")

# Normalizar nombres de columnas
df.columns = df.columns.str.lower()

# Eliminar columnas innecesarias
df.drop(columns=['passengerid', 'name', 'ticket', 'cabin'], inplace=True, errors='ignore')

# Manejo de valores nulos
df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
df['fare'].fillna(df['fare'].median(), inplace=True)

# Eliminar filas con NaN en la variable objetivo
df.dropna(subset=['survived'], inplace=True)
X = df.drop(columns=['survived'])
y = df['survived']

# Transformaciones de datos
numeric_features = ['age', 'fare', 'sibsp', 'parch']
categorical_features = ['pclass', 'sex', 'embarked']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# División en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Modelo optimizado
rf = RandomForestClassifier(random_state=42, n_estimators=500, max_depth=14, min_samples_split=5, min_samples_leaf=2)

titanic_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', rf)
])

# Entrenar modelo
titanic_pipeline.fit(X_train, y_train)

# Evaluación
train_acc = accuracy_score(y_train, titanic_pipeline.predict(X_train))
val_acc = accuracy_score(y_val, titanic_pipeline.predict(X_val))

print(f'Accuracy en entrenamiento: {train_acc:.4%}')
print(f'Accuracy en validación: {val_acc:.4%}')
print('Reporte de clasificación en validación:')
print(classification_report(y_val, titanic_pipeline.predict(X_val)))
