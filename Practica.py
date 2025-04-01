import pandas as pd
 
df = pd.read_csv("titanic.csv")
 
print(" Primeras filas:")
print(df.head())

print("\n  Información del dataset:")
print(df.info())

print("\n  Valores nulos por columna:")
print(df.isnull().sum())

print("\n Columnas disponibles:")
print(df.columns)

# En esta parte se eliminaron las variables por las siguientes razones:
# - 'Name': contiene datos únicos que no aportan a la predicción directamente.
# - 'Ticket': es un identificador no procesado, poco informativo.
# - 'Cabin': tiene muchos valores nulos.
# - 'PassengerId': es solo un identificador secuencial.
df.drop(columns=["Name", "Ticket", "Cabin", "PassengerId"], inplace=True, errors="ignore")

#  Se llenaron los  valores nulos que se lograron detectar .
if "Age" in df.columns:
    df["Age"].fillna(df["Age"].mean(), inplace=True)

if "Embarked" in df.columns:
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

if "Fare" in df.columns:
    df["Fare"].fillna(df["Fare"].mean(), inplace=True)

# Esta variable indica  si viajaba sola o acompañada la persona
df["family_size"] = df["sibsp"] + df["parch"] + 1

# Una variable binaria  indica  si tiene número de  bote o no 
df["has_boat"] = df["boat"].notnull().astype(int)

# Se agruparon los 10 destinos más comunes 
top_destinos = df["home.dest"].value_counts().nlargest(10).index
df["home.dest"] = df["home.dest"].where(df["home.dest"].isin(top_destinos), other="Otro")

# Codificación de variables categóricas
df = pd.get_dummies(df, columns=["sex", "embarked", "home.dest"], drop_first=True)

# Se confirmaron si las columnas finales que estenestan existentes 
print("\n  Columnas finales tras codificación:")
print(df.columns)

print("\n  Dataset limpio y listo para el modelo:")
print(df.head()) 

# Modelado con múltiples algoritmos 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
 
features = [
    "pclass", "age", "fare", "family_size", "has_boat",
    "sex_male", "embarked_Q", "embarked_S"
] + [col for col in df.columns if col.startswith("home.dest_")]
  
missing = [col for col in features if col not in df.columns]
if missing:
    print(f"  Faltan columnas necesarias para el modelo: {missing}")
else:
    df_model = df[features + ["survived"]].dropna()
    X = df_model[features]
    y = df_model["survived"]

    # Particion de los datos: 70% train, 30% test.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #   SVM y Red Neuronal
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
 
    modelos = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "Red Neuronal": MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42),
        "Análisis Discriminante": LinearDiscriminantAnalysis(),
        "SVM": SVC(kernel='rbf', C=10, gamma=0.1, probability=True, random_state=42)
    }

    print("\n Resultados de los modelos:\n")

    for nombre, modelo in modelos.items():
        print(f"  Entrenando: {nombre}")

        if nombre in ["SVM", "Red Neuronal"]:
            modelo.fit(X_train_scaled, y_train)
            y_pred = modelo.predict(X_test_scaled)
            y_train_pred = modelo.predict(X_train_scaled)
        else:
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            y_train_pred = modelo.predict(X_train)

        test_accuracy = accuracy_score(y_test, y_pred)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        print(f"{nombre}:")
        print(f"  Precisión en entrenamiento (train): {train_accuracy:.4f}")
        print(f"  Precisión en prueba (test): {test_accuracy:.4f}")

# Se establecio un parametro para indicar el rendimiento del modelo
        if test_accuracy >= 0.90:
            print("   🟢 ¡Este modelo alcanzó el 90% o más en test!\n")
        elif test_accuracy >= 0.86:
            print("   🟡 ¡Este modelo alcanzó al menos el 86% en test!\n")
        else:
            print("   🔴 Aún por debajo del 86% en test\n")
