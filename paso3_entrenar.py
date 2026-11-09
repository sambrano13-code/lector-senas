# Paso 3: entrenar el modelo con los ejemplos grabados
import csv
from collections import Counter

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

X, y = [], []
with open("datos.csv", encoding="utf-8") as f:
    for fila in csv.reader(f):
        if fila:
            y.append(fila[0])
            X.append([float(v) for v in fila[1:]])

# Si quedaron grabaciones de una versión vieja (con otra cantidad de datos),
# se usan solo las de la versión actual
largo_actual = len(X[-1])
viejas = sum(1 for fila in X if len(fila) != largo_actual)
if viejas:
    print(f"Ignorando {viejas} ejemplos grabados con una versión anterior.")
    y = [et for et, fila in zip(y, X) if len(fila) == largo_actual]
    X = [fila for fila in X if len(fila) == largo_actual]

X = np.array(X)
print("Ejemplos por seña:", dict(Counter(y)))
if len(set(y)) < 2:
    raise SystemExit("Necesitás grabar al menos 2 señas distintas.")

X_entr, X_prueba, y_entr, y_prueba = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

modelo = RandomForestClassifier(n_estimators=200, random_state=42)
modelo.fit(X_entr, y_entr)
print(f"\nPrecisión con ejemplos que no vio: {modelo.score(X_prueba, y_prueba):.0%}\n")
print(classification_report(y_prueba, modelo.predict(X_prueba), zero_division=0))

# Entrenar con todo y guardar
modelo.fit(X, y)
joblib.dump(modelo, "modelo_senas.pkl")
print("Modelo guardado en modelo_senas.pkl")
