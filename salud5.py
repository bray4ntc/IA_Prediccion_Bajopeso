import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from imblearn.over_sampling import SMOTE

# --- Paso 1: Cargar los datos ---
file_household = 'RECH0_2023.csv'  # Datos del hogar
file_children = 'RECH6_2023.csv'  # Datos de los niños

df_household = pd.read_csv(file_household)
df_children = pd.read_csv(file_children)

# --- Paso 2: Seleccionar columnas relevantes ---
columns_to_keep_household = [
    'HHID',  # Identificador único del hogar para unir datasets
    'HV040',  # Número de conglomerados
    'HV022',  # Estrato
    'HV024',  # Región
    'HV025',  # Área de residencia
    'HV009', 'HV010', 'HV011', 'HV012', 'HV013', 'HV014',  # Números totales en el hogar
    'UBIGEO',  # Ubicación geográfica (código)
    'NOMCCPP'  # Nombre del centro poblado
]

columns_to_keep_children = [
    'HHID',  # Identificador único del hogar para unir datasets
    'HC2',   # Peso en kilogramos
    'HC3',   # Altura en centímetros
    'HC8',   # Peso/Edad desviación estándar
    'HC61'   # Nivel educativo de la madre
]

df_household = df_household[columns_to_keep_household]
df_children = df_children[columns_to_keep_children]

# --- Paso 3: Unir datasets ---
df_combined = pd.merge(df_household, df_children, on='HHID')

# --- Paso 4: Limpieza de datos ---
# Reemplazar valores no válidos en HC8 por NaN
df_combined.replace({9998: np.nan}, inplace=True)
# Eliminar filas con valores faltantes
df_combined.dropna(inplace=True)

# Crear la variable objetivo: 'BajoPeso'
df_combined['BajoPeso'] = (df_combined['HC8'] < -2).astype(int)

# Eliminar HC8, ya que ahora su información está en 'BajoPeso'
df_combined.drop(columns=['HC8'], inplace=True)

# Convertir NOMCCPP a valores numéricos
nomccpp_mapping = {name: idx for idx, name in enumerate(df_combined['NOMCCPP'].unique())}
df_combined['NOMCCPP'] = df_combined['NOMCCPP'].map(nomccpp_mapping)

# --- Paso 5: Ingeniería de características ---
df_combined['DensidadHogar'] = df_combined['HV009'] / (df_combined['HV013'] + 1)  # Personas por habitación
df_combined['RatioNiños'] = df_combined['HV014'] / df_combined['HV009']  # Proporción de niños en el hogar

# --- Paso 6: Guardar el archivo combinado ---
output_file = 'dataset_salud_mejorado.csv'
df_combined.to_csv(output_file, index=False)
print(f"Archivo combinado creado: {output_file}")

# --- Paso 7: Preprocesamiento para el modelo ---
# Codificar variables categóricas
df_combined = pd.get_dummies(df_combined, columns=['HV022', 'HV024', 'HV025', 'HC61'], drop_first=True)

# Dividir en variables predictoras y objetivo
X = df_combined.drop(['BajoPeso', 'HHID', 'UBIGEO'], axis=1)  # Excluir identificadores irrelevantes
y = df_combined['BajoPeso']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Balanceo de clases con SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# --- Paso 8: Eliminar columnas constantes ---
selector_var = VarianceThreshold(threshold=0.0)  # Eliminar columnas con varianza cero
X_train_balanced = selector_var.fit_transform(X_train_balanced)
X_test = selector_var.transform(X_test)

# --- Paso 9: Selección de características ---
selector_kbest = SelectKBest(score_func=f_classif, k=10)  # Seleccionar las 10 mejores características
X_train_balanced = selector_kbest.fit_transform(X_train_balanced, y_train_balanced)
X_test = selector_kbest.transform(X_test)

# --- Paso 10: Escalado de características ---
scaler = StandardScaler()
X_train_balanced = scaler.fit_transform(X_train_balanced)
X_test = scaler.transform(X_test)

# --- Paso 11: Optimización de hiperparámetros ---
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

grid_search = GridSearchCV(LogisticRegression(max_iter=5000), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_balanced, y_train_balanced)

# Mejor modelo
best_model = grid_search.best_estimator_

# --- Paso 12: Entrenamiento y evaluación del modelo ---
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_score = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

print("Mejores hiperparámetros:", grid_search.best_params_)
print("Exactitud del modelo:", accuracy)
print("ROC-AUC Score:", roc_score)
print("\nMatriz de confusión:")
print(conf_matrix)
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))
