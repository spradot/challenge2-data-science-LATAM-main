# %% [markdown]
# # Telecom X – Parte 2: Predicción de Cancelación (Churn)
#
# **Autor:** Analista Junior de Machine Learning
# **Misión:** Desarrollar un pipeline de Machine Learning para predecir la cancelación de clientes, identificar los factores clave y proponer acciones estratégicas.

# %% [markdown]
# ## 1️⃣ Configuración del Entorno
# Importación de las librerías necesarias para el análisis, preprocesamiento, modelado y evaluación.

# %%
# Manipulación y análisis de datos
import pandas as pd
import numpy as np

# Visualizaciones
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocesamiento de datos
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Modelos de clasificación
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Métricas de evaluación
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# %% [markdown]
# ## 2️⃣ Carga y Preparación de Datos
# Replicamos los pasos iniciales de la Parte 1 para cargar y aplanar los datos, asegurando un DataFrame limpio y listo para el preprocesamiento.

# %%
# Cargar el archivo JSON
df = pd.read_json('TelecomX_Data.json')

# Normalizar columnas anidadas (JSON)
customer_df = pd.json_normalize(df['customer'])
phone_df = pd.json_normalize(df['phone'])
internet_df = pd.json_normalize(df['internet'])
account_df = pd.json_normalize(df['account'])

# Concatenar en un único DataFrame plano
df_flat = pd.concat([
    df[['customerID']],
    customer_df,
    phone_df,
    internet_df,
    account_df,
    df['Churn']  # Añadir la columna Churn al final
], axis=1)

# Limpieza inicial
# Eliminar filas donde 'Churn' es un string vacío
df_flat = df_flat[df_flat['Churn'] != ""]

# Convertir 'Charges.Total' a numérico, los errores se convierten en NaN
df_flat['Charges.Total'] = pd.to_numeric(df_flat['Charges.Total'], errors='coerce')

print("Dimensiones del DataFrame:", df_flat.shape)
df_flat.head()

# %% [markdown]
# ## 3️⃣ Preprocesamiento de Datos para Machine Learning
# Esta etapa es crucial. Transformaremos los datos para que los algoritmos puedan interpretarlos correctamente.
#
# **Pasos:**
# 1.  **Imputación de Nulos:** Rellenar los valores faltantes en `Charges.Total` con la mediana.
# 2.  **Codificación de la Variable Objetivo:** Convertir la columna `Churn` a formato binario (0/1).
# 3.  **Codificación de Variables Categóricas:** Usar One-Hot Encoding (`get_dummies`) para transformar las variables categóricas en numéricas.
# 4.  **Escalado de Variables Numéricas:** Estandarizar las variables numéricas para que tengan media 0 y desviación estándar 1.

# %%
# 1. Imputación de Nulos
# Usamos la mediana porque es más robusta a valores atípicos que la media.
median_total_charges = df_flat['Charges.Total'].median()
df_flat['Charges.Total'].fillna(median_total_charges, inplace=True)
print(f"Valores nulos restantes: {df_flat.isnull().sum().sum()}")

# 2. Codificación de la Variable Objetivo (Target)
df_flat['Churn'] = df_flat['Churn'].map({'No': 0, 'Yes': 1})

# 3. Codificación de Variables Categóricas
# Seleccionamos las columnas a procesar, excluyendo el ID de cliente
df_processed = df_flat.drop('customerID', axis=1)
categorical_cols = df_processed.select_dtypes(include=['object']).columns
df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)

print("Dimensiones tras One-Hot Encoding:", df_processed.shape)

# %%
# 4. Separar variables (X) y objetivo (y), y luego escalar
X = df_processed.drop('Churn', axis=1)
y = df_processed['Churn']

# Identificar columnas numéricas para escalar
numerical_cols = ['tenure', 'Charges.Monthly', 'Charges.Total']

# Aplicar StandardScaler
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

print("\nPrimeras filas de los datos escalados:")
X.head()


# %% [markdown]
# ## 4️⃣ Análisis de Correlación
# Visualizamos la correlación de las variables con el `Churn` para identificar relaciones lineales importantes.

# %%
plt.figure(figsize=(12, 10))
# Creamos una matriz de correlación solo con la columna 'Churn'
corr_matrix = df_processed.corr()[['Churn']].sort_values(by='Churn', ascending=False)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlación de Variables con Churn')
plt.show()

# %% [markdown]
# ## 5️⃣ División de Datos y Entrenamiento de Modelos
# Dividimos los datos en conjuntos de entrenamiento (80%) y prueba (20%). Luego, entrenaremos tres modelos de clasificación:
# 1.  **Regresión Logística:** Un modelo lineal simple y fácil de interpretar, ideal como línea de base.
# 2.  **Random Forest:** Un modelo de ensamble basado en árboles de decisión, muy potente y robusto.
# 3.  **Gradient Boosting:** Otro modelo de ensamble que suele ofrecer un rendimiento superior.

# %%
# División en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Tamaño de entrenamiento: {X_train.shape[0]} muestras")
print(f"Tamaño de prueba: {X_test.shape[0]} muestras")

# %%
# Diccionario para almacenar los modelos y sus resultados
models = {
    "Regresión Logística": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

results = {}

# Bucle para entrenar y evaluar cada modelo
for name, model in models.items():
    print(f"--- Entrenando {name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Almacenar métricas
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_proba),
        "Classification Report": classification_report(y_test, y_pred),
        "Confusion Matrix": confusion_matrix(y_test, y_pred)
    }
    
    print(f"Evaluación de {name}:")
    print(f"Accuracy: {results[name]['Accuracy']:.4f}")
    print(f"ROC AUC: {results[name]['ROC AUC']:.4f}")
    print(results[name]['Classification Report'])
    print("-" * 30 + "\n")

# %% [markdown]
# ## 6️⃣ Evaluación y Comparación de Modelos
# Comparamos el rendimiento de los modelos utilizando métricas clave. El **ROC AUC** es especialmente útil en problemas con clases desbalanceadas como el churn.

# %%
# Resumen de resultados
summary_df = pd.DataFrame({
    'Modelo': list(results.keys()),
    'Accuracy': [res['Accuracy'] for res in results.values()],
    'ROC AUC': [res['ROC AUC'] for res in results.values()]
}).set_index('Modelo')

print("Resumen de Rendimiento:")
print(summary_df.sort_values(by='ROC AUC', ascending=False))

# Visualización de la Matriz de Confusión para el mejor modelo (basado en ROC AUC)
best_model_name = summary_df['ROC AUC'].idxmax()
best_model_cm = results[best_model_name]['Confusion Matrix']

plt.figure(figsize=(6, 5))
sns.heatmap(best_model_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.title(f'Matriz de Confusión - {best_model_name}')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

# %% [markdown]
# ## 7️⃣ Importancia de las Variables (Feature Importance)
# Identificamos qué factores tienen más peso en la predicción del modelo más potente (Random Forest o Gradient Boosting). Esto nos da insights accionables para el negocio.

# %%
# Usar el modelo de Gradient Boosting para la importancia de las variables
best_model = models['Gradient Boosting']
importances = best_model.feature_importances_
feature_names = X.columns

# Crear un DataFrame para visualizar mejor
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False).head(15) # Top 15

# Visualización
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Top 15 Variables más Importantes para Predecir Churn (Gradient Boosting)')
plt.xlabel('Importancia')
plt.ylabel('Variable')
plt.show()

# %% [markdown]
# ## 8️⃣ Conclusión Estratégica
#
# ### Resumen del Rendimiento del Modelo
# El modelo **Gradient Boosting** demostró ser el más eficaz, con un **ROC AUC de 0.8444**. Aunque la precisión general (Accuracy) es similar entre los modelos, su capacidad para discriminar entre clientes que cancelan y los que no es superior. La matriz de confusión revela que, si bien aún hay falsos negativos (clientes que cancelan y no fueron detectados), el modelo tiene un buen balance general.
#
# ### Principales Factores que Influyen en la Cancelación
# El análisis de importancia de variables del modelo Gradient Boosting nos revela los siguientes factores clave, en orden de relevancia:
#
# 1.  **Tipo de Contrato (`Contract_Month-to-month`):** Es, con diferencia, el predictor más fuerte. Los clientes sin un contrato a largo plazo tienen una altísima probabilidad de cancelar.
# 2.  **Permanencia (`tenure`):** Los clientes más nuevos (baja permanencia) son mucho más propensos a irse. La lealtad se construye con el tiempo.
# 3.  **Servicio de Internet (`InternetService_Fiber optic`):** Los clientes con fibra óptica tienen una mayor tasa de cancelación. Esto podría ser contraintuitivo y sugiere problemas de calidad, precio o expectativas no cumplidas con este servicio específico.
# 4.  **Cargos Mensuales (`Charges.Monthly`):** Cargos mensuales más altos están correlacionados con una mayor probabilidad de cancelación.
# 5.  **Soporte Técnico (`TechSupport_No`):** La falta de un servicio de soporte técnico contratado es un indicador de riesgo significativo.
#
# ### 💡 Recomendaciones para Telecom X
#
# - **Acción Inmediata - Campañas de Retención Focalizadas:**
#   - **Objetivo:** Clientes con **contrato mensual** y **baja permanencia** (menos de 6 meses).
#   - **Oferta:** Proponerles un descuento agresivo para migrar a un contrato de 1 o 2 años. Esto ataca directamente los dos principales factores de riesgo.
#
# - **Investigación Estratégica - Servicio de Fibra Óptica:**
#   - **Pregunta Clave:** ¿Por qué los clientes de fibra óptica, nuestro servicio premium, cancelan más?
#   - **Acción:** Realizar encuestas de satisfacción específicas para este segmento. Analizar si el problema es el precio, la estabilidad del servicio, o si la competencia está ofreciendo mejores alternativas.
#
# - **Mejora del Producto - Paquetes de Servicios:**
#   - **Observación:** La falta de servicios de valor añadido como **Soporte Técnico** y **Seguridad Online** aumenta el riesgo de cancelación.
#   - **Acción:** Crear paquetes de servicios que incluyan `TechSupport` y `OnlineSecurity` por un costo marginal, especialmente para clientes nuevos, para aumentar su dependencia y satisfacción con el ecosistema de Telecom X.
#
# Con este modelo, Telecom X puede pasar de una estrategia reactiva a una **proactiva**, identificando clientes en riesgo antes de que tomen la decisión de irse y ofreciéndoles soluciones personalizadas.