# -*- coding: utf-8 -*-
"""
Telecom X – Parte 2: Predicción de Cancelación (Churn)

Este script implementa un pipeline de Machine Learning para predecir el churn de clientes.
Incluye carga de datos, preprocesamiento, entrenamiento de modelos, evaluación y
análisis de importancia de variables.
"""

# =============================================================================
# 1. Configuración del Entorno e Importación de Librerías
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# =============================================================================
# 2. Carga y Preparación de Datos
# =============================================================================
def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """
    Carga los datos desde un archivo JSON, los normaliza y realiza una limpieza inicial.
    
    Args:
        filepath (str): La ruta al archivo JSON.

    Returns:
        pd.DataFrame: Un DataFrame limpio y aplanado.
    """
    print("Iniciando la carga y preparación de datos...")
    df = pd.read_json(filepath)

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
        df['Churn']
    ], axis=1)

    # Limpieza inicial
    df_flat = df_flat[df_flat['Churn'] != ""]
    df_flat['Charges.Total'] = pd.to_numeric(df_flat['Charges.Total'], errors='coerce')
    
    print(f"Datos cargados. Dimensiones: {df_flat.shape}")
    return df_flat

# =============================================================================
# 3. Preprocesamiento de Datos para Machine Learning
# =============================================================================
def preprocess_for_ml(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    """
    Prepara el DataFrame para el modelado: imputa nulos, codifica variables
    y escala las características numéricas.

    Args:
        df (pd.DataFrame): El DataFrame limpio.

    Returns:
        tuple: Un tuple conteniendo el DataFrame de características (X) y la serie del objetivo (y).
    """
    print("Iniciando preprocesamiento para Machine Learning...")
    # Imputación de nulos con la mediana
    median_total_charges = df['Charges.Total'].median()
    df['Charges.Total'].fillna(median_total_charges, inplace=True)

    # Codificación de la variable objetivo
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

    # Codificación de variables categóricas (One-Hot Encoding)
    df_processed = df.drop('customerID', axis=1)
    categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)

    # Separación de variables (X) y objetivo (y)
    X = df_processed.drop('Churn', axis=1)
    y = df_processed['Churn']

    # Escalado de variables numéricas
    numerical_cols = ['tenure', 'Charges.Monthly', 'Charges.Total']
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    print("Preprocesamiento completado.")
    return X, y

# =============================================================================
# 4. Entrenamiento y Evaluación de Modelos
# =============================================================================
def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> (dict, dict):
    """
    Divide los datos, entrena múltiples modelos de clasificación y evalúa su rendimiento.

    Args:
        X (pd.DataFrame): DataFrame de características.
        y (pd.Series): Serie del objetivo.

    Returns:
        tuple: Un tuple con un diccionario de modelos entrenados y un diccionario de resultados.
    """
    print("Dividiendo datos y entrenando modelos...")
    # División en datos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Regresión Logística": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }
    
    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"--- Entrenando {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        trained_models[name] = model
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "ROC AUC": roc_auc_score(y_test, y_proba),
            "Classification Report": classification_report(y_test, y_pred),
            "Confusion Matrix": confusion_matrix(y_test, y_pred)
        }
        print(f"Evaluación de {name}:")
        print(f"Accuracy: {results[name]['Accuracy']:.4f}")
        print(f"ROC AUC: {results[name]['ROC AUC']:.4f}\n")
        
    return trained_models, results

# =============================================================================
# 5. Visualización de Resultados
# =============================================================================
def visualize_results(trained_models: dict, results: dict, X: pd.DataFrame):
    """
    Genera visualizaciones clave: resumen de rendimiento, matriz de confusión
    del mejor modelo y la importancia de las variables.

    Args:
        trained_models (dict): Diccionario de modelos entrenados.
        results (dict): Diccionario con las métricas de evaluación.
        X (pd.DataFrame): DataFrame de características para obtener los nombres de las columnas.
    """
    print("Generando visualizaciones...")
    # Resumen de rendimiento
    summary_df = pd.DataFrame({
        'Modelo': list(results.keys()),
        'Accuracy': [res['Accuracy'] for res in results.values()],
        'ROC AUC': [res['ROC AUC'] for res in results.values()]
    }).set_index('Modelo').sort_values(by='ROC AUC', ascending=False)
    
    print("\n--- Resumen de Rendimiento ---")
    print(summary_df)
    
    # Matriz de Confusión del mejor modelo
    best_model_name = summary_df['ROC AUC'].idxmax()
    best_model_cm = results[best_model_name]['Confusion Matrix']

    plt.figure(figsize=(6, 5))
    sns.heatmap(best_model_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.title(f'Matriz de Confusión - {best_model_name}')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.show()

    # Importancia de las Variables
    best_model = trained_models[best_model_name]
    importances = best_model.feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(15)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title(f'Top 15 Variables más Importantes ({best_model_name})')
    plt.xlabel('Importancia')
    plt.ylabel('Variable')
    plt.tight_layout()
    plt.show()

# =============================================================================
# 6. Conclusión Estratégica
# =============================================================================
def print_strategic_conclusions():
    """Imprime en consola las conclusiones y recomendaciones estratégicas."""
    conclusion = """
    =============================================================================
    8. Conclusión Estratégica
    =============================================================================

    ### Resumen del Rendimiento del Modelo
    El modelo Gradient Boosting demostró ser el más eficaz, con un ROC AUC superior.      
    Su capacidad para discriminar entre clientes que cancelan y los que no es la más robusta.

    ### Principales Factores que Influyen en la Cancelación
    1.  Contrato Mes a Mes: El predictor más fuerte de churn.
    2.  Permanencia (tenure): Clientes nuevos son más propensos a cancelar.
    3.  Servicio de Fibra Óptica: Sugiere posibles problemas de calidad/precio.
    4.  Cargos Mensuales Elevados: Factor de riesgo económico directo.
    5.  Falta de Soporte Técnico: Indica menor integración del cliente con los servicios. 

    ### Recomendaciones para Telecom X
    -   ACCIÓN INMEDIATA: Lanzar campañas de retención para clientes con contrato
        mensual y baja permanencia, ofreciendo descuentos para migrar a planes anuales.   
    -   INVESTIGACIÓN ESTRATÉGICA: Analizar la causa de la alta tasa de cancelación       
        en clientes con fibra óptica (encuestas, análisis de competencia).
    -   MEJORA DEL PRODUCTO: Crear paquetes de servicios que incluyan Soporte Técnico     
        y Seguridad Online para aumentar el valor percibido y la retención.
    """
    print(conclusion)

# =============================================================================
# Función Principal (main) para orquestar el pipeline
# =============================================================================
def main():
    """Función principal que ejecuta el pipeline completo."""
    filepath = 'TelecomX_Data.json'
    
    # Ejecutar el pipeline
    df_clean = load_and_prepare_data(filepath)
    X, y = preprocess_for_ml(df_clean)
    trained_models, results = train_and_evaluate(X, y)
    visualize_results(trained_models, results, X)
    print_strategic_conclusions()

if __name__ == '__main__':
    main()