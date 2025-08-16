# Telecom X - Análisis de Evasión de Clientes (Churn)

## 📌 Propósito del análisis
Este proyecto tiene como objetivo analizar los datos de clientes de la empresa Telecom X para identificar patrones y factores asociados a la evasión de clientes (churn). El análisis permite preparar los datos para modelos predictivos y entregar recomendaciones al negocio para reducir la pérdida de clientes.

## 📁 Estructura del proyecto
- `TelecomX_Data.json`: Archivo con los datos originales de clientes en formato JSON.
- `TelecomX_diccionario.md`: Diccionario de datos con la descripción de cada variable.
- `TelecomX_LATAM.ipynb`: Notebook principal con todo el flujo de análisis, limpieza, visualización y conclusiones.
- `README.md`: Este archivo, con la guía y documentación del proyecto.

## 📊 Ejemplos de gráficos e insights
A continuación, algunos ejemplos de visualizaciones generadas en el análisis:

- **Distribución de Churn:**
  ![Ejemplo Churn](https://i.imgur.com/4yQwQ7B.png)
- **Churn por tipo de contrato:**
  ![Ejemplo Contrato](https://i.imgur.com/8yQwQ7B.png)
- **Distribución de tenure según Churn:**
  ![Ejemplo Tenure](https://i.imgur.com/1yQwQ7B.png)

**Principales insights:**
- Los clientes con contrato mes a mes presentan mayor tasa de churn.
- El tiempo de permanencia (tenure) bajo se asocia a mayor evasión.
- No hay diferencia significativa de churn por género.

## ▶️ Instrucciones para ejecutar el notebook
1. Descarga o clona este repositorio.
2. Asegúrate de tener Python 3 y las siguientes librerías instaladas:
   - pandas
   - numpy
   - matplotlib
   - seaborn
3. Abre el archivo `TelecomX_LATAM.ipynb` en Jupyter Notebook, VS Code o Google Colab.
4. Ejecuta las celdas en orden para reproducir el análisis y visualizar los resultados.

---

¡Explora el notebook y descubre los factores clave detrás de la evasión de clientes en Telecom X!
