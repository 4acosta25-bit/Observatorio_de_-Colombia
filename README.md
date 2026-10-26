# 🇨🇴 Observatorio Avanzado de Análisis Económico y Social por Departamentos en Colombia

Aplicación analítica interactiva construida en Python para la exploración, estimación econométrica, modelado supervisado, reducción de dimensionalidad y simulación de política pública a nivel departamental en Colombia.

## 🚀 Características y Módulos del Sistema

1. **Exploración Espacial y Mapas Coropléticos:** Integración con GeoJSON oficial de Colombia y renderizado interactivo de variables socioeconómicas.
2. **Econometría y Series de Tiempo Multivariadas:** Estimación de modelos Vectores Autorregresivos (VAR) y Modelos de Corrección de Errores Vectoriales (VECM) con proyección temporal.
3. **Machine Learning y Deep Learning:** Algoritmos de regularización (Ridge y Lasso), Support Vector Regression (SVR) y perceptrones multicapa (MLP) para predicción supervisada.
4. **Minería de Datos (PCA):** Reducción de dimensionalidad con estandarización previa para agrupamiento departamental.
5. **Interpretabilidad Automatizada:** Estimación de importancia relativa de variables mediante árboles en ensambles (*Random Forest*).
6. **Simulador de Política Pública ("What-If"):** Análisis de impacto de choques presupuestales sobre variables macrométricas departamentales.

## 🛠️ Instalación y Ejecución Local

1. Clonar este repositorio o copiar la carpeta del proyecto.
2. Crear y activar un entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Linux/macOS
   # venv\Scripts\activate   # En Windows