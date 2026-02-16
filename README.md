# Proyecto B: Sistema de Predicción de Abandono de Clientes (Churn)

## 👥 Equipo
- **Integrante 1**: Claudio Poveda Sanchez 
- **Integrante 2**: Kendall Solano Solis 
- **Integrante 3**: Roberto Coto Guevara 

## 📋 Descripción del Proyecto
Sistema inteligente que predice qué clientes tienen mayor probabilidad de abandonar un servicio de telecomunicaciones, permitiendo implementar estrategias de retención oportunas.

## 🎯 Objetivos
- Analizar factores que influyen en el abandono de clientes
- Desarrollar modelos de clasificación binaria y scoring de riesgo con ANN
- Calcular ROI potencial de estrategias de retención
- Crear sistema de alertas tempranas para clientes en riesgo

## 📊 Dataset
- **Fuente**: Telco Customer Churn Dataset
- **URL**: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Registros**: 7,043 clientes
- **Variables**: 21 (demográficas, servicios, información de cuenta)
- **Variables principales**: antigüedad, tipo de contrato, método de pago, cargos mensuales, servicios adicionales

## 🔧 Instalación

### Requisitos Previos
- Python 3.8+
- pip
- Cuenta de Kaggle (para descargar dataset)

### Pasos de Instalación
```bash
# 1. Navegar al proyecto
cd ProyectoB_ChurnClientes

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar Kaggle API (si no lo has hecho)
# Descargar kaggle.json desde tu perfil de Kaggle
# Linux/Mac: mv kaggle.json ~/.kaggle/
# Windows: mv kaggle.json %HOMEPATH%\.kaggle\

# 5. Descargar dataset
python data/raw/download_data.py
```

## 🚀 Uso

### Notebooks (orden recomendado)
```bash
jupyter notebook notebooks/
```
1. `01_EDA_Churn.ipynb` - Análisis de tasas de churn por segmento
2. `02_Preprocesamiento.ipynb` - Manejo de desbalanceo de clases
3. `03_ANN_BinaryClass.ipynb` - Modelo de predicción churn
4. `04_ANN_RiskScore.ipynb` - Modelo de scoring de riesgo
5. `05_ROI_Analysis.ipynb` - Análisis de retorno de inversión

### Entrenar Modelos
```bash
python src/train/churn_binary.py
python src/train/risk_scorer.py
```

### API
```bash
cd api
uvicorn main:app --reload
```
Documentación: http://localhost:8000/docs

### Frontend
```bash
cd app
streamlit run Home.py
```
Disponible en: http://localhost:8501

## 📁 Estructura del Proyecto
```
ProyectoB_ChurnClientes/
├── data/
│   ├── raw/download_data.py
│   └── processed/
├── notebooks/
│   ├── 01_EDA_Churn.ipynb
│   ├── 02_Preprocesamiento.ipynb
│   ├── 03_ANN_BinaryClass.ipynb
│   ├── 04_ANN_RiskScore.ipynb
│   └── 05_ROI_Analysis.ipynb
├── src/
│   ├── data_prep.py
│   ├── config.py
│   └── train/
├── models/
├── api/
└── app/
```

## 🧪 Modelos Implementados

### Modelo 1: Clasificación Binaria (Churn Prediction)
- **Objetivo**: Predecir si el cliente abandonará (Sí/No)
- **Métricas objetivo**: 
  - Recall alto (no perder clientes en riesgo)
  - Precision aceptable (evitar falsos positivos costosos)

### Modelo 2: Risk Scoring
- **Objetivo**: Calcular probabilidad de churn (0.0 - 1.0)
- **Output**: Score continuo para priorizar intervenciones

## 📈 Análisis de ROI
El proyecto incluye análisis económico:
- Costo de adquisición de cliente (CAC)
- Valor de tiempo de vida del cliente (CLV)
- Costo de retención vs costo de adquisición
- ROI esperado de estrategias de retención

## 🛠️ Tecnologías
- TensorFlow/Keras, Pandas, NumPy, Scikit-learn
- Imbalanced-learn (SMOTE para desbalanceo)
- FastAPI, Streamlit, Plotly

---
**CUC - Inteligencia Artificial Aplicada - 2025**
