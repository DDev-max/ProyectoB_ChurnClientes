import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

st.markdown("""
    <style>
    .stApp{
background: linear-gradient(225deg, #273d51 0.000%, #263e57 3.030%, #24405c 6.061%, #234263 9.091%, #224469 12.121%, #21476f 15.152%, #214a76 18.182%, #214e7d 21.212%, #215284 24.242%, #21568b 27.273%, #215b91 30.303%, #226098 33.333%, #23659e 36.364%, #246aa5 39.394%, #266fab 42.424%, #2875b0 45.455%, #2a7bb6 48.485%, #2c80bb 51.515%, #2e86bf 54.545%, #318cc3 57.576%, #3392c7 60.606%, #3697ca 63.636%, #399dcd 66.667%, #3ca2cf 69.697%, #3fa7d0 72.727%, #42acd1 75.758%, #45b1d1 78.788%, #48b5d1 81.818%, #4bbad0 84.848%, #4ebdcf 87.879%, #51c1cd 90.909%, #54c4ca 93.939%, #56c7c7 96.970%, #59c9c4 100.000%);
    }
            

    </style>            
            
""", unsafe_allow_html=True)


@st.cache_resource
def cargar_recursos():
    df_test = pd.read_csv(r'data\processed\train-Modelo-2.csv')
    X_test = df_test.drop(columns='Churn')
    y_test = df_test['Churn']
    modelo = load_model(r'models\model2.keras')
    return X_test, y_test, modelo

X_test, y_test, modelo_cargado = cargar_recursos()

st.header("Evaluación del Modelo y Ajuste de Umbral")
st.markdown("""
Ajusta el **umbral de decisión** para observar cómo cambian la Precisión, el Recall y la Matriz de Confusión. 
Esto es crítico para balancear los falsos positivos frente a los falsos negativos.
""")

umbral = st.slider("Selecciona el Umbral de Decisión (Threshold)", 0.0, 1.0, 0.5, 0.05)

probs = modelo_cargado.predict(X_test).flatten()
y_pred = (probs >= umbral).astype(int)

report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred)

col1, col2, col3, col4 = st.columns(4)

precision = report['1']['precision']
recall = report['1']['recall']
f1_score = report['1']['f1-score']
accuracy = report['accuracy']

col1.metric("Accuracy", f"{accuracy:.2%}")
col2.metric("Precision (Churn)", f"{precision:.2%}")
col3.metric("Recall (Churn)", f"{recall:.2%}")
col4.metric("F1-Score (Churn)", f"{f1_score:.2%}")

st.divider()

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Matriz de Confusión")
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'], 
                yticklabels=['No Churn', 'Churn'], ax=ax_cm)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    st.pyplot(fig_cm)

with col_right:
    st.subheader("Reporte de Clasificación Completo")
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report.style.format(precision=3).background_gradient(cmap='Greens', subset=['precision', 'recall', 'f1-score']))
