import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

st.markdown("""
    <style>
    .stApp{
        background: linear-gradient(30deg, #8f5c3a 0.000%, #8f5c3a 2.941%, #93573a calc(2.941% + 1px), #93573a 5.882%, #96503a calc(5.882% + 1px), #96503a 8.824%, #994a3a calc(8.824% + 1px), #994a3a 11.765%, #9c433b calc(11.765% + 1px), #9c433b 14.706%, #9e3c3b calc(14.706% + 1px), #9e3c3b 17.647%, #a1343c calc(17.647% + 1px), #a1343c 20.588%, #a22d3d calc(20.588% + 1px), #a22d3d 23.529%, #a4263f calc(23.529% + 1px), #a4263f 26.471%, #a51f40 calc(26.471% + 1px), #a51f40 29.412%, #a51841 calc(29.412% + 1px), #a51841 32.353%, #a61243 calc(32.353% + 1px), #a61243 35.294%, #a60c45 calc(35.294% + 1px), #a60c45 38.235%, #a50747 calc(38.235% + 1px), #a50747 41.176%, #a40249 calc(41.176% + 1px), #a40249 44.118%, #a3004b calc(44.118% + 1px), #a3004b 47.059%, #a1004e calc(47.059% + 1px), #a1004e 50.000%, #9f0050 calc(50.000% + 1px), #9f0050 52.941%, #9d0053 calc(52.941% + 1px), #9d0053 55.882%, #9a0055 calc(55.882% + 1px), #9a0055 58.824%, #970058 calc(58.824% + 1px), #970058 61.765%, #94005b calc(61.765% + 1px), #94005b 64.706%, #91005d calc(64.706% + 1px), #91005d 67.647%, #8d0060 calc(67.647% + 1px), #8d0060 70.588%, #890063 calc(70.588% + 1px), #890063 73.529%, #850266 calc(73.529% + 1px), #850266 76.471%, #810768 calc(76.471% + 1px), #810768 79.412%, #7d0c6b calc(79.412% + 1px), #7d0c6b 82.353%, #79126e calc(82.353% + 1px), #79126e 85.294%, #751870 calc(85.294% + 1px), #751870 88.235%, #721f72 calc(88.235% + 1px), #721f72 91.176%, #6e2675 calc(91.176% + 1px), #6e2675 94.118%, #6a2d77 calc(94.118% + 1px), #6a2d77 97.059%, #673479 calc(97.059% + 1px) 100.000%);
    }

    </style>            
            
""", unsafe_allow_html=True)


@st.cache_resource
def cargar_modelos_y_pipe():
    mod_bin = load_model(r'models\model1.keras')
    mod_prob = load_model(r'models\model2.keras')
    pipe = joblib.load(r'models\Pipe_RiskScore.pkl')
    return mod_bin, mod_prob, pipe


modelo_binario, modelo_proba, preproc_proba = cargar_modelos_y_pipe()

st.header("Predicción Individual de Churn")
st.markdown("Ingrese los datos del cliente para evaluar su riesgo de abandono.")

tipo_modelo = st.radio(
    "Seleccione el tipo de modelo:",
    ["Binario (Clasificación)", "Probabilístico (Score de Riesgo)"],
    horizontal=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Variables Numéricas")
    tenure = st.number_input("Tenure (Meses)", min_value=0, max_value=100, value=1)
    total_charges = st.number_input("Cargos Totales", min_value=0.0, value=50.0)

with col2:
    st.subheader("Servicios y Binarias")
    gender = st.selectbox("Género", ["Female", "Male"])
    phone_service = st.selectbox("PhoneService", ["No", "Yes"])
    online_security = st.selectbox("OnlineSecurity", ["No", "Yes"])
    online_backup = st.selectbox("OnlineBackup", ["No", "Yes"])

with col3:
    st.subheader("Contrato y Pagos")
    device_protection = st.selectbox("DeviceProtection", ["No", "Yes"])
    tech_support = st.selectbox("TechSupport", ["No", "Yes"])
    streaming_tv = st.selectbox("StreamingTV", ["No", "Yes"])
    streaming_movies = st.selectbox("StreamingMovies", ["No", "Yes"])
    multiple_lines = st.selectbox("MultipleLines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("PaymentMethod", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

if st.button("Realizar Predicción"):
    input_data = pd.DataFrame({
        'gender': [gender],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaymentMethod': [payment_method],
        'TotalCharges': [total_charges]
    })

    try:
        datos_preprocesados = preproc_proba.transform(input_data)

        if tipo_modelo == "Binario (Clasificación)":
            pred = modelo_binario.predict(datos_preprocesados)
            clase = (pred > 0.5).astype(int)[0][0]
            resultado = "CHURN (Abandona)" if clase == 1 else "NO CHURN (Se queda)"
            color = "red" if clase == 1 else "green"
            st.markdown(f"### Resultado: :{color}[{resultado}]")
            
        else:
            prob = modelo_proba.predict(datos_preprocesados)[0][0]
            st.markdown(f"### Score de Riesgo: {prob:.2%}")
            
            st.progress(float(prob))
            
            if prob > 0.7:
                st.error("Riesgo Crítico: Se recomienda acción inmediata.")
            elif prob > 0.4:
                st.warning("Riesgo Moderado: Realizar seguimiento.")
            else:
                st.success("Riesgo Bajo: Cliente estable.")

    except Exception as e:
        st.error(f"Error en el preprocesamiento: {e}")
        st.info("Asegúrate de que las columnas del DataFrame coincidan con las que espera el Pipe_RiskScore.pkl")