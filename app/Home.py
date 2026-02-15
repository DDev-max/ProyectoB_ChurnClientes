import streamlit as st

st.set_page_config(
    page_title="Churn Analytics Hub",
    layout="wide"
)

st.markdown("""
    <style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .main-container {
        animation: fadeIn 1.2s ease-out;
    }

    .nav-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        border-bottom-left-radius: 0;
        border-bottom-right-radius: 0;
        color: white;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        cursor: pointer;
        height: 250px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    .nav-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }

    .nav-card h3 {
        color: white !important;
        margin-bottom: 10px;
    }
            
    .spanTitle{
        background: black;
        padding: 0.25rem;
        border-radius: 1rem;
        filter: invert(1);     
    }
            
    a{    
        text-decoration: none;
    }
            
    a p{
            color: black;
    }
            
    .stApp{
background: linear-gradient(0deg, #07001b 0.000%, #07001b 2.941%, #09001a calc(2.941% + 1px), #09001a 5.882%, #0c0119 calc(5.882% + 1px), #0c0119 8.824%, #0e0218 calc(8.824% + 1px), #0e0218 11.765%, #110318 calc(11.765% + 1px), #110318 14.706%, #140417 calc(14.706% + 1px), #140417 17.647%, #170517 calc(17.647% + 1px), #170517 20.588%, #1a0616 calc(20.588% + 1px), #1a0616 23.529%, #1c0816 calc(23.529% + 1px), #1c0816 26.471%, #1f0916 calc(26.471% + 1px), #1f0916 29.412%, #220b16 calc(29.412% + 1px), #220b16 32.353%, #250d17 calc(32.353% + 1px), #250d17 35.294%, #280e17 calc(35.294% + 1px), #280e17 38.235%, #2b1018 calc(38.235% + 1px), #2b1018 41.176%, #2e1219 calc(41.176% + 1px), #2e1219 44.118%, #31141a calc(44.118% + 1px), #31141a 47.059%, #34161b calc(47.059% + 1px), #34161b 50.000%, #37191c calc(50.000% + 1px), #37191c 52.941%, #3a1b1e calc(52.941% + 1px), #3a1b1e 55.882%, #3d1d1f calc(55.882% + 1px), #3d1d1f 58.824%, #402021 calc(58.824% + 1px), #402021 61.765%, #432323 calc(61.765% + 1px), #432323 64.706%, #452525 calc(64.706% + 1px), #452525 67.647%, #482827 calc(67.647% + 1px), #482827 70.588%, #4b2b29 calc(70.588% + 1px), #4b2b29 73.529%, #4d2e2c calc(73.529% + 1px), #4d2e2c 76.471%, #50312e calc(76.471% + 1px), #50312e 79.412%, #523431 calc(79.412% + 1px), #523431 82.353%, #543734 calc(82.353% + 1px), #543734 85.294%, #573a37 calc(85.294% + 1px), #573a37 88.235%, #593e3a calc(88.235% + 1px), #593e3a 91.176%, #5b413e calc(91.176% + 1px), #5b413e 94.118%, #5c4541 calc(94.118% + 1px), #5c4541 97.059%, #5e4845 calc(97.059% + 1px) 100.000%);    }
       
    </style>
""", unsafe_allow_html=True)


with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    st.title("Predicción de Abandono de Clientes")
    st.markdown("Sistema que predice qué clientes tienen mayor probabilidad de abandonar un servicio de telecomunicaciones.")

    st.write("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <a href="/Prediccion" target="_self" class="nav-card">
                <h3><span class="spanTitle">Predicción</span></h3>
                <p>Clasificacion binaria y probabilistica de abandono</p>
            </a>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <a class="nav-card"  href="/Analisis" target="_self"  style="background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 99%, #fecfef 100%);">
                <h3 style="color: #4a4a4a !important;"><span class="spanTitle">Análisis</span></h3>
                <p style="color: #4a4a4a;">Analisis exploratorio del dataset Telco</p>
            </a>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <a href="/Metricas" target="_self" class="nav-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                <h3 style="color: #4a4a4a !important;"><span class="spanTitle">Métricas</span></h3>
                <p style="color: #4a4a4a;">Precision, recall, accuracy ...</p>
            </a>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.write("---")