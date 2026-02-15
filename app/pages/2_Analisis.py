import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

st.set_page_config(page_title="Dashboard de Análisis de Churn", layout="wide")

st.markdown("""
    <style>
    .stApp{
    background: linear-gradient(135deg, #ffc695 0.000%, #ffc195 3.030%, #ffbb94 6.061%, #ffb694 9.091%, #ffb093 12.121%, #f5aa92 15.152%, #eaa491 18.182%, #de9e8f 21.212%, #d2988d 24.242%, #c6928b 27.273%, #b98d89 30.303%, #ac8787 33.333%, #9f8184 36.364%, #927b81 39.394%, #84757e 42.424%, #776f7b 45.455%, #6a6978 48.485%, #5c6374 51.515%, #4f5e70 54.545%, #42586c 57.576%, #365268 60.606%, #2a4d64 63.636%, #1e485f 66.667%, #13425b 69.697%, #083d56 72.727%, #003851 75.758%, #00334c 78.788%, #002e47 81.818%, #002942 84.848%, #00253c 87.879%, #002037 90.909%, #001c31 93.939%, #00182c 96.970%, #001426 100.000%);
    }
            
div[role="tablist"] .st-ca {
    background: white;
    color: black;
    padding: 1rem;
}
             
    .st-cn:focus, .st-cn:hover{
        color: white;
    }
                
    </style>            
            
""", unsafe_allow_html=True)

def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

@st.cache_data
def load_and_preprocess():
    df = pd.read_csv(r"data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['Churn_num'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

df = load_and_preprocess()

st.title("Análisis Exploratorio de Datos")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Distribuciones Categóricas", 
    "Distribuciones Numéricas", 
    "Análisis de Churn", 
    "Correlaciones",
    "Asociaciones Avanzadas"
])

with tab1:
    st.header("Frecuencia de Variables Categóricas")
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'customerID' in categorical_cols:
        categorical_cols.remove('customerID')

    n_cols = 3
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows*4))
    axes = axes.flatten()

    for i, col in enumerate(categorical_cols):
        df[col].value_counts().plot(kind='bar', ax=axes[i], color='steelblue')
        axes[i].set_title(f'Frecuencia de {col}')
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', rotation=45)

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    st.header("Distribución de Variables Numéricas")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, col in enumerate(['TotalCharges', 'MonthlyCharges', 'tenure']):
        sns.histplot(df[col], bins=30, kde=True, ax=axes[i])
        axes[i].set_title(f'Distribución de {col}')

    plt.tight_layout()
    st.pyplot(fig)

with tab3:
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Pie chart")
        fig_pie, ax_pie = plt.subplots()
        churn_counts = df['Churn'].value_counts()
        labels = ['No Churn', 'Churn']
        ax_pie.pie(churn_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#F44336'])
        ax_pie.set_title('Proporción de clientes que abandonan')
        st.pyplot(fig_pie)

    with col_b:
        st.subheader("Boxplots")
        var_box = st.selectbox("Selecciona variable para Boxplot", ['tenure', 'MonthlyCharges', 'TotalCharges'])
        fig_box, ax_box = plt.subplots()
        sns.boxplot(x='Churn', y=var_box, data=df, ax=ax_box)
        ax_box.set_title(f'{var_box} según Churn')
        plt.xticks([0,1], ['No', 'Sí'])
        st.pyplot(fig_box)

    st.divider()
    st.subheader("Barras Apiladas")
    
    stacked_options = [c for c in categorical_cols if c != 'Churn']
    selected_col = st.selectbox("Selecciona categoría para ver proporción de Churn", stacked_options)
    
    fig_stack, ax_stack = plt.subplots(figsize=(10,6))
    cross = pd.crosstab(df[selected_col], df['Churn'], normalize='index') * 100
    if 'Yes' in cross.columns:
        cross.columns = ['No Churn', 'Churn'] if len(cross.columns) == 2 else cross.columns
    
    ax_s = cross.plot(kind='bar', stacked=True, ax=ax_stack, color=['#4CAF50','#F44336'])
    plt.title(f'Proporción de Churn por {selected_col}')
    plt.ylabel('Porcentaje')
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='upper right')
    for container in ax_s.containers:
        ax_s.bar_label(container, label_type='center', fmt='%.1f%%')
    st.pyplot(fig_stack)

with tab4:
    st.header("Matrices de Correlación")
    
    st.subheader("Matriz de Correlación (Spearman) - Variables codificadas")
    cols_importantes = [
        'Churn', 'tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen',
        'Partner', 'Dependents', 'InternetService', 'TechSupport',
        'OnlineSecurity', 'Contract', 'PaperlessBilling'
    ]
    df_corr = df[cols_importantes].copy()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in num_cols:
        df_corr[col] = pd.to_numeric(df_corr[col], errors='coerce')
    si_no = ['Churn', 'Partner', 'Dependents', 'TechSupport', 'OnlineSecurity', 'PaperlessBilling']
    for col in si_no:
        df_corr[col] = df_corr[col].map({'Yes': 1, 'No': 0})
    df_corr['SeniorCitizen'] = pd.to_numeric(df_corr['SeniorCitizen'], errors='coerce')
    df_corr['Contract'] = df_corr['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    internet_dummies = pd.get_dummies(df_corr['InternetService'], prefix='Internet', drop_first=True)
    df_corr = pd.concat([df_corr.drop('InternetService', axis=1), internet_dummies], axis=1)
    df_corr = df_corr.dropna()
    
    fig_corr1, ax_corr1 = plt.subplots(figsize=(14, 12))
    corr_spearman = df_corr.corr(method='spearman')
    sns.heatmap(corr_spearman, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1, square=True, ax=ax_corr1)
    st.pyplot(fig_corr1)

    st.subheader("Correlación entre variables numéricas")
    df_num_only = df[num_cols].apply(pd.to_numeric, errors='coerce').dropna()
    fig_corr2, ax_corr2 = plt.subplots(figsize=(6,5))
    sns.heatmap(df_num_only.corr(), annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1, ax=ax_corr2)
    st.pyplot(fig_corr2)

with tab5:
    st.header("Asociaciones")
    
    col_c, col_d = st.columns(2)
    
    with col_c:
        st.subheader('Correlación punto-biserial con Churn')
        df['Churn_num'] = df['Churn'].map({'Yes': 1, 'No': 0})
        df_pb = df[num_cols + ['Churn_num']].dropna()
        corr_churn_pb = df_pb[num_cols].apply(lambda x: x.corr(df_pb['Churn_num'])).sort_values()
        
        fig_pb, ax_pb = plt.subplots()
        corr_churn_pb.plot(kind='barh', color='teal', ax=ax_pb)
        plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
        st.pyplot(fig_pb)

    with col_d:
        st.subheader('Asociación entre variables categóricas (Cramer\'s V)')
        cat_cols_v = [c for c in df.select_dtypes(include=['object']).columns if c != 'customerID']
        n_v = len(cat_cols_v)
        cramers_matrix = np.zeros((n_v, n_v))
        for i, col1 in enumerate(cat_cols_v):
            for j, col2 in enumerate(cat_cols_v):
                if i == j: cramers_matrix[i, j] = 1.0
                else:
                    ct = pd.crosstab(df[col1], df[col2])
                    cramers_matrix[i, j] = cramers_v(ct)

        fig_cv, ax_cv = plt.subplots(figsize=(10, 8))
        sns.heatmap(cramers_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=cat_cols_v, yticklabels=cat_cols_v, ax=ax_cv)
        st.pyplot(fig_cv)

    st.divider()
    st.subheader("Correlación / Asociación con Churn")
    
    corr_dict = {}
    df_clean_final = df[num_cols + ['Churn_num']].dropna()
    for col in num_cols:
        corr_dict[col] = df_clean_final[col].corr(df_clean_final['Churn_num'])
    
    cat_cols_assoc = ['Partner', 'Dependents', 'InternetService', 'TechSupport', 'OnlineSecurity', 'Contract', 'PaperlessBilling', 'SeniorCitizen']
    for col in cat_cols_assoc:
        if col in df.columns:
            ct = pd.crosstab(df[col], df['Churn'])
            corr_dict[col] = cramers_v(ct)

    corr_series = pd.Series(corr_dict).sort_values()
    fig_final, ax_final = plt.subplots(figsize=(10,6))
    corr_series.plot(kind='barh', color='coral', ax=ax_final)
    plt.axvline(0, color='black', linestyle='--')
    st.pyplot(fig_final)