"""
Script para descargar Telco Customer Churn Dataset
Requiere Kaggle API configurada

Configuración de Kaggle API:
1. Ir a https://www.kaggle.com/settings
2. Click en "Create New API Token"
3. Mover kaggle.json a ~/.kaggle/ (Linux/Mac) o %HOMEPATH%\.kaggle\ (Windows)
4. chmod 600 ~/.kaggle/kaggle.json (Linux/Mac)

Ejecutar desde raíz: python data/raw/download_data.py
"""

import os
import pandas as pd
import subprocess

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

# Dataset de Kaggle
KAGGLE_DATASET = "blastchar/telco-customer-churn"
KAGGLE_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

def check_kaggle_setup():
    """Verifica que Kaggle API esté configurada"""
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json):
        print("❌ Error: Kaggle API no configurada")
        print("\nPasos para configurar:")
        print("1. Ir a https://www.kaggle.com/settings")
        print("2. Click 'Create New API Token'")
        print("3. Mover kaggle.json a ~/.kaggle/")
        print("4. En Linux/Mac: chmod 600 ~/.kaggle/kaggle.json")
        return False
    return True

def download_from_kaggle():
    """Descarga dataset desde Kaggle"""
    print("Descargando dataset desde Kaggle...")
    
    try:
        # Descargar dataset
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", RAW_DIR],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error descargando: {result.stderr}")
            return False
        
        # Descomprimir
        import zipfile
        zip_path = os.path.join(RAW_DIR, "telco-customer-churn.zip")
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(RAW_DIR)
            os.remove(zip_path)
            print("✓ Dataset descargado y descomprimido")
        
        return True
        
    except FileNotFoundError:
        print("❌ Error: 'kaggle' command not found")
        print("Instalar con: pip install kaggle")
        return False

def process_churn_data():
    """Procesa el dataset de churn"""
    data_path = os.path.join(RAW_DIR, KAGGLE_FILE)
    
    if not os.path.exists(data_path):
        print(f"❌ Archivo no encontrado: {data_path}")
        return None
    
    print(f"\nCargando datos desde: {data_path}")
    df = pd.read_csv(data_path)
    
    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    print("\nColumnas del dataset:")
    print(df.columns.tolist())
    
    print("\nPrimeras filas:")
    print(df.head())
    
    print("\nInformación del dataset:")
    print(df.info())
    
    print("\nDistribución de Churn:")
    print(df['Churn'].value_counts())
    print(f"Tasa de churn: {(df['Churn'] == 'Yes').mean():.2%}")
    
    # Guardar versión procesada
    processed_path = os.path.join(PROCESSED_DIR, "telco_churn.csv")
    df.to_csv(processed_path, index=False)
    print(f"\n✓ Dataset guardado en: {processed_path}")
    
    return df

def main():
    """Función principal"""
    # Crear directorios
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    print("="*60)
    print("DESCARGA DE DATASET: Telco Customer Churn")
    print("="*60)
    
    # Verificar configuración
    if not check_kaggle_setup():
        print("\nAlternativa: Descargar manualmente desde:")
        print(f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}")
        print(f"Y colocar {KAGGLE_FILE} en {RAW_DIR}/")
        return
    
    # Descargar
    data_path = os.path.join(RAW_DIR, KAGGLE_FILE)
    if not os.path.exists(data_path):
        if not download_from_kaggle():
            return
    else:
        print(f"✓ Dataset ya existe: {data_path}")
    
    # Procesar
    print("\n" + "="*60)
    print("PROCESANDO DATASET")
    print("="*60)
    process_churn_data()
    
    print("\n" + "="*60)
    print("¡DESCARGA COMPLETADA!")
    print("="*60)
    print("\nSiguiente paso: notebooks/01_EDA_Churn.ipynb")

if __name__ == "__main__":
    main()
