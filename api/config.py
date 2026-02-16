"""
Configuración para convertir datos del CSV original a features numéricas

Archivo en su mayoria generado por IA para una mejor visualizacion y hacer mas intuitiva la comunicacion con el API que nos permite recibir
hacer un get y ver caracteristica de la variable

{
  "features": [0,1,0,0,12,1,0,1,0,0,1,...]
}

"""

# =====================================================
# CAMPOS DEL DATASET ORIGINAL
# =====================================================

FIELD_DESCRIPTIONS = {
    'gender': {
        'descripcion': 'Género del cliente',
        'tipo': 'categórico',
        'valores_validos': ['Male', 'Female'],
        'ejemplo': 'Female'
    },
    'SeniorCitizen': {
        'descripcion': 'Si el cliente es adulto mayor (65+ años)',
        'tipo': 'numérico',
        'valores_validos': [0, 1],
        'ejemplo': 0
    },
    'Partner': {
        'descripcion': 'Si el cliente tiene pareja',
        'tipo': 'categórico',
        'valores_validos': ['Yes', 'No'],
        'ejemplo': 'Yes'
    },
    'Dependents': {
        'descripcion': 'Si el cliente tiene dependientes (hijos, familia)',
        'tipo': 'categórico',
        'valores_validos': ['Yes', 'No'],
        'ejemplo': 'No'
    },
    'tenure': {
        'descripcion': 'Número de meses con la compañía',
        'tipo': 'numérico',
        'rango': '0 - 72 meses',
        'ejemplo': 12
    },
    'PhoneService': {
        'descripcion': 'Si tiene servicio telefónico',
        'tipo': 'categórico',
        'valores_validos': ['Yes', 'No'],
        'ejemplo': 'Yes'
    },
    'MultipleLines': {
        'descripcion': 'Si tiene múltiples líneas telefónicas',
        'tipo': 'categórico',
        'valores_validos': ['Yes', 'No', 'No phone service'],
        'ejemplo': 'No'
    },
    'InternetService': {
        'descripcion': 'Tipo de servicio de internet',
        'tipo': 'categórico',
        'valores_validos': ['DSL', 'Fiber optic', 'No'],
        'ejemplo': 'Fiber optic'
    },
    'OnlineSecurity': {
        'descripcion': 'Servicio de seguridad online',
        'tipo': 'categórico',
        'valores_validos': ['Yes', 'No', 'No internet service'],
        'ejemplo': 'No'
    },
    'OnlineBackup': {
        'descripcion': 'Servicio de respaldo online',
        'tipo': 'categórico',
        'valores_validos': ['Yes', 'No', 'No internet service'],
        'ejemplo': 'Yes'
    },
    'DeviceProtection': {
        'descripcion': 'Protección de dispositivos',
        'tipo': 'categórico',
        'valores_validos': ['Yes', 'No', 'No internet service'],
        'ejemplo': 'No'
    },
    'TechSupport': {
        'descripcion': 'Soporte técnico',
        'tipo': 'categórico',
        'valores_validos': ['Yes', 'No', 'No internet service'],
        'ejemplo': 'No'
    },
    'StreamingTV': {
        'descripcion': 'Servicio de streaming de TV',
        'tipo': 'categórico',
        'valores_validos': ['Yes', 'No', 'No internet service'],
        'ejemplo': 'Yes'
    },
    'StreamingMovies': {
        'descripcion': 'Servicio de streaming de películas',
        'tipo': 'categórico',
        'valores_validos': ['Yes', 'No', 'No internet service'],
        'ejemplo': 'Yes'
    },
    'Contract': {
        'descripcion': 'Tipo de contrato',
        'tipo': 'categórico',
        'valores_validos': ['Month-to-month', 'One year', 'Two year'],
        'ejemplo': 'Month-to-month'
    },
    'PaperlessBilling': {
        'descripcion': 'Si usa facturación electrónica',
        'tipo': 'categórico',
        'valores_validos': ['Yes', 'No'],
        'ejemplo': 'Yes'
    },
    'PaymentMethod': {
        'descripcion': 'Método de pago',
        'tipo': 'categórico',
        'valores_validos': [
            'Electronic check',
            'Mailed check',
            'Bank transfer (automatic)',
            'Credit card (automatic)'
        ],
        'ejemplo': 'Electronic check'
    },
    'MonthlyCharges': {
        'descripcion': 'Cargo mensual en dólares',
        'tipo': 'numérico',
        'rango': '$18.25 - $118.75',
        'ejemplo': 70.70
    },
    'TotalCharges': {
        'descripcion': 'Total de cargos acumulados',
        'tipo': 'numérico',
        'rango': '$18.80 - $8684.80',
        'ejemplo': 1500.50
    },
}

# Ejemplo de request válido (datos legibles del CSV)
EXAMPLE_REQUEST = {
    "gender": "Female",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
    "TotalCharges": 151.65
}

# Estadísticas para normalización (StandardScaler)
# Calculadas del dataset completo antes del split
SCALER_STATS = {
    'tenure': {'mean': 32.4218, 'std': 24.5453},
    'TotalCharges': {'mean': 2283.3004, 'std': 2266.7714}
}


def normalize_numeric(value: float, mean: float, std: float) -> float:
    """Normaliza un valor usando StandardScaler: (x - mean) / std"""
    return (value - mean) / std


def transform_to_features(data: dict) -> list:
    """
    Transforma los datos del CSV original a un array de features numéricas
    para alimentar al modelo.

    NOTA: El orden de las features debe coincidir EXACTAMENTE con el orden
    en que fueron entrenados los modelos (columnas de X_train).

    Esta función asume que el preprocesamiento incluyó one-hot encoding
    con pd.get_dummies(df, drop_first=True)

    Args:
        data: Diccionario con los datos en formato original del CSV

    Returns:
        Lista de valores numéricos para el modelo
    """
    features = []

    features.append(normalize_numeric(
        float(data['tenure']),
        SCALER_STATS['tenure']['mean'],
        SCALER_STATS['tenure']['std']
    ))
    features.append(normalize_numeric(
        float(data['TotalCharges']),
        SCALER_STATS['TotalCharges']['mean'],
        SCALER_STATS['TotalCharges']['std']
    ))

    features.append(1 if data['gender'] == 'Male' else 0)

    features.append(1 if data['PhoneService'] == 'Yes' else 0)

    features.append(1 if data['MultipleLines'] == 'No' else 0)
    features.append(1 if data['MultipleLines'] == 'No phone service' else 0)
    features.append(1 if data['MultipleLines'] == 'Yes' else 0)

    features.append(1 if data['InternetService'] == 'DSL' else 0)
    features.append(1 if data['InternetService'] == 'Fiber optic' else 0)
    features.append(1 if data['InternetService'] == 'No' else 0)

    features.append(1 if data['OnlineSecurity'] == 'No' else 0)
    features.append(1 if data['OnlineSecurity'] == 'No internet service' else 0)
    features.append(1 if data['OnlineSecurity'] == 'Yes' else 0)

    features.append(1 if data['OnlineBackup'] == 'No' else 0)
    features.append(1 if data['OnlineBackup'] == 'No internet service' else 0)
    features.append(1 if data['OnlineBackup'] == 'Yes' else 0)

    features.append(1 if data['DeviceProtection'] == 'No' else 0)
    features.append(1 if data['DeviceProtection'] == 'No internet service' else 0)
    features.append(1 if data['DeviceProtection'] == 'Yes' else 0)

    features.append(1 if data['TechSupport'] == 'No' else 0)
    features.append(1 if data['TechSupport'] == 'No internet service' else 0)
    features.append(1 if data['TechSupport'] == 'Yes' else 0)

    features.append(1 if data['StreamingTV'] == 'No' else 0)
    features.append(1 if data['StreamingTV'] == 'No internet service' else 0)
    features.append(1 if data['StreamingTV'] == 'Yes' else 0)

    features.append(1 if data['StreamingMovies'] == 'No' else 0)
    features.append(1 if data['StreamingMovies'] == 'No internet service' else 0)
    features.append(1 if data['StreamingMovies'] == 'Yes' else 0)

    features.append(1 if data['Contract'] == 'Month-to-month' else 0)
    features.append(1 if data['Contract'] == 'One year' else 0)
    features.append(1 if data['Contract'] == 'Two year' else 0)

    features.append(1 if data['PaymentMethod'] == 'Bank transfer (automatic)' else 0)
    features.append(1 if data['PaymentMethod'] == 'Credit card (automatic)' else 0)
    features.append(1 if data['PaymentMethod'] == 'Electronic check' else 0)
    features.append(1 if data['PaymentMethod'] == 'Mailed check' else 0)

    return features