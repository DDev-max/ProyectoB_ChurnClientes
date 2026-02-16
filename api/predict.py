import tensorflow as tf
import numpy as np
import os
import logging
import config
from schemas import ChurnPredictionResponse, RiskScoreResponse, CombinedPredictionResponse

logger = logging.getLogger(__name__)

# Rutas de los modelos
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL1_PATH = os.path.join(MODEL_DIR, 'model1.keras')
MODEL2_PATH = os.path.join(MODEL_DIR, 'model2.keras')

# Variables globales
churn_model = None  # Modelo 1: Clasificación de Churn (Yes/No)
risk_score_model = None  # Modelo 2: Score de riesgo (Bajo/Medio/Alto)


def load_models():
    """
    Carga los modelos de ML al iniciar la aplicación
    """
    global churn_model, risk_score_model
#Valida si todos los mdelos fueron cargados correctamente
    try:
        logger.info("Iniciando carga de modelos...")

        # Cargar Modelo 1
        if os.path.exists(MODEL1_PATH):
            churn_model = tf.keras.models.load_model(MODEL1_PATH)
            logger.info(f"Modelo 1 (Churn) cargado exitosamente")
            logger.info(f"Input shape esperado: {churn_model.input_shape}")
        else:
            logger.warning(f"Modelo 1 no encontrado en: {MODEL1_PATH}")

        # Cargar Modelo 2
        if os.path.exists(MODEL2_PATH):
            risk_score_model = tf.keras.models.load_model(MODEL2_PATH)
            logger.info(f"Modelo 2 (Risk Score) cargado exitosamente")
            logger.info(f"Input shape esperado: {risk_score_model.input_shape}")
        else:
            logger.warning(f"Modelo 2 no encontrado en: {MODEL2_PATH}")

        if churn_model and risk_score_model:
            logger.info("Todos los modelos cargados correctamente")
        else:
            logger.warning("Algunos modelos no están disponibles")

    except Exception as e:
        logger.error(f"Error al cargar modelos: {str(e)}")
        raise


def get_churn_prediction(customer_data: dict) -> ChurnPredictionResponse:
    """
    Predicion churn
    """
    if churn_model is None:
        raise ValueError("Modelo 1 (Churn) no está disponible")

    try:
        # Transformar datos del CSV a features numéricas
        features = config.transform_to_features(customer_data)

        # Convertir a numpy array con shape correcto
        input_data = np.array(features).reshape(1, -1)

        logger.info(f"Input shape: {input_data.shape}, Features: {len(features)}")

        # Predicción con el modelo
        probability = float(churn_model.predict(input_data, verbose=0)[0][0])

        # Clasificación binaria (threshold 0.5)
        prediction_class = int(probability > 0.5)
        churn_label = "Yes" if prediction_class == 1 else "No"

        #mostrar resultados redondeados en 4 decimales
        logger.info(f"Modelo 1 - Churn: {churn_label} (prob: {probability:.4f})")

        return ChurnPredictionResponse(
            prediction=prediction_class,
            probability=round(probability, 4),
            churn=churn_label,
            mensaje=f"Cliente {'con riesgo de' if churn_label == 'Yes' else 'sin riesgo de'} churn"
        )

    except Exception as e:
        logger.error(f"Error en predicción de churn: {str(e)}")
        raise


def get_risk_score_prediction(customer_data: dict) -> RiskScoreResponse:
    """
    Predicion el nivel de riesgo
    """
    if risk_score_model is None:
        raise ValueError("Modelo 2 (Risk Score) no está disponible")

    try:
        # Transformar datos del CSV a features numéricas
        features = config.transform_to_features(customer_data)

        # Convertir a numpy array
        input_data = np.array(features).reshape(1, -1)

        # Predicción de probabilidad
        probabilidad_churn = float(risk_score_model.predict(input_data, verbose=0)[0][0])

        # Determinar nivel de riesgo según bins
        if probabilidad_churn < 0.33:
            nivel_riesgo = "Bajo"
        elif probabilidad_churn < 0.66:
            nivel_riesgo = "Medio"
        else:
            nivel_riesgo = "Alto"

        logger.info(f"Modelo 2 - Risk: {nivel_riesgo} (prob: {probabilidad_churn:.4f})")

        return RiskScoreResponse(
            probabilidad_churn=round(probabilidad_churn, 4),
            nivel_riesgo=nivel_riesgo,
            mensaje=f"Cliente con riesgo {nivel_riesgo.lower()} de churn"
        )

    except Exception as e:
        logger.error(f"Error en predicción de risk score: {str(e)}")
        raise


def get_combined_prediction(customer_data: dict) -> CombinedPredictionResponse:
    """
    Prediccion con los 2 modelos
    """
    if churn_model is None or risk_score_model is None:
        raise ValueError("Uno o más modelos no están disponibles")

    try:
        # Transformar datos UNA SOLA VEZ
        features = config.transform_to_features(customer_data)
        input_data = np.array(features).reshape(1, -1)

        logger.info(f"📊 Ejecutando ambos modelos...")

        # ===== MODELO 1: Predicción de Churn =====
        churn_prob = float(churn_model.predict(input_data, verbose=0)[0][0])
        churn_class = int(churn_prob > 0.5)
        churn_label = "Yes" if churn_class == 1 else "No"

        # ===== MODELO 2: Score de Riesgo =====
        risk_prob = float(risk_score_model.predict(input_data, verbose=0)[0][0])

        if risk_prob < 0.33:
            nivel_riesgo = "Bajo"
        elif risk_prob < 0.66:
            nivel_riesgo = "Medio"
        else:
            nivel_riesgo = "Alto"

        logger.info(
            f"Combinado - Churn: {churn_label} ({churn_prob:.4f}) | "
            f"Risk: {nivel_riesgo} ({risk_prob:.4f})"
        )

        return CombinedPredictionResponse(
            churn=ChurnPredictionResponse(
                prediction=churn_class,
                probability=round(churn_prob, 4),
                churn=churn_label,
                mensaje=f"Cliente {'con riesgo de' if churn_label == 'Yes' else 'sin riesgo de'} churn"
            ),
            risk_score=RiskScoreResponse(
                probabilidad_churn=round(risk_prob, 4),
                nivel_riesgo=nivel_riesgo,
                mensaje=f"Cliente con riesgo {nivel_riesgo.lower()} de churn"
            )
        )

    except Exception as e:
        logger.error(f"Error en predicción combinada: {str(e)}")
        raise