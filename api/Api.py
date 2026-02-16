
#==================================================
#ESTE ARCHIVO NO HA SIDO TERMINADO
#==================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import tensorflow as tf
import numpy as np
import os
import logging
from typing import List, Optional

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title="API de Predicción de Churn",
    description="API REST para predecir churn de clientes y calcular scores de riesgo",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rutas de los modelos
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL1_PATH = os.path.join(MODEL_DIR, 'model1.keras')
MODEL2_PATH = os.path.join(MODEL_DIR, 'model2.keras')

# Variables globales para los modelos
churn_model = None
risk_score_model = None


# Modelos Pydantic para validación de datos
class PredictionRequest(BaseModel):
    """Modelo de entrada para las predicciones"""
    features: List[float] = Field(
        ...,
        description="Lista de características del cliente",
        example=[0.5, 1.2, 0.8, 3.5, 0.1, 2.3, 1.7, 0.9, 4.2, 0.6]
    )

    @validator('features')
    def validate_features(cls, v):
        if not v:
            raise ValueError("La lista de features no puede estar vacía")
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Todas las features deben ser números")
        return v


class ChurnPredictionResponse(BaseModel):
    """Respuesta de predicción de churn"""
    prediction: int = Field(..., description="Predicción binaria: 0 (no churn) o 1 (churn)")
    probability: float = Field(..., description="Probabilidad de churn (0.0 a 1.0)")
    churn: bool = Field(..., description="Booleano indicando si habrá churn")
    mensaje: str = Field(..., description="Mensaje descriptivo del resultado")


class RiskScoreResponse(BaseModel):
    """Respuesta de predicción de risk score"""
    risk_score: float = Field(..., description="Score de riesgo (0.0 a 1.0)")
    risk_level: str = Field(..., description="Nivel de riesgo: bajo, medio o alto")
    mensaje: str = Field(..., description="Mensaje descriptivo del resultado")


class CombinedPredictionResponse(BaseModel):
    """Respuesta combinada de ambas predicciones"""
    churn: ChurnPredictionResponse
    risk_score: RiskScoreResponse


@app.on_event("startup")
async def startup_event():
    #Cargar modelos al iniciar la aplicación
    global churn_model, risk_score_model
    try:
        logger.info("Iniciando carga de modelos...")
    except Exception as e:
        logger.error(f"Error al cargar modelos: {str(e)}")


@app.get("/", tags=["General"])
async def root():
    """
    Endpoint raíz con información de la API
    """
    return {
        "nombre": "API de Predicción de Churn",
        "version": "1.0.0",
        "framework": "FastAPI",
        "documentacion": {
            "swagger": "/docs",
            "redoc": "/redoc"
        },
        "endpoints": {
            "/predict/churn": "POST - Predice si un cliente abandonará (churn)",
            "/predict/risk_score": "POST - Predice el score de riesgo del cliente",
            "/predict/both": "POST - Realiza ambas predicciones",
            "/health": "GET - Verifica el estado de la API"
        },
        "ejemplo_request": {
            "features": [0.5, 1.2, 0.8, 3.5, 0.1, 2.3, 1.7, 0.9, 4.2, 0.6]
        }
    }



@app.post(
    "/predict/churn",
    response_model=ChurnPredictionResponse,
    tags=["Predicciones"],
    summary="Predicción de Churn",
    description="Predice si un cliente abandonará el servicio (churn)"
)
async def predict_churn(request: PredictionRequest):
    """
    Predice si un cliente abandonará el servicio.

    - **features**: Lista de características del cliente

    Retorna la predicción binaria, la probabilidad y un mensaje descriptivo.
    """
    if churn_model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo de churn no disponible"
        )

    try:
        # Convertir features a numpy array
        input_data = np.array(request.features).reshape(1, -1)

        # Realizar predicción
        prediction_prob = float(churn_model.predict(input_data, verbose=0)[0][0])
        prediction_class = int(prediction_prob > 0.5)

        logger.info(f"Predicción de churn: {prediction_class} (prob: {prediction_prob:.4f})")

        return ChurnPredictionResponse(
            prediction=prediction_class,
            probability=prediction_prob,
            churn=bool(prediction_class),
            mensaje="Cliente con riesgo de churn" if prediction_class == 1
                   else "Cliente sin riesgo de churn"
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error en los datos de entrada: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error en predicción de churn: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la predicción: {str(e)}"
        )


@app.post(
    "/predict/risk_score",
    response_model=RiskScoreResponse,
    tags=["Predicciones"],
    summary="Predicción de Risk Score",
    description="Calcula el score de riesgo del cliente"
)
async def predict_risk_score(request: PredictionRequest):
    """
    Calcula el score de riesgo del cliente.

    - **features**: Lista de características del cliente

    Retorna el score de riesgo (0.0-1.0), el nivel de riesgo (bajo/medio/alto)
    y un mensaje descriptivo.
    """
    if risk_score_model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo de risk score no disponible"
        )

    try:
        # Convertir features a numpy array
        input_data = np.array(request.features).reshape(1, -1)

        # Realizar predicción
        risk_score = float(risk_score_model.predict(input_data, verbose=0)[0][0])

        # Determinar nivel de riesgo
        if risk_score < 0.33:
            risk_level = "bajo"
        elif risk_score < 0.66:
            risk_level = "medio"
        else:
            risk_level = "alto"

        logger.info(f"Predicción de risk score: {risk_score:.4f} ({risk_level})")

        return RiskScoreResponse(
            risk_score=risk_score,
            risk_level=risk_level,
            mensaje=f"Cliente con riesgo {risk_level}"
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error en los datos de entrada: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error en predicción de risk score: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la predicción: {str(e)}"
        )


@app.post(
    "/predict/both",
    response_model=CombinedPredictionResponse,
    tags=["Predicciones"],
    summary="Predicción Combinada",
    description="Realiza predicción de churn y risk score en una sola llamada"
)
async def predict_both(request: PredictionRequest):
    """
    Realiza ambas predicciones (churn y risk score) en una sola llamada.

    - **features**: Lista de características del cliente

    Retorna tanto la predicción de churn como el risk score.
    """
    if churn_model is None or risk_score_model is None:
        raise HTTPException(
            status_code=503,
            detail="Uno o más modelos no disponibles"
        )

    try:
        # Convertir features a numpy array
        input_data = np.array(request.features).reshape(1, -1)

        # Predicción de churn
        churn_prob = float(churn_model.predict(input_data, verbose=0)[0][0])
        churn_class = int(churn_prob > 0.5)

        # Predicción de risk score
        risk_score = float(risk_score_model.predict(input_data, verbose=0)[0][0])

        # Determinar nivel de riesgo
        if risk_score < 0.33:
            risk_level = "bajo"
        elif risk_score < 0.66:
            risk_level = "medio"
        else:
            risk_level = "alto"

        logger.info(f"Predicción combinada - Churn: {churn_class}, Risk: {risk_score:.4f}")

        return CombinedPredictionResponse(
            churn=ChurnPredictionResponse(
                prediction=churn_class,
                probability=churn_prob,
                churn=bool(churn_class),
                mensaje="Cliente con riesgo de churn" if churn_class == 1
                       else "Cliente sin riesgo de churn"
            ),
            risk_score=RiskScoreResponse(
                risk_score=risk_score,
                risk_level=risk_level,
                mensaje=f"Cliente con riesgo {risk_level}"
            )
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error en los datos de entrada: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error en predicción combinada: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la predicción: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)