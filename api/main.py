from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import config
import predict
from schemas import CustomerData, ChurnPredictionResponse, RiskScoreResponse, CombinedPredictionResponse

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Crear interfaz
app = FastAPI(
    title="API de Predicción de Churn",
    description="""
    ## API REST para predecir churn de clientes de telecomunicaciones

    **Envía datos tal como aparecen en el CSV original** - la API los transforma automáticamente.

    ### Modelos:
    - **Modelo 1**: Clasificación binaria de Churn (Yes/No)
    - **Modelo 2**: Score de riesgo con niveles (Bajo/Medio/Alto)

    ### Endpoints disponibles:
    - `POST /predict/churn` - Predice si habrá churn
    - `POST /predict/risk_score` - Calcula nivel de riesgo
    - `POST /predict/both` - Ambas predicciones en una llamada
    - `GET /fields/info` - Ver descripción de todos los campos
    - `GET /fields/example` - Ver ejemplo de request válido
    """
)


# Carga de modelos al iniciar
@app.on_event("startup")
async def startup_event():
    predict.load_models()


# ENDPOINTS con fines de presentacion

@app.get("/", tags=["Información"])
async def root():
    """
    Información general de la API
    """
    return {
        "api": "API de Predicción de Churn",
        "descripcion": "Envía datos en formato CSV original - la API hace la transformación",
        "endpoints": {
            "POST /predict/churn": "Predice si habrá churn (Yes/No)",
            "POST /predict/risk_score": "Calcula nivel de riesgo (Bajo/Medio/Alto)",
            "POST /predict/both": "Ambas predicciones combinadas",
            "GET /fields/info": "Ver descripción de todos los campos",
            "GET /fields/example": "Ver ejemplo de request válido"
        },
        "documentacion": {
            "swagger": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/fields/info", tags=["Información"])
async def fields_info():
    """
    Muestra información detallada de todos los campos
    """
    return {
        "total_campos": len(config.FIELD_DESCRIPTIONS),
        "campos": config.FIELD_DESCRIPTIONS
    }


@app.get("/fields/example", tags=["Información"])
async def fields_example():
    """
    Muestra un ejemplo de request válido
    """
    return {
        "descripcion": "Ejemplo de datos válidos para enviar a la API",
        "ejemplo": config.EXAMPLE_REQUEST,
        "nota": "Estos datos corresponden a una fila del CSV original"
    }


# ENDPOINTS DE PREDICCION

@app.post(
    "/predict/churn",
    response_model=ChurnPredictionResponse,
    tags=["Predicciones"],
    summary="Predecir Churn (Modelo 1)"
)
async def predict_churn(customer: CustomerData):
    try:
        # Convertir Pydantic model a dict
        customer_dict = customer.dict()

        # Ejecutar predicción
        return predict.get_churn_prediction(customer_dict)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error en /predict/churn: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@app.post(
    "/predict/risk_score",
    response_model=RiskScoreResponse,
    tags=["Predicciones"],
    summary="Calcular Score de Riesgo (Modelo 2)"
)
async def predict_risk_score(customer: CustomerData):
    try:
        customer_dict = customer.dict()
        return predict.get_risk_score_prediction(customer_dict)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error en /predict/risk_score: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@app.post(
    "/predict/both",
    response_model=CombinedPredictionResponse,
    tags=["Predicciones"],
    summary="Predicción Completa (Ambos Modelos)"
)
async def predict_both(customer: CustomerData):
    try:
        customer_dict = customer.dict()
        return predict.get_combined_prediction(customer_dict)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error en /predict/both: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

# EJECUTAR SERVIDOR

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )