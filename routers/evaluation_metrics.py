from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from core.lstm_model import results


router = APIRouter()

@router.get("/mae")
def calculate_mae():
    """
    Calcula o MAE com base nos valores reais e previstos salvos no modelo.
    """
    try:
        #Verifica se os resultados existem
        if not results["real_values"] or not results["predicted_values"]:
            raise ValueError("Os valores reais e previstos ainda não foram calculados. Treine o modelo primeiro.")
        
        #Converte os resultados em arrays numpy

        y_real = np.array(results["real_values"])
        y_pred = np.array(results["predicted_values"])

        mae = mean_absolute_error(y_real, y_pred)

        return {"mae": mae}
    
    except Exception as ex:
        raise HTTPException(status_code=400, detail=f"Erro ao calcular MAE: {str(ex)}")
    

@router.get("/rmse")
def calculate_rmse():
    try:
        if not results["real_values"] or not results["predicted_values"]:
            raise ValueError("Os valores reais e previstos ainda não foram calculados. Treine o modelo primeiro.")
        
        y_real = np.array(results["real_values"])
        y_pred = np.array(results["predicted_values"])

        rmse = mean_squared_error(y_real, y_pred, squared=False)

        return {"rmse": rmse}
    
    except Exception as ex:
        raise HTTPException(status_code=400, detail=f"Erro ao calcular RMSE: {str(ex)}")