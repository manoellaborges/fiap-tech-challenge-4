from fastapi import APIRouter
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from utils.extract import fetch_historical_data


router = APIRouter()

class StockSymbol(BaseModel):
    symbol: str
    custom_data: list = None

model = load_model('stock_prediction_model.keras')
scaler = joblib.load('scaler.pkl')


@router.post("")
@router.post("/")
def predict(data: StockSymbol):
    if not data.symbol:
        return {"erro": "O campo symbol deve ser informado."}
    elif data.symbol != "AAPL":
        return {"aviso": "No momento, suporte apenas para o símbolo AAPL."}
    
    try:
        if data.custom_data:
            historical_data = np.array(data.custom_data)

            if len(historical_data) < 60:
                return {"erro": "Dados insuficientes. São necessários pelo menos 60 pontos de dados."}
            if not all(isinstance(x, (int, float)) for x in historical_data):
                return {"erro": "Todos os pontos de dados devem ser numéricos."}

            if np.any(historical_data > np.mean(historical_data) + 3 * np.std(historical_data)):
                return {"aviso": "Possíveis outliers detectados. Por favor, verifique seus dados."}

        else:
            historical_data = fetch_historical_data(data.symbol)

        input_data = scaler.transform(historical_data[-60:].reshape(-1, 1)).reshape(1, 60, 1)
        prediction = model.predict(input_data)
        unscaled_prediction = scaler.inverse_transform(prediction)[0][0]
        
        return {"predicted_price": float(f"{unscaled_prediction:.2f}")}
    except Exception as e:
        return {"error": str(e)}
    