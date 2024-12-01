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


@router.post("/")
def predict(data: StockSymbol):
    try:
        # Verificar se dados personalizados foram fornecidos
        if data.custom_data:
            historical_data = np.array(data.custom_data)

            # Validação básica
            if len(historical_data) < 60:
                return {"error": "Insufficient data. At least 60 data points are required."}
            if not all(isinstance(x, (int, float)) for x in historical_data):
                return {"error": "All data points must be numbers."}

            # Verificação de outliers (exemplo simples)
            if np.any(historical_data > np.mean(historical_data) + 3 * np.std(historical_data)):
                return {"warning": "Possible outliers detected. Please verify your data."}

            # Dados aprovados para uso
        else:
            # Buscar dados históricos automaticamente
            historical_data = fetch_historical_data(data.symbol)

        input_data = scaler.transform(historical_data[-60:].reshape(-1, 1)).reshape(1, 60, 1)
        prediction = model.predict(input_data)
        unscaled_prediction = scaler.inverse_transform(prediction)[0][0]
        
        return {"predicted_price": float(unscaled_prediction)}
    except Exception as e:
        return {"error": str(e)}
    