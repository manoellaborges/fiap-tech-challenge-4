from fastapi import APIRouter
from pydantic import BaseModel
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import datetime, timedelta


router = APIRouter()

class StockSymbol(BaseModel):
    symbol: str

model = load_model('stock_prediction_model.keras')
scaler = joblib.load('scaler.pkl')

def fetch_historical_data(symbol: str):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=120)
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    return data['Close'].values

@router.post("/")
def predict(data: StockSymbol):
    try:
        historical_data = fetch_historical_data(data.symbol)

        input_data = scaler.transform(historical_data[-60:].reshape(-1, 1)).reshape(1, 60, 1)
        prediction = model.predict(input_data)
        unscaled_prediction = scaler.inverse_transform(prediction)[0][0]
        
        return {"predicted_price": float(unscaled_prediction)}
    except Exception as e:
        return {"error": str(e)}
    