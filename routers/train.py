from fastapi import APIRouter, HTTPException
import yfinance as yf
import pandas as pd
from core.lstm_model import train_lstm

router = APIRouter()

@router.post("/{symbol}")
@router.post("/{symbol}/")
async def train_model(symbol: str, start_date: str = "2020-01-01", end_date: str = "2023-01-01"):
    """
    Treina o modelo LSTM e salva o arquivo.
    """
    
    if symbol != "AAPL":
        return {"erro": "No momento, suporte apenas para o símbolo AAPL."}

    try:
        train_lstm(symbol, start_date, end_date)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao treinar o modelo: {str(e)}")
    
    return {"message": "Modelo treinado com sucesso."}
