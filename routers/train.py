from fastapi import APIRouter, HTTPException
from models.lstm_model import train_lstm

router = APIRouter()

@router.post("/")
async def train_model(symbol: str = "AAPL", start_date: str = "2020-01-01", end_date: str = "2023-01-01"):
    """
    Treina o modelo LSTM e salva o arquivo.
    """
    try:
        model_path, mse = train_lstm(symbol, start_date, end_date)
        return {"message": "Modelo treinado com sucesso.", "model_path": model_path, "mse": mse}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao treinar o modelo: {str(e)}")