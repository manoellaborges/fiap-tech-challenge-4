from fastapi import APIRouter
import yfinance as yf
import pandas as pd

router = APIRouter()

@router.get("/{symbol}")
async def fetch_data(symbol: str, start_date: str = "2020-01-01", end_date: str = "2023-01-01"):
    """
    Baixa os dados históricos de preços de ações usando yfinance.
    """
    try:
        # Baixar dados usando yfinance
        df = yf.download(symbol, start=start_date, end=end_date)
        if df.empty:
            return {"error": f"Nenhum dado encontrado para o símbolo '{symbol}' no intervalo especificado."}

        # Ajustar o índice para ser uma coluna
        df.reset_index(inplace=True)

        # Salvar os dados em um arquivo CSV com cabeçalhos apropriados
        csv_path = f"data_{symbol}.csv"
        df.to_csv(csv_path, index=False)  # Salva sem índices extras

        # Convertendo o DataFrame para uma lista de dicionários
        preview = df.head().to_dict(orient="records")

        return {"message": f"Dados salvos em {csv_path}.", "preview": preview}
    except Exception as e:
        return {"error": str(e)}
