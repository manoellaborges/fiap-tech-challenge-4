from fastapi import FastAPI
from routers import predict, train, evaluation_metrics
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


app = FastAPI(title="Previsão de Preços de Ações com Modelo LSTM", 
              description="""Bem-vindo à documentação da API de Previsão de Preços de Ações, 
              desenvolvida como parte de um desafio tecnológico da faculdade FIAP. 
              Esta solução utiliza redes neurais recorrentes LSTM (Long Short-Term Memory) para prever preços futuros de ações, 
              demonstrando o potencial das tecnologias de aprendizado de máquina em aplicações financeiras.""")

#incluir rotas
app.include_router(predict.router, prefix="/predict", tags=["Predictions"])
app.include_router(train.router, prefix="/train", tags=["Model Training"])
app.include_router(evaluation_metrics.router, prefix="/evaluation", tags=["Evalution Metrics"])

@app.get("/")
async def root():
    return {"message": "Bem-vindo à API Financeira com LSTM!"}
