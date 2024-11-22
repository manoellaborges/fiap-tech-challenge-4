from fastapi import FastAPI
from routers import predict, train
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf


app = FastAPI(title="Finance API", description="API para coletar dados e treinar um modeelo LSTM.")

#incluir rotas
app.include_router(predict.router, prefix="/predict", tags=["Predictions"])
app.include_router(train.router, prefix="/train", tags=["Model Training"])

@app.get("/")
async def root():
    return {"message": "Bem-vindo à API Financeira com LSTM!"}
