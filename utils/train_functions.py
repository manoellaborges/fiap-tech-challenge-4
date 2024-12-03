import logging
import numpy as np
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import keras_tuner as kt
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
 
from utils.extract import extract_data
from utils.preprocessing import preprocess_data


def create_dataset(logger, dataset, time_step=60):
    """
    Cria um conjunto de dados para previsão de séries temporais.

    Esta função transforma um conjunto de dados de séries temporais em pares de entrada-saída adequados 
    para treinar um modelo de aprendizado de máquina. Cada sequência de entrada terá um comprimento de 
    `time_step`, e o valor correspondente será o valor imediatamente seguinte à sequência de entrada 
    no conjunto de dados original.

    Args:
        dataset (numpy.ndarray): O conjunto de dados de séries temporais original, onde cada linha é 
                                 um passo de tempo e cada coluna é uma característica. A função assume 
                                 que a variável alvo está na primeira coluna.
        time_step (int, optional): O número de passos de tempo a serem incluídos em cada sequência de 
                                   entrada. O padrão é 60.

    Returns:
        tuple: Uma tupla contendo dois arrays numpy:
            - dataX (numpy.ndarray): As sequências de entrada, onde cada sequência tem uma forma de 
                                     (time_step,).
            - dataY (numpy.ndarray): Os valores de saída correspondentes, onde cada valor é um escalar.
    """
    logger.debug("Creating dataset with time_step=%d", time_step)
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step-1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def evaluate_model(y_true, y_pred):
    """
    Avalia o desempenho de um modelo de regressão usando as métricas de Mean Squared Error (MSE),
    Mean Absolute Error (MAE) e R-squared (R2).

    Parâmetros:
    y_true (array-like): Valores verdadeiros da variável alvo.
    y_pred (array-like): Valores previstos da variável alvo.

    Retorna:
    tuple: Uma tupla contendo o MSE, MAE e o R2 score.
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2

def calculate_n_splits(data_length, min_test_size=0.1):
    """
    Calcula o número de divisões (splits) para uma série temporal com base no tamanho dos dados e no tamanho mínimo do teste.

    Args:
        data_length (int): O comprimento dos dados.
        min_test_size (float, opcional): A proporção mínima do tamanho do teste em relação ao comprimento dos dados. O padrão é 0.1 (10%).

    Returns:
        int: O número de divisões (splits) calculado.
    """
    test_size = int(data_length * min_test_size)
    n_splits = max(2, data_length // test_size)
    return n_splits

def build_model(hp):
    """
    Constrói e compila um modelo LSTM com hiperparâmetros ajustáveis.

    Args:
        hp (HyperParameters): Objeto que contém os hiperparâmetros para ajuste.

    Returns:
        keras.Model: O modelo LSTM compilado.

    O modelo consiste em:
        - Uma camada LSTM com unidades ajustáveis e `return_sequences=True`.
        - Uma segunda camada LSTM com unidades ajustáveis e `return_sequences=False`.
        - Uma camada densa com unidades ajustáveis.
        - Uma camada densa final com uma unidade de saída.

    O modelo é compilado com:
        - Otimizador Adam com taxa de aprendizado ajustável.
        - Função de perda 'mean_squared_error'.
    """
    model = Sequential()
    model.add(LSTM(
    units=hp.Int('units', min_value=32, max_value=128, step=32),
    return_sequences=True,
    input_shape=(60, 1)
    ))
    model.add(LSTM(
    units=hp.Int('units', min_value=32, max_value=128, step=32),
    return_sequences=False
    ))
    model.add(Dense(units=hp.Int('dense_units', min_value=16, max_value=64, step=16)))
    model.add(Dense(1))
    model.compile(
    optimizer=Adam(
    hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    ),
    loss='mean_squared_error'
    )
    return model

def run_pipeline(symbol, start_date, end_date):
    """
    Executa o pipeline de dados para o símbolo e intervalo de datas fornecidos.

    Esta função extrai dados brutos para o símbolo e intervalo de datas especificados,
    processa os dados e retorna os dados limpos.

    Args:
        symbol (str): O símbolo para o qual extrair os dados.
        start_date (str): A data de início para a extração de dados no formato 'YYYY-MM-DD'.
        end_date (str): A data de término para a extração de dados no formato 'YYYY-MM-DD'.

    Returns:
        DataFrame: Os dados limpos após o pré-processamento.
    """
    raw_data = extract_data(symbol, start_date, end_date)
    cleaned_data = preprocess_data(raw_data, symbol)
    return cleaned_data
