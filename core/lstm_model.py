import logging
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras_tuner as kt
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils.extract import extract_data
from utils.preprocessing import preprocess_data

# Configuração do logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='training.log', filemode='w')
logger = logging.getLogger()


def create_dataset(dataset, time_step=60):
    logger.debug("Creating dataset with time_step=%d", time_step)
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step-1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2

def calculate_n_splits(data_length, min_test_size=0.1):
    """
    Calcula o número de splits baseado no tamanho dos dados e no tamanho mínimo desejado para o conjunto de teste.
    :param data_length: Comprimento do conjunto de dados.
    :param min_test_size: Tamanho mínimo desejado para o conjunto de teste como uma fração do conjunto de dados.
    :return: Número de splits.
    """
    test_size = int(data_length * min_test_size)
    n_splits = max(2, data_length // test_size)  # Garantir pelo menos 2 splits
    return n_splits

def build_model(hp):
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

results = {
    "real_values": [],
    "predicted_values": []
}

def run_pipeline(symbol, start_date, end_date):
    raw_data = extract_data(symbol, start_date, end_date)
    cleaned_data = preprocess_data(raw_data, symbol)
    return cleaned_data

def train_lstm(symbol, start_date, end_date):
    logger.info("Starting training for symbol: %s", symbol)

    data = run_pipeline(symbol, start_date, end_date)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']].values)
    joblib.dump(scaler, 'scaler.pkl')

    time_step = 60
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    n_splits = calculate_n_splits(len(X), min_test_size=0.1)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mse_scores = []

    logger.info("Starting hyperparameter tuning")
    tuner = kt.RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=5,
        executions_per_trial=3,
        directory='tuner_dir',
        project_name='stock_prediction'
    )

    best_mse = float('inf')
    best_model = None

    for train_index, test_index in tscv.split(X):
        logger.info("Training split: %d", len(train_index))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[EarlyStopping(patience=3)])

        current_best_model = tuner.get_best_models(num_models=1)[0]# Avalia o modelo atual
        
        test_predictions = current_best_model.predict(X_test)
        test_predictions = scaler.inverse_transform(test_predictions)
        y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

        mse, mae, r2 = evaluate_model(y_test_unscaled, test_predictions)
        logger.info("Evaluation - MSE: %f, MAE: %f, R²: %f", mse, mae, r2)

        # best_model.save('stock_prediction_model.keras')

        # test_predictions = best_model.predict(X_test)
        # test_predictions = scaler.inverse_transform(test_predictions)
        # y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))
        # mse = mean_squared_error(y_test_unscaled, test_predictions)
        mse_scores.append(mse) # Atualiza o melhor modelo global com base no MSE
        
        if mse < best_mse:
            best_mse = mse
            best_model = current_best_model

        global results
        results["real_values"].extend(y_test_unscaled.flatten())
        results["predicted_values"].extend(test_predictions.flatten())
        logger.info("Finished split with MSE: %f", mse)

    # Salva o melhor modelo global
    if best_model:
        best_model.save('stock_prediction_model.keras')
        logger.info("Best model saved successfully with MSE: %f", best_mse)

    return mse
