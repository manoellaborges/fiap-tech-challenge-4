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

from utils.extract import extract_data
from utils.preprocessing import preprocess_data


def create_dataset(dataset, time_step=60):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step -1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


# Função para dividir dados por semestre
def dividir_por_semestre(dados):
    semestres = []
    anos = dados.index.year.unique()
    for ano in anos:
        semestre1 = dados[(dados.index >= f'{ano}-01-01') & (dados.index < f'{ano}-07-01')]
        semestre2 = dados[(dados.index >= f'{ano}-07-01') & (dados.index < f'{ano+1}-01-01')]
        semestres.append((semestre1, semestre2))
    return semestres


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


def train_lstm(symbol, start_date, end_date):
    data = extract_data(symbol, start_date, end_date)
    processed_data = preprocess_data(data, symbol)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(processed_data[['Close']].values)
    joblib.dump(scaler, 'scaler.pkl')

    time_step = 60
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    n_splits = calculate_n_splits(len(X), min_test_size=0.1)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mse_scores = []

 

    tuner = kt.RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=5,
        executions_per_trial=3,
        directory='tuner_dir',
        project_name='stock_prediction'
    )

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[EarlyStopping(patience=3)])

        best_model = tuner.get_best_models(num_models=1)[0]
        best_model.save('stock_prediction_model.keras')

        test_predictions = best_model.predict(X_test)
        test_predictions = scaler.inverse_transform(test_predictions)
        y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))
        mse = mean_squared_error(y_test_unscaled, test_predictions)
        mse_scores.append(mse)

        global results
        results["real_values"].extend(y_test_unscaled.flatten())
        results["predicted_values"].extend(test_predictions.flatten())
        

    avg_mse = np.mean(mse_scores)
    print(avg_mse)

    return avg_mse
