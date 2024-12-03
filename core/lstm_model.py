import mlflow
import mlflow.keras
import logging
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras_tuner as kt
from tensorflow.keras.optimizers import Adam
 
from utils.train_functions import create_dataset, evaluate_model, calculate_n_splits, build_model, run_pipeline

# Configuração do logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='training.log', filemode='w')
logger = logging.getLogger()

mlruns_path = os.path.join(os.getcwd(), "mlruns")
os.environ["MLFLOW_TRACKING_URI"] = f"file://{mlruns_path}"

results = {
    "real_values": [],
    "predicted_values": []
}

def train_lstm(symbol, start_date, end_date):
    """
    Treina um modelo LSTM para previsão de preços de ações.
    Parâmetros:
    symbol (str): O símbolo da ação a ser prevista.
    start_date (str): A data de início do período de treinamento.
    end_date (str): A data de término do período de treinamento.
    O treinamento envolve as seguintes etapas:
    1. Coleta e pré-processamento dos dados.
    2. Escalonamento dos dados de fechamento.
    3. Criação de conjuntos de dados de treinamento e teste.
    4. Divisão dos dados em múltiplas divisões de séries temporais.
    5. Ajuste de hiperparâmetros usando uma busca aleatória.
    6. Avaliação do modelo em cada divisão e seleção do melhor modelo.
    7. Salvamento do melhor modelo treinado.
    O modelo é salvo no arquivo 'stock_prediction_model.keras' e o escalador é salvo no arquivo 'scaler.pkl'.
    """
    try:
        logger.info("Starting training for symbol: %s", symbol)
        
        mlflow.log_param("symbol", symbol)
        mlflow.log_param("start_date", start_date)
        mlflow.log_param("end_date", end_date)

        data = run_pipeline(symbol, start_date, end_date)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[['Close']].values)
        joblib.dump(scaler, 'scaler.pkl')

        time_step = 60
        X, y = create_dataset(logger, scaled_data, time_step)
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
            mse_scores.append(mse)

            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            
            logger.info("Evaluation - MSE: %f, MAE: %f, R²: %f", mse, mae, r2)
            
            if mse < best_mse:
                best_mse = mse
                best_model = current_best_model

            global results
            results["real_values"].extend(y_test_unscaled.flatten())
            results["predicted_values"].extend(test_predictions.flatten())
            logger.info("Finished split with MSE: %f", mse)

        if best_model:
            best_model.save('stock_prediction_model.keras')
            logger.info("Best model saved successfully with MSE: %f", best_mse)
            mlflow.keras.log_model(best_model, "model")
            mlflow.log_artifact('stock_prediction_model.keras')
        
    finally:
        mlflow.end_run()  # Encerra o experimento MLflow

    return mse
