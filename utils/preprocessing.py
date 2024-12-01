import numpy as np


def add_technical_indicators(data):
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data.dropna(inplace=True)
    return data


def preprocess_data(data, symbol):
    """
    Preprocesses the given data by removing NaN values and handling outliers in the 'Close' column for the specified symbol.
    Parameters:
    data (pd.DataFrame): The input data containing stock prices with multi-level columns.
    symbol (str): The stock symbol for which the 'Close' prices need to be preprocessed.
    Returns:
    pd.DataFrame: The preprocessed data with outliers handled and NaN values forward-filled.
    """
    data.dropna(inplace=True)

    # 
    data = add_technical_indicators(data)

    deviations = 3
    mean = data[('Close', symbol)].mean()
    deviation_padrao = data[('Close', symbol)].std()
    outliers = (data[('Close', symbol)] > mean + deviations * deviation_padrao) | (data[('Close', symbol)] < mean - deviations * deviation_padrao)
    data.loc[outliers, ('Close', symbol)] = np.nan
    data[('Close', symbol)].fillna(method='ffill', inplace=True)

    return data
