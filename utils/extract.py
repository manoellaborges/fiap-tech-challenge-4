import yfinance as yf


def extract_data(symbol, start_date, end_date):
    """
    Extracts the closing price data for a given stock symbol within a specified date range.

    Parameters:
    symbol (str): The stock symbol to extract data for.
    start_date (str): The start date for the data extraction in 'YYYY-MM-DD' format.
    end_date (str): The end date for the data extraction in 'YYYY-MM-DD' format.

    Returns:
    numpy.ndarray: A numpy array containing the closing prices for the specified date range.
    """
    df = yf.download(symbol, start=start_date, end=end_date)
    # data = df[['Close']].values
    return df
