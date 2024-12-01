import yfinance as yf
from datetime import datetime, timedelta


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
    return df

def fetch_historical_data(symbol: str):
    """
    Fetch historical closing price data for a given stock symbol over the past 120 days.

    Args:
        symbol (str): The stock symbol to fetch data for.

    Returns:
        numpy.ndarray: An array of closing prices for the specified stock symbol.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=120)
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    return data['Close'].values
