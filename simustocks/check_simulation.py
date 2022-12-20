import logging
from datetime import datetime

import numpy as np
import numpy.typing as npt
import pandas_datareader.data as web

from simustocks.simulation import Simulation
from simustocks.stocks import Stocks

logging.basicConfig(
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# -------------- Take real stock prices
stocks = ["ge", "intc", "amd", "gold", "spy", "ko", "pep"]
start, end = datetime(2017, 1, 1), datetime(2020, 1, 1)
df = web.DataReader(stocks, "stooq", start=start, end=end)
df_prices = df["Close"]


# -------------- Simulate future prices
stocks_history = Stocks(df=df_prices)
expected_returns = {
    "ge": -0.8,
    "intc": 1,
    "amd": -0.1,
    "gold": -0.5,
    "spy": -0.99,
    "ko": 0.08,
    "pep": 0.01,
}
simulation = Simulation(stocks_history, er=expected_returns, m=254)
future_returns, future_cov, future_prices = simulation(order=12)

# -------------- Check that the simulated prices are correct
def get_returns(array: npt.NDArray) -> npt.NDArray:
    # array (n, k)  n > 1
    # output (n - 1, k)
    return np.diff(array, axis=0) / array[:-1, :]


recovered_returns = get_returns(future_prices)
np.allclose(recovered_returns, future_returns.T)
history_returns = get_returns(stocks_history.prices.T)

future_cov = np.cov(recovered_returns.T)
history_cov = np.cov(history_returns.T)
np.allclose(history_cov, future_cov)
