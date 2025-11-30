import numpy as np
import talib as ta
from crypto_data_downloader.utils import load_pkl
from trading_models.utils import D_TYPE

from fast_trading_simulator.simulate import simulate
from fast_trading_simulator.utils import plot_trades, pymoo_minimize

path = "futures_data_2025-08-01_2025-11-20.pkl"
data: D_TYPE = load_pkl(path, gz=True)
T = len(data["BTCUSDT"])
cache = {}


def loss_func(P, plot=False):
    sim_data = []
    for sym, arr in data.items():
        if len(arr) == T:
            time, close = arr.T
            period = int(P["period"])
            id = f"{sym}_{period}"
            x = cache.get(id) or close / ta.KAMA(close, period) - 1
            pos = np.zeros(T)
            pos[x > P["high_bar"]] = -1
            pos[x < P["low_bar"]] = 1
            sim_data.append([time, close, pos])

    sim_data = np.array(sim_data).transpose(2, 0, 1)
    trades = simulate(
        sim_data,
        timeout=P["timeout"],
        take_profit=P["take_profit"],
        stop_loss=P["stop_loss"],
        fee=1e-3,
        alloc_ratio=P["alloc_ratio"],
        use_ratio=P["use_ratio"],
    )
    if plot:
        plot_trades(trades)
    gain = trades[-1][-1] / trades[0][-1]
    return -gain


conf = {
    "period": [20, 100, 1],
    "high_bar": [0.01, 0.1, 0.001],
    "low_bar": [-0.1, -0.01, 0.001],
    "timeout": [10, 100, 1],
    "take_profit": [0.01, 0.1, 0.001],
    "stop_loss": [-1, -0.2, 0.001],
    "alloc_ratio": [0.01, 0.1, 0.001],
    "use_ratio": [0.1, 1, 0.001],
}
P0 = {
    "period": 82,
    "high_bar": 0.092,
    "low_bar": -0.017,
    "timeout": 87,
    "take_profit": 0.096,
    "stop_loss": -0.607,
    "alloc_ratio": 0.1,
    "use_ratio": 0.214,
}
if 0:
    pymoo_minimize(loss_func, conf)
else:
    loss_func(P0, plot=True)
