import numpy as np
from crypto_data_downloader.utils import save_pkl
from trading_models.utils import plot_general, shape

from fast_trading_simulator.simulate_v3 import SimModes, simulate
from fast_trading_simulator.utils import ActMap, make_market_n_obs

np.random.seed(0)

path = "./futures_data_2025-08-01_2025-11-20.pkl"
market, obs = make_market_n_obs(path)
SYMBOL, TIME, _ = market.shape

# pos, lev, timeout, take_profit, stop_loss = action
act_low = np.array([-1, 1, 10, 0.01, -0.5])
act_high = np.array([1, 100, 100, 0.1, -0.01])
assert np.all(act_low < act_high)


def rand_action():
    tanh_act = np.random.uniform(-1, 1, (SYMBOL, TIME, 5))
    return ActMap.from_tanh(tanh_act, act_low, act_high)


def simple_action():
    ob = obs.transpose(2, 0, 1)
    pos = np.where(ob[0] < -0.05, 1, 0)
    pos = np.where(ob[0] > 0.05, -1, pos)
    lev = np.full((SYMBOL, TIME), 1)
    timeout = np.full((SYMBOL, TIME), 10)
    take_profit = np.full((SYMBOL, TIME), 0.01)
    stop_loss = np.full((SYMBOL, TIME), -0.9)
    return np.array([pos, lev, timeout, take_profit, stop_loss]).transpose(1, 2, 0)


action = simple_action()
plots = {}
for mode in SimModes:
    res = simulate(mode, obs, market, action)
    plots[f"mode={mode.name}, duration_hist"] = res["duration"]
    plots[f"mode={mode.name}, profit_hist"] = res["profit"]
    plots[f"mode={mode.name}, log10(worth)"] = np.log10(res["worth"])
    res["tanh_act"] = ActMap.to_tanh(res["action"], act_low, act_high)
    print(shape(res))
    if mode == SimModes.PORTFOLIO:
        save_pkl(res, f"./ml_data_{mode.name}.pkl")
plot_general(plots, "simulate_v3", C=3)
