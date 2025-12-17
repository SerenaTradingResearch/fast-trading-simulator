import numpy as np
from trading_models.utils import plot_general, shape

from fast_trading_simulator.simulate_v3 import SimModes, simulate
from fast_trading_simulator.utils import make_market_n_obs

np.random.seed(0)

path = "./futures_data_2025-08-01_2025-11-20.pkl"
market, obs = make_market_n_obs(path)
SYMBOL, TIME, _ = market.shape

# pos, timeout, take_profit, stop_loss = action
act_low = np.array([-1, 10, 0.01, -0.5])
act_high = np.array([1, 100, 0.1, -0.01])
assert np.all(act_low < act_high)
tanh_act = np.random.uniform(-1, 1, (SYMBOL, TIME, 4))
action = (tanh_act + 1) / 2 * (act_high - act_low) + act_low

plots = {}
for mode in SimModes:
    res = simulate(mode, obs, market, action)
    plots[f"mode={mode.name}, duration_hist"] = res["duration"]
    plots[f"mode={mode.name}, profit_hist"] = res["profit"]
    print(shape(res))
plot_general(plots, "simulate_v3")
