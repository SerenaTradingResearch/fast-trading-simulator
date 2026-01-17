import numpy as np
from trading_models.utils import shape

from fast_trading_simulator.sim import map_trades, simulate
from fast_trading_simulator.utils import make_market_n_obs, plot_act_hist, rand_action

path = "./futures_data_2025-08-01_2025-11-20.pkl"
periods = [5, 10, 20, 40, 80, 160, 320]
MARKET, OBS = make_market_n_obs(path, periods=periods)
tot_fee = 1e-3
liq_fee = np.full(len(OBS), 0.02)
tanh_act, action = rand_action(OBS)
plot_act_hist(action)
trades = simulate(
    MARKET,
    action,
    tot_fee,
    liq_fee,
    clip_pr=True,
    alloc_ratio=1e-3,
    init_cash=1e8,
)
res, (obs, tanh_act) = map_trades(trades, (OBS, tanh_act))
profit = res["profit"]
print(shape([obs, tanh_act, profit]))
