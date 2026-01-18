# https://gemini.google.com/app/8c73a4b7d75ed94c
import matplotlib.pyplot as plt
import numpy as np
import torch as tc
from crypto_data_downloader.utils import load_pkl, save_pkl
from trading_models.stat import StandardScaler
from trading_models.utils import mlp, shape, tensor, to_np

from fast_trading_simulator.sim import map_trades, simulate
from fast_trading_simulator.utils import make_market_n_obs, plot_act_hist, rand_action


def make_obs_act_pr():
    path = "./futures_data_2025-08-01_2025-11-20.pkl"
    periods = [5, 10, 20, 40, 80, 160, 320]
    MARKET, OBS = make_market_n_obs(path, periods=periods)
    tot_fee = 1e-3
    liq_fee = np.full(len(OBS), 0.02)
    _, action = rand_action(OBS)
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
    res, (obs, action) = map_trades(trades, (OBS, action))
    profit = res["profit"][:, None]
    print(shape([obs, action, profit]))
    save_pkl((obs, action, profit), "data/obs_act_pr.pkl")


def safety_1st_loss(pr_pred: tc.Tensor, pr: tc.Tensor, amp=10.0):
    SE = (pr_pred - pr) ** 2
    ignore = pr.abs() < 0.01  # remove low abs noise
    wrong1 = pr_pred > pr  # overestimate profit: 10x
    wrong2 = wrong1 & (pr < 0)  # overestimate profit & loss: 100x
    SE = tc.where(ignore, 0, SE)
    SE = tc.where(wrong1, amp * SE, SE)
    SE = tc.where(wrong2, amp * SE, SE)
    return SE.mean()


# make_obs_act_pr()

obs, act, pr = tensor(load_pkl("data/obs_act_pr.pkl"))
pr_np = to_np(pr[:, 0])
obs_act = tc.concat((obs, act), dim=-1)
scaler = StandardScaler()
obs_act = scaler.fit_transform(obs_act)
hid = 64
net = mlp([obs_act.size(-1), hid, hid, 1])
opt = tc.optim.AdamW(net.parameters(), lr=1e-3)
losses = []
corr = 0
for e in range(10000):
    pr_pred = net(obs_act)
    loss = safety_1st_loss(pr_pred, pr, amp=10.0)
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())
    if e % 100 == 0:
        pr_pred = to_np(pr_pred[:, 0])
        corr = np.corrcoef(pr_np, pr_pred)[0, 1]
        plt.scatter(pr_np, pr_pred, s=1, c="y", label="pr_pred")
        plt.plot(pr_np, pr_np, c="b", label="pr")
        plt.hlines(0, -0.2, 0.2, colors="k")
        plt.xlim(-0.2, 0.2)
        plt.ylim(-0.2, 0.2)
        plt.legend()
        plt.savefig(f"data/Q_func_{hid}")
        plt.close()
    print(f"{e}, loss: {loss.item()}, corr: {corr}")
