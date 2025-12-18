from enum import Enum
from typing import Dict, List

import numba
import numpy as np


class SimModes(Enum):
    ALL_POINTS = 1
    PORTFOLIO = 2


def simulate(
    mode: SimModes,
    obs: np.ndarray,
    market: np.ndarray,
    action: np.ndarray,
    tot_fee=1e-3,
    min_pos=0.1,
    use_ratio=0.5,
    alloc_ratio=0.01,
    init_cash=10e3,
    min_cash=10,
    price_idx=1,
    clip_pr=False,
) -> Dict[str, np.ndarray]:
    if mode == SimModes.ALL_POINTS:
        trades = simulate_all_points(
            market,
            action,
            tot_fee,
            min_pos,
            price_idx,
            clip_pr,
        )
    if mode == SimModes.PORTFOLIO:
        trades = simulate_portfolio(
            market,
            action,
            tot_fee,
            min_pos,
            use_ratio,
            alloc_ratio,
            init_cash,
            min_cash,
            price_idx,
            clip_pr,
        )
    res = np.array(trades)
    keys = ["sym", "time", "cash", "duration", "profit", "worth"]
    sym, time = res[:, :2].T.astype(int)
    obs, action = obs[sym, time], action[sym, time]
    res = {k: res[..., i] for i, k in enumerate(keys)}
    return {**res, "obs": obs, "action": action}


@numba.njit
def find_profit(
    price1: float,
    price2: float,
    dt: int,
    action: np.ndarray,
    tot_fee: float,
    clip=False,
):
    pos, lev, timeout, take_profit, stop_loss = action
    lev = max(1, int(lev))
    pr = lev * (np.sign(pos) * (price2 / price1 - 1) - tot_fee)
    exit = dt >= timeout or pr >= take_profit or pr <= stop_loss
    if clip:
        pr = min(pr, take_profit)
    return max(-1, pr), exit


@numba.njit
def simulate_all_points(
    market: np.ndarray,
    action: np.ndarray,
    tot_fee=1e-3,
    min_pos=0.1,
    price_idx=1,
    clip_pr=False,
):
    SYMBOL, TIME, _ = market.shape
    done_trades = []

    for s1 in numba.prange(SYMBOL):
        for t1 in numba.prange(TIME - 1):
            pos = action[s1, t1, 0]
            if abs(pos) >= min_pos:
                price1 = market[s1, t1, price_idx]
                act = action[s1, t1]
                for t in range(t1 + 1, TIME):
                    dt = t - t1
                    price2 = market[s1, t, price_idx]
                    pr, exit = find_profit(price1, price2, dt, act, tot_fee, clip_pr)
                    if exit:
                        done_trades.append([s1, t1, 1.0, float(dt), pr, 0.0])
                        break
    return done_trades


@numba.njit
def simulate_portfolio(
    market: np.ndarray,
    action: np.ndarray,
    tot_fee=1e-3,
    min_pos=0.1,
    use_ratio=0.5,
    alloc_ratio=0.01,
    init_cash=10e3,
    min_cash=10,
    price_idx=1,
    clip_pr=False,
):
    max_open = int(use_ratio / alloc_ratio)
    SYMBOL, TIME, _ = market.shape

    worth = cash_left = init_cash
    open_trades: Dict[int, np.ndarray] = {}
    done_trades: List[np.ndarray] = []

    for t in range(TIME):

        for id, x in list(open_trades.items()):
            s1, t1, cash = int(x[0]), int(x[1]), x[2]
            dt = t - t1
            price1 = market[s1, t1, price_idx]
            price2 = market[s1, t, price_idx]
            act = action[s1, t1]
            pr, exit = find_profit(price1, price2, dt, act, tot_fee, clip_pr)
            if exit:
                worth += cash * pr
                cash_left += cash * (1 + pr)
                x[-3:] = float(dt), pr, worth
                del open_trades[id]
                done_trades.append(x)

        for s in range(SYMBOL):
            pos = action[s, t, 0]
            if abs(pos) >= min_pos and len(open_trades) < max_open:
                cash = int(min(cash_left, worth * alloc_ratio * abs(pos)))
                id = int(np.sign(pos) * (s + 1))
                if cash >= min_cash and id not in open_trades:
                    cash_left -= cash
                    open_trades[id] = np.array([s, t, cash, 0.0, 0.0, 0.0])

    return done_trades
