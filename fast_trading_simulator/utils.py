from typing import Callable, Dict

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from trading_models.utils import plot_general


def round_dx(x, dx):
    return round(round(x / dx) * dx, 10)


def pymoo_minimize(func: Callable, conf: Dict, algo=GA()):
    xl = [v[0] for v in conf.values()]
    xu = [v[1] for v in conf.values()]
    dx = [v[2] for v in conf.values()]
    best = np.inf

    class Prob(ElementwiseProblem):
        def __init__(s):
            super().__init__(n_var=len(xl), n_obj=1, xl=xl, xu=xu)

        def _evaluate(s, X: np.ndarray, out, *args, **kwargs):
            X = [round_dx(xi, dxi) for xi, dxi in zip(X, dx)]
            P = dict(zip(conf.keys(), X))
            loss = func(P)
            nonlocal best
            if loss < best:
                best = loss
                print(f"best: {best} {P}")
            out["F"] = loss

    minimize(Prob(), algo, seed=42)


def plot_trades(trades):
    trades = np.array(trades)
    plots = {
        f"worth ({len(trades)} trades)": trades[:, -1],
        "position_hist": trades[:, 2],
        "duration_hist": trades[:, -3],
        "profit_hist": trades[:, -2],
    }
    plot_general(plots, "simulate")
