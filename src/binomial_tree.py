"""
Cox-Ross-Rubinstein (CRR) binomial tree option pricer, supporting both
European and American exercise.

VALIDATION STRATEGY:
  - European binomial price should CONVERGE to the Black-Scholes closed
    form as the number of steps grows (the tree is a discretization of
    the same continuous-time model Black-Scholes solves exactly).
  - American price must be >= European price, ALWAYS, for the same
    parameters -- early exercise is an optional right, so it can only
    add value, never subtract it. This holds with certainty on the same
    tree, not just as a statistical tendency.
"""
import math
from dataclasses import dataclass

import numpy as np


def binomial_tree_price(S: float, K: float, r: float, sigma: float, T: float,
                        n_steps: int, option_type: str = "call",
                        exercise: str = "european", q: float = 0.0) -> float:
    """
    q: continuous dividend yield (annualized), 0.0 by default -- reduces
    to the original no-dividend tree exactly when q=0 (p's formula below
    becomes (exp(r*dt)-d)/(u-d), identical to before this parameter was
    added). With q > 0, the risk-neutral UP-probability p uses r-q, not
    r, as the tree's per-step drift: holding the stock earns the
    dividend yield too, so the risk-neutral drift that must be matched
    by u/d/p is reduced by q (same reasoning as Black-Scholes' d1/d2
    shift -- see black_scholes.py). u and d themselves (the tree's step
    SIZES) are unaffected by q; only which probability path through
    them is risk-neutral changes.
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    if exercise not in ("european", "american"):
        raise ValueError("exercise must be 'european' or 'american'")

    dt = T / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)  # risk-neutral probability

    # terminal stock prices at each of the n_steps+1 terminal nodes
    j = np.arange(n_steps + 1)
    terminal_prices = S * (u ** j) * (d ** (n_steps - j))

    if option_type == "call":
        values = np.maximum(terminal_prices - K, 0.0)
    else:
        values = np.maximum(K - terminal_prices, 0.0)

    discount = math.exp(-r * dt)

    # step backward through the tree
    for step in range(n_steps - 1, -1, -1):
        values = discount * (p * values[1:] + (1 - p) * values[:-1])

        if exercise == "american":
            j = np.arange(step + 1)
            prices_at_step = S * (u ** j) * (d ** (step - j))
            if option_type == "call":
                intrinsic = np.maximum(prices_at_step - K, 0.0)
            else:
                intrinsic = np.maximum(K - prices_at_step, 0.0)
            values = np.maximum(values, intrinsic)

    return float(values[0])
