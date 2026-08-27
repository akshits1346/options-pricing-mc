"""
Shared GBM path simulator for the Monte Carlo pricers in this project.

Simulates paths under the risk-neutral measure:
    S_{i+1} = S_i * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z_i)

ANTITHETIC VARIATES: when antithetic=True, for every independent draw
Z_i, a second path is also generated using -Z_i instead. This pairs
each path with its "mirror image" -- if the underlying moves are
negatively correlated across the pair, the AVERAGE of the two payoffs
tends to have lower variance than two independent draws would, for
payoffs that are monotonic in the underlying path (which covers both
the Asian and barrier options priced here). n_paths must be even when
antithetic=True, since paths come in pairs.
"""
import numpy as np


def simulate_gbm_paths(S0: float, r: float, sigma: float, T: float,
                        n_steps: int, n_paths: int,
                        antithetic: bool = False, seed: int = None) -> np.ndarray:
    """
    Returns an array of shape (n_paths, n_steps + 1): each row is one
    price path, column 0 is S0, columns 1..n_steps are the simulated
    prices at each monitoring point.
    """
    if antithetic and n_paths % 2 != 0:
        raise ValueError("n_paths must be even when antithetic=True (paths come in +/- pairs)")

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - 0.5 * sigma ** 2) * dt
    vol = sigma * np.sqrt(dt)

    if antithetic:
        half = n_paths // 2
        Z = rng.standard_normal((half, n_steps))
        Z_full = np.concatenate([Z, -Z], axis=0)  # each of the first `half` rows paired with its negation
    else:
        Z_full = rng.standard_normal((n_paths, n_steps))

    log_increments = drift + vol * Z_full
    log_paths = np.cumsum(log_increments, axis=1)
    log_paths = np.concatenate([np.zeros((n_paths, 1)), log_paths], axis=1)  # prepend log(S0/S0)=0 for column 0

    paths = S0 * np.exp(log_paths)
    return paths
