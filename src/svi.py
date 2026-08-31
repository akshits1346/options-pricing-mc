"""
SVI ("Stochastic Volatility Inspired", Gatheral) parametric fit for a
single-maturity implied volatility smile.

vol_surface.py (see its module docstring) checks a grid of
INDEPENDENTLY-SOLVED implied vols for arbitrage after the fact. SVI
takes a different approach to the same underlying problem: fit a smooth,
5-parameter curve THROUGH noisy market quotes, so the whole smile comes
from one coherent function rather than one solve per strike. The raw
SVI parametrization of TOTAL IMPLIED VARIANCE as a function of
log-moneyness k = ln(K/F) (F = forward price):

    w(k) = a + b * ( rho*(k - m) + sqrt((k - m)^2 + sigma^2) )

    a      : overall variance level
    b      : overall slope/wing steepness (b >= 0)
    rho    : skew / rotation of the smile (-1 < rho < 1)
    m      : horizontal shift of the smile's center
    sigma  : curvature at the center (sigma > 0)

WHY REUSE vol_surface.py's CHECKER RATHER THAN A NEW ANALYTIC FORMULA:
Gatheral & Jacquier's paper on arbitrage-free SVI gives an analytic
local condition for a slice to be free of butterfly arbitrage, but
re-deriving and hand-verifying that condition here from memory (no
internet access in this environment to check it against the source)
risks shipping a WRONG "rigor" check with unjustified confidence --
exactly the kind of unverifiable claim this project tries hard to
avoid elsewhere. Instead: evaluate the fitted SVI curve on a fine
strike grid and run it through vol_surface.py's check_butterfly_arbitrage(),
which IS already independently tested (see test_vol_surface.py) against
both a clean surface and a deliberately-broken one. This is standard
engineering practice (reuse a validated checker over restating an
unverified formula) and it also ties this file directly to the vol
surface work rather than duplicating its logic.
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from src.vol_surface import VolSurface


@dataclass
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float


def raw_svi_total_variance(k, a: float, b: float, rho: float, m: float, sigma: float):
    """w(k), vectorized (k can be a scalar or a numpy array)."""
    k = np.asarray(k, dtype=float)
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def svi_implied_vol(k, T: float, params: SVIParams):
    w = raw_svi_total_variance(k, params.a, params.b, params.rho, params.m, params.sigma)
    return np.sqrt(np.maximum(w, 1e-12) / T)


def fit_svi(strikes, implied_vols, T: float, S: float, r: float) -> SVIParams:
    """
    Least-squares fit of raw SVI parameters to observed (strike,
    implied_vol) points at ONE maturity T. Forward price F = S*e^(rT);
    log-moneyness k_i = ln(K_i / F). Fits in TOTAL VARIANCE space
    (w = iv^2 * T), which is what the SVI parametrization is defined in
    -- fitting iv directly would be fitting the wrong functional form.

    Bounds enforce the basic well-definedness constraints (b >= 0,
    |rho| < 1, sigma > 0) but do NOT by themselves guarantee the result
    is arbitrage-free -- that's a separate, explicit check, see
    check_svi_butterfly_arbitrage() below. A least-squares fit that
    satisfies these bounds can still violate no-arbitrage; enforcing
    that fully during the fit itself is exactly the harder problem
    "arbitrage-free SVI" methods solve, which this simpler two-step
    fit-then-check approach deliberately doesn't attempt.
    """
    strikes = np.asarray(strikes, dtype=float)
    implied_vols = np.asarray(implied_vols, dtype=float)
    if len(strikes) != len(implied_vols):
        raise ValueError("strikes and implied_vols must be the same length")
    if len(strikes) < 5:
        raise ValueError("need at least 5 quotes to fit SVI's 5 parameters")

    forward = S * math.exp(r * T)
    k = np.log(strikes / forward)
    w_observed = (implied_vols ** 2) * T

    def residuals(params):
        a, b, rho, m, sigma = params
        w_model = raw_svi_total_variance(k, a, b, rho, m, sigma)
        return w_model - w_observed

    a0 = float(np.mean(w_observed))
    initial = [a0, 0.1, 0.0, 0.0, 0.1]
    lower = [-np.inf, 0.0, -0.999, -np.inf, 1e-6]
    upper = [np.inf, np.inf, 0.999, np.inf, np.inf]

    result = least_squares(residuals, initial, bounds=(lower, upper))
    a, b, rho, m, sigma = result.x
    return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)


def check_svi_butterfly_arbitrage(params: SVIParams, T: float, S: float, r: float,
                                   k_min: float = -1.5, k_max: float = 1.5, n_points: int = 61):
    """
    Evaluates the fitted SVI curve on a fine, evenly-spaced grid of
    log-moneyness points spanning [k_min, k_max], converts to strikes,
    and runs the resulting (strike, implied_vol) grid through
    vol_surface.py's ALREADY-TESTED check_butterfly_arbitrage(). Returns
    the list of violations (empty == no butterfly arbitrage detected at
    this grid resolution -- a finite-resolution check, not a proof for
    every possible strike, same caveat that applies to any grid-based
    numerical check).
    """
    forward = S * math.exp(r * T)
    k_grid = np.linspace(k_min, k_max, n_points)
    strikes = forward * np.exp(k_grid)
    ivs = svi_implied_vol(k_grid, T, params)

    iv_grid = {(float(K), T): float(iv) for K, iv in zip(strikes, ivs)}
    surface = VolSurface.from_grid(S, r, iv_grid)
    return surface.check_butterfly_arbitrage()
