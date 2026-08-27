"""
Black-Scholes closed-form European option pricing, plus the standard
analytic Greeks (delta, gamma, vega, theta, rho).

This is the baseline every other pricer in this project gets checked
against: the binomial tree should converge to it as steps increase,
plain Monte Carlo should converge to it statistically, and the implied
volatility solver inverts it. Getting this exactly right matters more
than any other single file here.
"""
import math
from dataclasses import dataclass

from scipy.stats import norm


@dataclass
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def _d1_d2(S: float, K: float, r: float, sigma: float, T: float):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def black_scholes_price(S: float, K: float, r: float, sigma: float, T: float, option_type: str = "call") -> float:
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must both be positive")

    d1, d2 = _d1_d2(S, K, r, sigma, T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def black_scholes_greeks(S: float, K: float, r: float, sigma: float, T: float, option_type: str = "call") -> Greeks:
    """
    Analytic Greeks. gamma and vega are the same formula for calls and
    puts (they measure curvature/vol-sensitivity of the SAME underlying
    distribution regardless of which side you're priced on); delta,
    theta, and rho differ by option_type because they depend on the
    direction of the payoff.
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    d1, d2 = _d1_d2(S, K, r, sigma, T)
    pdf_d1 = norm.pdf(d1)
    sqrt_T = math.sqrt(T)

    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega = S * pdf_d1 * sqrt_T  # per unit change in sigma (not per 1% -- caller can /100 if needed)

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (-(S * pdf_d1 * sigma) / (2 * sqrt_T)
                 - r * K * math.exp(-r * T) * norm.cdf(d2))
        rho = K * T * math.exp(-r * T) * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1
        theta = (-(S * pdf_d1 * sigma) / (2 * sqrt_T)
                 + r * K * math.exp(-r * T) * norm.cdf(-d2))
        rho = -K * T * math.exp(-r * T) * norm.cdf(-d2)

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
