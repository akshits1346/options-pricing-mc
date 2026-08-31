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


def _d1_d2(S: float, K: float, r: float, sigma: float, T: float, q: float = 0.0):
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def black_scholes_price(S: float, K: float, r: float, sigma: float, T: float,
                         option_type: str = "call", q: float = 0.0) -> float:
    """
    q: continuous dividend yield (annualized), 0.0 by default -- every
    call site that doesn't pass q gets EXACTLY the original no-dividend
    formula (q=0 reduces d1/d2 and the S*exp(-qT) discount factor to
    their original forms identically, not approximately).

    With q > 0, the stock's forward drift under the risk-neutral measure
    is r-q rather than r (holding the stock earns you the dividend
    stream, so the NO-dividend drift r must be reduced by q to keep the
    total expected return under the risk-neutral measure at r), and the
    spot itself is discounted by exp(-qT) in the pricing formula (you
    don't receive the dividends the option holder would have, since an
    option isn't the stock).
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must both be positive")

    d1, d2 = _d1_d2(S, K, r, sigma, T, q)

    if option_type == "call":
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)


def black_scholes_greeks(S: float, K: float, r: float, sigma: float, T: float,
                          option_type: str = "call", q: float = 0.0) -> Greeks:
    """
    Analytic Greeks, with the same optional continuous dividend yield q
    as black_scholes_price (q=0.0 by default, reducing to the original
    no-dividend formulas exactly). gamma and vega are the same formula
    for calls and puts (they measure curvature/vol-sensitivity of the
    SAME underlying distribution regardless of which side you're priced
    on); delta, theta, and rho differ by option_type because they depend
    on the direction of the payoff. rho is UNCHANGED by q (it's the
    sensitivity to r, which only enters through the strike's discount
    factor K*exp(-rT), not through the S*exp(-qT) term at all).
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    d1, d2 = _d1_d2(S, K, r, sigma, T, q)
    pdf_d1 = norm.pdf(d1)
    sqrt_T = math.sqrt(T)
    div_discount = math.exp(-q * T)

    gamma = div_discount * pdf_d1 / (S * sigma * sqrt_T)
    vega = S * div_discount * pdf_d1 * sqrt_T  # per unit change in sigma (not per 1% -- caller can /100 if needed)

    if option_type == "call":
        delta = div_discount * norm.cdf(d1)
        theta = (-(S * div_discount * pdf_d1 * sigma) / (2 * sqrt_T)
                 - r * K * math.exp(-r * T) * norm.cdf(d2)
                 + q * S * div_discount * norm.cdf(d1))
        rho = K * T * math.exp(-r * T) * norm.cdf(d2)
    else:
        delta = div_discount * (norm.cdf(d1) - 1)
        theta = (-(S * div_discount * pdf_d1 * sigma) / (2 * sqrt_T)
                 + r * K * math.exp(-r * T) * norm.cdf(-d2)
                 - q * S * div_discount * norm.cdf(-d1))
        rho = -K * T * math.exp(-r * T) * norm.cdf(-d2)

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
