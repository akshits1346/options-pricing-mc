"""
Implied volatility surface: solve implied vol pointwise across a grid of
(strike, maturity) quotes, then check the surface for the two classic
static no-arbitrage violations a real vol surface must not have.

    - CALENDAR ARBITRAGE: at a fixed strike, total implied variance
      w(K, T) = sigma(K, T)^2 * T must be non-decreasing in T. If it
      ever decreases, a calendar spread (long the longer-dated option,
      short the shorter-dated one, same strike) is a risk-free
      arbitrage -- a standard result in vol surface construction (see
      e.g. Gatheral, "The Volatility Surface").

    - BUTTERFLY (STRIKE) ARBITRAGE: at a fixed maturity, price must be
      a convex function of strike. Equivalently (Breeden-Litzenberger),
      the risk-neutral density implied by the surface must be
      non-negative everywhere. Checked here as a discrete convexity
      test on three neighboring strikes via divided differences (valid
      for unevenly spaced strikes, unlike a fixed-step second
      difference): the slope of price between (K_lo, K_mid) must not
      exceed the slope between (K_mid, K_hi). If it does, the middle
      strike's price sits above the chord connecting its neighbors --
      a butterfly spread (long 1x K_lo, short 2x K_mid, long 1x K_hi)
      has negative cost and a non-negative payoff everywhere: a
      risk-free arbitrage.

Prices for the butterfly check are always repriced as CALLS from each
point's fitted implied vol (via black_scholes_price(..., option_type=
"call")), regardless of whether the original quote was a call or a put.
This makes the check well-defined even when the input mixes calls and
puts: comparing a raw put price against a raw call price directly would
conflate the strike-convexity question with put-call parity, which is a
separate (and, under Black-Scholes assumptions, always-satisfied)
relationship, not part of what this check is testing.

WHY THIS MATTERS: implied vols solved independently at each (K, T), as
this repo's implied_vol.py does pointwise, are NOT automatically
consistent with each other. Constructing a surface from independently-
solved points and then checking these two conditions is exactly the
gap between "I can invert Black-Scholes at a point" and "I have a
surface a real desk could quote off of."
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.black_scholes import black_scholes_price
from src.implied_vol import implied_volatility


@dataclass
class VolQuote:
    K: float
    T: float
    price: float
    option_type: str = "call"


class VolSurface:
    def __init__(self, S: float, r: float):
        self.S = S
        self.r = r
        self.points: Dict[Tuple[float, float], float] = {}  # (K, T) -> implied vol

    @classmethod
    def from_grid(cls, S: float, r: float, iv_grid: Dict[Tuple[float, float], float]) -> "VolSurface":
        """Build a surface directly from known (K, T) -> implied vol points,
        bypassing implied_volatility() entirely. Used to test the
        arbitrage checks in isolation against hand-constructed surfaces
        with a known-good or known-violating shape, independent of
        whether the solver itself is working (that's implied_vol.py's
        own test's job)."""
        surface = cls(S, r)
        surface.points = dict(iv_grid)
        return surface

    def fit(self, quotes: List[VolQuote]) -> "VolSurface":
        for q in quotes:
            result = implied_volatility(q.price, self.S, q.K, self.r, q.T, q.option_type)
            self.points[(q.K, q.T)] = result.implied_vol
        return self

    def iv(self, K: float, T: float) -> float:
        return self.points[(K, T)]

    def total_variance(self, K: float, T: float) -> float:
        return self.iv(K, T) ** 2 * T

    def strikes(self) -> List[float]:
        return sorted(set(k for k, _ in self.points))

    def maturities(self) -> List[float]:
        return sorted(set(t for _, t in self.points))

    def _call_price(self, K: float, T: float) -> float:
        return black_scholes_price(self.S, K, self.r, self.iv(K, T), T, option_type="call")

    def check_calendar_arbitrage(self, tol: float = 1e-10) -> List[dict]:
        violations = []
        for K in self.strikes():
            Ts = sorted(t for (k, t) in self.points if k == K)
            for T_prev, T_next in zip(Ts, Ts[1:]):
                w_prev = self.total_variance(K, T_prev)
                w_next = self.total_variance(K, T_next)
                if w_next < w_prev - tol:
                    violations.append({
                        "type": "calendar", "K": K,
                        "T_prev": T_prev, "T_next": T_next,
                        "w_prev": w_prev, "w_next": w_next,
                    })
        return violations

    def check_butterfly_arbitrage(self, tol: float = 1e-8) -> List[dict]:
        violations = []
        for T in self.maturities():
            Ks = sorted(k for (k, t) in self.points if t == T)
            if len(Ks) < 3:
                continue
            prices = {K: self._call_price(K, T) for K in Ks}
            for K_lo, K_mid, K_hi in zip(Ks, Ks[1:], Ks[2:]):
                slope_left = (prices[K_mid] - prices[K_lo]) / (K_mid - K_lo)
                slope_right = (prices[K_hi] - prices[K_mid]) / (K_hi - K_mid)
                if slope_right < slope_left - tol:
                    violations.append({
                        "type": "butterfly", "T": T,
                        "K_lo": K_lo, "K_mid": K_mid, "K_hi": K_hi,
                        "slope_left": slope_left, "slope_right": slope_right,
                    })
        return violations

    def is_arbitrage_free(self) -> bool:
        return not self.check_calendar_arbitrage() and not self.check_butterfly_arbitrage()
