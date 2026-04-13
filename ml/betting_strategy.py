"""
Betting Strategy Engine.

Combines (in priority order):
  1. Hi-Opt I/II exact advantage formula: BSE + 0.515 × TC (Humble & Cooper)
  2. OLS-predicted EV (from Monte Carlo training) — cross-validation
  3. Kelly Criterion (Humble & Cooper p.497-502) for optimal bet sizing
  4. Count-based bet spread (practical ramp for live play)
  5. Index plays (deviations from basic strategy — Hi-Opt I tables)

The goal: translate the true count into a recommended bet size and
any strategy deviations — in real time during live play.

"Betting your bankroll over and over [with Hi-Opt I], it is all you need
to turn your original stake into a blob." — Humble & Cooper, p.558
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from config import RainManConfig, DEFAULT_CONFIG

if TYPE_CHECKING:
    from ml.ols_model import OLSResults
    from counting.base_counter import BaseCounter


@dataclass
class BetRecommendation:
    """Live bet recommendation for the current state of the shoe."""
    recommended_bet: float
    min_bet: float
    max_bet: float
    true_count: float
    running_count: int
    predicted_ev: float        # Expected return per unit (from OLS)
    player_has_edge: bool
    kelly_bet: float           # Pure Kelly criterion bet
    bet_units: float           # Bet in terms of min-bet units
    decks_remaining: float
    penetration: float
    index_plays: list          # Deviations from basic strategy
    confidence: float          # Model confidence (from R²)

    def __str__(self) -> str:
        edge_str = f"EDGE +{self.predicted_ev:.2%}" if self.player_has_edge else f"house {-self.predicted_ev:.2%}"
        return (f"BET: ${self.recommended_bet:.0f} | "
                f"TC: {self.true_count:+.1f} | "
                f"RC: {self.running_count:+d} | "
                f"EV: {edge_str} | "
                f"Decks: {self.decks_remaining:.1f}")


class BettingStrategy:
    """
    Live betting strategy engine.

    Uses the trained OLS model to recommend bets in real time.
    Falls back to count-based spread if model is not available.
    """

    def __init__(self, config: RainManConfig = DEFAULT_CONFIG,
                 ols_results: Optional["OLSResults"] = None):
        self.config = config
        self.ols_results = ols_results

    def update_model(self, ols_results: "OLSResults") -> None:
        """Update with newly trained OLS results."""
        self.ols_results = ols_results

    def recommend_bet(self, counter: "BaseCounter",
                      bankroll: float) -> BetRecommendation:
        """
        Generate a bet recommendation for the current shoe state.

        Parameters
        ----------
        counter : BaseCounter
            Active card counter with current running/true count.
        bankroll : float
            Current player bankroll.
        """
        tc = counter.true_count
        rc = counter.running_count
        decks_left = counter.shoe.decks_remaining
        pen = counter.shoe.penetration_reached
        cfg = self.config

        # 1. Exact advantage from Hi-Opt I/II formula if available
        #    (Humble & Cooper: Advantage = BSE + 0.515 × TC, p.514-516)
        exact_ev = None
        if hasattr(counter, "player_advantage"):
            exact_ev = counter.player_advantage / 100.0  # convert % to fraction

        # 2. Get EV from OLS model (cross-validation / training-based)
        if self.ols_results is not None:
            ols_ev = self.ols_results.predict_ev(tc)
            confidence = self.ols_results.r_squared
        else:
            ols_ev = None
            confidence = 0.0

        # 3. Blend: Hi-Opt formula takes priority (it's based on Griffin's analysis);
        #    OLS adds Monte Carlo validation; fallback is empirical approximation.
        if exact_ev is not None:
            predicted_ev = exact_ev
            if ols_ev is not None and abs(exact_ev - ols_ev) < 0.05:
                # Both agree — high confidence; slight blend for stability
                predicted_ev = 0.7 * exact_ev + 0.3 * ols_ev
        elif ols_ev is not None:
            predicted_ev = ols_ev
        else:
            # Last resort: empirical approximation (~0.5% per TC unit)
            bse = cfg.counting.bse / 100.0
            predicted_ev = bse + 0.00515 * tc

        has_edge = predicted_ev > 0

        # 2. Kelly criterion bet (quarter Kelly for safety)
        if has_edge:
            # Kelly fraction = edge / variance (blackjack variance ≈ 1.3)
            variance = 1.3
            kelly_fraction = predicted_ev / variance
            kelly_quarter = kelly_fraction * 0.25
            kelly_bet = min(
                kelly_quarter * bankroll,
                cfg.betting.max_bet,
                bankroll * 0.05,  # Never bet more than 5% of bankroll
            )
            kelly_bet = max(kelly_bet, cfg.betting.min_bet)
        else:
            kelly_bet = cfg.betting.min_bet

        # 3. Count-based bet ramp (the "spread")
        spread = cfg.betting.bet_spread
        bet_units = 1
        for tc_threshold in sorted(spread.keys()):
            if tc >= tc_threshold:
                bet_units = spread[tc_threshold]

        spread_bet = min(cfg.betting.min_bet * bet_units, cfg.betting.max_bet)

        # 4. Final recommendation: blend Kelly and spread
        # If model is well-trained (R²>0.01), lean on Kelly; otherwise use spread
        if self.ols_results and self.ols_results.r_squared > 0.001:
            recommended = (0.5 * kelly_bet + 0.5 * spread_bet) if has_edge else cfg.betting.min_bet
        else:
            recommended = spread_bet if has_edge else cfg.betting.min_bet

        recommended = max(recommended, cfg.betting.min_bet)
        recommended = min(recommended, cfg.betting.max_bet)
        recommended = min(recommended, bankroll)

        # Round to nearest min_bet unit for practicality
        recommended = round(recommended / cfg.betting.min_bet) * cfg.betting.min_bet

        # 5. Index plays (Hi-Lo specific — deviations from basic strategy)
        index_plays = []
        if hasattr(counter, "get_index_plays"):
            index_plays = counter.get_index_plays()

        return BetRecommendation(
            recommended_bet=recommended,
            min_bet=cfg.betting.min_bet,
            max_bet=cfg.betting.max_bet,
            true_count=tc,
            running_count=rc,
            predicted_ev=predicted_ev,
            player_has_edge=has_edge,
            kelly_bet=kelly_bet,
            bet_units=bet_units,
            decks_remaining=decks_left,
            penetration=pen,
            index_plays=index_plays,
            confidence=confidence,
        )

    def should_take_insurance(self, counter: "BaseCounter") -> bool:
        """
        Insurance is profitable when TC >= +3 (Hi-Lo).
        More than 1/3 of remaining cards are 10-value.
        """
        return counter.true_count >= 3.0

    def wonging_threshold(self) -> float:
        """
        Back-counting (Wong): enter shoe only when TC >= this threshold.
        Standard: enter at TC >= +1, leave at TC <= -1.
        """
        return 1.0

    def should_wong_in(self, counter: "BaseCounter") -> bool:
        """True if conditions are favourable to start playing."""
        return counter.true_count >= self.wonging_threshold()

    def should_wong_out(self, counter: "BaseCounter") -> bool:
        """True if conditions are so negative we should leave the table."""
        return counter.true_count <= -2.0
