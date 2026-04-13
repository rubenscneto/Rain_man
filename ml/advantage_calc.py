"""
Exact Player Advantage Calculator.

Based on Griffin's formula as presented in Humble & Cooper,
"The World's Greatest Blackjack Book", Chapter 7-8:

    Player Advantage (%) = BSE + 0.515 × TC + SG

Where:
  BSE = Basic Strategy Expectation (depends on casino rules)
  TC  = Hi-Opt I True Count (running count / decks remaining)
  SG  = Strategy Gain from index play deviations (≈0 when BSE already assumed)
  0.515 = value of each true-count unit (Griffin's calculation, p.516)

"Every increment in the Hi-Opt I true count is worth approximately
0.5% (0.515) to the player." — Humble & Cooper, p.518

This module provides:
1. BSE lookup for common casino rule combinations
2. Exact advantage calculation
3. Kelly Criterion bet sizing using the exact advantage
4. Breakeven true count for each rule set
5. Comparison across counting systems
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Basic Strategy Expectations (BSE) by Rule Set
# Source: Humble & Cooper p.526 and standard blackjack EVs
# All values in % (positive = player advantage, negative = house edge)
# ---------------------------------------------------------------------------

@dataclass
class CasinoRules:
    """Description of a casino's blackjack rule set."""
    name: str
    num_decks: int
    dealer_hits_soft17: bool      # H17 (True) or S17 (False)
    double_after_split: bool      # DAS
    resplit_aces: bool
    surrender: str                # "none", "late", "early"
    blackjack_payout: float       # 1.5 = 3:2; 1.2 = 6:5
    bse: float                    # Basic Strategy Expectation (%)

    @property
    def breakeven_true_count(self) -> float:
        """Hi-Opt I TC needed to reach EV = 0 (break-even)."""
        return -self.bse / 0.515

    @property
    def advantage_at_tc(self):
        """Return a function: TC → player advantage %."""
        bse = self.bse
        return lambda tc: bse + 0.515 * tc


# Catalogue of common rule sets with their BSEs
CASINO_RULES: dict[str, CasinoRules] = {
    # Single-deck games (historical, rare today)
    "caesars_1d": CasinoRules(
        name="Caesars Palace 1-Deck",
        num_decks=1, dealer_hits_soft17=False,
        double_after_split=True, resplit_aces=False,
        surrender="late", blackjack_payout=1.5,
        bse=+0.17,   # Book p.529
    ),
    "stardust_1d": CasinoRules(
        name="Stardust 1-Deck",
        num_decks=1, dealer_hits_soft17=False,
        double_after_split=False, resplit_aces=False,
        surrender="none", blackjack_payout=1.5,
        bse=0.00,    # Book p.529
    ),
    "lv_downtown_1d": CasinoRules(
        name="Las Vegas Downtown 1-Deck H17",
        num_decks=1, dealer_hits_soft17=True,
        double_after_split=False, resplit_aces=False,
        surrender="none", blackjack_payout=1.5,
        bse=-0.19,   # Book p.529
    ),
    # Multi-deck (modern standard)
    "mgm_4d": CasinoRules(
        name="MGM Grand 4-Deck H17",
        num_decks=4, dealer_hits_soft17=True,
        double_after_split=True, resplit_aces=False,
        surrender="late", blackjack_payout=1.5,
        bse=-0.34,   # Book p.529
    ),
    "typical_6d_h17_das": CasinoRules(
        name="Typical 6-Deck H17 DAS Late Surrender",
        num_decks=6, dealer_hits_soft17=True,
        double_after_split=True, resplit_aces=False,
        surrender="late", blackjack_payout=1.5,
        bse=-0.54,   # Industry standard
    ),
    "typical_6d_s17_das": CasinoRules(
        name="Typical 6-Deck S17 DAS Late Surrender",
        num_decks=6, dealer_hits_soft17=False,
        double_after_split=True, resplit_aces=False,
        surrender="late", blackjack_payout=1.5,
        bse=-0.34,   # S17 is better for player
    ),
    "typical_8d_h17": CasinoRules(
        name="Typical 8-Deck H17 DAS",
        num_decks=8, dealer_hits_soft17=True,
        double_after_split=True, resplit_aces=False,
        surrender="late", blackjack_payout=1.5,
        bse=-0.57,
    ),
    "atlantic_city_8d": CasinoRules(
        name="Atlantic City 8-Deck S17 DAS LS",
        num_decks=8, dealer_hits_soft17=False,
        double_after_split=True, resplit_aces=False,
        surrender="late", blackjack_payout=1.5,
        bse=-0.43,
    ),
    "bad_rules_6to5": CasinoRules(
        name="6:5 Single-Deck (trap game!)",
        num_decks=1, dealer_hits_soft17=True,
        double_after_split=False, resplit_aces=False,
        surrender="none", blackjack_payout=1.2,  # 6:5 payout!
        bse=-1.39,   # The 6:5 BJ rule alone costs ~1.4%
    ),
}

DEFAULT_RULES = CASINO_RULES["typical_6d_h17_das"]


# ---------------------------------------------------------------------------
# Advantage Calculator
# ---------------------------------------------------------------------------

@dataclass
class AdvantageReport:
    """Full advantage analysis at a given true count."""
    true_count: float
    bse: float
    advantage_pct: float       # BSE + 0.515 * TC
    per_tc_unit_pct: float = 0.515
    casino_rules: Optional[str] = None
    breakeven_tc: float = 0.0
    kelly_fraction: float = 0.0
    kelly_quarter_fraction: float = 0.0
    recommended_bet_units: float = 0.0  # In terms of min-bet units

    def __str__(self) -> str:
        sign = "+" if self.advantage_pct >= 0 else ""
        edge = "PLAYER EDGE" if self.advantage_pct >= 0 else "house edge"
        return (
            f"TC={self.true_count:+.1f} | "
            f"Advantage: {sign}{self.advantage_pct:.3f}% ({edge}) | "
            f"Kelly: {self.kelly_fraction:.4f} | "
            f"Breakeven TC: {self.breakeven_tc:+.1f}"
        )


class AdvantageCalculator:
    """
    Computes exact player advantage using the Humble & Cooper formula.

    Incorporates:
      1. BSE for the specific casino rule set
      2. Hi-Opt I / Hi-Opt II true count contribution
      3. Kelly Criterion for optimal bet sizing
      4. Rule-adjustment bonuses (surrender, DAS, etc.)
    """

    # Approximate rule-set bonus adjustments (% added to BSE)
    # Source: Humble & Cooper Chapter 6 effects-of-rules table
    RULE_BONUSES: dict[str, float] = {
        "late_surrender":        +0.06,   # Per hand, flat betting
        "early_surrender":       +0.62,   # Very favourable, rare
        "das_allowed":           +0.14,   # Double after split
        "resplit_aces":          +0.05,
        "s17_vs_h17":            +0.20,   # Stand soft 17 better for player
        "blackjack_3to2":         0.00,   # Baseline
        "blackjack_6to5":        -1.39,   # Terrible — avoid these tables!
        "no_hole_card":          -0.11,   # European no-hole-card rule
        "ace_side_count":        +0.20,   # Hi-Opt I with Ace tracking
        "count_strategy_gain_tc1": +0.1,  # Strategy deviations at TC +1
        "count_strategy_gain_tc2": +0.3,  # Strategy deviations at TC +2
        "count_strategy_gain_tc3": +0.6,  # Strategy deviations at TC +3
    }

    def __init__(self, rules: CasinoRules = DEFAULT_RULES,
                 variance: float = 1.3):
        """
        Parameters
        ----------
        rules : CasinoRules
            The casino's rule set.
        variance : float
            Blackjack outcome variance per hand (empirical ≈ 1.3).
            Used for Kelly Criterion.
        """
        self.rules = rules
        self.variance = variance

    @classmethod
    def for_casino(cls, casino_key: str) -> "AdvantageCalculator":
        """Create calculator for a named casino rule set."""
        rules = CASINO_RULES.get(casino_key, DEFAULT_RULES)
        return cls(rules)

    def player_advantage(self, true_count: float,
                          include_strategy_gain: bool = True) -> float:
        """
        Compute player advantage at a given true count.

        Formula (Humble & Cooper p.514-624):
            Advantage (%) = BSE + 0.515 × TC [+ SG]

        Parameters
        ----------
        true_count : float
        include_strategy_gain : bool
            Include approximate strategy gain from index deviations.

        Returns
        -------
        float : advantage in % (positive = player has edge)
        """
        advantage = self.rules.bse + 0.515 * true_count
        if include_strategy_gain and abs(true_count) > 1:
            # Approximate strategy gain (Griffin's curve, p.250)
            # Quadratic approximation: peaks at ~6% gain at TC ±10
            sg = 0.015 * (abs(true_count) - 1) ** 1.5
            advantage += sg
        return advantage

    def kelly_bet(self, true_count: float,
                  bankroll: float,
                  min_bet: float,
                  max_bet: float,
                  kelly_fraction: float = 0.25) -> float:
        """
        Compute Kelly Criterion bet.

        Full Kelly: f* = edge / variance
        Fractional Kelly (recommended): f = (edge / variance) × fraction

        Humble & Cooper (p.497-502) recommend 1/4 Kelly for safety.
        Full Kelly maximises geometric growth but causes large variance.

        Parameters
        ----------
        true_count : float
        bankroll : float
        min_bet, max_bet : float
        kelly_fraction : float
            Fraction of full Kelly to bet (0.25 = quarter Kelly).
        """
        edge = self.player_advantage(true_count) / 100.0
        if edge <= 0:
            return min_bet  # No edge → minimum bet

        full_kelly = edge / self.variance
        fractional_kelly = full_kelly * kelly_fraction
        bet = fractional_kelly * bankroll

        # Apply table limits and safety cap
        bet = max(bet, min_bet)
        bet = min(bet, max_bet)
        bet = min(bet, bankroll * 0.05)  # Never risk more than 5% of bankroll
        return bet

    def report(self, true_count: float,
               bankroll: float = 1000.0,
               min_bet: float = 10.0,
               max_bet: float = 200.0) -> AdvantageReport:
        """Generate a full advantage report at the given true count."""
        adv = self.player_advantage(true_count)
        edge = adv / 100.0
        kelly_full = max(edge, 0) / self.variance
        kelly_quarter = kelly_full * 0.25
        bet = self.kelly_bet(true_count, bankroll, min_bet, max_bet)

        return AdvantageReport(
            true_count=true_count,
            bse=self.rules.bse,
            advantage_pct=adv,
            per_tc_unit_pct=0.515,
            casino_rules=self.rules.name,
            breakeven_tc=self.rules.breakeven_true_count,
            kelly_fraction=kelly_full,
            kelly_quarter_fraction=kelly_quarter,
            recommended_bet_units=bet / min_bet,
        )

    def print_ev_table(self, tc_range: range = range(-5, 9)) -> None:
        """Print EV at each true count level."""
        print(f"\n{'':=<60}")
        print(f"  ADVANTAGE TABLE — {self.rules.name}")
        print(f"  BSE={self.rules.bse:+.2f}% | "
              f"Break-even TC={self.rules.breakeven_true_count:+.1f}")
        print(f"{'':=<60}")
        print(f"  {'TC':>5}  {'Advantage':>10}  {'Edge':>12}  {'Bet (1k bankroll)':>18}")
        print(f"  {'-'*5}  {'-'*10}  {'-'*12}  {'-'*18}")
        for tc in tc_range:
            adv = self.player_advantage(float(tc))
            bet = self.kelly_bet(float(tc), 1000.0, 10.0, 200.0)
            edge_str = "PLAYER" if adv > 0 else "house "
            adv_str = f"{adv:+.3f}%"
            bar_len = int(abs(adv) * 10)
            bar = ("▲" if adv >= 0 else "▼") * min(bar_len, 12)
            print(f"  TC {tc:>+3}  {adv_str:>10}  {edge_str:>12}  "
                  f"${bet:>6.0f}  {bar}")
        print(f"{'':=<60}")

    @staticmethod
    def compare_rule_sets() -> None:
        """Print break-even TCs for all known rule sets."""
        print("\n" + "=" * 70)
        print("  RULE SET COMPARISON — Break-even True Count (Hi-Opt I)")
        print("=" * 70)
        print(f"  {'Casino / Rules':<40} {'BSE':>6}  {'Break-even TC':>13}")
        print(f"  {'-'*40}  {'-'*6}  {'-'*13}")
        for key, rules in sorted(CASINO_RULES.items(),
                                  key=lambda x: x[1].bse, reverse=True):
            btc = rules.breakeven_true_count
            print(f"  {rules.name:<40} {rules.bse:>+6.2f}%  {btc:>+10.1f}")
        print("=" * 70)
