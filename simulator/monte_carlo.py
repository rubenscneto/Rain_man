"""
Monte Carlo Simulation Engine.

Simulates thousands of blackjack hands to:
1. Estimate Expected Value (EV) at each true count level.
2. Build the training dataset for OLS regression.
3. Calculate Risk of Ruin (RoR) for a given bankroll/bet-spread.
4. Estimate optimal bet sizing.

Monte Carlo is the "training phase" — it runs entirely offline before
any live play, producing the regression coefficients the live system uses.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from simulator.deck import Shoe
from simulator.blackjack_game import BlackjackGame, basic_strategy_action
from config import RainManConfig, DEFAULT_CONFIG

if TYPE_CHECKING:
    from counting.base_counter import BaseCounter


@dataclass
class SimulationRecord:
    """One simulated hand's data — becomes a row in the training DataFrame."""
    hand_num: int
    running_count: int
    true_count: float
    decks_remaining: float
    player_total: int
    dealer_upcard: int
    is_soft: bool
    bet_placed: float
    net_result: float          # +bet or -bet or 0 (push)
    return_on_bet: float       # net_result / bet_placed  ∈ {-1, 0, +1, +1.5}
    penetration: float         # how deep into the shoe


@dataclass
class SimulationSummary:
    """Aggregate statistics from a full Monte Carlo run."""
    total_hands: int = 0
    total_wagered: float = 0.0
    total_net: float = 0.0
    house_edge_percent: float = 0.0   # Before counting
    player_edge_percent: float = 0.0  # With counting strategy
    ev_by_true_count: dict = field(default_factory=dict)
    ror_estimate: float = 0.0
    win_rate_per_100: float = 0.0     # Units won per 100 hands
    std_dev_per_hand: float = 0.0
    records: List[SimulationRecord] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert simulation records to a pandas DataFrame for OLS training."""
        if not self.records:
            return pd.DataFrame()
        data = [{
            "hand_num": r.hand_num,
            "running_count": r.running_count,
            "true_count": r.true_count,
            "decks_remaining": r.decks_remaining,
            "player_total": r.player_total,
            "dealer_upcard": r.dealer_upcard,
            "is_soft": int(r.is_soft),
            "bet_placed": r.bet_placed,
            "net_result": r.net_result,
            "return_on_bet": r.return_on_bet,
            "penetration": r.penetration,
        } for r in self.records]
        return pd.DataFrame(data)


def _make_counting_bet(true_count: float, config: RainManConfig) -> float:
    """
    Convert true count to bet size using the configured bet spread.
    This is the count-based betting ramp.
    """
    spread = config.betting.bet_spread
    bet_units = 1
    for tc_threshold in sorted(spread.keys()):
        if true_count >= tc_threshold:
            bet_units = spread[tc_threshold]
    return min(
        config.betting.min_bet * bet_units,
        config.betting.max_bet
    )


class MonteCarloSimulator:
    """
    Runs large-scale blackjack simulations to generate training data.

    Usage
    -----
    simulator = MonteCarloSimulator(config, counter)
    summary = simulator.run()
    df = summary.to_dataframe()   # Feed to OLS model
    """

    def __init__(self, config: RainManConfig = DEFAULT_CONFIG,
                 counter: Optional["BaseCounter"] = None):
        self.config = config
        self.counter = counter
        self._rng = np.random.default_rng(config.monte_carlo.random_seed)

    def run(self, verbose: bool = True) -> SimulationSummary:
        """
        Run the full Monte Carlo simulation.

        Returns a SimulationSummary with all hand records and aggregate stats.
        """
        n = self.config.monte_carlo.num_simulations
        cfg = self.config

        shoe = Shoe(
            num_decks=cfg.shoe.num_decks,
            penetration=cfg.shoe.penetration,
            seed=cfg.monte_carlo.random_seed,
        )
        game = BlackjackGame(shoe, cfg.rules, self.counter)

        records: List[SimulationRecord] = []
        total_wagered = 0.0
        total_net = 0.0

        if verbose:
            print(f"[Monte Carlo] Running {n:,} hands "
                  f"({cfg.shoe.num_decks} decks, "
                  f"penetration={cfg.shoe.penetration:.0%})...")

        for hand_num in range(n):
            if shoe.needs_reshuffle:
                shoe.shuffle()
                if self.counter:
                    self.counter.reset()

            tc = self.counter.true_count if self.counter else 0.0
            rc = self.counter.running_count if self.counter else 0
            decks_left = shoe.decks_remaining
            pen = shoe.penetration_reached

            bet = _make_counting_bet(tc, cfg)

            result = game.play_hand(bet)

            if result.player_hands:
                ph = result.player_hands[0]
                dealer_upcard = result.dealer_hand.cards[0].value
                rec = SimulationRecord(
                    hand_num=hand_num,
                    running_count=rc,
                    true_count=tc,
                    decks_remaining=decks_left,
                    player_total=ph.total,
                    dealer_upcard=dealer_upcard,
                    is_soft=ph.is_soft,
                    bet_placed=bet,
                    net_result=result.net_result,
                    return_on_bet=result.net_result / bet if bet > 0 else 0.0,
                    penetration=pen,
                )
                records.append(rec)
                total_wagered += bet
                total_net += result.net_result

            if verbose and (hand_num + 1) % 10_000 == 0:
                pct = (hand_num + 1) / n * 100
                running_edge = (total_net / total_wagered * 100
                                if total_wagered > 0 else 0)
                print(f"  {pct:.0f}% complete | "
                      f"Edge so far: {running_edge:+.2f}%")

        summary = self._compute_summary(records, total_wagered, total_net)
        if verbose:
            self._print_summary(summary)
        return summary

    def run_risk_of_ruin(self, verbose: bool = True) -> float:
        """
        Estimate Risk of Ruin via session-based simulation.

        Simulates `num_sessions` sessions of `hands_per_session` hands.
        RoR = fraction of sessions that go bankrupt.
        """
        cfg = self.config
        n_sessions = cfg.monte_carlo.num_sessions
        h_per_session = cfg.monte_carlo.hands_per_session
        starting_bankroll = cfg.betting.bankroll

        ruins = 0

        for _ in range(n_sessions):
            bankroll = starting_bankroll
            shoe = Shoe(
                num_decks=cfg.shoe.num_decks,
                penetration=cfg.shoe.penetration,
                seed=None,  # Different seed each session
            )
            if self.counter:
                self.counter.reset()
            game = BlackjackGame(shoe, cfg.rules, self.counter)

            for _ in range(h_per_session):
                if bankroll < cfg.betting.min_bet:
                    ruins += 1
                    break
                if shoe.needs_reshuffle:
                    shoe.shuffle()
                    if self.counter:
                        self.counter.reset()
                tc = self.counter.true_count if self.counter else 0.0
                bet = min(_make_counting_bet(tc, cfg), bankroll)
                result = game.play_hand(bet)
                bankroll += result.net_result

        ror = ruins / n_sessions
        if verbose:
            print(f"[RoR] Risk of Ruin: {ror:.1%} "
                  f"over {n_sessions} sessions of {h_per_session} hands "
                  f"(bankroll={starting_bankroll} units)")
        return ror

    def _compute_summary(self, records: List[SimulationRecord],
                          total_wagered: float,
                          total_net: float) -> SimulationSummary:
        summary = SimulationSummary()
        summary.total_hands = len(records)
        summary.total_wagered = total_wagered
        summary.total_net = total_net
        summary.records = records

        if total_wagered > 0:
            summary.house_edge_percent = -(total_net / total_wagered) * 100
            summary.player_edge_percent = (total_net / total_wagered) * 100

        returns = np.array([r.return_on_bet for r in records])
        summary.std_dev_per_hand = float(np.std(returns))
        summary.win_rate_per_100 = (
            float(np.mean(returns)) * 100
            if len(returns) > 0 else 0.0
        )

        # EV by true count bucket (integers -5 to +10)
        ev_by_tc: dict[int, list] = {}
        for r in records:
            tc_bucket = int(np.clip(np.round(r.true_count), -5, 10))
            ev_by_tc.setdefault(tc_bucket, []).append(r.return_on_bet)

        summary.ev_by_true_count = {
            tc: float(np.mean(vals))
            for tc, vals in sorted(ev_by_tc.items())
        }
        return summary

    def _print_summary(self, s: SimulationSummary) -> None:
        print("\n" + "=" * 55)
        print("  MONTE CARLO SIMULATION RESULTS")
        print("=" * 55)
        print(f"  Hands simulated   : {s.total_hands:>12,}")
        print(f"  Total wagered     : {s.total_wagered:>12,.1f} units")
        print(f"  Net result        : {s.total_net:>+12,.1f} units")
        print(f"  Player edge       : {s.player_edge_percent:>+11.3f}%")
        print(f"  Win rate/100 hands: {s.win_rate_per_100:>+11.3f} units")
        print(f"  Std dev per hand  : {s.std_dev_per_hand:>12.4f}")
        print()
        print("  EV by True Count:")
        print("  " + "-" * 30)
        for tc, ev in s.ev_by_true_count.items():
            bar_len = int(abs(ev) * 200)
            bar = ("+" if ev >= 0 else "-") * min(bar_len, 20)
            print(f"  TC {tc:>+3}: {ev:>+.4f}  {bar}")
        print("=" * 55)
