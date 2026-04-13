"""
Hi-Opt II Card Counting System — Level 2.

Based on "The World's Greatest Blackjack Book" (Humble & Cooper).
Performance (Braun simulation, p.243-244):
  - Playing Efficiency:  0.671  (best in class)
  - Betting Correlation: 0.91
  - 1-deck flat betting:  0.9%
  - 1-deck 1:4 spread:    2.3%
  - 1-deck 1:8 spread:    3.1%

"The Hi-Opt II is slightly more complex [than Hi-Opt I]. It is a
second-level system counting three more cards than the Hi-Opt I."
— Humble & Cooper, p.362

Tag values (level 2 — cards have values ±1 and ±2):
  2, 3, 6, 7   →  +1
  4, 5         →  +2  (the most powerful low cards — note the doubled tag)
  10, J, Q, K  →  -2  (also doubled compared to Hi-Opt I)
  8, 9, A      →   0  (neutral; Ace tracked separately)

Because the deck sums to 0 (4×1 + 4×1 + 4×1 + 4×1 + 4×2 + 4×2
                              - 4×2 - 4×2 - 4×2 - 4×2 = 0),
Hi-Opt II is balanced. True count = RC / decks_remaining.

Hi-Opt II is the "preferred system among professional Blackjack players"
according to Humble & Cooper (p.364), but requires more mental effort
than Hi-Opt I due to the multi-level tags. Recommended for experienced
counters who already know Hi-Opt I well.
"""

from __future__ import annotations
from simulator.deck import Card, Shoe
from counting.base_counter import BaseCounter
from counting.hi_opt1 import BSE_TABLE


# ---------------------------------------------------------------------------
# Hi-Opt II Card Tags (Level 2)
# ---------------------------------------------------------------------------

HI_OPT2_TAGS: dict[str, int] = {
    "2": +1,
    "3": +1,
    "4": +2,   # Doubled — 4 and 5 are the most powerful low cards
    "5": +2,   # Doubled
    "6": +1,
    "7": +1,
    "8":  0,
    "9":  0,
    "10": -2,  # Doubled — 10-value cards strongly favour dealer
    "J":  -2,
    "Q":  -2,
    "K":  -2,
    "A":   0,  # Not counted — tracked by Ace side count
}

# Full-deck sum check: 4(+1)+4(+1)+4(+2)+4(+2)+4(+1)+4(+1) - 4(2)-4(2)-4(2)-4(2) = 0 ✓


class HiOpt2Counter(BaseCounter):
    """
    Hi-Opt II Level-2 counting system.

    More powerful than Hi-Opt I for both betting and playing efficiency.
    Recommended for players who have already mastered Hi-Opt I and want
    to extract the maximum possible edge.

    Advantage formula same as Hi-Opt I but calibrated for Hi-Opt II's
    stronger correlation:
        Advantage (%) = BSE + 0.515 × TC   (approximately — TC here is
        already more precise due to better betting correlation 0.91 vs 0.86)
    """

    def __init__(self, shoe: Shoe, bse: float = -0.54,
                 use_ace_side_count: bool = True):
        """
        Parameters
        ----------
        shoe : Shoe
        bse : float
            Basic Strategy Expectation (default: -0.54% for 6-deck H17).
        use_ace_side_count : bool
            Track Aces separately (recommended). Default: True.
        """
        super().__init__(shoe)
        self.bse = bse
        self.use_ace_side_count = use_ace_side_count
        self._ace_count: int = 0
        self._cards_seen: int = 0

    @property
    def name(self) -> str:
        return "Hi-Opt II"

    @property
    def is_balanced(self) -> bool:
        return True

    def reset(self) -> None:
        super().reset()
        self._ace_count = 0
        self._cards_seen = 0

    def card_value(self, card: Card) -> int:
        return HI_OPT2_TAGS.get(card.rank, 0)

    def see_card(self, card: Card) -> None:
        self._running_count += self.card_value(card)
        self._cards_seen += 1
        if card.is_ace:
            self._ace_count += 1

    @property
    def ace_side_count(self) -> int:
        return self._ace_count

    @property
    def ace_adjusted_true_count(self) -> float:
        """
        True count adjusted for Ace richness.
        Same logic as Hi-Opt I — Aces are neutral in the main count
        but their presence matters for bet sizing.
        """
        tc = self.true_count
        if not self.use_ace_side_count:
            return tc
        decks_remaining = self.shoe.decks_remaining
        expected_aces = decks_remaining * 4
        aces_remaining = (self.shoe.num_decks * 4) - self._ace_count
        ace_deviation = (aces_remaining - expected_aces) / decks_remaining
        return tc + ace_deviation

    @property
    def player_advantage(self) -> float:
        """
        Exact player advantage:
            Advantage (%) = BSE + 0.515 × ace_adjusted_TC

        Hi-Opt II's 0.91 betting correlation (vs Hi-Opt I's 0.86) means
        the true count more accurately reflects the actual deck composition,
        making the edge calculation more reliable.
        """
        return self.bse + 0.515 * self.ace_adjusted_true_count

    def should_take_insurance(self) -> bool:
        """Insurance profitable when TC >= +3 (same threshold as Hi-Opt I)."""
        return self.true_count >= 3

    @property
    def level(self) -> int:
        return 2  # Level-2 system (cards have values ±1 and ±2)

    def __repr__(self) -> str:
        return (f"HiOpt2(RC={self._running_count}, "
                f"TC={self.true_count:+.2f}, "
                f"adv={self.player_advantage:+.3f}%, "
                f"aces_seen={self._ace_count})")
