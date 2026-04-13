"""
Omega II Card Counting System — Multi-level balanced system.

Tag values (level 2 — cards have values beyond ±1):
  2, 3, 7  →  +1
  4, 5, 6  →  +2
  9        →  -1
  10, J, Q, K  →  -2
  8, A     →   0  (Ace is NOT counted — tracked separately)

Because Aces are not counted, a separate Ace side count is maintained.
This system is more powerful than Hi-Lo but harder to use at speed.
Correlation with player edge ≈ 0.92 (vs Hi-Lo ≈ 0.97 for betting,
but Omega II correlates better with playing efficiency).

Reference: "Blackjack for Blood" (Snyder, 1983);
           "The World's Greatest Blackjack Book" (Humble & Cooper, 1980).
"""

from simulator.deck import Card, Shoe
from counting.base_counter import BaseCounter


OMEGA2_TAGS: dict[str, int] = {
    "2": +1, "3": +1, "7": +1,
    "4": +2, "5": +2, "6": +2,
    "9": -1,
    "10": -2, "J": -2, "Q": -2, "K": -2,
    "8":  0, "A": 0,  # Aces tracked separately
}

# Expected aces per deck = 4/52 ≈ 0.0769
ACES_PER_DECK = 4 / 52


class Omega2Counter(BaseCounter):
    """
    Omega II balanced multi-level counting system.

    Maintains a separate Ace side count for bet sizing adjustment.
    The ace-adjusted true count is used as the primary ML feature.
    """

    def __init__(self, shoe: Shoe):
        super().__init__(shoe)
        self._ace_count: int = 0       # Side count of Aces seen
        self._cards_seen: int = 0

    @property
    def name(self) -> str:
        return "Omega II"

    @property
    def is_balanced(self) -> bool:
        return True  # Sum of all 52 tags = 0 (Aces are 0)

    @property
    def ace_side_count(self) -> int:
        return self._ace_count

    @property
    def ace_adjusted_true_count(self) -> float:
        """
        Adjust the true count for Ace richness/poorness.

        If more Aces remain than expected, the deck is "Ace-rich" →
        add to true count (good for player).
        If fewer Aces remain, subtract from true count.
        """
        tc = self.true_count
        decks_remaining = self.shoe.decks_remaining
        # Expected aces remaining
        expected_aces = decks_remaining * 4
        # Actual aces remaining = 4*num_decks - aces_seen
        aces_remaining = (self.shoe.num_decks * 4) - self._ace_count
        ace_deviation = (aces_remaining - expected_aces) / decks_remaining
        return tc + ace_deviation

    def reset(self) -> None:
        super().reset()
        self._ace_count = 0
        self._cards_seen = 0

    def card_value(self, card: Card) -> int:
        return OMEGA2_TAGS.get(card.rank, 0)

    def see_card(self, card: Card) -> None:
        self._running_count += self.card_value(card)
        self._cards_seen += 1
        if card.is_ace:
            self._ace_count += 1

    def __repr__(self) -> str:
        return (f"Omega2(RC={self._running_count}, "
                f"TC={self.true_count:+.2f}, "
                f"ace_adj_TC={self.ace_adjusted_true_count:+.2f}, "
                f"aces_seen={self._ace_count})")
