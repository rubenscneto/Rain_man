"""
Knock-Out (KO) Card Counting System — Unbalanced.

Tag values:
  2, 3, 4, 5, 6, 7  →  +1  (7 is added vs Hi-Lo!)
  8, 9              →   0
  10, J, Q, K, A   →  -1

Because 7 is included as +1, the full-deck sum = +4 per deck (NOT zero).
This means it's UNBALANCED — no true count conversion needed.
Instead, a "key count" (IRC = Initial Running Count) is used.

Advantage: Simpler in live play — no division needed.
Disadvantage: Slightly less accurate than Hi-Lo for bet sizing.

Reference: "Knock-Out Blackjack" (Fucito & Renzey, 1998).
"""

from simulator.deck import Card, Shoe
from counting.base_counter import BaseCounter


KO_TAGS: dict[str, int] = {
    "2": +1, "3": +1, "4": +1, "5": +1, "6": +1, "7": +1,
    "8":  0, "9":  0,
    "10": -1, "J": -1, "Q": -1, "K": -1, "A": -1,
}


class KOCounter(BaseCounter):
    """
    Knock-Out (KO) unbalanced counting system.

    Uses a "key count" threshold instead of true count.
    IRC (Initial Running Count) = -4 * (num_decks - 1)
    Pivot point (where player has edge) ≈ +1 for 1 deck, +4 for 6 decks.
    """

    def __init__(self, shoe: Shoe):
        super().__init__(shoe)
        # KO starts with a negative IRC to compensate for unbalanced tags
        self._irc = -4 * (shoe.num_decks - 1)
        self._running_count = self._irc
        self._pivot = 4 * (shoe.num_decks - 1)

    @property
    def name(self) -> str:
        return "KO (Knock-Out)"

    @property
    def is_balanced(self) -> bool:
        return False  # Intentionally unbalanced: sum = +4 per deck

    @property
    def true_count(self) -> float:
        """
        KO doesn't use a traditional true count.
        We approximate it for compatibility with the OLS model.
        """
        decks = self.shoe.decks_remaining
        if decks <= 0:
            return 0.0
        # Convert running count back to a Hi-Lo equivalent true count
        # by adjusting for the irc offset
        adjusted_rc = self._running_count - self._irc
        return adjusted_rc / decks

    @property
    def key_count(self) -> int:
        """
        The KO key count for the current number of decks.
        When RC >= key_count, the player has an edge.
        """
        return self._pivot

    @property
    def has_edge(self) -> bool:
        """True when the running count indicates a player advantage."""
        return self._running_count >= self.key_count

    def reset(self) -> None:
        """Reset to IRC for new shoe."""
        self._running_count = self._irc

    def card_value(self, card: Card) -> int:
        return KO_TAGS.get(card.rank, 0)

    def see_card(self, card: Card) -> None:
        self._running_count += self.card_value(card)

    def __repr__(self) -> str:
        edge_str = "EDGE" if self.has_edge else "no edge"
        return (f"KO(RC={self._running_count}, IRC={self._irc}, "
                f"key={self.key_count}, {edge_str})")
