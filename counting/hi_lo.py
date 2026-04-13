"""
Hi-Lo Card Counting System (the most widely used balanced system).

Tag values:
  2, 3, 4, 5, 6  →  +1  (low cards, favour dealer — we want them out)
  7, 8, 9        →   0  (neutral)
  10, J, Q, K, A →  -1  (high cards, favour player — we want them in)

A positive true count means MORE high cards remain in the shoe → player advantage.
A negative true count means MORE low cards remain → house advantage.

Expected edge per true count unit ≈ +0.5% per TC point above zero.

Reference: "Beat the Dealer" (Thorp, 1966);
           "The World's Greatest Blackjack Book" (Humble & Cooper, 1980).
"""

from simulator.deck import Card
from counting.base_counter import BaseCounter

# Hi-Lo tag for each rank
HI_LO_TAGS: dict[str, int] = {
    "2": +1, "3": +1, "4": +1, "5": +1, "6": +1,
    "7":  0, "8":  0, "9":  0,
    "10": -1, "J": -1, "Q": -1, "K": -1, "A": -1,
}


class HiLoCounter(BaseCounter):
    """
    Hi-Lo balanced counting system.

    This is the workhorse of card counting — simple to learn, powerful enough
    to give a long-run edge. The true count feeds directly into the OLS model
    as the primary regressor: E[return | TC] = α + β·TC.
    """

    @property
    def name(self) -> str:
        return "Hi-Lo"

    @property
    def is_balanced(self) -> bool:
        return True  # Sum of all 52 tags = 0

    def card_value(self, card: Card) -> int:
        return HI_LO_TAGS.get(card.rank, 0)

    def see_card(self, card: Card) -> None:
        self._running_count += self.card_value(card)

    # --- Deviations from basic strategy based on true count ---
    # These index plays (departures from basic strategy) are ordered by
    # frequency and profitability ("Illustrious 18" from Don Schlesinger).
    def get_index_plays(self) -> list[dict]:
        """
        Return the top index plays for the current true count.
        These are deviations from basic strategy that are profitable
        when the true count crosses certain thresholds.
        """
        tc = self.true_count
        plays = []

        # Illustrious 18 — most important index plays
        INDEX_PLAYS = [
            # (player_hand, dealer_upcard, threshold, action_above, action_below)
            # e.g. Insurance: take if TC >= +3
            {"name": "Insurance",    "threshold": 3,  "direction": ">="},
            {"name": "16 vs 10",     "threshold": 0,  "direction": ">=", "note": "Stand"},
            {"name": "15 vs 10",     "threshold": 4,  "direction": ">=", "note": "Stand"},
            {"name": "10 vs 10",     "threshold": 4,  "direction": ">=", "note": "Double"},
            {"name": "12 vs 3",      "threshold": 2,  "direction": ">=", "note": "Stand"},
            {"name": "12 vs 2",      "threshold": 3,  "direction": ">=", "note": "Stand"},
            {"name": "11 vs A",      "threshold": 1,  "direction": ">=", "note": "Double"},
            {"name": "9 vs 2",       "threshold": 1,  "direction": ">=", "note": "Double"},
            {"name": "10 vs A",      "threshold": 4,  "direction": ">=", "note": "Double"},
            {"name": "9 vs 7",       "threshold": 3,  "direction": ">=", "note": "Double"},
            {"name": "16 vs 9",      "threshold": 5,  "direction": ">=", "note": "Stand"},
            {"name": "13 vs 2",      "threshold": -1, "direction": "<=", "note": "Hit"},
            {"name": "12 vs 4",      "threshold": 0,  "direction": ">=", "note": "Stand"},
            {"name": "12 vs 5",      "threshold": -2, "direction": "<=", "note": "Hit"},
            {"name": "12 vs 6",      "threshold": -1, "direction": "<=", "note": "Hit"},
        ]

        for play in INDEX_PLAYS:
            threshold = play["threshold"]
            direction = play["direction"]
            triggered = (tc >= threshold if direction == ">=" else tc <= threshold)
            plays.append({
                "play": play["name"],
                "note": play.get("note", ""),
                "threshold": threshold,
                "triggered": triggered,
                "current_tc": tc,
            })
        return [p for p in plays if p["triggered"]]
