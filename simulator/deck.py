"""
Deck and Shoe simulation.

A shoe (a caixinha) typically holds 4-8 standard 52-card decks.
Physical size (~30cm) is irrelevant to the algorithm — what matters is
num_decks and penetration depth.
"""

import random
from dataclasses import dataclass
from typing import List, Optional


SUITS = ["♠", "♥", "♦", "♣"]
RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

# Blackjack point values
RANK_VALUES: dict[str, int] = {
    "A": 11,  # Ace: 11 or 1 (handled in hand evaluation)
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
    "7": 7, "8": 8, "9": 9,
    "10": 10, "J": 10, "Q": 10, "K": 10,
}


@dataclass(frozen=True)
class Card:
    """Immutable playing card."""
    rank: str
    suit: str

    @property
    def value(self) -> int:
        return RANK_VALUES[self.rank]

    @property
    def is_ace(self) -> bool:
        return self.rank == "A"

    @property
    def is_face_card(self) -> bool:
        return self.rank in ("J", "Q", "K")

    @property
    def is_ten_value(self) -> bool:
        """10-value cards: 10, J, Q, K."""
        return self.value == 10

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"

    def __repr__(self) -> str:
        return f"Card({self.rank}{self.suit})"


def build_deck() -> List[Card]:
    """Build a standard 52-card deck."""
    return [Card(rank, suit) for suit in SUITS for rank in RANKS]


class Shoe:
    """
    Multi-deck shoe — the physical 'caixinha' that holds the cards.

    Parameters
    ----------
    num_decks : int
        Number of 52-card decks in the shoe (typically 4, 6, or 8).
    penetration : float
        Fraction of the shoe dealt before reshuffling (0.0–1.0).
        A penetration of 0.75 means 75% of cards are dealt before reshuffle.
        Higher penetration = more card counting advantage.
    seed : int | None
        Random seed for reproducibility (Monte Carlo training).
    """

    def __init__(self, num_decks: int = 6, penetration: float = 0.75,
                 seed: Optional[int] = None):
        self.num_decks = num_decks
        self.penetration = penetration
        self.rng = random.Random(seed)
        self._cards: List[Card] = []
        self._dealt_count: int = 0
        self._total_cards: int = num_decks * 52
        # Cut card position: reshuffle when this many cards remain
        self._cut_card_position = int(self._total_cards * (1 - penetration))
        self.shuffle()

    def shuffle(self) -> None:
        """Build and shuffle the shoe."""
        self._cards = build_deck() * self.num_decks
        self.rng.shuffle(self._cards)
        self._dealt_count = 0

    @property
    def cards_remaining(self) -> int:
        return len(self._cards)

    @property
    def decks_remaining(self) -> float:
        """Approximate decks remaining — needed for true count calculation."""
        return max(self.cards_remaining / 52.0, 0.5)

    @property
    def needs_reshuffle(self) -> bool:
        """True when the cut card is reached."""
        return self.cards_remaining <= self._cut_card_position

    @property
    def penetration_reached(self) -> float:
        """How deep into the shoe we are (0.0 = fresh, 1.0 = fully dealt)."""
        return self._dealt_count / self._total_cards

    def deal(self) -> Card:
        """Deal one card from the top of the shoe."""
        if not self._cards:
            self.shuffle()
        card = self._cards.pop()
        self._dealt_count += 1
        return card

    def cards_dealt(self) -> int:
        return self._dealt_count

    def __len__(self) -> int:
        return self.cards_remaining

    def __repr__(self) -> str:
        return (f"Shoe(decks={self.num_decks}, remaining={self.cards_remaining}, "
                f"penetration={self.penetration_reached:.1%})")


class Hand:
    """
    A blackjack hand (player or dealer).
    Handles soft totals (Ace = 11 or 1) correctly.
    """

    def __init__(self):
        self.cards: List[Card] = []
        self.is_split_hand: bool = False
        self.doubled: bool = False

    def add_card(self, card: Card) -> None:
        self.cards.append(card)

    @property
    def total(self) -> int:
        """Best hand total — reduces Aces from 11→1 to avoid busting."""
        total = sum(c.value for c in self.cards)
        aces = sum(1 for c in self.cards if c.is_ace)
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1
        return total

    @property
    def is_soft(self) -> bool:
        """True if hand contains an Ace counted as 11."""
        total_hard = sum(c.value if not c.is_ace else 1 for c in self.cards)
        return any(c.is_ace for c in self.cards) and (total_hard + 10 <= 21)

    @property
    def is_bust(self) -> bool:
        return self.total > 21

    @property
    def is_blackjack(self) -> bool:
        return len(self.cards) == 2 and self.total == 21

    @property
    def is_pair(self) -> bool:
        return len(self.cards) == 2 and self.cards[0].value == self.cards[1].value

    @property
    def can_double(self) -> bool:
        return len(self.cards) == 2 and not self.is_split_hand

    def __str__(self) -> str:
        cards_str = " ".join(str(c) for c in self.cards)
        soft_str = " (soft)" if self.is_soft else ""
        return f"[{cards_str}] = {self.total}{soft_str}"

    def __repr__(self) -> str:
        return f"Hand({self.cards}, total={self.total})"
