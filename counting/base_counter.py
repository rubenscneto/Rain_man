"""
Abstract base class for all card counting systems.

All counting systems share the same interface:
  - see_card(card)  : update the running count
  - true_count      : property returning count adjusted for remaining decks
  - reset()         : shoe was reshuffled
"""

from abc import ABC, abstractmethod
from simulator.deck import Card, Shoe


class BaseCounter(ABC):
    """Abstract card counter."""

    def __init__(self, shoe: Shoe):
        self.shoe = shoe
        self._running_count: int = 0

    @property
    def running_count(self) -> int:
        return self._running_count

    @property
    def true_count(self) -> float:
        """
        True Count = Running Count / Decks Remaining.

        This normalization (invented for Hi-Lo by Stanford Wong) allows
        comparing counts across different shoe depths.
        Gauss-Markov requires this standardization for unbiased OLS estimates.
        """
        decks = self.shoe.decks_remaining
        if decks <= 0:
            return 0.0
        return self._running_count / decks

    def reset(self) -> None:
        """Call when the shoe is reshuffled."""
        self._running_count = 0

    @abstractmethod
    def see_card(self, card: Card) -> None:
        """Update count when a card is observed."""
        ...

    @abstractmethod
    def card_value(self, card: Card) -> int:
        """Return the count tag for a given card (+1, 0, -1, -2, etc.)."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def is_balanced(self) -> bool:
        """
        A balanced system (e.g., Hi-Lo) sums to 0 over a full deck.
        An unbalanced system (e.g., KO) does not require true count conversion.
        """
        ...

    def __repr__(self) -> str:
        return (f"{self.name}(RC={self._running_count}, "
                f"TC={self.true_count:+.2f}, "
                f"decks_left={self.shoe.decks_remaining:.1f})")
