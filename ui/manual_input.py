"""
Manual Card Input.

Fallback when screen reading is unavailable or fails.
The user types cards as they are dealt (e.g., "AS", "10H", "KD").

Supported formats:
  AS, A♠         → Ace of Spades
  10H, 10♥       → 10 of Hearts
  KD, K♦         → King of Diamonds
  2C, 2♣         → 2 of Clubs
  done, d        → end of round
  shuffle, s     → shoe reshuffled
  quit, q        → exit
"""

from __future__ import annotations
from typing import List, Optional
from simulator.deck import Card, RANKS
from screen_reader.card_detector import parse_card_text, DetectedCard


SUIT_MAP = {
    "S": "♠", "H": "♥", "D": "♦", "C": "♣",
    "♠": "♠", "♥": "♥", "♦": "♦", "♣": "♣",
}

HELP_TEXT = """
Card Input Format:
  Rank + Suit:  AS=A♠, KH=K♥, QD=Q♦, JC=J♣
  10s:          10S, 10H, 10D, 10C
  Number cards: 2S, 3H ... 9C

Commands:
  done / d      → Signal end of hand (start counting)
  shuffle / s   → Shoe was reshuffled (reset count)
  count / c     → Show current count
  help / ?      → Show this help
  quit / q      → Exit live mode
"""


class ManualCardInput:
    """
    Interactive manual card input handler.

    Prompts the user to type cards as they appear on the table.
    Validates input and converts to Card objects.
    """

    def __init__(self, rich_console=None):
        self.console = rich_console  # Optional Rich console for pretty output

    def prompt_cards(self, prompt: str = "Cards dealt") -> List[Card]:
        """
        Prompt user to enter all cards for a position (player hand or dealer).
        Returns a list of Card objects.

        Example input: "AS 10H" → [Card(A♠), Card(10♥)]
        """
        self._print(f"\n{prompt} (e.g. 'AS 10H' or type one card at a time, 'done' to finish):")
        cards = []
        while True:
            try:
                raw = input("  > ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            cmd = raw.lower()
            if cmd in ("done", "d", ""):
                break
            if cmd in ("help", "?"):
                self._print(HELP_TEXT)
                continue
            if cmd in ("quit", "q", "exit"):
                return []

            # Parse one or more cards from the input
            tokens = raw.split()
            for token in tokens:
                card = self._parse_single(token)
                if card:
                    cards.append(card)
                    self._print(f"  + {card} added")
                else:
                    self._print(f"  ? Could not parse '{token}' — try 'AS', '10H', 'KD'")

        return cards

    def prompt_single_card(self, prompt: str = "Enter card") -> Optional[Card]:
        """Prompt for exactly one card."""
        while True:
            try:
                raw = input(f"  {prompt} > ").strip()
            except (EOFError, KeyboardInterrupt):
                return None
            cmd = raw.lower()
            if cmd in ("quit", "q"):
                return None
            if cmd in ("help", "?"):
                self._print(HELP_TEXT)
                continue
            card = self._parse_single(raw)
            if card:
                return card
            self._print(f"  ? Cannot parse '{raw}'. Format: AS, 10H, KD, 2C")

    def prompt_command(self, prompt: str = "Command") -> str:
        """
        General command prompt. Returns the raw command string.
        Used in the main game loop.
        """
        try:
            return input(f"{prompt} > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "quit"

    def _parse_single(self, text: str) -> Optional[Card]:
        """Parse a single card string into a Card object."""
        text = text.upper().strip()
        if not text:
            return None

        # Normalize suit letters
        for letter, symbol in SUIT_MAP.items():
            if text.endswith(letter) and len(text) > 1:
                text = text[:-len(letter)] + symbol
                break

        result = parse_card_text(text)
        if result:
            rank, suit = result
            if rank in RANKS and suit in ("♠", "♥", "♦", "♣"):
                return Card(rank, suit)
        return None

    def _print(self, msg: str) -> None:
        if self.console:
            self.console.print(msg)
        else:
            print(msg)
