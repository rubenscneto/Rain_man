"""
Fast Hotkey Input Mode — single keypress per card, no Enter needed.

This is the PRIMARY input method for live dealer games where typing
full card notation ("AS", "10H") is too slow.

Key Map (designed for right-hand, number row):
──────────────────────────────────────────────
  Hi-Opt I value  Key     Cards
  ─────────────── ──────  ─────────────────────
       +1         3,4,5,6   (low cards, literally press the card number)
        0         2,7,8,9   (neutral cards)
       -1         t         Ten-value: 10, J, Q, K  (press 't')
        0         a         Ace  (tracked by side count)
  ───────────────────────────────────────────────
  CONTROL KEYS:
       n  or  Space    → Next hand / clear this round's cards
       s               → Shoe reshuffled — reset count
       i               → Insurance question (should I take it?)
       b               → Show current bet recommendation
       c               → Show current count (RC / TC)
       d               → Debug: print full state
       q  or  Esc      → Quit watch mode

SPEED ADVANTAGE:
  Full notation mode: "AS" + Enter + "KH" + Enter = ~3 seconds
  Hotkey mode:        'a' + 't'                   = ~0.3 seconds  ×10 faster!

The keys 3,4,5,6,2,7,8,9 correspond to their actual card face values,
so the mapping is intuitive — just press the card you see!
For 10/J/Q/K: all are 't' (for "Ten-value").
For Ace: 'a'.
"""

from __future__ import annotations
import sys
import threading
from typing import Callable, Optional
from dataclasses import dataclass

try:
    from pynput import keyboard as pynput_kb
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False

from simulator.deck import Card


@dataclass
class HotkeyEvent:
    """A card or control event from a keypress."""
    event_type: str   # "card", "next_hand", "shuffle", "query", "quit"
    card: Optional[Card] = None
    query: str = ""   # for event_type="query": "insurance", "bet", "count"


# Map from key char → (rank, suit_placeholder, count_description)
# Suit is always ♠ as placeholder — for counting only rank matters
_HOTKEY_CARD_MAP: dict[str, tuple[str, str]] = {
    "a": ("A",  "♠"),   # Ace
    "2": ("2",  "♠"),
    "3": ("3",  "♠"),
    "4": ("4",  "♠"),
    "5": ("5",  "♠"),
    "6": ("6",  "♠"),
    "7": ("7",  "♠"),
    "8": ("8",  "♠"),
    "9": ("9",  "♠"),
    "t": ("10", "♠"),   # Ten-value (10/J/Q/K) → same Hi-Opt I tag
    "0": ("10", "♠"),   # Alternative: use 0 for ten-value
    "j": ("J",  "♠"),
    "q": ("Q",  "♠"),   # NOTE: 'q' also = quit — we handle this in context
    "k": ("K",  "♠"),
}

_CONTROL_KEYS = {
    "n": "next_hand",
    " ": "next_hand",
    "s": "shuffle",
    "i": "insurance",
    "b": "bet",
    "c": "count",
    "d": "debug",
}


class HotkeyListener:
    """
    Non-blocking keyboard listener using pynput.

    Usage
    -----
    def on_card(card):
        counter.see_card(card)
        print(f"Tag: {counter.card_value(card):+d}  RC={counter.running_count:+d}")

    listener = HotkeyListener(on_card=on_card, on_control=on_control)
    listener.start()   # non-blocking
    # ... game runs ...
    listener.stop()
    """

    def __init__(self, on_card: Optional[Callable[[Card], None]] = None,
                 on_control: Optional[Callable[[str], None]] = None,
                 verbose: bool = True):
        self.on_card = on_card
        self.on_control = on_control
        self.verbose = verbose
        self._listener = None
        self._quit_called = False

    def start(self) -> None:
        """Start non-blocking hotkey listener."""
        if not PYNPUT_AVAILABLE:
            print("[HotkeyListener] pynput not available — using fallback terminal input.")
            return
        self._listener = pynput_kb.Listener(
            on_press=self._on_press,
            suppress=False,   # Don't consume keys globally
        )
        self._listener.start()
        if self.verbose:
            self._print_help()

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
            self._listener = None

    def _on_press(self, key) -> Optional[bool]:
        """Called for each keypress."""
        try:
            char = key.char.lower() if hasattr(key, "char") and key.char else None
        except AttributeError:
            char = None

        # Special keys
        if key == pynput_kb.Key.space:
            char = " "
        elif key == pynput_kb.Key.esc:
            char = "esc"

        if char is None:
            return

        # Quit
        if char in ("esc",):
            self._quit_called = True
            if self.on_control:
                self.on_control("quit")
            return False  # Stop listener

        # Card key
        if char in _HOTKEY_CARD_MAP:
            # 'q' is ambiguous — treat as Queen, not quit (use Esc to quit)
            rank, suit = _HOTKEY_CARD_MAP[char]
            card = Card(rank, suit)
            if self.on_card:
                self.on_card(card)
            return

        # Control key
        if char in _CONTROL_KEYS:
            if self.on_control:
                self.on_control(_CONTROL_KEYS[char])
            return

    def _print_help(self) -> None:
        print("\n" + "─" * 50)
        print("  HOTKEY MODE — single key per card")
        print("─" * 50)
        print("  3 4 5 6     → +1 count  (small cards)")
        print("  2 7 8 9     →  0 count  (neutral)")
        print("  t (or 0)    → -1 count  (Ten/J/Q/K)")
        print("  a           →  0 count  (Ace — side counted)")
        print("─" * 50)
        print("  n / Space   → Next hand")
        print("  s           → Shoe reshuffled")
        print("  i           → Insurance?")
        print("  b           → Show bet recommendation")
        print("  c           → Show current count")
        print("  Esc         → Quit")
        print("─" * 50 + "\n")

    @property
    def is_running(self) -> bool:
        return self._listener is not None and self._listener.is_alive()


class TerminalHotkeyInput:
    """
    Fallback hotkey input for when pynput is not available.
    Uses blocking single-character reads via tty/termios (Unix only).
    Slightly slower than pynput (needs Enter for some systems) but portable.
    """

    def __init__(self):
        self._available = sys.platform != "win32"

    def get_char(self, prompt: str = "") -> str:
        """Read a single character without requiring Enter (Unix only)."""
        if not self._available:
            return input(prompt).strip().lower()[:1]
        import tty
        import termios
        if prompt:
            print(prompt, end="", flush=True)
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch.lower()

    def read_card_blocking(self) -> Optional[Card]:
        """Read one card from a single keypress."""
        char = self.get_char()
        if char in _HOTKEY_CARD_MAP:
            rank, suit = _HOTKEY_CARD_MAP[char]
            return Card(rank, suit)
        return None


def make_hotkey_listener(on_card, on_control, verbose=True) -> HotkeyListener:
    """Factory: create and return the best available hotkey listener."""
    return HotkeyListener(on_card=on_card, on_control=on_control, verbose=verbose)
