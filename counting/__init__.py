"""
Card counting systems package.

Systems available (ordered by complexity):
  hi_lo    — Hi-Lo balanced (simplest, good for beginners)
  ko       — Knock-Out unbalanced (no true count needed)
  hi_opt1  — Hi-Opt I (Humble & Cooper primary recommendation)
  hi_opt2  — Hi-Opt II (level-2, most powerful)
  omega2   — Omega II (multi-level balanced alternative)

Reference: "The World's Greatest Blackjack Book" (Humble & Cooper, 1980)
"""

from counting.base_counter import BaseCounter
from counting.hi_lo import HiLoCounter
from counting.ko import KOCounter
from counting.hi_opt1 import HiOpt1Counter
from counting.hi_opt2 import HiOpt2Counter
from counting.omega2 import Omega2Counter
from simulator.deck import Shoe


SYSTEM_INFO = {
    "hi_lo": {
        "class": HiLoCounter,
        "level": 1,
        "betting_correlation": 0.97,
        "playing_efficiency": 0.51,
        "description": "Hi-Lo: simplest balanced system. 2-6=+1, 10-A=-1.",
    },
    "ko": {
        "class": KOCounter,
        "level": 1,
        "betting_correlation": 0.98,
        "playing_efficiency": 0.55,
        "description": "Knock-Out: unbalanced, no true count needed. 2-7=+1, 10-A=-1.",
    },
    "hi_opt1": {
        "class": HiOpt1Counter,
        "level": 1,
        "betting_correlation": 0.86,   # 0.96 with Ace side count
        "playing_efficiency": 0.615,
        "description": "Hi-Opt I (Humble & Cooper): 3-6=+1, 10-K=-1. Best simple+powerful combo.",
    },
    "hi_opt2": {
        "class": HiOpt2Counter,
        "level": 2,
        "betting_correlation": 0.91,
        "playing_efficiency": 0.671,   # Highest playing efficiency available
        "description": "Hi-Opt II: level-2. 2,3,6,7=+1; 4,5=+2; 10-K=-2. Most powerful.",
    },
    "omega2": {
        "class": Omega2Counter,
        "level": 2,
        "betting_correlation": 0.92,
        "playing_efficiency": 0.67,
        "description": "Omega II: multi-level balanced. 2,3,7=+1; 4,5,6=+2; 9=-1; 10-K=-2.",
    },
}


def make_counter(system: str, shoe: Shoe, bse: float = -0.54) -> BaseCounter:
    """
    Factory: create a counter by name.

    Parameters
    ----------
    system : str
        Counting system key (hi_lo, ko, hi_opt1, hi_opt2, omega2).
    shoe : Shoe
        The shoe the counter will track.
    bse : float
        Basic Strategy Expectation for the casino — only used by
        Hi-Opt I/II for the advantage formula (BSE + 0.515 × TC).

    Returns
    -------
    BaseCounter
    """
    info = SYSTEM_INFO.get(system.lower())
    if info is None:
        raise ValueError(
            f"Unknown counting system: {system!r}. "
            f"Choose from: {list(SYSTEM_INFO)}"
        )
    cls = info["class"]
    # Hi-Opt I and II accept a bse parameter
    if system.lower() in ("hi_opt1", "hi_opt2"):
        return cls(shoe, bse=bse)
    return cls(shoe)


def list_systems() -> None:
    """Print a summary of all available counting systems."""
    print("\n" + "=" * 65)
    print("  AVAILABLE CARD COUNTING SYSTEMS")
    print("=" * 65)
    print(f"  {'System':<10} {'Level':>5}  {'Bet.Corr':>8}  {'Play.Eff':>8}  Note")
    print("  " + "-" * 63)
    for key, info in SYSTEM_INFO.items():
        print(f"  {key:<10} {info['level']:>5}  "
              f"{info['betting_correlation']:>8.3f}  "
              f"{info['playing_efficiency']:>8.3f}  "
              f"{info['description'][:35]}")
    print("=" * 65)
    print("  Recommended: hi_opt1 (best balance of simplicity + power)")
    print("  Professional: hi_opt2 (highest performance, more complex)")


__all__ = [
    "BaseCounter", "HiLoCounter", "KOCounter",
    "HiOpt1Counter", "HiOpt2Counter", "Omega2Counter",
    "make_counter", "list_systems", "SYSTEM_INFO",
]
