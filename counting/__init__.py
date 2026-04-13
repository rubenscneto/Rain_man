"""Card counting systems package."""
from counting.base_counter import BaseCounter
from counting.hi_lo import HiLoCounter
from counting.ko import KOCounter
from counting.omega2 import Omega2Counter
from simulator.deck import Shoe


def make_counter(system: str, shoe: Shoe) -> BaseCounter:
    """Factory: create a counter by name."""
    systems = {
        "hi_lo": HiLoCounter,
        "ko": KOCounter,
        "omega2": Omega2Counter,
    }
    cls = systems.get(system.lower())
    if cls is None:
        raise ValueError(f"Unknown counting system: {system!r}. "
                         f"Choose from: {list(systems)}")
    return cls(shoe)


__all__ = ["BaseCounter", "HiLoCounter", "KOCounter", "Omega2Counter", "make_counter"]
