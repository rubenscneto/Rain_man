"""
Rain Man - Blackjack Card Counting ML System
Central configuration file.

Based on concepts from "The World's Greatest Blackjack Book" (Humble & Cooper)
and standard econometric methodology (OLS, Gauss-Markov, Monte Carlo).
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ShoeConfig:
    """Configuration for the card shoe (a caixinha de cartas)."""
    num_decks: int = 6          # Standard casino: 4, 6, or 8 decks
    penetration: float = 0.75   # Fraction of shoe dealt before reshuffle (0.5 to 0.85)
    # NOTE: Physical shoe size (~30cm) does not affect counting —
    # only num_decks and penetration matter algorithmically.


@dataclass
class BlackjackRules:
    """Standard Vegas/casino blackjack rules."""
    blackjack_payout: float = 1.5   # 3:2 payout (some casinos pay 6:5 = 1.2, worse)
    dealer_hits_soft17: bool = True  # H17 rule (more common, worse for player)
    double_after_split: bool = True  # DAS allowed
    resplit_aces: bool = False       # RSA not always allowed
    surrender_allowed: bool = True   # Late surrender
    max_splits: int = 3              # Max re-splits per hand


@dataclass
class BettingConfig:
    """Betting parameters for Kelly criterion and spread."""
    bankroll: float = 1000.0         # Starting bankroll in units
    min_bet: float = 10.0            # Table minimum
    max_bet: float = 200.0           # Table maximum (20:1 spread = aggressive)
    kelly_fraction: float = 0.25     # Fractional Kelly (full Kelly is too risky)
    # True count thresholds for bet ramping (Hi-Lo)
    bet_spread: dict = field(default_factory=lambda: {
        -10: 1,   # Min bet below TC -1
         0:  1,   # Neutral: min bet
         1:  2,   # TC +1: 2 units
         2:  4,   # TC +2: 4 units
         3:  8,   # TC +3: 8 units
         4: 12,   # TC +4: 12 units
         5: 20,   # TC +5+: max bet
    })


@dataclass
class MonteCarloConfig:
    """Monte Carlo simulation parameters."""
    num_simulations: int = 100_000  # Number of hands to simulate for training
    num_sessions: int = 1_000       # Number of sessions for RoR (Risk of Ruin) calc
    hands_per_session: int = 200    # Hands per session
    random_seed: int = 42


@dataclass
class MLConfig:
    """Machine learning / econometrics configuration."""
    # OLS regression
    confidence_level: float = 0.95
    # Gauss-Markov tests
    heteroskedasticity_test: bool = True   # Breusch-Pagan test
    autocorrelation_test: bool = True      # Durbin-Watson test
    normality_test: bool = True            # Jarque-Bera test
    # Training
    train_size: float = 0.8               # Train/test split
    polynomial_degree: int = 2            # Polynomial features for OLS


@dataclass
class ScreenReaderConfig:
    """Screen capture and OCR configuration."""
    capture_region: dict = field(default_factory=lambda: {
        "top": 0, "left": 0, "width": 1920, "height": 1080
    })
    ocr_confidence: float = 0.6
    capture_interval_ms: int = 500  # Polling interval in milliseconds


@dataclass
class CountingSystem:
    name: Literal["hi_lo", "ko", "omega2"] = "hi_lo"


@dataclass
class RainManConfig:
    """Master configuration — all subsystems."""
    shoe: ShoeConfig = field(default_factory=ShoeConfig)
    rules: BlackjackRules = field(default_factory=BlackjackRules)
    betting: BettingConfig = field(default_factory=BettingConfig)
    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    screen: ScreenReaderConfig = field(default_factory=ScreenReaderConfig)
    counting: CountingSystem = field(default_factory=CountingSystem)
    verbose: bool = True


# Default global config instance
DEFAULT_CONFIG = RainManConfig()
