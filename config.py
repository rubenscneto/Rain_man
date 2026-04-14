"""
Rain Man - Blackjack Card Counting ML System
Central configuration file.

Based on concepts from "The World's Greatest Blackjack Book" (Humble & Cooper)
and standard econometric methodology (OLS, Gauss-Markov, Monte Carlo).
"""

from dataclasses import dataclass, field
from typing import Literal


# Casino rule set keys (see ml/advantage_calc.py for full definitions)
# "evolution_8d_s17_das" = Evolution Gaming Blackjack A confirmado nos prints:
#   • "Dealer must stand on 17" → S17
#   • "BLACKJACK PAYS 3:2"
#   • Estimativa visual: 8 baralhos
CASINO_RULE_KEY: str = "evolution_8d_s17_das"


@dataclass
class ShoeConfig:
    """Configuration for the card shoe (a caixinha de cartas)."""
    num_decks: int = 8          # Evolution Gaming Blackjack A: 8 baralhos (estimado visualmente)
    penetration: float = 0.75   # Fraction of shoe dealt before reshuffle (0.5 to 0.85)
    # NOTE: Physical shoe size (~30cm) does not affect counting —
    # only num_decks and penetration matter algorithmically.


@dataclass
class BlackjackRules:
    """Standard Vegas/casino blackjack rules."""
    blackjack_payout: float = 1.5    # 3:2 payout (some casinos pay 6:5 = 1.2, worse)
    dealer_hits_soft17: bool = False  # S17: dealer PARA no 17 mole (confirmado nos prints)
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
    # Systems ordered by complexity/power:
    #   hi_lo    — simplest, good for beginners (Hi-Lo balanced)
    #   ko       — unbalanced, no true count needed
    #   hi_opt1  — Humble & Cooper primary system (most recommended)
    #   hi_opt2  — Level-2, most powerful, harder to use
    #   omega2   — Multi-level balanced alternative
    name: Literal["hi_lo", "ko", "hi_opt1", "hi_opt2", "omega2"] = "hi_opt1"
    # BSE for the casino you're playing at (used by Hi-Opt I/II advantage formula)
    # Overridden automatically if casino_rules_key is set in RainManConfig
    bse: float = -0.43   # Evolution Gaming 8d S17 DAS (ponto de equilíbrio: TC ≈ +0.83)


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
    # Casino rules key — matches keys in ml/advantage_calc.py CASINO_RULES dict
    # Set this to match the casino where you're playing for accurate edge calculations.
    casino_rules_key: str = CASINO_RULE_KEY
    verbose: bool = True


# Default global config instance
DEFAULT_CONFIG = RainManConfig()
