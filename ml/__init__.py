"""ML / Econometrics package."""
from ml.ols_model import OLSModel, OLSResults
from ml.gauss_markov import GaussMarkovTester, GaussMarkovReport
from ml.betting_strategy import BettingStrategy, BetRecommendation
from ml.trainer import Trainer

__all__ = [
    "OLSModel", "OLSResults",
    "GaussMarkovTester", "GaussMarkovReport",
    "BettingStrategy", "BetRecommendation",
    "Trainer",
]
