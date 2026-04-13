"""
Training Orchestrator.

Ties together Monte Carlo simulation → OLS regression → Gauss-Markov tests
to produce a trained betting model.

Training pipeline:
  1. Run Monte Carlo (N hands)  → records DataFrame
  2. Fit OLS on records         → coefficients + EV curve
  3. Run Gauss-Markov tests     → validate assumptions
  4. Compute RoR estimate       → risk profile
  5. Save model to disk

The trained model is then loaded during live play.
"""

from __future__ import annotations
import os
import json
import pickle
import numpy as np
from typing import Optional, TYPE_CHECKING

from config import RainManConfig, DEFAULT_CONFIG
from simulator.deck import Shoe
from simulator.monte_carlo import MonteCarloSimulator, SimulationSummary
from counting import make_counter
from ml.ols_model import OLSModel, OLSResults
from ml.gauss_markov import GaussMarkovTester, GaussMarkovReport

if TYPE_CHECKING:
    from counting.base_counter import BaseCounter


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "trained_model.pkl")
STATS_PATH = os.path.join(os.path.dirname(__file__), "..", "training_stats.json")


class Trainer:
    """
    Full training pipeline for the Rain Man system.

    Usage
    -----
    trainer = Trainer(config)
    model = trainer.train()          # run Monte Carlo + OLS
    trainer.save()                   # persist to disk
    """

    def __init__(self, config: RainManConfig = DEFAULT_CONFIG):
        self.config = config
        self.ols_model: Optional[OLSModel] = None
        self.ols_results: Optional[OLSResults] = None
        self.gm_report: Optional[GaussMarkovReport] = None
        self.simulation_summary: Optional[SimulationSummary] = None
        self.ror: float = 0.0

    def train(self, verbose: bool = True) -> OLSResults:
        """
        Full training pipeline.

        Returns
        -------
        OLSResults
            Fitted OLS model with EV curve, coefficients, and statistics.
        """
        cfg = self.config
        if verbose:
            print("\n" + "#" * 60)
            print("#  RAIN MAN — TRAINING PIPELINE")
            print(f"#  System: {cfg.counting.name.upper()}")
            print(f"#  Shoe: {cfg.shoe.num_decks} decks, "
                  f"{cfg.shoe.penetration:.0%} penetration")
            print("#" * 60)

        # Step 1: Build shoe + counter
        shoe = Shoe(
            num_decks=cfg.shoe.num_decks,
            penetration=cfg.shoe.penetration,
            seed=cfg.monte_carlo.random_seed,
        )
        counter = make_counter(cfg.counting.name, shoe)

        # Step 2: Monte Carlo simulation
        if verbose:
            print("\n[Step 1/4] Monte Carlo Simulation...")
        simulator = MonteCarloSimulator(config=cfg, counter=counter)
        summary = simulator.run(verbose=verbose)
        self.simulation_summary = summary

        # Step 3: OLS regression
        if verbose:
            print("\n[Step 2/4] OLS Regression...")
        df = summary.to_dataframe()
        if df.empty:
            raise RuntimeError("Simulation produced no data.")

        self.ols_model = OLSModel(
            polynomial_degree=cfg.ml.polynomial_degree,
            confidence_level=cfg.ml.confidence_level,
        )
        self.ols_results = self.ols_model.fit(df, verbose=verbose)

        # Step 4: Gauss-Markov tests
        if verbose:
            print("\n[Step 3/4] Gauss-Markov Assumptions Tests...")
        tester = GaussMarkovTester(
            residuals=self.ols_results.residuals,
            X=self.ols_results.y_pred_train.reshape(-1, 1),
            y_pred=self.ols_results.y_pred_train,
            alpha=1 - cfg.ml.confidence_level,
        )
        self.gm_report = tester.run_all(verbose=verbose)

        # Step 5: Risk of Ruin
        if verbose:
            print("\n[Step 4/4] Risk of Ruin Estimation...")
        # Reset shoe and counter for RoR
        shoe2 = Shoe(num_decks=cfg.shoe.num_decks, penetration=cfg.shoe.penetration)
        counter2 = make_counter(cfg.counting.name, shoe2)
        simulator2 = MonteCarloSimulator(config=cfg, counter=counter2)
        self.ror = simulator2.run_risk_of_ruin(verbose=verbose)

        if verbose:
            self._print_training_summary()

        return self.ols_results

    def save(self, model_path: str = MODEL_PATH,
             stats_path: str = STATS_PATH) -> None:
        """Persist the trained model and statistics to disk."""
        if self.ols_model is None:
            raise RuntimeError("No trained model to save. Call train() first.")

        # Save sklearn model + results
        with open(model_path, "wb") as f:
            pickle.dump({
                "ols_model": self.ols_model,
                "ols_results": self.ols_results,
                "config": self.config,
            }, f)

        # Save human-readable stats as JSON
        stats = {
            "counting_system": self.config.counting.name,
            "num_decks": self.config.shoe.num_decks,
            "penetration": self.config.shoe.penetration,
            "hands_simulated": self.simulation_summary.total_hands if self.simulation_summary else 0,
            "player_edge_percent": self.simulation_summary.player_edge_percent if self.simulation_summary else 0,
            "r_squared": self.ols_results.r_squared if self.ols_results else 0,
            "breakeven_tc": self.ols_results.breakeven_true_count() if self.ols_results else 0,
            "risk_of_ruin": self.ror,
            "ev_curve": {str(k): v for k, v in self.ols_results.ev_curve.items()} if self.ols_results else {},
            "gm_valid": self.gm_report.overall_valid if self.gm_report else None,
        }
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\n[Trainer] Model saved to {model_path}")
        print(f"[Trainer] Stats saved to {stats_path}")

    @classmethod
    def load(cls, model_path: str = MODEL_PATH,
             config: RainManConfig = DEFAULT_CONFIG) -> "Trainer":
        """Load a previously trained model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No trained model at {model_path}. "
                                    "Run python main.py --mode train first.")
        with open(model_path, "rb") as f:
            data = pickle.load(f)

        trainer = cls(config=data.get("config", config))
        trainer.ols_model = data["ols_model"]
        trainer.ols_results = data["ols_results"]
        print(f"[Trainer] Model loaded from {model_path}")
        return trainer

    def _print_training_summary(self) -> None:
        print("\n" + "=" * 60)
        print("  TRAINING COMPLETE — SUMMARY")
        print("=" * 60)
        if self.ols_results:
            print(f"  R²                  : {self.ols_results.r_squared:.6f}")
            print(f"  Break-even TC       : {self.ols_results.breakeven_true_count():+.1f}")
        if self.simulation_summary:
            print(f"  Simulated hands     : {self.simulation_summary.total_hands:,}")
            print(f"  Player edge         : {self.simulation_summary.player_edge_percent:+.3f}%")
        print(f"  Risk of Ruin (RoR)  : {self.ror:.1%}")
        if self.gm_report:
            print(f"  GM assumptions valid: {self.gm_report.overall_valid}")
        print("=" * 60)
        print("\nReady for live play. Run: python main.py --mode live")
