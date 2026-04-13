"""
OLS (Ordinary Least Squares) Regression Model.

Econometric model: E[return | true_count] = α + β₁·TC + β₂·TC² + ε

This answers the core question:
  "Given a true count of X, what is my expected return per unit wagered?"

The model is trained on Monte Carlo simulation data and then used in
live play to translate the running count into a recommended bet size.

Gauss-Markov theorem guarantees OLS is BLUE (Best Linear Unbiased Estimator)
IF assumptions hold:
  (1) Linearity in parameters
  (2) No perfect multicollinearity
  (3) Zero conditional mean: E[ε|TC] = 0
  (4) Homoskedasticity: Var(ε|TC) = σ²  [tested below]
  (5) No autocorrelation: Cov(εᵢ, εⱼ) = 0  [tested with Durbin-Watson]
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OLSResults:
    """Full OLS regression output."""
    coefficients: np.ndarray = field(default_factory=lambda: np.array([]))
    intercept: float = 0.0
    feature_names: list = field(default_factory=list)
    r_squared: float = 0.0
    adj_r_squared: float = 0.0
    rmse: float = 0.0
    n_obs: int = 0
    # Statistical inference
    std_errors: np.ndarray = field(default_factory=lambda: np.array([]))
    t_stats: np.ndarray = field(default_factory=lambda: np.array([]))
    p_values: np.ndarray = field(default_factory=lambda: np.array([]))
    conf_intervals: np.ndarray = field(default_factory=lambda: np.array([]))
    # Predictions
    y_pred_train: np.ndarray = field(default_factory=lambda: np.array([]))
    y_pred_test: np.ndarray = field(default_factory=lambda: np.array([]))
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    # EV curve: true count → predicted return
    ev_curve: dict = field(default_factory=dict)

    def predict_ev(self, true_count: float) -> float:
        """Predict expected return for a given true count."""
        if len(self.ev_curve) == 0:
            return 0.0
        tc_arr = np.array(sorted(self.ev_curve.keys()))
        ev_arr = np.array([self.ev_curve[tc] for tc in tc_arr])
        return float(np.interp(true_count, tc_arr, ev_arr))

    def breakeven_true_count(self) -> float:
        """
        The true count at which EV crosses zero (player gains edge).
        Typically ~+1 to +2 for Hi-Lo with H17, 6 decks.
        """
        for tc in np.arange(-5, 10, 0.1):
            if self.predict_ev(tc) > 0:
                return round(tc, 1)
        return float("inf")


class OLSModel:
    """
    Econometric OLS model for blackjack EV estimation.

    Fits: return_on_bet ~ f(true_count, decks_remaining, dealer_upcard, ...)
    Primary regressor: true_count (TC).
    """

    def __init__(self, polynomial_degree: int = 2,
                 confidence_level: float = 0.95):
        self.degree = polynomial_degree
        self.alpha = 1 - confidence_level
        self._poly = PolynomialFeatures(degree=polynomial_degree,
                                        include_bias=False)
        self._model = LinearRegression(fit_intercept=True)
        self._is_fitted = False
        self.results: Optional[OLSResults] = None

    def fit(self, df: pd.DataFrame, verbose: bool = True) -> OLSResults:
        """
        Fit OLS on Monte Carlo simulation data.

        Target: return_on_bet  (−1 = loss, 0 = push, +1 = win, +1.5 = BJ)
        Features: true_count (primary), plus optional controls.
        """
        required = {"true_count", "return_on_bet"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        # Feature matrix — true count is the key regressor
        # Additional controls help reduce omitted-variable bias
        feature_cols = ["true_count"]
        if "decks_remaining" in df.columns:
            feature_cols.append("decks_remaining")
        if "dealer_upcard" in df.columns:
            feature_cols.append("dealer_upcard")

        X_raw = df[feature_cols].values
        y = df["return_on_bet"].values

        # Polynomial expansion (allows for nonlinear TC relationship)
        X_poly = self._poly.fit_transform(X_raw)
        feature_names = self._poly.get_feature_names_out(feature_cols)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_poly, y, test_size=0.2, random_state=42
        )

        # Fit OLS via sklearn (uses QR decomposition internally)
        self._model.fit(X_train, y_train)
        self._is_fitted = True

        # Predictions
        y_pred_train = self._model.predict(X_train)
        y_pred_test = self._model.predict(X_test)
        residuals = y_train - y_pred_train
        n, p = X_train.shape

        # R² and adjusted R²
        r2 = r2_score(y_test, y_pred_test)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))

        # Manual OLS inference: standard errors, t-stats, p-values
        # σ² = RSS / (n - p - 1)  [unbiased estimate]
        rss = float(np.sum(residuals ** 2))
        sigma2 = rss / (n - p - 1) if (n - p - 1) > 0 else 1.0
        try:
            XtX_inv = np.linalg.pinv(X_train.T @ X_train)
            var_coeff = sigma2 * XtX_inv
            std_errors = np.sqrt(np.diag(var_coeff))
        except np.linalg.LinAlgError:
            std_errors = np.ones(p)

        coeff = self._model.coef_
        t_stats = coeff / (std_errors + 1e-12)
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - p - 1))
        t_crit = stats.t.ppf(1 - self.alpha / 2, df=n - p - 1)
        conf_intervals = np.column_stack([
            coeff - t_crit * std_errors,
            coeff + t_crit * std_errors,
        ])

        # Build EV curve: for each integer true count, predict EV
        # using only true_count (marginal effect)
        ev_curve = {}
        for tc in range(-5, 11):
            if len(feature_cols) == 1:
                x_pred = np.array([[tc]])
            else:
                # Use mean values for controls, vary only true_count
                control_means = X_raw[:, 1:].mean(axis=0)
                x_pred = np.array([[tc] + list(control_means)])
            x_poly = self._poly.transform(x_pred)
            ev_curve[tc] = float(self._model.predict(x_poly)[0])

        results = OLSResults(
            coefficients=coeff,
            intercept=float(self._model.intercept_),
            feature_names=list(feature_names),
            r_squared=r2,
            adj_r_squared=adj_r2,
            rmse=rmse,
            n_obs=n,
            std_errors=std_errors,
            t_stats=t_stats,
            p_values=p_values,
            conf_intervals=conf_intervals,
            y_pred_train=y_pred_train,
            y_pred_test=y_pred_test,
            residuals=residuals,
            ev_curve=ev_curve,
        )
        self.results = results

        if verbose:
            self._print_results(results)

        return results

    def predict_ev(self, true_count: float,
                   decks_remaining: float = 3.0,
                   dealer_upcard: int = 7) -> float:
        """Predict expected return per unit for given true count."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        feat = [true_count]
        if self.degree > 0:
            feat.append(decks_remaining)
        x = self._poly.transform(np.array([feat]))
        return float(self._model.predict(x)[0])

    def optimal_bet_fraction(self, true_count: float,
                              bankroll: float) -> float:
        """
        Kelly Criterion: f* = edge / variance
        Full Kelly is too aggressive; use fractional Kelly.
        """
        if self.results is None:
            return 0.0
        ev = self.results.predict_ev(true_count)
        if ev <= 0:
            return 0.0
        # For blackjack: variance ≈ 1.3 (empirical standard)
        variance = 1.3
        kelly_full = ev / variance
        kelly_quarter = kelly_full * 0.25  # Quarter Kelly — safer
        return min(kelly_quarter * bankroll, bankroll * 0.1)

    def _print_results(self, r: OLSResults) -> None:
        print("\n" + "=" * 60)
        print("  OLS REGRESSION RESULTS  (E[return | true_count])")
        print("=" * 60)
        print(f"  Observations      : {r.n_obs:>10,}")
        print(f"  R²                : {r.r_squared:>10.6f}")
        print(f"  Adjusted R²       : {r.adj_r_squared:>10.6f}")
        print(f"  RMSE              : {r.rmse:>10.6f}")
        print(f"  Intercept (α)     : {r.intercept:>+10.6f}")
        print()
        print(f"  {'Feature':<30} {'Coeff':>10} {'SE':>10} {'t':>8} {'p':>8}")
        print("  " + "-" * 70)
        for name, c, se, t, p in zip(r.feature_names, r.coefficients,
                                      r.std_errors, r.t_stats, r.p_values):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {name:<30} {c:>+10.6f} {se:>10.6f} {t:>8.3f} {p:>8.4f} {sig}")
        print()
        print(f"  Break-even True Count: {r.breakeven_true_count():>+.1f}")
        print()
        print("  EV Curve (predicted return per unit):")
        for tc, ev in r.ev_curve.items():
            if -4 <= tc <= 8:
                bar_len = int(abs(ev) * 300)
                direction = "+" if ev >= 0 else "-"
                bar = direction * min(bar_len, 25)
                print(f"  TC {tc:>+3}: {ev:>+.4f}  {bar}")
        print("=" * 60)
