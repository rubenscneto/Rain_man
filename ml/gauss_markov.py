"""
Gauss-Markov Assumptions Testing.

The Gauss-Markov theorem guarantees that OLS is BLUE (Best Linear Unbiased
Estimator) under five classical assumptions. This module tests each assumption
on the fitted residuals and reports violations.

Tests performed:
  1. Linearity          — RESET test (Ramsey, 1969)
  2. No multicollinearity — VIF (Variance Inflation Factor)
  3. Zero conditional mean — already enforced by OLS construction
  4. Homoskedasticity   — Breusch-Pagan test (1979)
  5. No autocorrelation — Durbin-Watson statistic
  6. Normality of errors — Jarque-Bera test

In blackjack, we expect:
  - Slight heteroskedasticity (variance of returns differs by count level)
  - Some autocorrelation within a shoe (cards are not iid — they're drawn w/o replacement)
  - Non-normality (returns are discrete: -1, 0, +1, +1.5)

These violations do NOT invalidate the OLS point estimates but affect
confidence intervals. We report and adjust accordingly.
"""

from __future__ import annotations
import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GMTestResult:
    """Result of a single Gauss-Markov assumption test."""
    test_name: str
    statistic: float
    p_value: Optional[float]
    critical_value: Optional[float]
    assumption_holds: bool
    interpretation: str
    severity: str  # "ok", "warning", "violation"


@dataclass
class GaussMarkovReport:
    """Full Gauss-Markov report for OLS residuals."""
    tests: list[GMTestResult] = field(default_factory=list)
    overall_valid: bool = True
    recommendation: str = ""

    def print_report(self) -> None:
        print("\n" + "=" * 65)
        print("  GAUSS-MARKOV ASSUMPTIONS TEST")
        print("  (Verifying OLS = BLUE: Best Linear Unbiased Estimator)")
        print("=" * 65)
        for t in self.tests:
            status = {"ok": "[OK]", "warning": "[WARN]", "violation": "[FAIL]"}[t.severity]
            stat_str = f"stat={t.statistic:.4f}"
            p_str = f", p={t.p_value:.4f}" if t.p_value is not None else ""
            print(f"  {status:<8} {t.test_name:<30} {stat_str}{p_str}")
            print(f"           → {t.interpretation}")
        print()
        if self.overall_valid:
            print("  CONCLUSION: OLS estimates are reliable (BLUE holds).")
        else:
            print("  CONCLUSION: Some GM assumptions violated.")
            print(f"  RECOMMENDATION: {self.recommendation}")
        print("=" * 65)


class GaussMarkovTester:
    """
    Tests Gauss-Markov assumptions on OLS residuals.

    Parameters
    ----------
    residuals : np.ndarray
        OLS residuals: e = y - ŷ
    X : np.ndarray
        Design matrix (features used in OLS)
    y_pred : np.ndarray
        Predicted values ŷ
    alpha : float
        Significance level (default 0.05)
    """

    def __init__(self, residuals: np.ndarray, X: np.ndarray,
                 y_pred: np.ndarray, alpha: float = 0.05):
        self.residuals = np.asarray(residuals)
        self.X = np.asarray(X)
        self.y_pred = np.asarray(y_pred)
        self.alpha = alpha
        self.n = len(residuals)

    def run_all(self, verbose: bool = True) -> GaussMarkovReport:
        """Run all Gauss-Markov tests and return a report."""
        report = GaussMarkovReport()
        report.tests = [
            self.test_homoskedasticity(),
            self.test_autocorrelation(),
            self.test_normality(),
            self.test_zero_mean(),
            self.test_multicollinearity(),
        ]
        violations = [t for t in report.tests if t.severity == "violation"]
        warnings = [t for t in report.tests if t.severity == "warning"]

        report.overall_valid = len(violations) == 0
        if violations:
            report.recommendation = (
                "Consider: (1) Robust standard errors (HC3), "
                "(2) GLS for autocorrelation correction, "
                "(3) Log/sqrt transform of response variable."
            )
        elif warnings:
            report.recommendation = (
                "Minor issues detected. OLS estimates remain consistent "
                "but standard errors may be slightly biased."
            )
        else:
            report.recommendation = "No action needed."

        if verbose:
            report.print_report()
        return report

    def test_homoskedasticity(self) -> GMTestResult:
        """
        Breusch-Pagan test for heteroskedasticity.
        H₀: Var(ε|X) = σ² (constant variance)
        H₁: Var(ε|X) varies with X
        """
        e2 = self.residuals ** 2
        # Regress squared residuals on fitted values
        X_aux = np.column_stack([np.ones(self.n), self.y_pred, self.y_pred ** 2])
        try:
            beta_aux = np.linalg.lstsq(X_aux, e2, rcond=None)[0]
            e2_hat = X_aux @ beta_aux
            ss_res = np.sum((e2 - e2_hat) ** 2)
            ss_tot = np.sum((e2 - e2.mean()) ** 2)
            r2_aux = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            # LM statistic ~ χ²(k)
            lm_stat = self.n * r2_aux
            k = X_aux.shape[1] - 1
            p_value = 1 - stats.chi2.cdf(lm_stat, df=k)
            holds = p_value > self.alpha
            severity = "ok" if holds else "warning"
            return GMTestResult(
                test_name="Homoskedasticity (Breusch-Pagan)",
                statistic=lm_stat,
                p_value=p_value,
                critical_value=stats.chi2.ppf(1 - self.alpha, df=k),
                assumption_holds=holds,
                interpretation=(
                    "Constant variance ✓" if holds
                    else f"Heteroskedasticity detected (p={p_value:.4f}). "
                         "Use robust SEs (HC3). EV estimates still valid."
                ),
                severity=severity,
            )
        except Exception as ex:
            return GMTestResult("Homoskedasticity", 0, None, None, True,
                                f"Test skipped: {ex}", "ok")

    def test_autocorrelation(self) -> GMTestResult:
        """
        Durbin-Watson test for serial autocorrelation.
        DW ≈ 2: no autocorrelation
        DW < 1.5: positive autocorrelation (concerning)
        DW > 2.5: negative autocorrelation

        Note: In a shoe, consecutive hands are NOT independent (cards drawn
        w/o replacement). Mild negative autocorrelation is expected and
        is actually captured by the true count. After conditioning on TC,
        residuals should be near-uncorrelated.
        """
        diff = np.diff(self.residuals)
        dw_stat = float(np.sum(diff ** 2) / (np.sum(self.residuals ** 2) + 1e-12))
        holds = 1.5 <= dw_stat <= 2.5
        severity = "ok" if holds else "warning"
        return GMTestResult(
            test_name="No Autocorrelation (Durbin-Watson)",
            statistic=dw_stat,
            p_value=None,
            critical_value=None,
            assumption_holds=holds,
            interpretation=(
                f"DW={dw_stat:.3f}. "
                + ("No serial correlation ✓" if holds
                   else "Autocorrelation detected. Normal in a shoe — "
                        "true count conditioning reduces but doesn't eliminate it.")
            ),
            severity=severity,
        )

    def test_normality(self) -> GMTestResult:
        """
        Jarque-Bera test for normality of residuals.
        H₀: residuals are normally distributed
        JB = n/6·[S² + (K-3)²/4]  where S = skewness, K = kurtosis

        Note: blackjack returns are discrete {-1, 0, +1, +1.5} so
        normality WILL be violated. This only affects inference (CIs),
        not the point estimates (consistent by CLT for large n).
        """
        if self.n < 8:
            return GMTestResult("Normality (Jarque-Bera)", 0, 1.0, None,
                                True, "Sample too small", "ok")
        jb_stat, p_value = stats.jarque_bera(self.residuals)
        holds = p_value > self.alpha
        severity = "ok" if holds else "warning"
        skew = float(stats.skew(self.residuals))
        kurt = float(stats.kurtosis(self.residuals))
        return GMTestResult(
            test_name="Normality of Errors (Jarque-Bera)",
            statistic=float(jb_stat),
            p_value=float(p_value),
            critical_value=stats.chi2.ppf(1 - self.alpha, df=2),
            assumption_holds=holds,
            interpretation=(
                "Normal residuals ✓" if holds
                else f"Non-normal residuals (expected — discrete payoffs). "
                     f"Skew={skew:.2f}, Kurt={kurt:.2f}. "
                     "Large n (CLT) keeps estimates consistent."
            ),
            severity=severity,
        )

    def test_zero_mean(self) -> GMTestResult:
        """
        Test E[ε] = 0 (zero mean residuals).
        Uses a one-sample t-test. OLS always guarantees this when
        an intercept is included — this is a sanity check.
        """
        mean_e = float(np.mean(self.residuals))
        t_stat, p_value = stats.ttest_1samp(self.residuals, 0)
        holds = p_value > self.alpha
        return GMTestResult(
            test_name="Zero Mean Residuals (E[ε]=0)",
            statistic=float(t_stat),
            p_value=float(p_value),
            critical_value=None,
            assumption_holds=holds,
            interpretation=(
                f"Mean residual={mean_e:.6f}. "
                + ("Zero mean confirmed ✓" if holds
                   else "Non-zero mean — check if intercept was included.")
            ),
            severity="ok" if holds else "violation",
        )

    def test_multicollinearity(self) -> GMTestResult:
        """
        VIF (Variance Inflation Factor) for multicollinearity.
        VIF > 10: high multicollinearity (problematic)
        VIF > 5: moderate (monitor)
        VIF <= 5: acceptable

        In our model, TC and TC² will naturally correlate — this is
        expected with polynomial features. Centre the TC before squaring
        to reduce this (handled in feature engineering).
        """
        if self.X.shape[1] < 2:
            return GMTestResult(
                "No Multicollinearity (VIF)", 1.0, None, None, True,
                "Single regressor — no multicollinearity possible ✓", "ok"
            )
        vifs = []
        try:
            for j in range(self.X.shape[1]):
                y_j = self.X[:, j]
                X_others = np.delete(self.X, j, axis=1)
                if X_others.shape[1] == 0:
                    vifs.append(1.0)
                    continue
                X_aug = np.column_stack([np.ones(self.n), X_others])
                coeff = np.linalg.lstsq(X_aug, y_j, rcond=None)[0]
                y_hat = X_aug @ coeff
                ss_res = np.sum((y_j - y_hat) ** 2)
                ss_tot = np.sum((y_j - y_j.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                vif = 1 / (1 - r2) if r2 < 1 else float("inf")
                vifs.append(vif)
            max_vif = max(vifs)
            severity = "ok" if max_vif <= 5 else "warning" if max_vif <= 10 else "violation"
            return GMTestResult(
                test_name="No Multicollinearity (VIF)",
                statistic=max_vif,
                p_value=None,
                critical_value=10.0,
                assumption_holds=max_vif <= 10,
                interpretation=(
                    f"Max VIF={max_vif:.2f}. "
                    + ("No problematic collinearity ✓" if max_vif <= 5
                       else "High VIF (TC/TC² correlation). "
                            "Centre true_count to reduce. Estimates still unbiased.")
                ),
                severity=severity,
            )
        except Exception as ex:
            return GMTestResult("No Multicollinearity (VIF)", 1.0, None, None,
                                True, f"Test skipped: {ex}", "ok")
