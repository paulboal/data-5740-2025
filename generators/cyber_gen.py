# cybersec_data_generator.py
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# -------------------------------
# Core generator
# -------------------------------

@dataclass
class CyberDataGenConfig:
    n_rows: int = 240                       # total rows (dept x quarter or just iid rows)
    seed: Optional[int] = 42
    noise_scale: float = 2.0                # higher -> noisier incidents (lower R^2)
    with_interactions: bool = True
    inject_data_issues: bool = True         # add a few realistic data-hygiene problems
    departments: List[str] = field(default_factory=lambda: [
        "Sales","Marketing","Finance","HR","IT","Operations","Customer Support","R&D","Legal",
        "Procurement","Facilities","Security","Product","Data Science","Compliance","Quality Assurance",
        "Supply Chain","Logistics","Engineering","DevOps","Business Development","Accounting","Training",
        "Public Relations","Risk Management","Internal Audit","Investor Relations","E-commerce",
        "Field Services","Clinical Operations"
    ])
    quarters: List[str] = field(default_factory=lambda: [
        "Q1-2023","Q2-2023","Q3-2023","Q4-2023","Q1-2024","Q2-2024","Q3-2024","Q4-2024"
    ])

    # Default linear model coefficients (interpretable units)
    # incidents  ~  β0 + Σ β_i * feature_i + (optional) interactions + noise
    coeffs: Dict[str, float] = field(default_factory=lambda: {
        "intercept": 5.0,
        "org_size": 0.0007,                  # per employee
        "vuln_count": 0.015,                 # per open vulnerability
        "mean_time_to_patch": 0.03,          # per day
        "phishing_sim_click_rate": 20.0,     # 0..1
        "cloud_misconfig_count": 0.05,
        "login_failures_per_user": 2.0,      # per user per quarter
        "training_completion_rate": -25.0,   # 0..1  (strong negative as requested)
        "mfa_coverage": -8.0,                # 0..1
        "endpoint_coverage": -7.0,           # 0..1
        # Optional interactions (only used if with_interactions=True)
        "vuln_x_endpoint_gap": 0.004,        # vuln_count * (1 - endpoint_coverage)
    })


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def generate_cybersec_data(cfg: CyberDataGenConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)

    # Choose department-quarter pairs to reach ~n_rows
    combos = [(d, q) for d in cfg.departments for q in cfg.quarters]
    if cfg.n_rows <= len(combos):
        rng.shuffle(combos)
        combos = combos[:cfg.n_rows]
    else:
        # sample with replacement if more rows than unique combos
        combos = list(rng.choice(combos, size=cfg.n_rows, replace=True))

    rows = []
    for dept, quarter in combos:
        # Latent department “size/budget/risk” profiles to induce structure
        dept_hash = abs(hash(dept)) % 10_000
        dept_scale = 0.6 + (dept_hash % 4000) / 4000.0  # 0.6..1.6

        org_size = int(rng.integers(300, 4000) * dept_scale)  # employees
        it_budget_per_emp = rng.normal(4500, 700) + (dept_scale - 1.0) * 600

        # Controls & behavior (with sensible relationships)
        mfa_coverage = _clip01(rng.normal(0.78, 0.10) + 0.00003 * (it_budget_per_emp - 4500))
        endpoint_coverage = _clip01(rng.normal(0.82, 0.08) + 0.00004 * (it_budget_per_emp - 4500))

        phishing_sim_click_rate = _clip01(
            rng.beta(2, 8) + (1 - mfa_coverage) * 0.08 + rng.normal(0, 0.01)
        )

        training_completion_rate = _clip01(
            rng.normal(0.77, 0.08)
            + (it_budget_per_emp - 4500) / 10000
            - phishing_sim_click_rate * 0.20
        )

        mean_time_to_patch = max(
            1.0,
            rng.normal(45, 9)
            - (it_budget_per_emp - 4500) / 120
            - (endpoint_coverage - 0.80) * 35
        )

        login_failures_per_user = max(
            0.0, rng.normal(1.8, 0.45) + (1 - mfa_coverage) * 1.0
        )

        vuln_count = max(0, rng.poisson(lam=(org_size / 60) * rng.uniform(0.9, 1.1)))
        cloud_misconfig_count = max(0, rng.poisson(2 + (1 - endpoint_coverage) * 8))

        # Optional interaction(s)
        vuln_x_endpoint_gap = vuln_count * (1 - endpoint_coverage)

        # Build deterministic part via coefficients
        b = cfg.coeffs
        mu = (
            b.get("intercept", 0.0)
            + b.get("org_size", 0.0) * org_size
            + b.get("vuln_count", 0.0) * vuln_count
            + b.get("mean_time_to_patch", 0.0) * mean_time_to_patch
            + b.get("phishing_sim_click_rate", 0.0) * phishing_sim_click_rate
            + b.get("cloud_misconfig_count", 0.0) * cloud_misconfig_count
            + b.get("login_failures_per_user", 0.0) * login_failures_per_user
            + b.get("training_completion_rate", 0.0) * training_completion_rate
            + b.get("mfa_coverage", 0.0) * mfa_coverage
            + b.get("endpoint_coverage", 0.0) * endpoint_coverage
        )

        if cfg.with_interactions:
            mu += b.get("vuln_x_endpoint_gap", 0.0) * vuln_x_endpoint_gap

        # Noise term (controls R^2 visually)
        eps = rng.normal(0, cfg.noise_scale)
        incidents = max(0.0, mu + eps)           # enforce non-negative
        incidents = int(round(incidents))        # integer count

        rows.append({
            "department": dept,
            "quarter": quarter,
            "org_size": int(org_size),
            "it_budget_per_emp": round(float(it_budget_per_emp), 2),
            "mfa_coverage": round(float(mfa_coverage), 3),
            "endpoint_coverage": round(float(endpoint_coverage), 3),
            "vuln_count": int(vuln_count),
            "mean_time_to_patch": round(float(mean_time_to_patch), 1),
            "phishing_sim_click_rate": round(float(phishing_sim_click_rate), 3),
            "training_completion_rate": round(float(training_completion_rate), 3),
            "login_failures_per_user": round(float(login_failures_per_user), 2),
            "cloud_misconfig_count": int(cloud_misconfig_count),
            "vuln_x_endpoint_gap": round(float(vuln_x_endpoint_gap), 3),
            "security_incidents": incidents,
        })

    df = pd.DataFrame(rows)

    # Optional: introduce a few data-quality issues (for teaching “hygiene”)
    if cfg.inject_data_issues and len(df) >= 50:
        rng = np.random.default_rng((cfg.seed or 0) + 123)
        # 1) a few missing trainings
        miss_ix = rng.choice(df.index, size=min(12, len(df)//15), replace=False)
        df.loc[miss_ix, "training_completion_rate"] = np.nan
        # 2) a few out-of-range phishing rates
        bad_ix = rng.choice(df.index, size=min(8, len(df)//20), replace=False)
        df.loc[bad_ix, "phishing_sim_click_rate"] = df.loc[bad_ix, "phishing_sim_click_rate"] + 1.1
        # 3) text in numeric field
        text_ix = rng.choice(df.index, size=min(5, len(df)//40), replace=False)
        df.loc[text_ix, "vuln_count"] = "N/A"
        # 4) leverage point: huge org with great controls but mid incidents
        df.loc[df.index.min(), ["org_size","mfa_coverage","endpoint_coverage","security_incidents"]] = [25000, 0.99, 0.99, 12]

    return df


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    # 1) Use defaults
    cfg = CyberDataGenConfig(
        n_rows=200,           # rows you want
        noise_scale=1.5,      # lower -> tighter fit (higher R^2), higher -> noisier
        seed=7,
        with_interactions=True,
        inject_data_issues=True
    )
    df = generate_cybersec_data(cfg)
    print(df.head(10))

    # 2) Custom coefficients (e.g., make training even stronger, reduce vuln effect)
    custom_coeffs = {
        "intercept": 3.0,
        "org_size": 0.0006,
        "vuln_count": 0.010,
        "mean_time_to_patch": 0.035,
        "phishing_sim_click_rate": 25.0,
        "cloud_misconfig_count": 0.06,
        "login_failures_per_user": 1.8,
        "training_completion_rate": -30.0,   # stronger negative
        "mfa_coverage": -8.0,
        "endpoint_coverage": -7.0,
        "vuln_x_endpoint_gap": 0.004
    }
    cfg2 = CyberDataGenConfig(n_rows=120, noise_scale=1.0, seed=21, with_interactions=True, inject_data_issues=False, coeffs=custom_coeffs)
    df2 = generate_cybersec_data(cfg2)

    # Save to CSVs for your lab
    df.to_csv("cybersec_regression_lab_data_generated.csv", index=False)
    df2.to_csv("cybersec_regression_lab_data_generated_custom.csv", index=False)
    print("\nSaved: cybersec_regression_lab_data_generated.csv, cybersec_regression_lab_data_generated_custom.csv")

    # Optional: quick sanity check regression (requires statsmodels)
    try:
        import statsmodels.formula.api as smf
        m = smf.ols("security_incidents ~ org_size + vuln_count + mean_time_to_patch + phishing_sim_click_rate + "
                    "cloud_misconfig_count + login_failures_per_user + training_completion_rate + "
                    "mfa_coverage + endpoint_coverage + vuln_x_endpoint_gap",
                    data=df2).fit()
        print("\nQuick OLS summary (custom set):")
        print(m.summary().tables[1])  # coefficients table
        print(f"\nAdj R^2: {m.rsquared_adj:0.3f}")
    except Exception as e:
        print("Install statsmodels for the quick check, or ignore this:", e)