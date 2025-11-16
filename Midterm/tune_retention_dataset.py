#!/usr/bin/env python3
"""
tune_retention_dataset.py
-------------------------
Make instructor-tunable synthetic targets for the construction midterm dataset.

USAGE (examples):
  # 1) Quick pass: just scale milestone & safety effects and write a new CSV
  python tune_retention_dataset.py \
      --in midterm_construction_projects.csv \
      --out midterm_construction_projects_custom.csv \
      --seed 42 \
      --beta-milestones 0.06 \
      --beta-safety -0.12

  # 2) Calibrate to target *bivariate* correlations (approximate)
  python tune_retention_dataset.py \
      --in midterm_construction_projects.csv \
      --out midterm_construction_projects_custom.csv \
      --seed 7 \
      --target-corr-milestones 0.15 \
      --target-corr-safety -0.20 \
      --max-iters 30 \
      --verbose

NOTES
- This script leaves the original *features* untouched and regenerates ONLY:
    * next12mo_spend (continuous target)
    * retained_12mo   (binary retention, optional logit link; same knobs)
- You can control the coefficients explicitly (beta-*) or ask the script to
  iteratively calibrate them to reach target *bivariate* correlations.
- Calibration is heuristic (binary search–like scaling on betas) and aims for
  tolerance ±0.01 by default. It’s fast and “good enough” for teaching use.

"""

import argparse
import numpy as np
import pandas as pd

def build_signal(df: pd.DataFrame,
                 beta_milestones: float,
                 beta_safety: float,
                 other_scale: float = 1.0) -> np.ndarray:
    """Constructs the latent signal for next12mo_spend with tunable betas."""
    # Safe fills for a few occasionally-missing features
    sat = df["customer_satisfaction"].fillna(3.5)
    milestones = df["on_time_milestones_pct"].fillna(85) / 10.0  # scale ~[3,10]
    pmexp = df["pm_experience_years"].fillna(8) / 10.0
    disc = df["discount_pct"].fillna(5)

    signal = (
        0.18*np.log(df["project_size_usd"]) +
        0.22*sat +
        0.06*df["prior_relationship_years"] -
        0.20*df["cost_overrun_pct"] -
        0.15*df["time_overrun_pct"] -
        0.03*df["payment_delay_days"]/30 -
        0.02*df["n_change_orders"] -
        0.015*df["competition_count"] +
        (beta_milestones * milestones) +     # <-- tunable
        (beta_safety * df["safety_incidents"]) +  # <-- tunable (negative expected)
        -0.01*disc +
        0.02*pmexp +
        0.03*(df["is_union_site"]==0).astype(int)
    )
    return other_scale * signal

def build_logit(df: pd.DataFrame,
                beta_milestones_logit: float,
                beta_safety_logit: float) -> np.ndarray:
    """Logit for retained_12mo with its own knobs (can mirror continuous)."""
    sat = df["customer_satisfaction"].fillna(3.5)
    milestones = df["on_time_milestones_pct"].fillna(85) / 10.0
    pmexp = df["pm_experience_years"].fillna(8) / 10.0
    logit = (
        -1.2
        + 0.6*(sat-3.5)
        + 0.25*np.log1p(df["prior_relationship_years"])
        - 0.9*df["cost_overrun_pct"]
        - 0.7*df["time_overrun_pct"]
        - 0.25*df["payment_delay_days"]/30
        + (beta_milestones_logit * milestones)      # <-- tunable
        + (beta_safety_logit * df["safety_incidents"])  # <-- tunable (negative expected)
        + 0.1*(pmexp)
    )
    return logit

def regenerate_targets(df: pd.DataFrame,
                       beta_milestones: float,
                       beta_safety: float,
                       beta_milestones_logit: float = None,
                       beta_safety_logit: float = None,
                       noise_sd: float = 0.25,
                       seed: int = 42):
    """Regenerate next12mo_spend and retained_12mo given the chosen betas."""
    rng = np.random.default_rng(seed)

    if beta_milestones_logit is None:
        beta_milestones_logit = beta_milestones * 0.8
    if beta_safety_logit is None:
        beta_safety_logit = beta_safety * 0.6

    signal = build_signal(df, beta_milestones, beta_safety)
    noise = rng.normal(0, noise_sd, len(df))

    baseline = 250_000
    df = df.copy()
    df["next12mo_spend"] = np.round(baseline * np.exp(signal/5 + noise), -2).clip(0, 5_000_000)

    logit = build_logit(df, beta_milestones_logit, beta_safety_logit)
    prob = 1.0/(1.0 + np.exp(-logit))
    df["retained_12mo"] = (rng.random(len(df)) < prob).astype(int)

    return df

def bivariate_corr(df: pd.DataFrame) -> dict:
    """Convenience function: bivariate correlations with the target."""
    corr_m = df[["next12mo_spend","on_time_milestones_pct"]].corr().iloc[0,1]
    corr_s = df[["next12mo_spend","safety_incidents"]].corr().iloc[0,1]
    return {"milestones": float(corr_m), "safety": float(corr_s)}

def calibrate(df: pd.DataFrame,
              target_corr_milestones: float = None,
              target_corr_safety: float = None,
              init_beta_milestones: float = 0.06,
              init_beta_safety: float = -0.12,
              seed: int = 42,
              tol: float = 0.01,
              max_iters: int = 25,
              noise_sd: float = 0.25,
              verbose: bool = False):
    """
    Heuristically scale betas to hit desired *bivariate* correlations (± tol).
    We adjust each beta independently with a multiplicative factor using a
    bracketed search.
    """
    beta_m = init_beta_milestones
    beta_s = init_beta_safety

    def adjust(beta, direction):
        # coarse multiplicative scaling step
        return beta * (1.25 if direction > 0 else 0.8)

    for _ in range(max_iters):
        # Re-generate with current betas
        df_try = regenerate_targets(df, beta_m, beta_s, noise_sd=noise_sd, seed=seed)
        corrs = bivariate_corr(df_try)

        done_m = True if target_corr_milestones is None else (abs(corrs["milestones"] - target_corr_milestones) <= tol)
        done_s = True if target_corr_safety is None else (abs(corrs["safety"] - target_corr_safety) <= tol)

        if verbose:
            print(f"[iter] betas: milestones={beta_m:.4f}, safety={beta_s:.4f} | "
                  f"corrs: milestones={corrs['milestones']:.3f}, safety={corrs['safety']:.3f}")

        if done_m and done_s:
            return df_try, beta_m, beta_s, corrs

        # Adjust milestones beta
        if target_corr_milestones is not None and not done_m:
            direction = 1 if corrs["milestones"] < target_corr_milestones else -1
            beta_m = adjust(beta_m, direction)

        # Adjust safety beta (more negative correlation desired ⇒ more negative beta)
        if target_corr_safety is not None and not done_s:
            # If we want, say, -0.20 and current is -0.12, we need "more negative" (direction = -1)
            direction = -1 if corrs["safety"] > target_corr_safety else 1
            # Boost magnitude away from zero in the correct sign
            beta_s = adjust(beta_s, -1 if direction < 0 else 1)

    # Return last attempt if not converged
    return df_try, beta_m, beta_s, corrs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input CSV with features (original midterm dataset).")
    ap.add_argument("--out", dest="out_path", required=True, help="Output CSV with regenerated targets.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--noise-sd", type=float, default=0.25, help="Noise SD for target generation (higher = noisier).")

    # Direct coefficient control
    ap.add_argument("--beta-milestones", type=float, default=None, help="Coefficient for on_time_milestones_pct (continuous target).")
    ap.add_argument("--beta-safety", type=float, default=None, help="Coefficient for safety_incidents (continuous target).")
    ap.add_argument("--beta-milestones-logit", type=float, default=None, help="Coefficient for milestones in retention logit.")
    ap.add_argument("--beta-safety-logit", type=float, default=None, help="Coefficient for safety in retention logit.")

    # Correlation targeting (overrides betas if provided)
    ap.add_argument("--target-corr-milestones", type=float, default=None, help="Desired bivariate corr(target, milestones).")
    ap.add_argument("--target-corr-safety", type=float, default=None, help="Desired bivariate corr(target, safety_incidents).")
    ap.add_argument("--tol", type=float, default=0.01, help="Correlation tolerance.")
    ap.add_argument("--max-iters", type=int, default=25, help="Max iterations for calibration.")
    ap.add_argument("--verbose", action="store_true", help="Print iterative progress.")

    args = ap.parse_args()

    df = pd.read_csv(args.in_path)

    # If user gave target correlations, calibrate; else, use betas provided or defaults
    if args.target_corr_milestones is not None or args.target_corr_safety is not None:
        df_out, beta_m, beta_s, corrs = calibrate(
            df=df,
            target_corr_milestones=args.target_corr_milestones,
            target_corr_safety=args.target_corr_safety,
            init_beta_milestones=0.06 if args.beta_milestones is None else args.beta_milestones,
            init_beta_safety=-0.12 if args.beta_safety is None else args.beta_safety,
            seed=args.seed,
            tol=args.tol,
            max_iters=args.max_iters,
            noise_sd=args.noise_sd,
            verbose=args.verbose
        )
        print(f"Final betas: milestones={beta_m:.4f}, safety={beta_s:.4f}")
        print(f"Achieved corrs: milestones={corrs['milestones']:.3f}, safety={corrs['safety']:.3f}")
    else:
        # Direct betas path
        beta_m = 0.06 if args.beta_milestones is None else args.beta_milestones
        beta_s = -0.12 if args.beta_safety is None else args.beta_safety
        df_out = regenerate_targets(df, beta_m, beta_s,
                                    beta_milestones_logit=args.beta_milestones_logit,
                                    beta_safety_logit=args.beta_safety_logit,
                                    noise_sd=args.noise_sd,
                                    seed=args.seed)
        corrs = bivariate_corr(df_out)
        print(f"Set betas: milestones={beta_m:.4f}, safety={beta_s:.4f}")
        print(f"Corrs: milestones={corrs['milestones']:.3f}, safety={corrs['safety']:.3f}")

    # Write output
    df_out.to_csv(args.out_path, index=False)
    print(f"Wrote: {args.out_path}")

if __name__ == "__main__":
    main()
