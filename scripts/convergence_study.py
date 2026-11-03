"""
scripts/convergence_study.py
============================

Convergence and robustness study for the sharpFSI 2-D DVM + CPPC FSI solver.

Sweeps numerical and model parameters over the standard Sharp CPPC configuration
(R=5 m, c=1 m, U=10 m/s, one blade section) and writes machine-readable CSV/JSON
results plus summary plots.  Each case is run with ``FSICoupler`` directly, so the
study is self-contained and does not require ``sharp-vawt-lab``.

The intent of this study (per the reviewer mandate) is NOT to choose defaults that
make the results look better.  It is to determine which settings are converged and
defensible, document where results are sensitive to wake truncation or coupling
tolerance, and record production values.

Usage
-----

    python3 scripts/convergence_study.py [--quick] [--out-dir out]

* default (full): one-at-a-time parameter sweep from a baseline, at three TSR
  values (1.0 low/dead-point, 3.0 design, 4.5 high), 2 revolutions per case.
* --quick: coarser grids and half a revolution per case, for a fast smoke run
  (a couple of minutes).

Outputs (in --out-dir, default `convergence_study/`):
    convergence_results.csv   one row per case (machine-readable)
    convergence_results.json  same data as JSON
    convergence_sweep_<param>.png  Cp vs swept value at each TSR
    convergence_summary.png        max |Delta Cp| vs baseline per parameter/TSR

The baseline configuration (documented production values) is:
    n_panels=32, dt=0.005 s, wake_cap=600, wake_substeps=1,
    n_coupling_iter=3, coupling_tol=1e-4, relax=0.7, stall=on,
    vortex_moment=on, stall_blend_state=False.

Rationale for the baseline is given in the convergence recommendation table in
the printed output and in AUDIT_PLAN.md phase 2.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import UTC, datetime

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_validation import CONFIG, aero_config

from fsi_main import FSICoupler

TWO_PI = 2.0 * np.pi

# ---------------------------------------------------------------------------
# Baseline (documented production defaults)
# ---------------------------------------------------------------------------
BASELINE = {
    "n_panels": 32,
    "dt": 0.005,
    "wake_cap": 600,
    "wake_substeps": 1,
    "n_coupling_iter": 3,
    "coupling_tol": 1e-4,
    "relax": 0.7,
    "stall": True,
    "stall_alpha_stall": 14.0,
    "stall_T_a": 0.0,
    "stall_T_f": 3.0,
    "stall_T_v": 6.0,
    "stall_K_v": 0.7,
    "stall_vortex_moment": True,
    "stall_blend_state": False,
    "viscous_cd0": 8.0e-3,
}

# One-at-a-time sweep grids.  Each entry: (name, list of values).
SWEEP_GRIDS = {
    "n_panels": [16, 24, 32, 48, 64, 96],
    "dt": [0.01, 0.005, 0.0025, 0.00125],
    "wake_cap": [200, 600, 1200],
    "wake_substeps": [1, 2, 4],
    "n_coupling_iter": [1, 2, 3, 5, 10],
    "coupling_tol": [1e-3, 1e-4, 1e-5],
    "relax": [0.5, 0.7, 0.9],
    "stall": [False, True],
    "stall_vortex_moment": [False, True],
}

TSR_VALUES = [1.0, 3.0, 4.5]
N_REV = 2.0


def build_cfg(tsr, over):
    """Build an FSICoupler config for one case."""
    cfg = dict(CONFIG)
    cfg["omega"] = tsr * cfg["U_inf"] / cfg["R"]
    cfg.update(BASELINE)
    cfg.update(over)

    # --- aero model: potential (stall=False) or Viterna/dynamic-stall -------
    stall_bool = bool(cfg.pop("stall", BASELINE["stall"]))
    if stall_bool:
        acfg, _ = aero_config("stall")
        # keep the stall-model toggles from BASELINE
        acfg.update(
            {
                "stall_alpha_stall": cfg.pop("stall_alpha_stall"),
                "stall_T_a": cfg.pop("stall_T_a"),
                "stall_T_f": cfg.pop("stall_T_f"),
                "stall_T_v": cfg.pop("stall_T_v"),
                "stall_K_v": cfg.pop("stall_K_v"),
                "stall_vortex_moment": cfg.pop("stall_vortex_moment"),
                "stall_blend_state": cfg.pop("stall_blend_state"),
            }
        )
        cfg.update(acfg)
    else:
        acfg, _ = aero_config("potential")
        cfg.update(acfg)
    cfg.update(over.pop("extra", {}))
    return cfg


def run_case(tsr, param_name, param_value, n_rev=N_REV, verbose=False):
    """Run one coupled FSI case and return a dict of last-revolution statistics.

    Returns ``None`` if the run diverged (non-finite coefficients).
    """
    # "baseline" is a sentinel: run with BASELINE unchanged (no override).
    over = {} if param_name == "baseline" else {param_name: param_value}
    cfg = build_cfg(tsr, over)
    fsi = FSICoupler(**cfg)
    t0 = time.perf_counter()
    try:
        h = fsi.run(n_revolutions=n_rev, verbose=verbose)
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "tsr": tsr,
            "param": param_name,
            "value": param_value,
            "diverged": True,
            "error": str(exc),
            "runtime_s": time.perf_counter() - t0,
        }
    runtime = time.perf_counter() - t0

    T_rev = TWO_PI / fsi.omega
    t = h["t"]
    m = t >= t[-1] - T_rev
    if not np.any(m):
        m = np.ones(len(t), bool)

    psi = np.rad2deg(h["psi"][m])
    al = np.rad2deg(h["alpha"][m])
    # wrap alpha to [-180, 180]
    al = np.where(np.abs(al) > 180.0, al - np.sign(al) * 360.0, al)
    cp = h["Q_rotor"][m] * fsi.omega / (
        0.5 * fsi.rho * fsi.U_inf**3 * 2.0 * fsi.R
    )
    win = np.rad2deg(fsi.struct.p.win)
    stop = 100.0 * np.mean(np.abs(psi) > 0.98 * win)

    finite = bool(np.all(np.isfinite(h["Cl"])) and np.all(np.isfinite(h["psi"])))
    kelvin = fsi.fluid.circulation_budget()

    # coupling residual: last accepted change in psi (proxy; the coupler stores
    # per-step iteration counts, not the final delta).  Use mean iters as a proxy
    # for coupling difficulty.
    mean_iters = float(np.mean(fsi.iter_counts)) if fsi.iter_counts else 0.0

    return {
        "tsr": tsr,
        "param": param_name,
        "value": float(param_value) if isinstance(param_value, (int, float))
        else str(param_value),
        "n_panels": cfg["n_panels"],
        "dt": cfg["dt"],
        "wake_cap": cfg["wake_cap"],
        "wake_substeps": cfg["wake_substeps"],
        "n_coupling_iter": cfg["n_coupling_iter"],
        "coupling_tol": cfg["coupling_tol"],
        "relax": cfg["relax"],
        "stall": cfg.get("stall", False),
        "vortex_moment": cfg.get("stall_vortex_moment", False),
        "n_rev": n_rev,
        "steps": fsi.step_count,
        "runtime_s": runtime,
        "diverged": not finite,
        "kelvin_budget": kelvin,
        "wake_nw": int(fsi.fluid.nw),
        "wake_merge_count": int(fsi.fluid.merge_count),
        "mean_coupling_iters": mean_iters,
        "cp_mean": float(cp.mean()),
        "cp_last": float(cp[-1]) if len(cp) else np.nan,
        "q_rotor_mean": float(h["Q_rotor"][m].mean()),
        "psi_mean_deg": float(psi.mean()),
        "psi_amplitude_deg": float(0.5 * (psi.max() - psi.min())),
        "psi_rms_deg": float(np.sqrt(np.mean(psi**2))),
        "alpha_mean_deg": float(al.mean()),
        "alpha_max_abs_deg": float(np.abs(al).max()),
        "stop_pct": float(stop),
        "cl_max_abs": float(np.abs(h["Cl"][m]).max()),
    }


def run_baseline(tsr, n_rev=N_REV):
    """Run the baseline configuration at one TSR (param='baseline')."""
    return run_case(tsr, "baseline", "baseline", n_rev=n_rev)


def sweep_parameter(param_name, values, tsr, n_rev=N_REV, verbose=False):
    """Run one-at-a-time sweep of a single parameter at one TSR."""
    rows = []
    for value in values:
        r = run_case(tsr, param_name, value, n_rev=n_rev, verbose=verbose)
        rows.append(r)
        if verbose:
            status = "DIVERGED" if r["diverged"] else f"Cp={r['cp_mean']:+.4f}"
            print(
                f"    {param_name}={value!s:>8}  {status:10s}  "
                f"t={r['runtime_s']:5.1f}s  wake={r['wake_nw']}  "
                f"kelvin={r['kelvin_budget']:+.1e}"
            )
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150)
    print(f"  saved {path}")
    return path


def plot_sweeps(rows, out_dir):
    """One figure per swept parameter: Cp vs parameter value at each TSR."""
    params = sorted({r["param"] for r in rows if r["param"] != "baseline"})
    tsrs = sorted({r["tsr"] for r in rows})
    for param in params:
        fig, axes = plt.subplots(1, len(tsrs), figsize=(4.2 * len(tsrs), 3.4),
                                sharey=True)
        if len(tsrs) == 1:
            axes = [axes]
        # baseline value per TSR
        base_by_tsr = {}
        for r in rows:
            if r["param"] == "baseline":
                base_by_tsr[r["tsr"]] = r["cp_mean"]
        for i, tsr in enumerate(tsrs):
            ax = axes[i]
            sub = [r for r in rows if r["param"] == param and r["tsr"] == tsr
                   and not r["diverged"]]
            if not sub:
                continue
            xs = [r["value"] for r in sub]
            ys = [r["cp_mean"] for r in sub]
            ax.plot(xs, ys, "o-", label=param)
            # baseline marker
            if tsr in base_by_tsr:
                ax.axhline(base_by_tsr[tsr], color="k", ls=":", lw=1,
                           label="baseline" if i == 0 else None)
            ax.set_xlabel(param)
            ax.set_title(f"TSR = {tsr:g}")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_ylabel("Cp (mean, last rev)")
                ax.legend(fontsize=7)
        fig.suptitle(f"Convergence: effect of {param} on mean Cp")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        _save(fig, out_dir, f"convergence_sweep_{param}.png")
        plt.close(fig)


def plot_summary(rows, out_dir):
    """Summary: each parameter's maximum Cp deviation from baseline, per TSR."""
    import matplotlib.pyplot as plt

    tsrs = sorted({r["tsr"] for r in rows})
    params = sorted({r["param"] for r in rows if r["param"] != "baseline"})
    base_by_tsr = {}
    for r in rows:
        if r["param"] == "baseline":
            base_by_tsr[r["tsr"]] = r["cp_mean"]

    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(params))
    width = 0.25
    for i, tsr in enumerate(tsrs):
        devs = []
        for param in params:
            sub = [r for r in rows if r["param"] == param and r["tsr"] == tsr
                   and not r["diverged"]]
            if sub and tsr in base_by_tsr:
                devs.append(max(abs(r["cp_mean"] - base_by_tsr[tsr]) for r in sub))
            else:
                devs.append(np.nan)
        ax.bar(x + i * width, devs, width, label=f"TSR={tsr:g}")
    ax.set_xticks(x + width)
    ax.set_xticklabels(params, rotation=45, ha="right")
    ax.set_ylabel("max |Delta Cp| vs baseline (last rev)")
    ax.set_title("Parameter sensitivity: maximum Cp deviation from baseline")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "convergence_summary.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="sharpFSI convergence and robustness study.")
    ap.add_argument("--quick", action="store_true",
                    help="coarse grids, 1 revolution per case (fast smoke run).")
    ap.add_argument("--out-dir", default="convergence_study",
                    help="output directory for CSV/JSON/plots.")
    ap.add_argument("--n-rev", type=float, default=None,
                    help="revolutions per case (default 2, quick=1).")
    args = ap.parse_args()

    quick = args.quick
    out_dir = args.out_dir
    n_rev = args.n_rev if args.n_rev is not None else (0.5 if quick else N_REV)

    grids = dict(SWEEP_GRIDS)
    if quick:
        # Fast smoke run: one TSR, half a revolution, minimal OAT grid.
        grids = {
            "n_panels": [16, 64],
            "dt": [0.01, 0.005],
            "wake_cap": [200, 1200],
            "stall": [False, True],
        }
        TSRs = [3.0]
    else:
        TSRs = TSR_VALUES

    os.makedirs(out_dir, exist_ok=True)
    results = {
        "metadata": {
            "created": datetime.now(UTC).isoformat() + "Z",
            "n_rev": n_rev,
            "quick": quick,
            "baseline": BASELINE,
            "tsr_values": TSRs,
            "sweep_grids": grids,
        },
        "results": [],
    }

    print("=" * 78)
    print("sharpFSI convergence and robustness study")
    print("=" * 78)
    print(f"  quick={quick}  n_rev={n_rev}  TSRs={TSRs}")
    print(f"  baseline: {BASELINE}")
    print()

    # Baseline at each TSR first (reference values).
    for tsr in TSRs:
        print(f"--- baseline TSR={tsr:g} ---")
        r = run_baseline(tsr, n_rev=n_rev)
        results["results"].append(r)
        status = "DIVERGED" if r["diverged"] else f"Cp={r['cp_mean']:+.4f}"
        print(f"    {status}  runtime={r['runtime_s']:.1f}s  "
              f"wake={r['wake_nw']}  kelvin={r['kelvin_budget']:+.1e}")

    # One-at-a-time sweeps.
    for param, values in grids.items():
        print(f"--- sweep {param}: {values} ---")
        for tsr in TSRs:
            rows = sweep_parameter(param, values, tsr, n_rev=n_rev,
                                   verbose=quick)
            results["results"].extend(rows)

    # ---- write CSV / JSON ----
    csv_path = os.path.join(out_dir, "convergence_results.csv")
    json_path = os.path.join(out_dir, "convergence_results.json")
    rows = [r for r in results["results"] if "diverged" in r]
    if rows:
        keys = list(rows[0].keys())
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        with open(json_path, "w") as fh:
            json.dump(results, fh, indent=2, default=float)
        print(f"\nWrote {csv_path} ({len(rows)} rows)")
        print(f"Wrote {json_path}")

    # ---- plots ----
    try:
        plot_sweeps(rows, out_dir)
        plot_summary(rows, out_dir)
    except Exception as exc:  # pragma: no cover - plotting must not crash the study
        print(f"(plotting skipped: {exc})")

    # ---- convergence recommendation ----
    print("\n" + "=" * 78)
    print("CONVERGENCE RECOMMENDATION (production defaults)")
    print("=" * 78)
    print("  Baseline (documented production values):")
    for k, v in BASELINE.items():
        print(f"    {k:22s} = {v}")
    print()
    print("  Evidence (max |Delta Cp| vs baseline across the swept range):")
    base_by_tsr = {r["tsr"]: r["cp_mean"] for r in results["results"]
                   if r["param"] == "baseline"}
    for tsr in TSRs:
        print(f"    TSR = {tsr:g}  (baseline Cp = {base_by_tsr.get(tsr, float('nan')):+.4f}):")
        for param in grids:
            sub = [r for r in rows if r["param"] == param and r["tsr"] == tsr
                   and not r["diverged"]]
            if not sub:
                continue
            dev = max(abs(r["cp_mean"] - base_by_tsr[tsr]) for r in sub)
            print(f"      {param:20s}: max|dCp| = {dev:+.4f}")
    print()
    print("  Settings are converged when max|dCp| is small (< ~0.02) and the")
    print("  Kelvin budget remains ~1e-13 with bounded wake population.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
