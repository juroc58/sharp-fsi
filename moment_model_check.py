"""
moment_model_check.py
=====================
Side-by-side comparison of the *separated pitching-moment model* used by the
FSI stall layer and by the BEM, plus a data-driven check against the FSI's
recorded moment.

Both codes model the aft shift of the load in separated flow, but they apply it
to different quantities:

  BEM   xcp = 0.25c + 0.25c*(1 - f)   and the *lift* Cl is placed there, so
        Cm(1/4c) = -0.25*(1 - f)*Cl + Cm_table

  FSI   the separated normal force Cn_sep acts at cp_sep = 0.5c, i.e.
        Cm(1/4c) = -(cp_sep - 0.25)*Cn_sep*(1 - f_s)      (the polar's own Cm)

with the SAME separation parameter (the FSI's f uses the identical
``_fstat`` closure as the BEM's, and case D feeds the BEM this same polar).

At alpha = 90 deg the two differ qualitatively: Cl -> 0, so the BEM's moment
vanishes, whereas a flat plate normal to the flow has Cn = 2 acting at mid-chord,
so the moment about the quarter chord is -0.5.  The FSI model is the physical
one there; this script quantifies the difference over the whole stalled range.

Also bins the FSI's *recorded* Cm from its history file against alpha, which
validates the load integration end to end (the recorded moment should reproduce
the polar in quasi-steady separated flow).  For that part the history must come
from a run with the **stall layer on** — a potential run reports the DVM's own
~zero Cm.  Without an argument the script picks a stall history if one is
present (`fsi_history_bem_stall_tsr1p00.txt`), else the validation history.

Usage:  python3 moment_model_check.py [history_file]
"""

from __future__ import annotations

import sys
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from aero_correction import StaticPolar

TWO_PI = 2.0 * np.pi


def fstat(cl, cla=2.0 * np.pi):
    """Separation parameter f in [0,1] from the normal-force reference pair."""
    a = np.linspace(-np.pi, np.pi, len(cl))
    Cl_att = cla * np.sin(a) * np.cos(a)
    Cl_sep = 2.0 * np.sin(a) * np.cos(a)
    den = Cl_att - Cl_sep
    good = np.abs(den) > 1e-3
    f = np.where(good, (cl - Cl_sep) / np.where(good, den, 1.0), 1.0)
    # same degeneracy guard as StaticPolar.viterna: at |alpha| = 90 deg the
    # section is fully separated (n = 0), not attached
    f = np.where(np.abs(np.cos(a)) < 1e-6, 0.0, f)
    return np.clip(f, 0.0, 1.0)


def main():
    if len(sys.argv) > 1:
        hist = sys.argv[1]
    else:
        # prefer a stalled run: the binned check compares the recorded moment
        # against the *separated-polar* moment, which only exists when the stall
        # layer is active (a potential run reports the DVM's own ~zero Cm)
        for cand in ('fsi_history_bem_stall_tsr1p00.txt',
                     'fsi_history_bem_potential_tsr1p00.txt',
                     'fsi_validation_history_tsr3p00_stall.txt',
                     'fsi_validation_history.txt'):
            if os.path.exists(cand):
                hist = cand
                break
        else:
            hist = 'fsi_validation_history.txt'
    pol = StaticPolar.viterna(alpha_stall=14.0, cd0=8.0e-3)
    a = pol.alpha
    ad = np.degrees(a)
    m = (ad >= 0.0) & (ad <= 180.0)

    cl, cd, cm = pol.cl, pol.cd, pol.cm
    f = fstat(cl)
    s, c = np.sin(a), np.cos(a)
    Cn_sep = s * (1.98 * s * s + 2.0 * c * c)

    Cm_fsi = cm                                   # what the FSI/its polar uses
    Cm_bem_arm = -0.25 * (1.0 - f) * cl           # BEM: lift placed at xcp
    Cm_bem_full = -0.25 * (1.0 - f) * Cn_sep      # same, but with the normal force

    print("=" * 78)
    print("separated pitching-moment models (about the quarter chord)")
    print("=" * 78)
    print(f"polar: {pol.name}")
    print("\n  alpha   Cl     Cd      Cn_sep    f     Cm_FSI   Cm_BEM(Cl arm) "
          " Cm_BEM(Cn arm)")
    for d in (10, 14, 20, 30, 45, 60, 75, 90, 110, 135, 160, 170):
        i = int(np.argmin(np.abs(ad - d)))
        print(f"  {d:5.0f}  {cl[i]:+6.3f} {cd[i]:6.3f}  {Cn_sep[i]:+7.3f} "
              f"{f[i]:5.2f}  {Cm_fsi[i]:+7.3f}  {Cm_bem_arm[i]:+12.3f}  "
              f"{Cm_bem_full[i]:+12.3f}")

    print(f"\n  mean |Cm_FSI - Cm_BEM(Cl arm)|  over 0-90 deg : "
          f"{np.mean(np.abs(Cm_fsi[m & (ad <= 90)] - Cm_bem_arm[m & (ad <= 90)])):.4f}")
    print(f"  mean |Cm_FSI - Cm_BEM(Cn arm)|  over 0-90 deg : "
          f"{np.mean(np.abs(Cm_fsi[m & (ad <= 90)] - Cm_bem_full[m & (ad <= 90)])):.4f}")
    i90 = int(np.argmin(np.abs(ad - 90.0)))
    print(f"  at 90 deg: Cm_FSI = {Cm_fsi[i90]:+.4f} (Cn_sep = 2 at mid-chord), "
          f"Cm_BEM(Cl arm) = {Cm_bem_arm[i90]:+.4f} (Cl -> 0)")

    # ------------------------------------------------ data-driven FSI check
    have_hist = True
    try:
        d = np.genfromtxt(hist, names=True)
        alpha_h = d['alpha']
        cm_h = d['Cm']
        W = d['W']
        good = np.abs(W) > 4.0
    except Exception as e:                                        # pragma: no cover
        have_hist = False
        print(f"\n(no usable history in {hist}: {e})")

    if have_hist:
        # signed alpha bins over the last quarter of the run (steady cycle),
        # live wind only: the model moment is odd in alpha, so |alpha| bins
        # would average +Cm and -Cm together
        n0 = int(0.75 * len(alpha_h))
        bins = np.arange(-180.0, 180.1, 30.0)
        cen = 0.5 * (bins[1:] + bins[:-1])
        an = np.degrees(((alpha_h + np.pi) % TWO_PI) - np.pi)
        got = np.full(len(cen), np.nan)
        mod = np.full(len(cen), np.nan)
        cnt = np.zeros(len(cen), int)
        for i in range(len(cen)):
            sel = good & (np.arange(len(an)) >= n0)
            sel &= (an >= bins[i]) & (an < bins[i + 1])
            cnt[i] = int(sel.sum())
            if cnt[i] > 3:
                got[i] = cm_h[sel].mean()
                mod[i] = np.interp(cen[i], ad, cm)
        print("\n  FSI recorded Cm vs signed alpha (last 25 % of the run, "
              "live wind |W| > 4 m/s):")
        print("    alpha bin   : " + " ".join(f"{c:7.0f}" for c in cen))
        print("    samples     : " + " ".join(f"{n:7d}" for n in cnt))
        print("    recorded    : " + " ".join(f"{v:+7.3f}" for v in got))
        print("    polar model : " + " ".join(f"{v:+7.3f}" for v in mod))
        v = (~np.isnan(got)) & (~np.isnan(mod)) & (cnt > 5)
        if v.sum():
            print(f"    mean |recorded - polar| over populated bins = "
                  f"{np.mean(np.abs(got[v] - mod[v])):.4f}"
                  f"   (max |Cm| = {np.nanmax(np.abs(mod)):.3f})")

    # --------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot(ad[m], cm[m], 'k-', lw=2, label='FSI / polar $C_m$')
    ax[0].plot(ad[m], Cm_bem_arm[m], 'C0--', lw=1.5,
               label='BEM: lift placed at $x_{cp}=0.25c+0.25c(1-f)$')
    ax[0].plot(ad[m], Cm_bem_full[m], 'C2:', lw=1.5,
               label='same $x_{cp}$, normal force')
    ax[0].axhline(0, color='k', lw=0.6)
    ax[0].set_xlabel('$\\alpha$ (deg)')
    ax[0].set_ylabel('$C_m$ about the quarter chord')
    ax[0].set_title('separated pitching-moment models')
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    if have_hist:
        ax[1].plot(ad[m], cm[m], 'k-', lw=1.5, label='polar (model)')
        v = ~np.isnan(got)
        ax[1].plot(cen[v], got[v], 'ro-', label='FSI recorded (binned)')
        ax[1].set_xlabel('$|\\alpha|$ (deg)')
        ax[1].set_ylabel('$C_m$')
        ax[1].set_title('FSI load integration vs its own polar')
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3)
    fig.suptitle('separated pitching moment: model comparison', fontsize=12)
    fig.tight_layout()
    fig.savefig('moment_model_check.png', dpi=140)
    plt.close(fig)
    print("\nsaved moment_model_check.png")


if __name__ == '__main__':
    main()
