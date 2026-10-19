"""
compare_with_bem.py
===================
Compare the sharpFSI 2-D DVM/FSI suite with the independent BEM solver
sharp-vawt-lab (https://github.com/juroc58/sharp-vawt-lab, physics core
vawt_core.py).

The reference solver is a separate project and is *not* vendored here: this
script locates a local clone via ref_solver.py (set SHARP_VAWT_LAB to its root,
or clone it as a sibling directory).

The rotor / structural data are identical in the two codes (R=5 m, c=1 m,
N=1 blade, ar=0.35, sp=0.25, mb=5 kg with CG at 0.25c, counterweight
mc=2.5 kg at xcw=+0.35c, pitch limits +-25 deg, stop wstop=50 rad/s, pivot
damping cb=0.15, U=10 m/s, fixed rotor speed omega = TSR*U/R).  The BEM is
run as a strictly TWO-DIMENSIONAL model (H = 1 m) so that its forces, moments
and torque are per unit span and can be compared directly with the 2-D DVM.
Tip loss, flow-curvature corrections and arm drag are switched off in every
case (all of them are 3-D / empirical corrections that have no counterpart in
the 2-D DVM).

Three/four BEM aerodynamic models are compared:

  A  matched quasi-steady   polar Cl = 2*pi*sin(alpha), Cd = 0, dynamic stall
                            OFF, no induction.  Same steady aerodynamics as the
                            DVM -> isolates the numerical + structural
                            implementation of the two codes.
  B  matched unsteady       same polar, but the classical Theodorsen
                            apparent-mass lift Cl_nc = (pi/2)(c/W)*alpha_dot is
                            enabled (stall lags and vortex lift disabled).
                            That is the same unsteady physics the DVM resolves
                            from the unsteady Bernoulli pressure, so this case
                            also tests the unsteady implementation.
  D  matched stall          the SAME static polar the FSI stall layer uses
                            (StaticPolar.viterna) plus sharp-vawt-lab's Noll-Ham
                            dynamic stall, with the time constants translated
                            to the FSI convention (its tau = c/W, ours
                            c/2W, so ds_Tf = T_f/2, ds_Tv = T_v/2), K_v and the
                            separated moment matched.  This is the apples-to-
                            apples stalled comparison: it isolates the
                            *flow/coupling* difference with the same aero
                            model on both sides.
  C  real aero              NACA 0012 polar tables + full Noll-Ham dynamic
                            stall + a fixed induction a_fix = 0.2.

The FSI aerodynamic model is selected on the command line; the BEM cases are
always all run so every comparison is available.

Usage:  python3 compare_with_bem.py [tsr] [n_rev] [aero]
        aero = potential | stall | naca   (see run_validation.py)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_solver                                                     # noqa: E402

REF_ROOT = ref_solver.find()
if REF_ROOT is None:
    raise SystemExit("compare_with_bem.py: " + ref_solver.INSTALL_HINT)
if REF_ROOT not in sys.path:
    sys.path.insert(0, REF_ROOT)

from vawt_core import (Geometry, MassProps, AeroConfig, SimConfig,   # noqa: E402
                       PolarDatabase, CycloturbineSim)

from fsi_main import FSICoupler                                       # noqa: E402
from run_validation import CONFIG, aero_config                        # noqa: E402
from aero_correction import StaticPolar                               # noqa: E402

POLAR_DIR = None          # resolved lazily, only if case C is requested


def _require_polar_dir():
    """Resolve the reference NACA 0012 tables, failing loudly if missing.

    Called only for the 'real aero' case C, so that importing this module does
    not itself require the polar tables to be present.  Fails loudly rather than
    letting PolarDatabase fall back to a surrogate, which would silently
    invalidate the comparison.
    """
    global POLAR_DIR
    if POLAR_DIR is None:
        POLAR_DIR = ref_solver.polar_dir(REF_ROOT)
    if POLAR_DIR is None:
        raise SystemExit(
            f"no NACA 0012 polar tables under {REF_ROOT} (looked in "
            "airfoil_polars/naca0012 and polars/naca0012); the 'real aero' "
            "case C would otherwise silently use a surrogate polar")
    return POLAR_DIR

TWO_PI = 2.0 * np.pi

# The FSI stall layer's time constants are in semi-chords (tau = c/2W);
# sharp-vawt-lab's are in chords (tau = c/W), hence the factor 1/2 when
# translating FSI -> BEM.  K_v, the separated moment and the drag model are
# matched too so that case D compares the SAME aero model on both sides.
STALL_FSI = dict(stall=True, stall_alpha_stall=14.0, stall_T_a=0.0,
                 stall_T_f=3.0, stall_T_v=6.0, stall_K_v=0.5,
                 stall_cm_sep_const=0.05, viscous_cd0=8.0e-3)
STALL_BEM = dict(ds_Ta=1.0e-6, ds_Tf=1.5, ds_Tv=3.0, ds_Kv=0.5, ds_Knc=0.0)


def viterna_polar():
    return StaticPolar.viterna(alpha_stall=STALL_FSI['stall_alpha_stall'],
                               cd0=STALL_FSI['viscous_cd0'])


# --------------------------------------------------------------------------
# matched configuration
# --------------------------------------------------------------------------
def geometry():
    # H = 1 m  ->  the BEM is a per-unit-span (2-D) model like the DVM
    return Geometry(R=5.0, c=1.0, H=1.0, N=1, ar=0.35, sp=0.25,
                    arm_d=0.0, n_arm=2, arm_cd=1.0)


def mass_props():
    # xcw = -sp - dcw*c = -0.25 + 0.60 = 0.35 c   (matches r_cw = 0.35 in CONFIG)
    return MassProps(mb=5.0, xbcg=0.25, mc=2.5, dcw=-0.60,
                     balance=0, bias_deg=0.0, J_hub=0.02)


def flat_plate(alpha):
    """Exact thin-flat-plate polars (alpha in radians)."""
    return 2.0 * np.pi * np.sin(alpha), np.zeros_like(alpha)


def _aero(case):
    """AeroConfig + polar database for each comparison case."""
    g = geometry()
    common = dict(polar_dir='', use_tip_loss=False, use_dmst=False,
                  use_dynamic_inflow=False, use_flow_curvature=False,
                  cd_add=0.0, win_deg=25.0, wstop=50.0, cb=0.15, kc=0.0,
                  mu_c=0.0, r_bearing=0.0)
    if case == 'real':
        ae = AeroConfig(use_dynamic_stall=True, a_fix=0.2, **common)
        pdb = PolarDatabase(_require_polar_dir(), g)
    elif case == 'viterna':
        ae = AeroConfig(use_dynamic_stall=True, a_fix=0.0, **STALL_BEM,
                        **common)
        pol = viterna_polar()
        pdb = PolarDatabase.from_callable(
            lambda a: pol.eval(a)[:2], cla=TWO_PI, geo=g)
    else:
        unsteady = (case == 'unsteady')
        ae = AeroConfig(use_dynamic_stall=unsteady, a_fix=0.0,
                        ds_Ta=1e-3, ds_Kv=0.0, ds_Knc=1.0, **common)
        pdb = PolarDatabase.from_callable(flat_plate, cla=2.0 * np.pi, geo=g)
    return g, ae, pdb


def run_bem(tsr, case, n_rev=4.0, dt=1e-3, stride=5, U=10.0, rho=1.225):
    g = geometry()
    omega = tsr * U / g.R
    T_rev = TWO_PI / omega
    sim_cfg = SimConfig(U=U, tsr=tsr, free=False, psi0_deg=5.0, dt=dt,
                        T_max=n_rev * T_rev, stride=stride, rho=rho,
                        n_rev_fixed=int(round(n_rev)))
    _, ae, pdb = _aero(case)
    sim = CycloturbineSim(g, mass_props(), ae, sim_cfg, polar_db=pdb)
    res = sim.run(store_history=True)
    h = res['history']
    return dict(t=h['t'], phi=h['th'] % TWO_PI, phi_un=h['th'], psi=h['psi'],
                alpha=h['alpha'], Cl=None, torque=h['Tsum'], raw=h, res=res)


def run_fsi(tsr, n_rev=4.0, tag='', aero='potential'):
    cfg = dict(CONFIG)
    cfg['omega'] = tsr * cfg['U_inf'] / cfg['R']
    acfg, akey = aero_config(aero)
    cfg.update(acfg)
    if akey == 'stall':
        cfg['stall_polar'] = viterna_polar()
    if akey != 'potential':
        # match the FSI stall layer to the BEM's dynamic-stall model (case D)
        cfg.update({k: v for k, v in STALL_FSI.items() if k != 'stall'})
    fsi = FSICoupler(**cfg)
    h = fsi.run(n_revolutions=n_rev, verbose=False)
    fsi.save_history(f'fsi_history_bem{tag}.txt')
    return fsi, h, akey


# --------------------------------------------------------------------------
# comparison helpers
# --------------------------------------------------------------------------
def last_rev(t, T_rev):
    return t >= t[-1] - T_rev


def binned(phi, y, nb=72):
    """Azimuth-bin average over [0, 2pi)."""
    idx = np.minimum((phi / TWO_PI * nb).astype(int), nb - 1)
    c = np.bincount(idx, weights=y, minlength=nb)
    n = np.bincount(idx, minlength=nb).astype(float)
    return (np.arange(nb) + 0.5) / nb * 360.0, c / np.maximum(n, 1.0)


def wrap_deg(a):
    return (np.degrees(a) + 180.0) % 360.0 - 180.0


def stats(name, phi, psi, alpha, torque, T_rev, stop_deg=25.0, out=None):
    """Print last-revolution statistics for one already-masked series."""
    m = np.ones(len(phi), bool)
    a = wrap_deg(alpha)
    line = (f"  {name:18s} <psi>={np.degrees(psi[m]).mean():+6.2f}  "
            f"psi [{np.degrees(psi[m]).min():+6.2f},{np.degrees(psi[m]).max():+6.2f}]  "
            f"<a>={a[m].mean():+7.2f}  "
            f"|a|max={np.abs(a[m]).max():6.1f}  "
            f"stop={100*np.mean(np.abs(np.degrees(psi[m])) > 0.98*stop_deg):5.1f}%  "
            f"<T>={torque[m].mean():+8.3f}")
    print(line)
    if out is not None:
        out[name] = line


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    tsr = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    n_rev = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
    aero = sys.argv[3] if len(sys.argv) > 3 else 'potential'

    print("=" * 78)
    print(f"sharpFSI (2-D DVM + CPPC) vs sharp-vawt-lab BEM   "
          f"TSR = {tsr:.2f}, {n_rev:.0f} revolutions, FSI aero = {aero}")
    print("=" * 78)

    # tag every artifact with the aero model as well as the TSR: otherwise a
    # potential run and a stalled run at the same TSR silently overwrite each
    # other's history and figure
    # tag every artifact with the aero model as well as the TSR: otherwise a
    # potential run and a stalled run at the same TSR silently overwrite each
    # other's history and figure
    akey = aero_config(aero)[1]
    tag = f"_{akey}_tsr{tsr:.2f}".replace('.', 'p')
    fsi, h_fsi, akey = run_fsi(tsr, n_rev, tag, aero)
    omega = fsi.omega
    T_rev = TWO_PI / omega
    t = h_fsi['t']
    m = last_rev(t, T_rev)
    cpl = h_fsi['Q_rotor'] * omega / (0.5 * fsi.rho * fsi.U_inf ** 3
                                      * 2.0 * fsi.R)
    print(f"FSI: {fsi.step_count} steps, {fsi.fluid.nw} wake vortices, "
          f"mean coupling iters {np.mean(fsi.iter_counts):.2f}, "
          f"<Cp>={cpl[m].mean():+.4f}")
    if akey != 'potential':
        k = h_fsi['kappa'][m]
        print(f"     stall layer: polar='{fsi.stall_polar.name}', "
              f"T_f={fsi.stall_model.T_f:g} T_v={fsi.stall_model.T_v:g} "
              f"K_v={fsi.stall_model.K_v:g}, "
              f"kappa in [{k.min():+.3f},{k.max():+.3f}]")

    cases = {}
    for case in ('quasi', 'unsteady', 'viterna', 'real'):
        cases[case] = run_bem(tsr, case, n_rev)
        r = cases[case]
        st = r['res']
        print(f"BEM[{case:8s}] {len(r['t'])} rows, cp={st['cp']:+.4f}, "
              f"stop={st['stop_pct']:.1f}%, aoa_max={st['aoa_max']:.1f} deg, "
              f"energy resid (abs)={abs(st['energy_balance']):.2e}")

    # align to the common length (both end at n_rev revolutions)
    n = min(len(t), *(len(cases[c]['t']) for c in cases))

    # ---------------------------------------------------------------- table
    print("\n" + "-" * 78)
    print(f"last-revolution statistics (azimuth window = last {T_rev:.3f} s)")
    rows = {}
    m_f = last_rev(t[:n], T_rev)
    stats(f'FSI ({akey})', h_fsi['phi'][:n][m_f], h_fsi['psi'][:n][m_f],
          h_fsi['alpha'][:n][m_f], h_fsi['Q_rotor'][:n][m_f], T_rev, out=rows)
    for case in ('quasi', 'unsteady', 'viterna', 'real'):
        c = cases[case]
        ph = c['phi'][:n]
        m_c = _mask_last(c, n, T_rev)
        stats(f"BEM {case}", ph[m_c], c['psi'][:n][m_c], c['alpha'][:n][m_c],
              c['torque'][:n][m_c], T_rev, out=rows)

    # ------------------------------------------------------- differences
    print("\n" + "-" * 78)
    print("difference vs the FSI (last revolution, pointwise on the common grid)")
    ref_psi, ref_al = h_fsi['psi'][:n], h_fsi['alpha'][:n]
    idx_f = np.flatnonzero(m_f)
    for case in ('quasi', 'unsteady', 'viterna', 'real'):
        c = cases[case]
        m_c = _mask_last(c, n, T_rev)
        idx_c = np.flatnonzero(m_c)
        k = min(len(idx_f), len(idx_c))
        i_f, i_c = idx_f[:k], idx_c[:k]
        dpsi = wrap_deg(c['psi'][:n][i_c] - ref_psi[i_f])
        dal = wrap_deg(c['alpha'][:n][i_c] - ref_al[i_f])
        dT = c['torque'][:n][i_c] - h_fsi['Q_rotor'][:n][i_f]
        print(f"  {case:9s} rms(dpsi)={np.sqrt(np.mean(dpsi**2)):6.2f} deg  "
              f"max|dpsi|={np.abs(dpsi).max():7.2f} deg   "
              f"rms(dalpha)={np.sqrt(np.mean(dal**2)):6.2f} deg  "
              f"<dT>={dT.mean():+7.3f} Nm")
        # azimuth-binned (removes small timing offsets)
        pb, pb_r = binned(c['phi'][:n][m_c], wrap_deg(c['psi'][:n][m_c]))
        _, pf_r = binned(h_fsi['phi'][:n][m_f], wrap_deg(h_fsi['psi'][:n][m_f]))
        if len(pb_r) == len(pf_r):
            d = pb_r - pf_r
            print(f"            azimuth-binned rms(dpsi)={np.sqrt(np.mean(d**2)):6.2f} deg")

    # ------------------------------------------------------------- history
    for case in ('quasi', 'unsteady', 'viterna', 'real'):
        c = cases[case]
        name = {'quasi': f'bem_history_matched{tag}.txt',
                'unsteady': f'bem_history_unsteady{tag}.txt',
                'viterna': f'bem_history_viterna{tag}.txt',
                'real': f'bem_history_real{tag}.txt'}[case]
        np.savetxt(name, np.column_stack(
            [c['t'], c['phi_un'], c['psi'], c['alpha'], c['torque']]),
            header='t th psi alpha Tsum', comments='# ')
        print(f"saved {name}")

    # ---------------------------------------------------------------- plots
    make_plots(tsr, n_rev, T_rev, t[:n], m_f, h_fsi, cases, n, tag, akey,
               fsi_polar=fsi.stall_polar)
    print(f"saved comparison_fsi_vs_bem{tag}.png")


def _mask_last(c, n, T_rev):
    t = c['t'][:n]
    return t >= t[-1] - T_rev


def make_plots(tsr, n_rev, T_rev, t, m_f, h_fsi, cases, n, tag='',
               akey='potential', fsi_polar=None):
    styles = {'quasi': ('C0', '-', 'BEM A matched (quasi-steady)'),
              'unsteady': ('C1', '-', 'BEM B matched (+apparent mass)'),
              'viterna': ('C2', '-', 'BEM D matched stall (same polar)'),
              'real': ('C3', '-', 'BEM C real aero (NACA0012 + DS, a=0.2)')}
    phi = np.degrees(h_fsi['phi'][:n])
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # 1. pitch vs azimuth
    ax = axes[0, 0]
    ax.plot(phi[m_f], np.degrees(h_fsi['psi'][:n][m_f]), 'k-', lw=2.2,
            label=f'FSI (DVM, {akey})')
    for case, (col, ls, lab) in styles.items():
        c = cases[case]
        mc = _mask_last(c, n, T_rev)
        ax.plot(np.degrees(c['phi'][:n][mc]), np.degrees(c['psi'][:n][mc]),
                color=col, ls=ls, lw=1.3, label=lab)
    ax.axhline(25, color='r', ls=':', lw=0.8)
    ax.axhline(-25, color='r', ls=':', lw=0.8)
    ax.set_xlabel('azimuth $\\phi$ (deg)')
    ax.set_ylabel('pitch $\\psi$ (deg)')
    ax.set_title('pitch angle, last revolution')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 2. effective alpha
    ax = axes[0, 1]
    ax.plot(phi[m_f], wrap_deg(h_fsi['alpha'][:n][m_f]), 'k-', lw=2.2,
            label='FSI (DVM)')
    for case, (col, ls, lab) in styles.items():
        c = cases[case]
        mc = _mask_last(c, n, T_rev)
        ax.plot(np.degrees(c['phi'][:n][mc]), wrap_deg(c['alpha'][:n][mc]),
                color=col, ls=ls, lw=1.3, label=lab)
    ax.axhline(0, color='k', ls=':', lw=0.7)
    ax.set_xlabel('azimuth $\\phi$ (deg)')
    ax.set_ylabel('effective $\\alpha$ (deg)')
    ax.set_title('effective angle of attack')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 3. Cl vs alpha (FSI, with the static polar for reference)
    ax = axes[1, 0]
    ax.plot(wrap_deg(h_fsi['alpha'][:n][m_f]), h_fsi['Cl'][:n][m_f], 'k.',
            ms=2, label=f'FSI ({akey})')
    a_ref = np.linspace(-95, 95, 400)
    ax.plot(a_ref, 2 * np.pi * np.sin(np.deg2rad(a_ref)), 'g--', lw=1,
            label='$2\\pi\\sin\\alpha$')
    if fsi_polar is not None:
        ax.plot(a_ref, fsi_polar.eval(np.deg2rad(a_ref))[0], 'm-', lw=1.2,
                label='static stall polar')
    ax.set_xlabel('effective $\\alpha$ (deg)')
    ax.set_ylabel('$C_l$')
    ax.set_title(f'FSI instantaneous $C_l$ (last rev, {akey})')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(-100, 100)

    # 4. rotor torque
    ax = axes[1, 1]
    ax.plot(phi[m_f], h_fsi['Q_rotor'][:n][m_f], 'k-', lw=2.2,
            label=f'FSI (DVM, {akey})')
    for case, (col, ls, lab) in styles.items():
        c = cases[case]
        mc = _mask_last(c, n, T_rev)
        ax.plot(np.degrees(c['phi'][:n][mc]), c['torque'][:n][mc],
                color=col, ls=ls, lw=1.3, label=lab)
    ax.axhline(0, color='k', ls=':', lw=0.7)
    ax.set_xlabel('azimuth $\\phi$ (deg)')
    ax.set_ylabel('$Q_{rotor}$ (N m per unit span)')
    ax.set_title('rotor-axis torque')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f'sharpFSI (2-D DVM + CPPC) vs sharp-vawt-lab BEM   '
                 f'(TSR={tsr:.2f}, U=10 m/s, H=1 m, last revolution)',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(f'comparison_fsi_vs_bem{tag}.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
