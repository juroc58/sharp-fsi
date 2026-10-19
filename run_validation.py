"""
run_validation.py
=================
Validation run for the 2D VAWT FSI suite: a number of rotor revolutions of the
matched config (R=5 m, c=1 m, U=10 m/s, TSR=1), saving the time history and
producing the two validation figures:

  fsi_validation_plots.png :
      left  - phase portrait  (pitch psi vs azimuth phi)
      right - aerodynamic hysteresis (Cl vs effective alpha)
  fsi_time_history.png     - pitch / alpha / coefficients / moment histories

Usage:  python3 run_validation.py [n_rev] [dt] [tsr] [aero]

  n_rev : rotor revolutions                     (default 4)
  dt    : time step, s                          (default 0.005)
  tsr   : tip-speed ratio omega*R/U             (default 1.0 = matched config)
  aero  : aerodynamic model
            potential  pure potential flow, no stall (the matched config for
                       the sharp-vawt-lab comparison)                [default]
            stall      potential flow + built-in Viterna/dynamic-stall
                       correction layer (aero_correction.py)
            naca       + dynamic stall on a tabulated NACA 0012 polar
                       (needs the Sheldahl-Klimas CSV; see
                       StaticPolar.naca0012)

Output file names are tagged with the TSR and the aero model when they differ
from the defaults, so runs never overwrite each other.
"""

from __future__ import annotations

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from fsi_main import FSICoupler

CONFIG = dict(
    chord=1.0, sp=0.25, ar=0.35, R=5.0, omega=2.0, U_inf=10.0, rho=1.225,
    dt=0.005, induction=0.0,
    m_blade=5.0, cg_blade=0.25, m_cw=2.5, r_cw=0.35,
    c_b=0.15, mu_c=0.0, win_deg=25.0, wstop=50.0, psi0_deg=5.0,
    n_panels=32, wake_cap=600, core_bound=0.002, core_wake=0.02,
    wake_substeps=1, viscous_cd0=0.0,
    n_coupling_iter=3, coupling_tol=1e-4, relax=0.7,
)


def aero_config(name='potential'):
    """Return the FSICoupler kwargs for a named aerodynamic model."""
    name = (name or 'potential').lower()
    if name in ('potential', 'none', 'inviscid'):
        return {'stall': False}, 'potential'
    if name in ('stall', 'viterna'):
        return dict(stall=True, stall_alpha_stall=14.0,
                    viscous_cd0=8.0e-3), 'stall'
    if name in ('naca', 'naca0012'):
        from aero_correction import StaticPolar
        polar = StaticPolar.naca0012()
        return dict(stall=True, stall_polar=polar,
                    viscous_cd0=8.0e-3), 'naca'
    raise SystemExit(f"unknown aero model '{name}' "
                     "(use potential | stall | naca)")


def run_validation(n_rev=4.0, dt=None, tsr=None, aero='potential'):
    cfg = dict(CONFIG)
    if dt is not None:
        cfg['dt'] = dt
    if tsr is not None:
        cfg['omega'] = tsr * cfg['U_inf'] / cfg['R']
    acfg, akey = aero_config(aero)
    cfg.update(acfg)
    # output tag: the default (matched, TSR = 1, potential) run keeps the
    # canonical names
    tag = ''
    if tsr is not None:
        tag += f"_tsr{tsr:.2f}".replace('.', 'p')
    if akey != 'potential':
        tag += f"_{akey}"
    print("=" * 66)
    print("2D VAWT FSI validation run")
    print("=" * 66)
    for k in ('chord', 'R', 'omega', 'U_inf', 'dt', 'n_panels', 'wake_cap',
              'm_blade', 'm_cw', 'r_cw', 'win_deg'):
        print(f"  {k:12s} = {cfg[k]}")
    print(f"  TSR          = {cfg['omega']*cfg['R']/cfg['U_inf']:.2f}")
    print(f"  aero model   = {akey}")
    fsi = FSICoupler(**cfg)
    print(f"  psi_eq       = {np.rad2deg(fsi.struct.p.psi_eq):.2f} deg")
    print(f"  fn_ratio     = {fsi.struct.p.fn_ratio:.2f}")
    print(f"  I_p          = {fsi.struct.p.I_p:.4f} kg m^2")
    print("-" * 66)
    h = fsi.run(n_revolutions=n_rev, verbose=True)
    fsi.save_history(f'fsi_validation_history{tag}.txt')
    return fsi, h, tag, akey


def summarize(fsi, h, akey='potential'):
    p = fsi.struct.p
    T_rev = 2 * np.pi / fsi.omega
    t = h['t']
    m = t >= t[-1] - T_rev
    print("\n" + "-" * 66)
    print("last-revolution summary")
    print(f"  <psi>       = {np.rad2deg(h['psi'][m].mean()):+7.2f} deg   "
          f"range [{np.rad2deg(h['psi'][m].min()):+7.2f}, "
          f"{np.rad2deg(h['psi'][m].max()):+7.2f}]")
    print(f"  <alpha>     = {np.rad2deg(h['alpha'][m].mean()):+7.2f} deg   "
          f"range [{np.rad2deg(h['alpha'][m].min()):+7.2f}, "
          f"{np.rad2deg(h['alpha'][m].max()):+7.2f}]")
    print(f"  <Cl>        = {h['Cl'][m].mean():+7.3f}   "
          f"range [{h['Cl'][m].min():+7.3f}, {h['Cl'][m].max():+7.3f}]")
    print(f"  <Cd>        = {h['Cd'][m].mean():+7.4f}")
    print(f"  <Cm>        = {h['Cm'][m].mean():+7.4f}")
    if akey != 'potential' and 'kappa' in h:
        print(f"  <kappa>     = {h['kappa'][m].mean():+7.4f}   "
              f"range [{h['kappa'][m].min():+7.4f}, {h['kappa'][m].max():+7.4f}]")
    print(f"  <Q_aero>    = {h['Q_aero'][m].mean():+7.3f} N m   "
          f"<Q_cent> = {h['Q_cent'][m].mean():+7.3f} N m")
    print(f"  stop time   = {100*np.mean(np.abs(h['psi'])>p.win*0.98):.1f} %")
    # rotor torque / power (2-D result interpreted per unit span)
    Wmag = np.abs(h['W'])
    area = 2.0 * fsi.R          # per unit span, frontal area
    cp = h['Q_rotor'] * fsi.omega / (0.5 * fsi.rho * fsi.U_inf ** 3 * area)
    print(f"  <Q_rotor>   = {h['Q_rotor'][m].mean():+7.3f} N m/span   "
          f"<Cp> = {cp[m].mean():+7.4f}")
    al = np.abs(np.rad2deg(h['alpha']))
    al = np.minimum(al, 360.0 - al)
    if akey == 'potential':
        print(f"  model validity (last rev): |alpha|>15 deg for "
              f"{100*np.mean(al[m] > 15.0):.1f} % of steps (outside the "
              f"attached-flow validity of a pure potential model), "
              f"|W|<2 m/s for {100*np.mean(Wmag[m] < 2.0):.1f} % "
              f"(coefficients not meaningful there)")
    else:
        print(f"  aero regime (last rev): |alpha|>15 deg for "
              f"{100*np.mean(al[m] > 15.0):.1f} % of steps (handled by the "
              f"stall correction), |W|<2 m/s for "
              f"{100*np.mean(Wmag[m] < 2.0):.1f} %")
    print(f"  wake vortices at end = {fsi.fluid.nw} "
          f"(merges {fsi.fluid.merge_count})")
    print(f"  Kelvin budget = {fsi.fluid.circulation_budget():.2e}")


def make_validation_figure(fsi, h, tag='', akey='potential'):
    T_rev = 2 * np.pi / fsi.omega
    t = h['t']
    phi = np.rad2deg(h['phi'])
    psi = np.rad2deg(h['psi'])
    alpha = np.rad2deg(h['alpha'])
    cl, cd, cm = h['Cl'], h['Cd'], h['Cm']
    n_rev = int(np.floor(t[-1] / T_rev))

    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 2, figure=fig)

    # ---- phase portrait -------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    cols = plt.cm.viridis(np.linspace(0.15, 1.0, max(n_rev, 1)))
    phi_un = np.unwrap(h['phi'])
    for r in range(n_rev):
        m = (t >= r * T_rev) & (t < (r + 1) * T_rev)
        ax.plot(np.rad2deg(phi_un[m]), psi[m], color=cols[r], lw=1.4,
                label=f'rev {r+1}')
    ax.axhline(0, color='k', ls=':', lw=0.8)
    for x in (0, 90, 180, 270, 360):
        ax.axvline(x, color='gray', ls=':', lw=0.5, alpha=0.5)
    ax.set_xlabel('azimuth $\\phi$ (deg)')
    ax.set_ylabel('pitch $\\psi$ (deg)')
    ax.set_title('phase portrait: pitch vs azimuth')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ---- hysteresis -----------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    for r in range(n_rev):
        m = (t >= r * T_rev) & (t < (r + 1) * T_rev)
        ax.plot(alpha[m], cl[m], color=cols[r], lw=1.2, label=f'rev {r+1}')
    a_ref = np.linspace(-95, 95, 400)
    if fsi.stall_polar is not None:
        ax.plot(a_ref, fsi.stall_polar.eval(np.deg2rad(a_ref))[0], 'k-',
                lw=1, alpha=0.7, label='static stall polar')
    ax.plot(a_ref, 2 * np.pi * np.sin(np.deg2rad(a_ref)), 'k--', lw=1,
            alpha=0.5, label='2$\\pi$sin$\\alpha$')
    ax.set_xlabel('effective $\\alpha$ (deg)')
    ax.set_ylabel('$C_l$')
    ax.set_title('$C_l$ vs $\\alpha$: ' +
                 ('potential-flow hysteresis' if akey == 'potential'
                  else 'with stall correction'))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(-100, 100)

    fig.suptitle(f'2D VAWT FSI validation: {n_rev} revolutions, '
                 f'TSR = {fsi.omega*fsi.R/fsi.U_inf:.2f} '
                 f'(DVM + CPPC, aero = {akey})',
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(f'fsi_validation_plots{tag}.png', dpi=160)
    plt.close(fig)
    print(f"saved fsi_validation_plots{tag}.png")


def make_history_figure(fsi, h, tag=''):
    t = h['t']
    T_rev = 2 * np.pi / fsi.omega
    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    ax = axes[0]
    ax.plot(t, np.rad2deg(h['psi']), 'b-', lw=1)
    ax.axhline(0, color='k', ls=':', lw=0.8)
    ax.axhline(fsi.struct.p.win_deg, color='r', ls='--', lw=0.8)
    ax.axhline(-fsi.struct.p.win_deg, color='r', ls='--', lw=0.8)
    ax.set_ylabel('$\\psi$ (deg)')
    ax.set_title('pitch angle')
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t, np.rad2deg(h['alpha']), 'r-', lw=1)
    ax.axhline(0, color='k', ls=':', lw=0.8)
    ax.set_ylabel('$\\alpha_{eff}$ (deg)')
    ax.set_title('effective angle of attack')
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(t, h['Cl'], 'b-', lw=1, label='$C_l$')
    ax.plot(t, h['Cd'], 'r-', lw=1, label='$C_d$')
    if 'kappa' in h and np.any(np.abs(h['kappa'] - 1.0) > 1e-12):
        ax.plot(t, h['kappa'], 'g-', lw=0.8, alpha=0.8,
                label='$\\kappa$ (stall scaling)')
    ax.axhline(0, color='k', ls=':', lw=0.8)
    ax.set_ylabel('coefficient')
    ax.legend()
    ax.set_title('aerodynamic coefficients')
    ax.grid(alpha=0.3)

    ax = axes[3]
    ax.plot(t, h['Q_aero'], 'b-', lw=1, label='$Q_{aero}$')
    ax.plot(t, h['Q_cent'], 'g--', lw=1, label='$Q_{cent}$')
    ax.axhline(0, color='k', ls=':', lw=0.8)
    ax.set_ylabel('moment (N m)')
    ax.legend()
    ax.set_title('pitch moments')
    ax.grid(alpha=0.3)

    ax = axes[4]
    ax.plot(t, h['Q_rotor'], 'm-', lw=1, label='$Q_{rotor}$ (per span)')
    ax.axhline(0, color='k', ls=':', lw=0.8)
    ax.axhline(h['Q_rotor'].mean(), color='b', ls='--', lw=0.8,
               label=f"mean = {h['Q_rotor'].mean():+.2f} N m")
    ax.set_xlabel('time (s)')
    ax.set_ylabel('torque (N m)')
    ax.legend()
    ax.set_title('rotor-axis torque')
    ax.grid(alpha=0.3)
    for a in axes:
        for r in range(1, int(t[-1] / T_rev) + 1):
            a.axvline(r * T_rev, color='gray', ls=':', lw=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(f'fsi_time_history{tag}.png', dpi=150)
    plt.close(fig)
    print(f"saved fsi_time_history{tag}.png")


def main():
    n_rev = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    dt = float(sys.argv[2]) if len(sys.argv) > 2 else None
    tsr = float(sys.argv[3]) if len(sys.argv) > 3 else None
    aero = sys.argv[4] if len(sys.argv) > 4 else 'potential'
    fsi, h, tag, akey = run_validation(n_rev, dt, tsr, aero)
    summarize(fsi, h, akey)
    make_validation_figure(fsi, h, tag, akey)
    make_history_figure(fsi, h, tag)


if __name__ == "__main__":
    main()
