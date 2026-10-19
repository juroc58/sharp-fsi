"""
design_study.py
===============
Design study / parameter sweep for the 2-D VAWT FSI section.

Two sweeps are provided:

  1. TSR sweep with the baseline counterweight -- the rotor power coefficient
     Cp(TSR) curve (the single most useful design curve), plus the pitch
     statistics at each TSR.
  2. Counterweight sweep (m_cw, r_cw) at a chosen design TSR -- a map of
     <Cp>, pitch-bias, stop fraction and angle-of-attack RMS, used to retune
     the CPPC counterweight for the design point.

Both use short (default 2 revolution) runs of the full coupled solver and
report last-revolution statistics, so the results are cycle-averaged.

Usage:
  python3 design_study.py [mode] [aero] [n_rev]

  mode : tsr | cw | both      (default both)
  aero : potential | stall | naca   (default stall)

Outputs: design_tsr_sweep.csv, design_cw_sweep.csv, design_study_tsr.png,
         design_study_cw.png
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fsi_main import FSICoupler
from run_validation import CONFIG, aero_config

TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------
def run_point(tsr, m_cw=None, r_cw=None, n_rev=2.0, aero='stall', **over):
    """One coupled run; returns last-revolution cycle statistics."""
    cfg = dict(CONFIG)
    cfg['omega'] = tsr * cfg['U_inf'] / cfg['R']
    if m_cw is not None:
        cfg['m_cw'] = m_cw
    if r_cw is not None:
        cfg['r_cw'] = r_cw
    cfg.update(over)
    acfg, akey = aero_config(aero)
    cfg.update(acfg)
    fsi = FSICoupler(**cfg)
    h = fsi.run(n_revolutions=n_rev, verbose=False)

    T_rev = TWO_PI / fsi.omega
    t = h['t']
    m = t >= t[-1] - T_rev
    psi = np.rad2deg(h['psi'][m])
    al = np.rad2deg(h['alpha'][m])
    al = np.where(np.abs(al) > 180.0, al - np.sign(al) * 360.0, al)
    cp = h['Q_rotor'] * fsi.omega / (0.5 * fsi.rho * fsi.U_inf ** 3
                                     * 2.0 * fsi.R)
    win = np.rad2deg(fsi.struct.p.win)
    stop = 100.0 * np.mean(np.abs(psi) > 0.98 * win)
    return dict(
        tsr=tsr, m_cw=fsi.struct.p.m_cw, r_cw=fsi.struct.p.xi_cw,
        fn_ratio=fsi.struct.p.fn_ratio,
        psi_eq_deg=np.rad2deg(fsi.struct.p.psi_eq),
        psi_mean=psi.mean(), psi_min=psi.min(), psi_max=psi.max(),
        stop_pct=stop, alpha_mean=al.mean(), alpha_rms=float(np.sqrt(
            np.mean(al ** 2))), alpha_absmax=float(np.abs(al).max()),
        cp_mean=float(cp[m].mean()), q_mean=float(h['Q_rotor'][m].mean()),
        cl_max=float(np.abs(h['Cl'][m]).max()),
        steps=fsi.step_count, secs=getattr(fsi, 'wall_time', float('nan')))


KEYS = ('tsr', 'm_cw', 'r_cw', 'fn_ratio', 'psi_eq_deg', 'psi_mean', 'psi_min',
        'psi_max', 'stop_pct', 'alpha_mean', 'alpha_rms', 'alpha_absmax',
        'cp_mean', 'q_mean', 'cl_max', 'steps', 'secs')


def _header(rows):
    return ','.join(KEYS) + '\n' + '\n'.join(
        ','.join(f'{r[k]:.6g}' if isinstance(r[k], float) else str(r[k])
                 for k in KEYS) for r in rows) + '\n'


def _line(r):
    return (f"TSR={r['tsr']:4.2f}  m_cw={r['m_cw']:4.2f} xi_cw={r['r_cw']:4.2f}  "
            f"psi_eq={r['psi_eq_deg']:+6.2f} fn={r['fn_ratio']:4.2f}  "
            f"<psi>={r['psi_mean']:+6.2f} [{r['psi_min']:+6.2f},"
            f"{r['psi_max']:+6.2f}]  stop={r['stop_pct']:5.1f}%  "
            f"<a>={r['alpha_mean']:+7.2f} a_rms={r['alpha_rms']:6.2f}  "
            f"<Cp>={r['cp_mean']:+.4f}  <Q>={r['q_mean']:+8.2f}")


# --------------------------------------------------------------------------
def tsr_sweep(aero='stall', n_rev=2.0, tsr_list=(1.0, 1.5, 2.0, 2.5, 3.0,
                                                 3.5, 4.0), **cw):
    print('=' * 78)
    print(f'TSR sweep (aero={aero}, {n_rev} revs, counterweight={cw or "baseline"})')
    print('=' * 78)
    rows = []
    for tsr in tsr_list:
        r = run_point(tsr, n_rev=n_rev, aero=aero, **cw)
        rows.append(r)
        print('  ' + _line(r))
    with open('design_tsr_sweep.csv', 'w') as f:
        f.write(_header(rows))
    print('saved design_tsr_sweep.csv')
    return rows


def cw_sweep(tsr=3.0, aero='stall', n_rev=2.0,
             m_list=(0.5, 1.0, 1.5, 2.5, 3.5),
             r_list=(0.05, 0.15, 0.25, 0.35, 0.50)):
    print('=' * 78)
    print(f'counterweight sweep at TSR={tsr} (aero={aero}, {n_rev} revs)')
    print('=' * 78)
    rows = []
    for mc in m_list:
        for rc in r_list:
            r = run_point(tsr, m_cw=mc, r_cw=rc, n_rev=n_rev, aero=aero)
            rows.append(r)
            print('  ' + _line(r))
    with open('design_cw_sweep.csv', 'w') as f:
        f.write(_header(rows))
    print('saved design_cw_sweep.csv')
    return rows


# --------------------------------------------------------------------------
def plot_tsr(rows, best=None):
    if not rows:
        return
    tsr = np.array([r['tsr'] for r in rows])
    cp = np.array([r['cp_mean'] for r in rows])
    stop = np.array([r['stop_pct'] for r in rows])
    ps = np.array([r['psi_mean'] for r in rows])
    ar = np.array([r['alpha_rms'] for r in rows])
    lab = ('baseline counterweight' if best is None else
           f"$m_{{cw}}$={best['m_cw']:.2f} kg, $r_{{cw}}$={best['r_cw']:.2f} m")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    ax[0].plot(tsr, cp, 'o-', color='C3', label=lab)
    i = int(np.argmax(cp))
    ax[0].plot(tsr[i], cp[i], 'k*', ms=14,
               label=f'max in sweep: TSR={tsr[i]:.2f}, Cp={cp[i]:.3f}')
    ax[0].set_xlabel('TSR = $\\omega R/U$')
    ax[0].set_ylabel('$C_P$')
    ax[0].set_title('power coefficient')
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].plot(tsr, stop, 's-', color='C0')
    ax[1].set_xlabel('TSR')
    ax[1].set_ylabel('stop fraction (%)')
    ax[1].set_title('pitch stop engagement')
    ax[1].grid(alpha=0.3)
    ax2 = ax[1].twinx()
    ax2.plot(tsr, ps, '^-', color='C2')
    ax2.set_ylabel('$<\\psi>$ (deg)', color='C2')
    ax[2].plot(tsr, ar, 'd-', color='C1')
    ax[2].set_xlabel('TSR')
    ax[2].set_ylabel('$\\alpha_{rms}$ (deg)')
    ax[2].set_title('angle-of-attack amplitude')
    ax[2].grid(alpha=0.3)
    fig.suptitle('sharpFSI design study: TSR sweep (2-D DVM + CPPC)')
    fig.tight_layout()
    fig.savefig('design_study_tsr.png', dpi=140)
    plt.close(fig)
    print('saved design_study_tsr.png')


def plot_cw(rows, tsr):
    if not rows:
        return
    ms = sorted({r['m_cw'] for r in rows})
    rs = sorted({r['r_cw'] for r in rows})
    cp = np.full((len(rs), len(ms)), np.nan)
    stop = np.full_like(cp, np.nan)
    ps = np.full_like(cp, np.nan)
    for r in rows:
        i, j = rs.index(r['r_cw']), ms.index(r['m_cw'])
        cp[i, j] = r['cp_mean']
        stop[i, j] = r['stop_pct']
        ps[i, j] = r['psi_mean']
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for a, Z, ttl, cmap in ((ax[0], cp, '$C_P$', 'viridis'),
                            (ax[1], stop, 'stop fraction (%)', 'magma'),
                            (ax[2], ps, '$<\\psi>$ (deg)', 'coolwarm')):
        im = a.imshow(Z, origin='lower', aspect='auto', cmap=cmap,
                      extent=[min(ms) - 0.25, max(ms) + 0.25,
                              min(rs) - 0.05, max(rs) + 0.05])
        a.set_xlabel('$m_{cw}$ (kg)')
        a.set_ylabel('$\\xi_{cw}$ (m)')
        a.set_title(ttl)
        fig.colorbar(im, ax=a)
        for i, r_ in enumerate(rs):
            for j, m_ in enumerate(ms):
                a.text(m_, r_, f'{Z[i, j]:.2f}' if ttl != '$C_P$'
                       else f'{Z[i, j]:.3f}', ha='center', va='center',
                       fontsize=7, color='w')
        a.plot([2.5], [0.35], 'kx', ms=9, mew=2)
    ax[0].plot([], [], 'kx', label='baseline (TSR=1) counterweight')
    ax[0].legend(fontsize=7, loc='upper right')
    fig.suptitle(f'counterweight map at TSR={tsr:.1f} '
                 '($m_{cw}$ = counterweight mass, $\\xi_{cw}$ = its lever arm;'
                 '  x = TSR=1 baseline)')
    fig.tight_layout()
    fig.savefig('design_study_cw.png', dpi=140)
    plt.close(fig)
    print('saved design_study_cw.png')


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'both'
    aero = sys.argv[2] if len(sys.argv) > 2 else 'stall'
    n_rev = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    best = None
    if mode in ('cw', 'both'):
        rows_cw = cw_sweep(3.0, aero, n_rev)
        plot_cw(rows_cw, 3.0)
        best = max(rows_cw, key=lambda r: r['cp_mean'])
        print(f"\nbest counterweight at TSR=3: m_cw={best['m_cw']:.2f} kg, "
              f"xi_cw={best['r_cw']:.2f} m  -> <Cp>={best['cp_mean']:+.4f}, "
              f"psi_eq={best['psi_eq_deg']:+.2f} deg, stop={best['stop_pct']:.1f}%, "
              f"fn_ratio={best['fn_ratio']:.2f}")
    if mode in ('tsr', 'both'):
        cw = dict(m_cw=best['m_cw'], r_cw=best['r_cw']) if best else {}
        plot_tsr(tsr_sweep(aero, n_rev, **cw), best=best)


if __name__ == '__main__':
    main()
