"""
pitch_moment_check.py
=====================
Diagnostic for the one remaining quantitative discrepancy between the FSI and the
BEM: the *pitch amplitude* (the FSI swings roughly 60 % of the BEM's, while the
rotor torque agrees to a few percent).

The FSI records the aerodynamic moment about the pitch axis directly,
``Q_aero = -Mz_pivot``.  The BEM does not output it, but both codes integrate the
*same* 1-DOF rocking-pitch equation

    I_p psidd = Q_cent(psi) + Q_aero - c_b psid + Q_stop(psi, psid)

    Q_cent(psi) = M w^2 Rp * s_t(psi),   s_t = -xi_g cos(psi) - ar_len sin(psi)

(verified identical in F6/F7, and the derived I_p, xi_g, ar_len, Rp, psi_eq and
fn_ratio are identical in the two codes -- this script prints both sets), so the
BEM's aerodynamic pitch moment can be reconstructed by inversion:

    Q_aero = I_p psidd - Q_cent(psi) + c_b psid - Q_stop(psi, psid)

The reconstruction is validated *on the FSI itself*: inverting the FSI's own
recorded psi(t)/psid(t) and comparing with its recorded Q_aero isolates the error
of the numerical differentiation alone.  That error is then the uncertainty band
on the reconstructed BEM moment.

Usage:  python3 pitch_moment_check.py [tsr] [aero]
        tsr  = 1.0 (default) or 3.0 ...
        aero = stall (default, the matched stalled comparison) | potential
"""

from __future__ import annotations

import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from compare_with_bem import (run_fsi, run_bem, geometry, mass_props,
                              last_rev, wrap_deg, TWO_PI)
from structural_solver import CPPCParams
from run_validation import CONFIG, aero_config


# --------------------------------------------------------------------------
def bem_params(tsr, U=10.0):
    """Derived mass/stiffness parameters as the BEM computes them."""
    g, mp = geometry(), mass_props()
    ar_len = g.ar * g.c
    Rp = g.R - ar_len
    xb = -g.sp * g.c + mp.xbcg * g.c
    xcw = -g.sp * g.c - mp.dcw * g.c
    M = mp.mb + mp.mc
    xg = (mp.mb * xb + mp.mc * xcw) / M
    Ip = (mp.mb * (g.c ** 2 / 12.0 + xb ** 2 + ar_len ** 2)
          + mp.mc * (xcw ** 2 + ar_len ** 2))
    wstop = 50.0
    return dict(M=M, xg=xg, ar_len=ar_len, Rp=Rp, Ip=Ip, xb=xb, xcw=xcw,
                c_b=0.15, win=np.deg2rad(25.0), omega=tsr * U / g.R,
                k_s=Ip * wstop ** 2, c_s=2.0 * Ip * wstop)


def fsi_params(tsr, U=10.0, aero='stall'):
    cfg = dict(CONFIG)
    cfg['omega'] = tsr * U / cfg['R']
    acfg, _ = aero_config(aero)
    cfg.update(acfg)
    if 'r_cw' in cfg:                       # CPPCParams calls it xi_cw
        cfg['xi_cw'] = cfg.pop('r_cw')
    fields = CPPCParams.__dataclass_fields__
    return CPPCParams(**{k: v for k, v in cfg.items() if k in fields})


def invert_Qaero(t, psi, psd, p):
    """Q_aero from the rocking-pitch equation; p is a CPPCParams or a dict."""
    if isinstance(p, dict):
        M, xg, ar, Rp, Ip = p['M'], p['xg'], p['ar_len'], p['Rp'], p['Ip']
        c_b, k_s, win, c_s, omega = (p['c_b'], p['k_s'], p['win'], p['c_s'],
                                     p['omega'])
    else:
        M, xg, ar, Rp, Ip = p.M, p.xi_g, p.ar_len, p.Rp, p.I_p
        c_b, k_s, win, c_s, omega = p.c_b, p.k_s, p.win, p.c_s, p.omega
    psi = np.asarray(psi)
    psd = np.asarray(psd)
    psa = np.gradient(psd, t)
    Qc = M * omega ** 2 * Rp * (-xg * np.cos(psi) - ar * np.sin(psi))
    Qs = np.where(psi > win, -k_s * (psi - win) - c_s * psd,
                  np.where(psi < -win, -k_s * (psi + win) - c_s * psd, 0.0))
    return Ip * psa - Qc + c_b * psd - Qs


def self_error(t, psi, psd, Q_ref, p):
    """Accuracy of the inversion+differentiation on the FSI's own data."""
    Q = invert_Qaero(t, psi, psd, p)
    err = Q - Q_ref
    rms = float(np.sqrt(np.mean(err ** 2)))
    amp = float(np.ptp(Q_ref))
    return Q, rms, amp


def stat_line(name, t, psi, Q, T_rev):
    m = t >= t[-1] - T_rev
    q = Q[m]
    pd = np.gradient(psi, t)[m]
    # work done by the aerodynamic moment on the pitch DOF per cycle
    w = float(np.trapezoid(q * pd, t[m]))
    return (f"  {name:16s} Q_aero rms={np.sqrt(np.mean(q**2)):8.3f}  "
            f"ptp={np.ptp(q):8.3f}  min/max=[{q.min():+8.2f},{q.max():+8.2f}]  "
            f"work/cycle={w:+9.3f} J")


# --------------------------------------------------------------------------
def main():
    tsr = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    aero = sys.argv[2] if len(sys.argv) > 2 else 'stall'
    n_rev = 4.0

    print("=" * 78)
    print(f"pitch-axis aerodynamic moment: FSI vs BEM   TSR={tsr:.2f}, aero={aero}")
    print("=" * 78)

    # ---- parameter cross-check -------------------------------------------
    bp = bem_params(tsr)
    fp = fsi_params(tsr, aero=aero)
    print("derived parameter cross-check (FSI | BEM):")
    print(f"  M      = {fp.M:10.6f} | {bp['M']:10.6f}   kg")
    print(f"  xi_g   = {fp.xi_g:10.6f} | {bp['xg']:10.6f}   m")
    print(f"  ar_len = {fp.ar_len:10.6f} | {bp['ar_len']:10.6f}   m")
    print(f"  Rp     = {fp.Rp:10.6f} | {bp['Rp']:10.6f}   m")
    print(f"  I_p    = {fp.I_p:10.6f} | {bp['Ip']:10.6f}   kg m^2")
    print(f"  psi_eq = {np.rad2deg(fp.psi_eq):10.4f} | "
          f"{np.rad2deg(np.arctan2(-bp['xg'], bp['ar_len'])):10.4f}   deg")
    print(f"  k_s    = {fp.k_s:10.3f}   c_b = {fp.c_b:g}   win = "
          f"{np.rad2deg(fp.win):.1f} deg")

    # ---- FSI -------------------------------------------------------------
    akey = aero_config(aero)[1]
    tag = f"_{akey}_tsr{tsr:.2f}".replace('.', 'p')
    fsi, h, akey = run_fsi(tsr, n_rev, tag, aero)
    t, psi, psd = h['t'], h['psi'], h['psi_dot']
    Q_ref = h['Q_aero']
    omega = fsi.omega
    T_rev = TWO_PI / omega
    Q_inv, err, amp = self_error(t, psi, psd, Q_ref, fsi.struct.p)
    print(f"\nFSI self-check of the inversion: rms(Q_inv - Q_recorded)="
          f"{err:.4f} N m  on a ptp of {amp:.1f} N m  ({100*err/amp:.2f} %)")

    # ---- BEM -------------------------------------------------------------
    res = {}
    for case in ('quasi', 'unsteady', 'viterna', 'real'):
        r = run_bem(tsr, case, n_rev)
        raw = r['raw']
        res[case] = r
        try:
            r['psd'] = np.asarray(raw['psd'])
        except Exception:
            r['psd'] = np.gradient(r['psi'], r['t'])

    print("\nlast-revolution aerodynamic pitch moment about the pitch axis:")
    print(stat_line(f'FSI ({akey})', t, psi, Q_ref, T_rev))
    print(stat_line('FSI (inverted)', t, psi, Q_inv, T_rev))
    for case in ('quasi', 'unsteady', 'viterna', 'real'):
        r = res[case]
        Qb = invert_Qaero(r['t'], r['psi'], r['psd'], fp)
        res[case]['Q'] = Qb
        print(stat_line(f'BEM {case}', r['t'], r['psi'], Qb, T_rev))

    # ------------------------------------------------ moment vs alpha table
    print("\nmoment about the pitch axis binned against the effective alpha "
          "(last revolution, N m):")
    nb = 8
    edges = np.linspace(-60, 60, nb + 1)
    cen = 0.5 * (edges[1:] + edges[:-1])
    def prof(run_a, Qrun):
        a = wrap_deg(run_a['alpha'])
        m = last_rev(run_a['t'], T_rev)
        idx = np.digitize(a[m], edges) - 1
        out = np.full(nb, np.nan)
        for i in range(nb):
            s = idx == i
            if s.sum():
                out[i] = Qrun[m][s].mean()
        return out
    fsi_run = dict(t=t, alpha=h['alpha'])
    p_fsi = prof(fsi_run, Q_ref)
    p_vit = prof(res['viterna'], res['viterna']['Q'])
    p_rea = prof(res['real'], res['real']['Q'])
    print("    alpha (deg)   : " + " ".join(f"{c:+7.1f}" for c in cen))
    print("    FSI           : " + " ".join(f"{v:+7.2f}" for v in p_fsi))
    print("    BEM viterna   : " + " ".join(f"{v:+7.2f}" for v in p_vit))
    print("    BEM real      : " + " ".join(f"{v:+7.2f}" for v in p_rea))

    # ---------------------------------------------------------------- plot
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    styles = {'quasi': ('C0', 'BEM A quasi-steady'),
              'unsteady': ('C1', 'BEM B (+apparent mass)'),
              'viterna': ('C2', 'BEM D matched stall'),
              'real': ('C3', 'BEM C real aero')}

    a0 = ax[0]
    m = last_rev(t, T_rev)
    a0.plot(np.degrees(h['phi'][m]), Q_ref[m], 'k-', lw=2.4,
            label=f'FSI ({akey}) [recorded]')
    a0.plot(np.degrees(h['phi'][m]), Q_inv[m], 'y--', lw=1.1,
            label='FSI [inverted] (self-check)')
    for case, (col, lab) in styles.items():
        r = res[case]
        mc = last_rev(r['t'], T_rev)
        a0.plot(np.degrees(r['phi'][mc]), r['Q'][mc], color=col, lw=1.3,
                label=lab)
    a0.axhline(0, color='k', ls=':', lw=0.7)
    a0.set_xlabel('azimuth $\\phi$ (deg)')
    a0.set_ylabel('$Q_{aero}$ about the pitch axis (N m)')
    a0.set_title('aerodynamic pitch moment')
    a0.legend(fontsize=7)
    a0.grid(alpha=0.3)

    a1 = ax[1]
    a1.plot(wrap_deg(h['alpha'][m]), Q_ref[m], 'k.', ms=2, label=f'FSI ({akey})')
    for case, (col, lab) in styles.items():
        r = res[case]
        mc = last_rev(r['t'], T_rev)
        a1.plot(wrap_deg(r['alpha'][mc]), r['Q'][mc], '.', color=col, ms=2,
                label=lab)
    a1.set_xlim(-70, 70)
    a1.set_xlabel('effective $\\alpha$ (deg)')
    a1.set_ylabel('$Q_{aero}$ (N m)')
    a1.set_title('pitch moment vs angle of attack')
    a1.legend(fontsize=7)
    a1.grid(alpha=0.3)

    a2 = ax[2]
    x = np.arange(nb)
    a2.plot(x, p_fsi, 'ko-', label=f'FSI ({akey})')
    a2.plot(x, p_vit, 'C2s-', label='BEM D matched stall')
    a2.plot(x, p_rea, 'C3^-', label='BEM C real aero')
    a2.set_xticks(x)
    a2.set_xticklabels([f'{c:.0f}' for c in cen])
    a2.set_xlabel('$\\alpha$ bin (deg)')
    a2.set_ylabel('mean $Q_{aero}$ (N m)')
    a2.set_title('cycle-mean pitch moment vs $\\alpha$')
    a2.legend(fontsize=7)
    a2.grid(alpha=0.3)

    fig.suptitle(f'pitch-axis aerodynamic moment diagnostic   '
                 f'(TSR={tsr:.2f}, {aero}, last revolution)', fontsize=12)
    fig.tight_layout()
    out = 'pitch_moment_check_' + f'tsr{tsr:.2f}'.replace('.', 'p') + '.png'
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nsaved {out}")


if __name__ == '__main__':
    main()
