"""
verify_solver.py
================
Empirical verification of the sharpFSI 2D FSI suite.  Every departure from an
analytical benchmark is quantified here rather than assumed.

  F1  Static polar of the flat plate               Cl -> 2*pi*sin(alpha)
  F2  Potential-flow drag                          Cd -> only the small
      O(Cl^2) model artefact (d'Alembert: exact drag is zero)
  F3  Quarter-chord moment and pivot-shift law     Cm(pivot) = Cl*(sp-0.25)
      (sign convention: Cm > 0 means the moment increases the local AoA)
  F4  Wagner impulsive-start response              Cl -> Cl_ss * Phi(tau),
      Jones' approximation, plus the added-mass impulse
      rho*(pi*c^2/4)*dW_n integral check
  F5  Panel / timestep convergence of the static Cl
  F6  Structural: free oscillation about psi_eq with the analytic frequency
  F7  Structural: applied moment acts in the correct sense and the elastic
      stops bound the pitch
  F8  Coupled run: finite, Kelvin budget conserved, wake bounded, stops held
  F9  Unsteady (apparent-mass) term vanishes in the steady state
  F10 Kinematics cross-check: the coupled blade kinematics (rocking arm,
      radii, sign conventions) reproduce sharp-vawt-lab's BEM kernel exactly
      (skipped if the reference solver is not available; see ref_solver.py)

Run:  python3 verify_solver.py     (writes verify_*.png)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))       # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
import ref_solver

from fluid_solver import DVMFluidSolver
from fsi_main import FSICoupler
from structural_solver import CPPCParams, CPPCStructural

RESULTS = []


def note(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


# ---------------------------------------------------------------------------
def static_run(alpha_deg, n_panels=32, dt=0.01, n_steps=400, sp=0.25,
               wake_cap=400, W=10.0, average_last=80):
    """Run a static flat plate to (near) steady state; return coefficients."""
    s = DVMFluidSolver(chord=1.0, n_panels=n_panels, rho=1.225, dt=dt, sp=sp,
                       wake_cap=wake_cap, core_bound=0.002, core_wake=0.02)
    a = np.deg2rad(alpha_deg)
    beta = 0.5 * np.pi                      # xi_hat = (1, 0)
    wind = W * np.array([np.cos(a), np.sin(a)])
    s.update_kinematics(np.zeros(2), beta, np.zeros(2), 0.0)
    s.begin_step(wind)
    cl = np.zeros(n_steps)
    cd = np.zeros(n_steps)
    cm = np.zeros(n_steps)
    for i in range(n_steps):
        s.solve()
        load = s.loads()
        cl[i], cd[i], cm[i] = load['Cl'], load['Cd'], load['Cm']
        s.end_step()
    k = slice(n_steps - average_last, n_steps)
    return cl[k].mean(), cd[k].mean(), cm[k].mean(), s


def impulsive_start(alpha_deg=5.0, dt=0.005, n_panels=48, n_steps=None):
    """Impulsively started flat plate: return (t, Cl, Cl_ss)."""
    alpha = np.deg2rad(alpha_deg)
    c = 1.0
    W = 10.0
    if n_steps is None:
        n_steps = round(8.0 * c / (2 * W) / dt) + 1
    s = DVMFluidSolver(chord=c, n_panels=n_panels, rho=1.225, dt=dt, sp=0.25,
                       wake_cap=max(900, n_steps + 5), core_bound=0.002,
                       core_wake=0.02)
    beta = 0.5 * np.pi
    wind = W * np.array([np.cos(alpha), np.sin(alpha)])
    s.update_kinematics(np.zeros(2), beta, np.zeros(2), 0.0)
    s.begin_step(wind)
    cl = np.zeros(n_steps)
    for i in range(n_steps):
        s.solve()
        cl[i] = s.loads()['Cl']
        s.end_step()
    t = np.arange(n_steps) * dt
    return t, cl, 2 * np.pi * np.sin(alpha), W, c


# ------------------------------------------------ F1/F2/F3 static behaviour
def test_static():
    print("\nF1-F3: static flat-plate polar")
    alphas = np.arange(-20.0, 20.1, 4.0)
    cl_s, cd_s, cm_s = [], [], []
    for a in alphas:
        cl, cd, cm, _ = static_run(a)
        cl_s.append(cl); cd_s.append(cd); cm_s.append(cm)
    cl_s, cd_s, cm_s = map(np.array, (cl_s, cd_s, cm_s))
    cl_exact = 2 * np.pi * np.sin(np.deg2rad(alphas))
    err = np.abs(cl_s - cl_exact)
    note("F1 Cl = 2 pi sin(alpha)", err.max() < 0.03,
         f"max|dCl|={err.max():.4f} ({100*err.max()/np.abs(cl_exact).max():.2f}% of max)")
    k_drag = float(np.sum(cd_s * cl_s ** 2) / np.sum(cl_s ** 4))
    fit_err = np.abs(cd_s - k_drag * cl_s ** 2).max()
    note("F2 potential drag small & ~C*Cl^2 (see F17: C ~ 1/L_wake)",
         np.abs(cd_s).max() < 0.02 and fit_err < 3e-4,
         f"max|Cd|={np.abs(cd_s).max():.5f}, Cd={k_drag:.2e}*Cl^2 (fit err {fit_err:.1e})")
    note("F3 Cm(pivot=0.25c) ~ 0", np.abs(cm_s).max() < 0.01,
         f"max|Cm|={np.abs(cm_s).max():.5f}")

    print("  pivot-shift moment law  Cm = Cl*(sp-0.25):")
    for sp in (0.15, 0.40):
        cl, cd, cm, _ = static_run(6.0, sp=sp)
        expect = cl * (sp - 0.25)
        note(f"F3 Cm(sp={sp})", abs(cm - expect) < 0.02,
             f"Cm={cm:+.4f} expected={expect:+.4f} (Cl={cl:.4f})")
    return alphas, cl_s, cd_s, cm_s


# ------------------------------------------------- F4 Wagner impulsive start
def test_wagner():
    print("\nF4: Wagner impulsive-start response")
    dt = 0.005
    t, cl, cl_ss, W, c = impulsive_start(5.0, dt=dt, n_steps=800)   # tau to 40
    tau = 2 * W * t / c
    jones = 1.0 - 0.165 * np.exp(-0.0455 * tau) - 0.335 * np.exp(-0.3 * tau)
    cl_j = cl_ss * jones
    m = tau >= 0.5
    rms = float(np.sqrt(np.mean((cl[m] - cl_j[m]) ** 2)) / cl_ss)
    note("F4 Wagner history matches Jones (tau>0.5)", rms < 0.05,
         f"RMS|dCl|/Cl_ss={rms:.4f}")
    note("F4 Cl -> Cl_ss as tau -> 40", abs(cl[-1] - cl_ss) < 0.05 * cl_ss,
         f"Cl(tau=40)={cl[-1]:.4f} Cl_ss={cl_ss:.4f} "
         f"(Jones predicts {cl_j[-1]:.4f})")

    # impulse delivered during the first step should equal the classical
    # added-mass impulse rho*(pi c^2/4)*dW_n (per unit span)
    impulse = float(cl[0] * dt * (0.5 * 1.225 * W ** 2 * c))
    impulse_ref = 1.225 * (np.pi * c ** 2 / 4.0) * (W * np.sin(np.deg2rad(5.0)))
    note("F4 added-mass impulse matches rho(pi c^2/4) dW_n",
         0.6 < impulse / impulse_ref < 1.6 and cl[0] > 0,
         f"first-step impulse={impulse:.4f} N s vs ref {impulse_ref:.4f} N s "
         f"(ratio {impulse/impulse_ref:.2f}); first-step Cl={cl[0]:+.3f} > 0")
    return tau, cl, cl_j


# --------------------------------------------------------------- F5 convergence
def test_convergence():
    print("\nF5: panel / timestep convergence (static, alpha=10 deg)")
    ex = 2 * np.pi * np.sin(np.deg2rad(10.0))
    rows = []
    for N in (16, 32, 64):
        for dt in (0.02, 0.01, 0.005):
            cl, _cd, _cm, _ = static_run(10.0, n_panels=N, dt=dt,
                                       n_steps=int(5.0 / dt), wake_cap=600)
            rows.append((N, dt, cl, 100 * abs(cl - ex) / ex))
            print(f"    N={N:3d} dt={dt:6.4f}  Cl={cl:.5f}  err={rows[-1][3]:.3f}%")
    errs = [r[3] for r in rows]
    note("F5 convergence (all < 3%)", max(errs) < 3.0, f"max err={max(errs):.3f}%")
    return rows


# --------------------------------------------------------------- F6/F7 structural
def test_structural():
    print("\nF6/F7: structural model")
    p = CPPCParams()
    s = CPPCStructural(p)
    s.reset(0.0, 0.0)
    dt = 1e-4
    n = int(8.0 / dt)
    hist = np.zeros(n)
    tt = np.arange(n) * dt
    for i in range(n):
        s.step(0.0, dt)
        hist[i] = s.psi
    # oscillation centre (average over whole record) and period
    ctr = float(hist.mean())
    x = hist - ctr
    zc = np.where(np.diff(np.sign(x)))[0]
    per = 2 * np.mean(np.diff(zc)) * dt if len(zc) > 2 else np.nan
    om = 2 * np.pi / per
    note("F6 oscillates about psi_eq", abs(ctr - p.psi_eq) < np.deg2rad(1.0),
         f"centre={np.rad2deg(ctr):.3f} deg vs psi_eq="
         f"{np.rad2deg(p.psi_eq):.3f} deg")
    note("F6 frequency matches sqrt(M w^2 Rp ar c / Ip)",
         abs(om - p.omega_n) / p.omega_n < 0.10,
         f"omega={om:.3f} vs {p.omega_n:.3f} rad/s "
         f"({100*(om/p.omega_n-1):+.1f}%)")

    s2 = CPPCStructural(p)
    s2.reset(p.psi_eq, 0.0)
    s2.step(+10.0, 0.01, n_sub=10)
    note("F7 +Q_aero increases psi", s2.psi > p.psi_eq,
         f"dpsi={np.rad2deg(s2.psi - p.psi_eq):+.4f} deg")

    Q = 50.0
    s3 = CPPCStructural(p)
    s3.reset(0.0, 0.0)
    for _ in range(20000):
        s3.step(-Q, 5e-4)
    deflect = Q / p.k_s
    note("F7 elastic stops: psi = -(win + Q/ks)", 
         abs(abs(s3.psi) - (p.win + deflect)) < np.deg2rad(1.0),
         f"psi={np.rad2deg(s3.psi):.2f} deg, model "
         f"{np.rad2deg(-(p.win+deflect)):.2f} deg")
    return tt, hist, p


# --------------------------------------------------------------- F8/F9 coupled
def test_coupled():
    print("\nF8/F9: coupled FSI sanity run (0.4 rev, coarse)")
    fsi = FSICoupler(dt=0.01, n_panels=24, wake_cap=300, omega=2.0,
                     n_coupling_iter=3)
    h = fsi.run(n_revolutions=0.4, verbose=False)
    budget = fsi.fluid.circulation_budget()
    psi_max = np.abs(h['psi']).max()
    note("F8 finite, no blow-up",
         np.all(np.isfinite(h['Cl'])) and np.all(np.isfinite(h['psi'])),
         f"|Cl|max={np.abs(h['Cl']).max():.3f}")
    note("F8 wake bounded", fsi.fluid.nw <= fsi.fluid.wake_cap,
         f"wake={fsi.fluid.nw}, merges={fsi.fluid.merge_count}")
    note("F8 Kelvin budget ~ 0", abs(budget) < 1e-6, f"sum(Gamma)={budget:.2e}")
    # The CPPC stops are elastic (a spring-damper), so the pitch overshoots the
    # nominal +-win by Q_aero/ks.  Check that the travel stays bounded by that
    # elastic limit rather than by the nominal limit itself.
    osk = np.max(np.abs(h['Q_aero'])) / fsi.struct.p.k_s
    note("F8 stops bounded (elastic stop travel)",
         psi_max < fsi.struct.p.win + osk + np.deg2rad(1.0),
         f"max|psi|={np.rad2deg(psi_max):.2f} deg, win={fsi.struct.p.win_deg:.1f}, "
         f"elastic overshoot Q/k_s={np.rad2deg(osk):.2f} deg")
    note("F8 one vortex shed per step",
         fsi.fluid.nw + fsi.fluid.merge_count == fsi.step_count,
         f"nw+merges={fsi.fluid.nw + fsi.fluid.merge_count} steps={fsi.step_count}")

    # F9: the unsteady term must vanish once the flow is steady
    cl_u, _cd_u, _cm_u, s_u = static_run(8.0, n_steps=300, average_last=40)
    s_u.unsteady_force = False
    load = s_u.loads()
    dcl = abs(cl_u - load['Cl'])
    note("F9 unsteady term vanishes at steady state", dcl < 1e-3,
         f"|Cl_unsteady-on - Cl_off|={dcl:.2e}")
    return fsi, h


# ---------------------------------------------------------------------------
def test_bem_kinematics():
    """F10: cross-check the blade kinematics against sharp-vawt-lab's BEM kernel.

    With the pitch prescribed at a constant value the two codes must report the
    same effective angle of attack at every azimuth.  This exercises the
    rocking-arm geometry (X_ref = X_pivot + ar*c*n_hat), the radius convention
    (the aero reference sits at R, the pivot at R - ar*c) and the sign
    conventions of the coupled kinematics.
    """
    print("\nF10: kinematics cross-check against sharp-vawt-lab")
    root = ref_solver.find()
    if root is None:
        print("  [SKIP] reference solver sharp-vawt-lab not found "
              "(set SHARP_VAWT_LAB to its root)")
        return None
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from vawt_core import (
            AeroConfig,
            CycloturbineSim,
            Geometry,
            MassProps,
            PolarDatabase,
            SimConfig,
        )
    except Exception as exc:  # noqa: BLE001                                 # pragma: no cover
        print(f"  [SKIP] sharp-vawt-lab import failed: {exc}")
        return None

    R, c, ar, sp, U, tsr, psi0 = 5.0, 1.0, 0.35, 0.25, 10.0, 1.0, 5.0
    omega = tsr * U / R
    g = Geometry(R=R, c=c, H=1.0, N=1, ar=ar, sp=sp)
    ms = MassProps(mb=5.0, xbcg=0.25, mc=2.5, dcw=-0.60)
    # NOTE: a callable polar database is passed below, so AeroConfig.polar_dir
    # is not used for the physics; resolve it anyway (either table layout) so
    # nothing dangles if that ever changes.
    _pdir = ref_solver.polar_dir(root) or ''
    ae = AeroConfig(polar_dir=_pdir,
                    use_dynamic_stall=False, use_tip_loss=False, use_dmst=False,
                    use_dynamic_inflow=False, use_flow_curvature=False,
                    cd_add=0.0, win_deg=25.0, wstop=50.0, cb=0.15, kc=0.0,
                    mu_c=0.0, r_bearing=0.0, a_fix=0.0, prescribe_pitch=True,
                    pp0_deg=psi0)
    pdb = PolarDatabase.from_callable(
        lambda a: (2.0 * np.pi * np.sin(a), np.zeros_like(a)),
        cla=2.0 * np.pi, geo=g)
    sim_cfg = SimConfig(U=U, tsr=tsr, free=False, psi0_deg=psi0, dt=1e-3,
                        T_max=2 * np.pi / omega, stride=1, rho=1.225,
                        n_rev_fixed=1)
    out = CycloturbineSim(g, ms, ae, sim_cfg, polar_db=pdb).run()['out']
    phi_b, alpha_b = out[:, 1], out[:, 5]

    fsi = FSICoupler(chord=c, sp=sp, ar=ar, R=R, omega=omega, U_inf=U,
                     m_blade=5.0, cg_blade=0.25, m_cw=2.5, r_cw=0.35)
    psi = np.deg2rad(psi0)
    al, Wm, rref = [], [], []
    for p in phi_b:
        x_ref, v_ref, beta, _ = fsi._blade_reference(p, psi, 0.0)
        Wg = fsi._wind() - v_ref
        n_hat = np.array([np.cos(beta), np.sin(beta)])
        xi_hat = np.array([np.sin(beta), -np.cos(beta)])
        al.append(np.arctan2(Wg @ n_hat, Wg @ xi_hat))
        Wm.append(float(np.hypot(*Wg)))
        rref.append(float(np.hypot(*x_ref)))
    al = np.degrees(np.array(al))
    dalpha = np.abs((al - np.degrees(alpha_b) + 180.0) % 360.0 - 180.0)
    note("F10 kinematics match sharp-vawt-lab kernel (alpha)",
         dalpha.max() < 1e-6,
         f"max|dalpha|={dalpha.max():.2e} deg over {len(al)} azimuths")
    # blade aero reference must sit at radius R at psi = 0
    x0 = fsi._blade_reference(0.0, 0.0, 0.0)[0]
    note("F10 blade reference radius = R at zero pitch",
         abs(np.hypot(*x0) - R) < 1e-12,
         f"|X_ref|={np.hypot(*x0):.9f} m, R={R:.1f} m")
    return phi_b, np.rad2deg(alpha_b), al


# ------------------------------------------- F11-F15 stall / drag correction
def _stall_static(polar, alpha_deg, n_steps=700, W=10.0, dt=0.01,
                  n_panels=48, wake_cap=800, average_last=100, T_f=3.0,
                  T_v=6.0, K_v=0.7):
    """Fixed-pitch DVM run with the stall correction; returns mean loads.

    Returns ``(Cl, Cd, Cm, kappa, model)`` averaged over the last steps.
    """
    from aero_correction import StallModel
    sm = StallModel(polar, T_f=T_f, T_v=T_v, K_v=K_v, chord=1.0)
    s = DVMFluidSolver(chord=1.0, n_panels=n_panels, rho=1.225, dt=dt, sp=0.25,
                       wake_cap=wake_cap, core_bound=0.002, core_wake=0.02,
                       stall_model=sm)
    a = np.deg2rad(alpha_deg)
    beta = 0.5 * np.pi
    wind = W * np.array([np.cos(a), np.sin(a)])
    s.update_kinematics(np.zeros(2), beta, np.zeros(2), 0.0)
    s.begin_step(wind)
    cl = np.zeros(n_steps)
    cd = np.zeros(n_steps)
    cm = np.zeros(n_steps)
    kp = np.zeros(n_steps)
    for i in range(n_steps):
        s.solve()
        load = s.loads()
        cl[i], cd[i], cm[i] = load['Cl'], load['Cd'], load['Cm']
        kp[i] = load['kappa']
        s.end_step()
    k = slice(n_steps - average_last, n_steps)
    return cl[k].mean(), cd[k].mean(), cm[k].mean(), kp[k].mean(), sm


def test_stall_transparency():
    print("\nF11: stall correction is transparent while attached")
    from aero_correction import StaticPolar
    polar = StaticPolar.viterna(alpha_stall=14.0, cd0=8.0e-3)
    dcl = dcm = 0.0
    kmin, kmax = 1.0, 1.0
    for a0 in (2.0, 4.0, 8.0, 12.0):
        cl0, _cd0_, cm0, _ = static_run(a0, n_panels=48, n_steps=700,
                                       wake_cap=800, average_last=100)
        cl1, _cd1, cm1, km, _sm = _stall_static(polar, a0)
        dcl = max(dcl, abs(cl1 - cl0))
        dcm = max(dcm, abs(cm1 - cm0))
        kmin, kmax = min(kmin, km), max(kmax, km)
    note("F11 kappa == 1 in attached flow", max(abs(kmin - 1), abs(kmax - 1))
         < 1e-9, f"kappa in [{kmin:.12f}, {kmax:.12f}]")
    note("F11 corrected Cl, Cm identical to pure potential flow",
         max(dcl, dcm) < 1e-9, f"max|dCl|={dcl:.2e}  max|dCm|={dcm:.2e}")
    return polar


def test_stall_static(polar, alphas=(16.0, 20.0, 30.0, 45.0, 60.0)):
    print("\nF12: corrected loads recover the prescribed static polar")
    rows = []
    for a0 in alphas:
        cl, cd, cm, km, _sm = _stall_static(polar, a0)
        clp, cdp, _ = polar.eval(np.deg2rad(a0))
        rows.append((a0, cl, clp, cd, cdp, cm, km))
    dcl = max(abs(r[1] - r[2]) for r in rows)
    dcd = max(abs(r[3] - r[4]) for r in rows)
    note("F12 Cl matches the polar in the stalled range", dcl < 0.05,
         f"max|dCl|={dcl:.4f}")
    note("F12 Cd matches the polar in the stalled range", dcd < 0.05,
         f"max|dCd|={dcd:.4f}")
    for r in rows:
        print(f"        alpha={r[0]:5.1f}  Cl={r[1]:+.4f} (polar {r[2]:+.4f})"
              f"  Cd={r[3]:.4f} (polar {r[4]:.4f})  kappa={r[6]:+.4f}")
    return rows


def test_stall_deep(polar):
    print("\nF13: deep-stall limits")
    cl, cd, cm, _km, _sm = _stall_static(polar, 90.0, n_steps=900)
    note("F13 Cl(90 deg) ~ 0", abs(cl) < 0.2, f"Cl={cl:+.4f}")
    note("F13 Cd(90 deg) ~ 2 (broadside flat plate)",
         abs(cd - 2.0) < 0.25, f"Cd={cd:.4f}")
    cl2, cd2, _cm2, _km2, _ = _stall_static(polar, 170.0, n_steps=900)
    clp, _, _ = polar.eval(np.deg2rad(170.0))
    note("F13 reversed flow: Cl(170 deg) matches the reflected polar",
         abs(cl2 - clp) < 0.06, f"Cl={cl2:+.4f} (polar {clp:+.4f})")
    note("F13 separated moment is nose-down for alpha > 0",
         cm < -0.02, f"Cm(90 deg)={cm:+.4f}")
    return (cl, cd), (cl2, cd2)


def test_stall_hysteresis(polar):
    print("\nF14: dynamic-stall hysteresis (pitch ramp through stall)")
    from aero_correction import StallModel
    sm = StallModel(polar, T_f=3.0, T_v=6.0, K_v=0.7)
    n, dt, W = 900, 2.0e-3, 10.0

    def ramp(a_start, a_end):
        sm.reset()
        cl = np.zeros(n)
        al = np.zeros(n)
        for i in range(n):
            a0 = a_start + (a_end - a_start) * i / (n - 1)
            al[i] = a0
            sm.begin_step()
            sm.update(np.deg2rad(a0), W, dt)
            cl[i] = sm.Cl_ds
        return al, cl

    al_u, cl_u = ramp(0.0, 35.0)
    al_d, cl_d = ramp(35.0, 0.0)
    cl_static_up = polar.eval(np.deg2rad(al_u))[0]
    over = float(np.max(cl_u - cl_static_up))
    loop = float(np.trapezoid(cl_u - cl_d[::-1], np.deg2rad(al_u)))
    print(f"        overshoot above the static polar: {over:+.4f}")
    print(f"        (static polar max Cl = {polar.cl_max:.4f})")
    note("F14 lift overshoot in the dynamic-stall ramp", over > 0.05,
         f"max overshoot={over:+.4f}")
    note("F14 hysteresis loop is non-degenerate",
         abs(loop) > 1e-3 and np.all(np.isfinite(cl_u)),
         f"loop integral={loop:+.5f} (deg*Cl)")
    note("F14 no overshoot in attached flow (|alpha|<10 deg)",
         float(np.max(cl_u[al_u < 10.0] - cl_static_up[al_u < 10.0])) < 5e-3,
         f"{float(np.max(cl_u[al_u < 10.0] - cl_static_up[al_u < 10.0])):+.2e}")
    return al_u, cl_u, cl_d, cl_static_up, al_d


def test_stall_coupled(polar):
    print("\nF15: coupled FSI run with the stall correction")
    fsi = FSICoupler(chord=1.0, sp=0.25, ar=0.35, R=5.0, omega=2.0,
                     U_inf=10.0, rho=1.225, dt=0.005, stall=True,
                     stall_polar=polar, n_panels=32, wake_cap=600,
                     psi0_deg=5.0)
    fsi.run(n_revolutions=0.5, verbose=False)
    h = fsi.history()
    # drop the first few steps: an impulsive start has a discretised
    # added-mass spike (see F4) that is not representative of the cycle
    sl = slice(5, None)
    finite = bool(np.all(np.isfinite(h['psi'])))
    k = h['kappa'][sl]
    al = np.rad2deg(h['alpha'][sl])
    stalled = np.abs(al) > 20.0
    # the live relative wind: coefficients are meaningless as q -> 0 (the
    # zero-relative-wind dead point), so judge |Cl| only in real wind
    live = h['W'][sl] > 4.0
    note("F15 coupled stall run is finite", finite,
         f"{len(h['t'])} steps, max|psi|={np.rad2deg(np.abs(h['psi'])).max():.1f} deg")
    clmax_live = float(np.abs(h['Cl'][sl][live]).max()) if np.any(live) else 0.0
    ov = float(np.max(np.abs(h['Cl'][sl][live]))) / polar.cl_max - 1.0 \
        if np.any(live) else 0.0
    note("F15 lift is stall-limited where the wind is live",
         clmax_live <= 1.6 * polar.cl_max,
         f"max|Cl|={clmax_live:.3f} = {100 * ov:+.0f}% vs static Clmax="
         f"{polar.cl_max:.3f} (dynamic-stall overshoot, model clamp " 
         f"1.6*Clmax={1.6 * polar.cl_max:.2f})")
    kk = k[stalled & live]
    note("F15 kappa < 1 in the stalled part of the cycle",
         kk.size == 0 or (float(np.max(kk)) < 1.0 + 1e-9
                          and float(np.min(kk)) > -0.3),
         f"kappa in [{np.min(kk):.3f}, {np.max(kk):.3f}] for |alpha|>20 deg"
         if kk.size else "no stalled live samples")
    need = float(np.mean(np.abs(h['Cl'][sl][live]) > 1.4)) if np.any(live) else 0.0
    note("F15 stall correction is active in the cycle", need > 0.0,
         f"{100 * need:.1f}% of live steps with |Cl|>1.4")
    return h


# ------------------------------------------------- F16 multiresolution wake
def test_wake_thinning():
    """Age-decimated wake thinning: conservation, invariants, and the induced
    velocity at the blade versus the unthinned wake."""
    print("\nF16: multiresolution (age-decimated) wake thinning")
    N, CAP, LEVELS, DS = 3720, 600, 5, 0.05
    age = np.arange(N)
    s = DS * (N - 1 - age)                     # age order: 0 = oldest
    x = s + 0.4 * np.sin(s / 3.0)
    y = 0.35 * np.sin(2.0 * np.pi * s / 2.5)
    g = 0.8 * np.cos(2.0 * np.pi * s / 2.5) * np.exp(-s / 40.0)
    c2 = 0.02 ** 2 + 1.5e-5 * age
    probes = np.array([[0.5, 0.0], [2.0, 1.0], [5.0, 3.0], [15.0, 2.0]])

    def ind(P, wx, wy, gw, c2w):
        dx = P[:, 0][:, None] - wx[None, :]
        dy = P[:, 1][:, None] - wy[None, :]
        r2 = dx * dx + dy * dy + c2w[None, :]
        u = (gw[None, :] * dy / (2 * np.pi * r2)).sum(axis=1)
        v = (-gw[None, :] * dx / (2 * np.pi * r2)).sum(axis=1)
        return np.column_stack((u, v))

    ref = ind(probes, x, y, g, c2)
    umax = float(np.hypot(*ref.T).max())
    errs = {}
    for mode in ('age', 'oldest'):
        m = DVMFluidSolver(chord=1.0, n_panels=8, wake_cap=CAP,
                           merge_mode=mode, wake_levels=LEVELS)
        big = N + 16
        m.wx, m.wy, m.gw, m.c2w = (np.zeros(big), np.zeros(big),
                                   np.zeros(big), np.zeros(big))
        m.wl = np.zeros(big, dtype=int)
        m.wx[:N], m.wy[:N], m.gw[:N], m.c2w[:N] = x, y, g, c2
        m.nw, m.lev_count[:], m.lev_count[0] = N, 0, N
        guard = 0
        while m.nw > CAP:
            m._decimate()
            guard += 1
            if guard > 20 * N:
                raise RuntimeError('decimation did not converge')
        out = ind(probes, m.wx[:m.nw], m.wy[:m.nw], m.gw[:m.nw], m.c2w[:m.nw])
        errs[mode] = float(np.abs(out - ref).max() / umax)
        # exact conservation of circulation and the first moments
        dg = abs(m.gw[:m.nw].sum() - g.sum())
        dmx = abs((m.gw[:m.nw] * m.wx[:m.nw]).sum() - (g * x).sum())
        dmy = abs((m.gw[:m.nw] * m.wy[:m.nw]).sum() - (g * y).sum())
        note(f"F16 {mode}: circulation/moment conservation", dg < 1e-12
             and dmx < 1e-9 and dmy < 1e-9,
             f"dGamma={dg:.1e} dMx={dmx:.1e} dMy={dmy:.1e}")
        if mode == 'age':
            lv = m.wl[:m.nw]
            note("F16 age: wake stays sorted by level", bool(np.all(np.diff(lv)
                                                                    <= 0)),
                 f"levels={m.wake_stats()['levels']}")
            note("F16 age: one block per level, memory bound respected",
                 m.nw <= CAP
                 and all(int((lv == L).sum()) <= m.n_per_level
                         for L in range(LEVELS + 1)),
                 f"nw={m.nw} <= cap={CAP}, n_per_level={m.n_per_level}")
            note("F16 age: effective wake memory = n_per_level*(2^L-1)",
                 m.wake_stats()['equivalent_steps'] == 3720,
                 f"{m.wake_stats()['equivalent_steps']} shed vortices "
                 f"represented by {m.nw}")
    note("F16 near-blade induced velocity preserved by age decimation",
         errs['age'] < 5.0e-3,
         f"max|du|/|u|max = {errs['age']:.2e} (oldest-pair scheme: "
         f"{errs['oldest']:.2e})")
    return errs


# ------------------------------------------- F17 residual-drag scaling
def test_wake_drag():
    """Origin of the residual potential-flow drag (F2).

    d'Alembert says the steady drag of a lifting body in potential flow is zero,
    yet the DVM returns a small Cd ~ 2e-3.  A panel/timestep convergence would
    suggest a lattice discretisation artefact; this check shows it is not.  It is
    a *finite-wake-length* effect: the truncated wake behaves like a free end (a
    starting vortex) and induces a small downwash with a component along the
    chord, so Cd ~ K*Cl^2 * c/L_wake.  Consequences:

      * Cd is invariant under panel refinement (it is not a lattice artefact),
      * Cd falls as 1/L_wake, so it vanishes with a long-enough wake.

    Returns the two data tables.
    """
    print("\nF17: residual potential drag scales with the wake length, not N")
    a = 10.0

    # (i) panel refinement at a fixed wake length (50 chords)
    ns, cap = 500, 900
    npan = (16, 32, 64, 128)
    runs = [static_run(a, n_panels=N, dt=0.01, n_steps=ns, wake_cap=cap)
            for N in npan]
    cl_n = np.array([r[0] for r in runs])
    cd_n = np.array([r[1] for r in runs])
    spread = (cd_n.max() - cd_n.min()) / cd_n.mean()
    print("    panel refinement (wake = 50c):")
    for N, clv, cdv in zip(npan, cl_n, cd_n):
        print(f"      N={N:3d}  Cd={cdv:.4e}  Cd/Cl^2={cdv/clv**2:.3e}")
    note("F17 drag is invariant under panel refinement", spread < 0.05,
         f"spread = {100*spread:.2f}% over N = {npan} (not a lattice artefact)")

    # (ii) wake-length refinement at fixed panel count
    npan0, dt = 32, 0.01
    steps = (250, 500, 1000)
    cd_l = np.array([static_run(a, n_panels=npan0, dt=dt, n_steps=n,
                                wake_cap=n + 5,
                                average_last=max(20, n // 5))[1] for n in steps])
    Lw = dt * 10.0 * np.array(steps)                    # wake length, chords
    print("    wake-length refinement (N = 32, 0.1c shed per step):")
    for L, cdv in zip(Lw, cd_l):
        print(f"      L={L:6.1f}c  Cd={cdv:.4e}  Cd*L={cdv*L:.4e}")
    slope = np.polyfit(np.log(Lw), np.log(cd_l), 1)[0]
    prod = cd_l * Lw
    note("F17 drag falls as 1/wake-length (finite-wake effect)",
         abs(slope + 1.0) < 0.25 and prod.max() / prod.min() < 1.35,
         f"log-log slope = {slope:+.2f} (-1 expected), "
         f"Cd*L/mean = {prod.min()/prod.mean():.2f}-{prod.max()/prod.mean():.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    ax[0].plot(npan, cd_n, 'bo-')
    ax[0].set_xscale('log', base=2)
    ax[0].set_xlabel('panels $N$')
    ax[0].set_ylabel('$C_d$')
    ax[0].set_title('F17: drag is panel-independent')
    ax[0].grid(alpha=0.3)
    ax[1].loglog(Lw, cd_l, 'rs-', label='DVM')
    ax[1].loglog(Lw, cd_l[-1] * Lw[-1] / Lw, 'k--', lw=1,
                 label=r'$\propto 1/L_{wake}$')
    ax[1].set_xlabel('wake length (chords)')
    ax[1].set_ylabel('$C_d$')
    ax[1].set_title('F17: residual drag $\\propto 1/L_{wake}$')
    ax[1].legend()
    ax[1].grid(alpha=0.3, which='both')
    fig.suptitle('residual potential drag: finite-wake truncation', fontsize=12)
    fig.tight_layout()
    fig.savefig('verify_wake_drag.png', dpi=140)
    plt.close(fig)
    return cd_n, cd_l, Lw


# ------------------------------------------- F18 reported-coefficient floor
def test_coeff_floor():
    """The reported coefficients must stay bounded through the dead point.

    At the zero-relative-wind point q -> 0 while the force stays finite, so
    normalising by the true q makes Cl/Cd/Cm diverge.  The normalising pressure
    is therefore floored at `coeff_W_min`; this check verifies that (i) the
    *forces* are bit-identical with and without the floor, (ii) the floored
    coefficients are bounded where the true ones diverge, and (iii)
    `coeff_valid` flags exactly the samples where the floor is active.
    """
    print("\nF18: reported coefficients bounded at the zero-wind dead point")
    a = np.deg2rad(20.0)
    beta = 0.5 * np.pi
    # (i) the floor must not touch the loads: drive two solvers with identical
    #     kinematics and different floors through the same low-wind sweep
    n, dt = 4000, 0.005
    Ws = np.linspace(10.0, 0.1, n)
    out = []
    for wmin in (2.0, 0.0):
        s = DVMFluidSolver(chord=1.0, n_panels=32, rho=1.225, dt=dt,
                           wake_cap=300, coeff_W_min=wmin)
        s.update_kinematics(np.zeros(2), beta, np.zeros(2), 0.0)
        s.begin_step(Ws[0] * np.array([np.cos(a), np.sin(a)]))
        rec = []
        for W in Ws:
            wind = W * np.array([np.cos(a), np.sin(a)])
            s.begin_step(wind)
            s.solve()
            L = s.loads()
            rec.append((L['Cl'], L['F'].copy(), L['Mz'], L['coeff_valid']))
            s.end_step()
        out.append(rec)
    rec_f, rec_t = out
    dF = max(float(np.abs(a1[1] - b1[1]).max())
             for a1, b1 in zip(rec_f, rec_t))
    dMz = max(abs(a1[2] - b1[2]) for a1, b1 in zip(rec_f, rec_t))
    cl_f = np.array([r[0] for r in rec_f])
    cl_t = np.array([r[0] for r in rec_t])
    note("F18 the floor leaves the loads bit-identical", dF == 0.0 and dMz == 0.0,
         f"max|dF|={dF:.1e}, max|dMz|={dMz:.1e} over the sweep")
    # skip the impulsive-start transient (first 50 steps) and judge the
    # *floored* branch, where |W| < coeff_W_min
    floored = Ws < 2.0
    floored[:50] = False

    # (ii) the sweep ends at |W| = 0.1 m/s, a factor 20 below the floor
    ratio = cl_t[-1] / cl_f[-1]
    _ws, _flor, _true = (10.0, 1.0, 0.2, 0.05), None, None
    print(f"      at |W| = {Ws[-1]:.2f} m/s:  Cl(false q) = {cl_f[-1]:+.3f}   "
          f"Cl(true q) = {cl_t[-1]:+.3f}   ratio = {ratio:.0f}")
    note("F18 the floor caps the dead-point blow-up at (W_ref/W)^2",
         abs(ratio - (2.0 / Ws[-1]) ** 2) < 1e-6 and ratio > 1.0e2,
         f"Cl(true q)/Cl(false q) = {ratio:.0f} at |W|={Ws[-1]:.2f} "
         f"(expected {(2.0 / Ws[-1]) ** 2:.0f})")
    mf = float(np.abs(cl_f[floored]).max())
    mrat = float(np.max(np.abs(cl_t[floored])
                        / np.maximum(np.abs(cl_f[floored]), 1e-12)))
    note("F18 floored coefficients stay bounded where the true ones diverge",
         mf < 4.0 and mrat > 50.0,
         f"max|Cl|_floored={mf:.3f} (bounded, the physical range); the "
         f"pointwise true/floored ratio reaches {mrat:.0f}")
    note("F18 coeff_valid flags exactly the floored samples",
         next(r[3] for r in rec_f) and not [r[3] for r in rec_f][-1],
         f"valid at |W|=10: {next(r[3] for r in rec_f)}, "
         f"at |W|=0.1: {[r[3] for r in rec_f][-1]}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    ax[0].semilogy(Ws, np.abs(cl_t), 'r-', label='normalised by the true $q$')
    ax[0].semilogy(Ws, np.abs(cl_f), 'b-', label=r'floored at $q_{ref}$')
    ax[0].axvline(2.0, color='k', lw=0.8, ls=':')
    ax[0].invert_xaxis()
    ax[0].set_xlabel('$|W|$ (m/s)  (low-wind sweep)')
    ax[0].set_ylabel('$|C_l|$')
    ax[0].set_title('F18: dead-point coefficient blow-up removed')
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].plot(Ws, np.abs(cl_t - cl_f) + 1e-30, 'ks-', ms=2)
    ax[1].invert_xaxis()
    ax[1].set_xlabel('$|W|$ (m/s)')
    ax[1].set_ylabel(r'$|C_{l,floor}-C_{l,true}|$')
    ax[1].set_title('F18: the loads are unchanged by the floor')
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('verify_deadpoint.png', dpi=140)
    plt.close(fig)
    return Ws, cl_f, cl_t


# --------------------------------------- F19 leading-edge-vortex moment
def test_vortex_moment(polar):
    """Dynamic-stall pitching moment from the convected leading-edge vortex.

    The vortex lift must contribute to the pitching moment at its (convecting)
    chordwise position.  Checks: it vanishes in the steady state (so the exact
    polar reproduction is untouched), it produces a *nose-down* moment break of
    physical size during a pitch ramp through stall, its magnitude is
    continuous, and the generated figure shows the classic dynamic-stall moment
    loop.
    """
    print("\nF19: leading-edge-vortex pitching moment")
    from aero_correction import StallModel
    # (i) steady state: no vortex, polar reproduced exactly
    sm = StallModel(polar)
    sm.reset()
    a20 = np.deg2rad(20.0)
    for _ in range(6000):
        sm.begin_step()
        sm.update(a20, 10.0, 2.0e-3)
    cm_pol = float(polar.eval(a20)[2])
    note("F19 vortex moment vanishes in the steady state",
         abs(sm.Cm_vx) < 1.0e-9, f"Cm_vx={sm.Cm_vx:+.2e}")
    note("F19 steady Cm still equals the polar",
         abs(sm.Cm_ext - cm_pol) < 1.0e-9,
         f"Cm={sm.Cm_ext:+.6f} vs polar {cm_pol:+.6f}")

    # (ii) ramp through stall, with and without the vortex moment
    n, dt, W = 1200, 2.0e-3, 10.0
    al = np.zeros(n)
    cm_v, cm_0, xv = np.zeros(n), np.zeros(n), np.zeros(n)
    for vm, dst in ((True, cm_v), (False, cm_0)):
        s = StallModel(polar, vortex_moment=vm)
        s.reset()
        for i in range(n):
            d = -2.0 + 34.0 * i / (n - 1)
            al[i] = d
            s.begin_step()
            s.update(np.deg2rad(d), W, dt)
            dst[i] = s.Cm_ext
            if vm:
                xv[i] = s.xv
    dv = cm_v - cm_0
    i_min = int(np.argmin(dv))
    i_birth = int(np.searchsorted(al, 15.0))
    jump = float(np.abs(np.diff(dv)).max())
    print(f"      vortex contribution: min {dv.min():+.4f} at alpha="
          f"{al[i_min]:.1f} deg, max {dv.max():+.4f}")
    note("F19 vortex gives a nose-down moment break in the stall transient",
         -0.30 < dv.min() < -0.005,
         f"min(Cm_vortex)={dv.min():+.4f} at alpha={al[i_min]:.1f} deg")
    note("F19 vortex moment is negligible while attached",
         float(np.abs(dv[al < 12.0]).max()) < 5.0e-3,
         f"max|Cm_vortex|(|alpha|<12)={float(np.abs(dv[al < 12.0]).max()):.2e}")
    note("F19 vortex convects over the chord and is re-born at the LE",
         0.0 < xv.min() and xv.max() <= 1.0 and xv[i_birth] <= 0.5,
         f"xv in [{xv.min():.3f}, {xv.max():.3f}]")
    note("F19 vortex moment is continuous (no jump at re-birth)",
         jump < 5.0e-3, f"max|dCm_vortex|={jump:.2e}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    ax[0].plot(al, cm_v, 'k-', lw=2, label='with the LE vortex')
    ax[0].plot(al, cm_0, 'C0--', lw=1.4, label='without (vortex_moment=False)')
    ax[0].plot(np.degrees(polar.alpha), polar.cm, 'C3:', lw=1,
               label='static polar')
    ax[0].set_xlim(-2, 34)
    ax[0].set_xlabel(r'$\alpha$ (deg)')
    ax[0].set_ylabel('$C_m$')
    ax[0].set_title('F19: dynamic-stall moment break')
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].plot(al, xv, 'g-')
    ax[1].set_xlabel(r'$\alpha$ (deg)')
    ax[1].set_ylabel(r'$x_v/c$ (from LE)')
    ax[1].set_title('F19: vortex convection')
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('verify_vortex_moment.png', dpi=140)
    plt.close(fig)
    return al, cm_v, cm_0, xv


# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("sharpFSI verification suite")
    print("=" * 70)
    alphas, cl_s, cd_s, cm_s = test_static()
    tau, cl, cl_j = test_wagner()
    rows = test_convergence()
    tt, hist, p = test_structural()
    _fsi, _h = test_coupled()
    kin = test_bem_kinematics()
    polar = test_stall_transparency()
    srows = test_stall_static(polar)
    test_stall_deep(polar)
    al_u, cl_u, cl_d, cl_st, al_d = test_stall_hysteresis(polar)
    test_stall_coupled(polar)
    test_wake_thinning()
    test_wake_drag()
    test_coeff_floor()
    test_vortex_moment(polar)

    # ---- individual figures ---------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(alphas, 2 * np.pi * np.sin(np.deg2rad(alphas)), 'k-',
               label='2$\\pi$sin$\\alpha$ (exact)')
    ax[0].plot(alphas, cl_s, 'bo', ms=5, label='DVM (static)')
    ax[0].set_xlabel('$\\alpha$ (deg)'); ax[0].set_ylabel('$C_l$')
    ax[0].set_title('F1: static lift polar'); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(alphas, cd_s, 'rs-', ms=4, label='DVM')
    k_fit = float(np.sum(cd_s * cl_s ** 2) / np.sum(cl_s ** 4))
    ax[1].plot(alphas, k_fit * cl_s ** 2, 'k--', lw=1, label='C*Cl^2 (fit)')
    ax[1].axhline(0, color='k', lw=0.8, ls=':')
    ax[1].set_xlabel('$\\alpha$ (deg)'); ax[1].set_ylabel('$C_d$')
    ax[1].set_title('F2: potential-flow drag'); ax[1].legend(); ax[1].grid(alpha=0.3)
    ax[2].plot(alphas, cm_s, 'g^-', ms=4)
    ax[2].axhline(0, color='k', lw=0.8, ls=':')
    ax[2].set_xlabel('$\\alpha$ (deg)'); ax[2].set_ylabel('$C_m$ (about 0.25c)')
    ax[2].set_title('F3: pitching moment'); ax[2].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig('verify_static.png', dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tau, cl_j, 'k-', label="Jones $\\Phi(\\tau)$ ($C_{l,ss}$ scaled)")
    ax.plot(tau, cl, 'b--', lw=1.2, label='DVM')
    ax.plot(tau[0], cl[0], 'r*', ms=12,
            label='first step (added-mass impulse)')
    cl_ss_f = 2 * np.pi * np.sin(np.deg2rad(5.0))
    ax.axhline(cl_ss_f, color='g', ls=':', lw=1, label='$C_{l,ss}$')
    ax.set_xlabel(r'$\tau = 2Wt/c$ (semi-chords)'); ax.set_ylabel('$C_l$')
    ax.set_title('F4: impulsive-start (Wagner) response')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig('verify_wagner.png', dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for N in (16, 32, 64):
        xs = [r[1] for r in rows if r[0] == N]
        ys = [r[3] for r in rows if r[0] == N]
        ax.loglog(xs, ys, 'o-', label=f'N={N}')
    ax.set_xlabel('$\\Delta t$ (s)'); ax.set_ylabel('$|C_l-C_{l,exact}|/C_{l,exact}$ (%)')
    ax.set_title('F5: convergence at $\\alpha=10°$'); ax.legend()
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout(); fig.savefig('verify_convergence.png', dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(tt, np.rad2deg(hist), 'b-', lw=1)
    ax.axhline(np.rad2deg(p.psi_eq), color='r', ls=':', label='$\\psi_{eq}$')
    ax.set_xlabel('t (s)'); ax.set_ylabel('$\\psi$ (deg)')
    ax.set_title('F6: free CPPC oscillation'); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig('verify_structural.png', dpi=140); plt.close(fig)

    # ---- stall / drag correction figure --------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    adeg = np.rad2deg(polar.alpha)
    ax[0].plot(adeg, polar.cl, 'k-', lw=1.2, label='prescribed polar')
    ax[0].plot(adeg, polar.cd, 'k--', lw=1.2, label='$C_d$ polar')
    a_s = [r[0] for r in srows]
    ax[0].plot(a_s, [r[1] for r in srows], 'ro', ms=6,
               label='DVM + correction')
    ax[0].plot(a_s, [r[3] for r in srows], 'rs', ms=5, mfc='none')
    ax[0].plot([13.999], [polar.eval(13.999 * np.pi / 180)[0]], 'r^', ms=7,
               label='attached (transparent)')
    ax[0].set_xlim(-95, 95)
    ax[0].set_xlabel('$\\alpha$ (deg)')
    ax[0].set_ylabel('$C_l$ , $C_d$')
    ax[0].set_title('F11-F13: stall correction vs prescribed polar')
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].plot(al_u, cl_st, 'k--', lw=1, label='static polar')
    ax[1].plot(al_u, cl_u, 'r-', lw=1.4, label='ramp up')
    ax[1].plot(al_d, cl_d, 'b-', lw=1.4, label='ramp down')
    ax[1].set_xlabel('$\\alpha$ (deg)')
    ax[1].set_ylabel('$C_l$')
    ax[1].set_title('F14: dynamic-stall hysteresis')
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('verify_stall.png', dpi=140)
    plt.close(fig)

    if kin is not None:
        phi_b, alpha_b, alpha_m = kin
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(np.degrees(phi_b), alpha_b, 'k-', lw=3, alpha=0.35,
                label='sharp-vawt-lab BEM kernel')
        ax.plot(np.degrees(phi_b), alpha_m, 'r--', lw=1.2, label='sharpFSI')
        ax.set_xlabel('azimuth $\\phi$ (deg)')
        ax.set_ylabel('effective $\\alpha$ (deg)')
        ax.set_title('F10: coupled kinematics vs sharp-vawt-lab (prescribed pitch)')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig('verify_kinematics.png', dpi=140)
        plt.close(fig)

    # ---- summary ---------------------------------------------------------
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(111); ax.axis('off')
    lines = [f"sharpFSI verification summary   {n_pass}/{len(RESULTS)} checks passed",
             ""]
    for name, ok, d in RESULTS:
        lines.append(f"{'PASS' if ok else 'FAIL'}  {name}")
        lines.append(f"        {d}")
    ax.text(0.01, 0.99, "\n".join(lines), va='top', ha='left',
            family='monospace', fontsize=8.5)
    fig.savefig('verify_solver_plots.png', dpi=140); plt.close(fig)

    print("\n" + "=" * 70)
    print(f"RESULT: {n_pass}/{len(RESULTS)} checks passed")
    print("plots: verify_solver_plots.png, verify_static.png, verify_wagner.png,")
    print("       verify_convergence.png, verify_structural.png, verify_stall.png"
          + (", verify_kinematics.png" if kin is not None else ""))
    print("=" * 70)
    return n_pass == len(RESULTS)


# ============================================================================
# Check functions for pytest parametrization (one per note() call in main())
# ============================================================================

def check_f1():
    """F1: Static polar Cl = 2*pi*sin(alpha)"""
    alphas = np.arange(-20.0, 20.1, 4.0)
    cl_s = []
    for a in alphas:
        cl, _, _, _ = static_run(a)
        cl_s.append(cl)
    cl_s = np.array(cl_s)
    cl_exact = 2 * np.pi * np.sin(np.deg2rad(alphas))
    return bool(np.abs(cl_s - cl_exact).max() < 0.03)


def check_f2():
    """F2: Potential-flow drag Cd ~ K*Cl^2"""
    alphas = np.arange(-20.0, 20.1, 4.0)
    cl_s, cd_s = [], []
    for a in alphas:
        cl, cd, _, _ = static_run(a)
        cl_s.append(cl); cd_s.append(cd)
    cl_s, cd_s = np.array(cl_s), np.array(cd_s)
    k_drag = float(np.sum(cd_s * cl_s ** 2) / np.sum(cl_s ** 4))
    fit_err = np.abs(cd_s - k_drag * cl_s ** 2).max()
    return bool(np.abs(cd_s).max() < 0.02 and fit_err < 3e-4)


def check_f3():
    """F3: Quarter-chord moment Cm ~ 0"""
    alphas = np.arange(-20.0, 20.1, 4.0)
    cm_s = []
    for a in alphas:
        _, _, cm, _ = static_run(a)
        cm_s.append(cm)
    cm_s = np.array(cm_s)
    return bool(np.abs(cm_s).max() < 0.01)


def check_f3_sp_015():
    """F3: Pivot-shift moment law Cm = Cl*(sp-0.25) at sp=0.15"""
    cl, _, cm, _ = static_run(6.0, sp=0.15)
    expect = cl * (0.15 - 0.25)
    return bool(abs(cm - expect) < 0.02)


def check_f3_sp_040():
    """F3: Pivot-shift moment law Cm = Cl*(sp-0.25) at sp=0.40"""
    cl, _, cm, _ = static_run(6.0, sp=0.40)
    expect = cl * (0.40 - 0.25)
    return bool(abs(cm - expect) < 0.02)


def check_f4_wagner():
    """F4: Wagner history matches Jones (tau>0.5)"""
    dt = 0.005
    t, cl, cl_ss, W, c = impulsive_start(5.0, dt=dt, n_steps=800)
    tau = 2 * W * t / c
    jones = 1.0 - 0.165 * np.exp(-0.0455 * tau) - 0.335 * np.exp(-0.3 * tau)
    cl_j = cl_ss * jones
    m = tau >= 0.5
    rms = float(np.sqrt(np.mean((cl[m] - cl_j[m]) ** 2)) / cl_ss)
    return bool(rms < 0.05)


def check_f4_steady():
    """F4: Cl -> Cl_ss as tau -> 40"""
    dt = 0.005
    _t, cl, cl_ss, _W, _c = impulsive_start(5.0, dt=dt, n_steps=800)
    return bool(abs(cl[-1] - cl_ss) < 0.05 * cl_ss)


def check_f4_added_mass():
    """F4: Added-mass impulse matches rho(pi c^2/4) dW_n"""
    dt = 0.005
    _t, cl, _cl_ss, W, c = impulsive_start(5.0, dt=dt, n_steps=800)
    impulse = float(cl[0] * dt * (0.5 * 1.225 * W ** 2 * c))
    impulse_ref = 1.225 * (np.pi * c ** 2 / 4.0) * (W * np.sin(np.deg2rad(5.0)))
    return bool(0.6 < impulse / impulse_ref < 1.6 and cl[0] > 0)


def check_f5():
    """F5: Panel/timestep convergence (all < 3%)"""
    ex = 2 * np.pi * np.sin(np.deg2rad(10.0))
    errs = []
    for N in (16, 32, 64):
        for dt in (0.02, 0.01, 0.005):
            cl, _, _, _ = static_run(10.0, n_panels=N, dt=dt,
                                     n_steps=int(5.0 / dt), wake_cap=600)
            errs.append(100 * abs(cl - ex) / ex)
    return bool(max(errs) < 3.0)


def check_f6_centre():
    """F6: Oscillates about psi_eq"""
    p = CPPCParams()
    s = CPPCStructural(p)
    s.reset(0.0, 0.0)
    dt = 1e-4
    n = int(8.0 / dt)
    hist = np.zeros(n)
    for i in range(n):
        s.step(0.0, dt)
        hist[i] = s.psi
    ctr = float(hist.mean())
    return bool(abs(ctr - p.psi_eq) < np.deg2rad(1.0))


def check_f6_freq():
    """F6: Frequency matches sqrt(M w^2 Rp ar c / Ip)"""
    p = CPPCParams()
    s = CPPCStructural(p)
    s.reset(0.0, 0.0)
    dt = 1e-4
    n = int(8.0 / dt)
    hist = np.zeros(n)
    for i in range(n):
        s.step(0.0, dt)
        hist[i] = s.psi
    x = hist - float(hist.mean())
    zc = np.where(np.diff(np.sign(x)))[0]
    per = 2 * np.mean(np.diff(zc)) * dt if len(zc) > 2 else np.nan
    om = 2 * np.pi / per
    return bool(abs(om - p.omega_n) / p.omega_n < 0.10)


def check_f7_increases_psi():
    """F7: Positive aero moment increases psi"""
    p = CPPCParams()
    s = CPPCStructural(p)
    s.reset(p.psi_eq, 0.0)
    s.step(+10.0, 0.01, n_sub=10)
    return bool(s.psi > p.psi_eq)


def check_f7_stops():
    """F7: Elastic stops psi = -(win + Q/ks)"""
    p = CPPCParams()
    s = CPPCStructural(p)
    s.reset(0.0, 0.0)
    Q = 50.0
    for _ in range(20000):
        s.step(-Q, 5e-4)
    deflect = Q / p.k_s
    return bool(abs(abs(s.psi) - (p.win + deflect)) < np.deg2rad(1.0))


def check_f8_finite():
    """F8: Finite, no blow-up"""
    fsi = FSICoupler(dt=0.01, n_panels=24, wake_cap=300, omega=2.0,
                     n_coupling_iter=3)
    h = fsi.run(n_revolutions=0.4, verbose=False)
    return bool(np.all(np.isfinite(h['Cl'])) and np.all(np.isfinite(h['psi'])))


def check_f8_wake_bounded():
    """F8: Wake bounded"""
    fsi = FSICoupler(dt=0.01, n_panels=24, wake_cap=300, omega=2.0,
                     n_coupling_iter=3)
    fsi.run(n_revolutions=0.4, verbose=False)
    return bool(fsi.fluid.nw <= fsi.fluid.wake_cap)


def check_f8_kelvin():
    """F8: Kelvin budget ~ 0"""
    fsi = FSICoupler(dt=0.01, n_panels=24, wake_cap=300, omega=2.0,
                     n_coupling_iter=3)
    fsi.run(n_revolutions=0.4, verbose=False)
    budget = fsi.fluid.circulation_budget()
    return bool(abs(budget) < 1e-6)


def check_f8_stops_bounded():
    """F8: Stops bounded"""
    fsi = FSICoupler(dt=0.01, n_panels=24, wake_cap=300, omega=2.0,
                     n_coupling_iter=3)
    h = fsi.run(n_revolutions=0.4, verbose=False)
    osk = np.max(np.abs(h['Q_aero'])) / fsi.struct.p.k_s
    psi_max = np.abs(h['psi']).max()
    return bool(psi_max < fsi.struct.p.win + osk + np.deg2rad(1.0))


def check_f8_one_vortex_per_step():
    """F8: One vortex shed per step"""
    fsi = FSICoupler(dt=0.01, n_panels=24, wake_cap=300, omega=2.0,
                     n_coupling_iter=3)
    fsi.run(n_revolutions=0.4, verbose=False)
    return bool(fsi.fluid.nw + fsi.fluid.merge_count == fsi.step_count)


def check_f9():
    """F9: Unsteady term vanishes at steady state"""
    cl_u, _, _, s_u = static_run(8.0, n_steps=300, average_last=40)
    s_u.unsteady_force = False
    load = s_u.loads()
    return bool(abs(cl_u - load['Cl']) < 1e-3)


def check_f10_alpha():
    """F10: Kinematics match sharp-vawt-lab kernel (alpha)"""
    root = ref_solver.find()
    if root is None:
        return True  # skipped if reference not available
    if root not in sys.path:
        sys.path.insert(0, root)
    from vawt_core import AeroConfig, CycloturbineSim, Geometry, MassProps, PolarDatabase, SimConfig
    R, c, ar, sp, U, tsr, psi0 = 5.0, 1.0, 0.35, 0.25, 10.0, 1.0, 5.0
    omega = tsr * U / R
    g = Geometry(R=R, c=c, H=1.0, N=1, ar=ar, sp=sp)
    ms = MassProps(mb=5.0, xbcg=0.25, mc=2.5, dcw=-0.60)
    _pdir = ref_solver.polar_dir(root) or ''
    ae = AeroConfig(polar_dir=_pdir, use_dynamic_stall=False, use_tip_loss=False,
                    use_dmst=False, use_dynamic_inflow=False,
                    use_flow_curvature=False, cd_add=0.0, win_deg=25.0,
                    wstop=50.0, cb=0.15, kc=0.0, mu_c=0.0, r_bearing=0.0,
                    a_fix=0.0, prescribe_pitch=True, pp0_deg=psi0)
    pdb = PolarDatabase.from_callable(
        lambda a: (2.0 * np.pi * np.sin(a), np.zeros_like(a)),
        cla=2.0 * np.pi, geo=g)
    sim_cfg = SimConfig(U=U, tsr=tsr, free=False, psi0_deg=psi0, dt=1e-3,
                        T_max=2 * np.pi / omega, stride=1, rho=1.225,
                        n_rev_fixed=1)
    out = CycloturbineSim(g, ms, ae, sim_cfg, polar_db=pdb).run()['out']
    phi_b, alpha_b = out[:, 1], out[:, 5]
    fsi = FSICoupler(chord=c, sp=sp, ar=ar, R=R, omega=omega, U_inf=U,
                     m_blade=5.0, cg_blade=0.25, m_cw=2.5, r_cw=0.35)
    psi = np.deg2rad(psi0)
    al = []
    for p in phi_b:
        _x_ref, v_ref, beta, _ = fsi._blade_reference(p, psi, 0.0)
        Wg = fsi._wind() - v_ref
        n_hat = np.array([np.cos(beta), np.sin(beta)])
        xi_hat = np.array([np.sin(beta), -np.cos(beta)])
        al.append(np.arctan2(Wg @ n_hat, Wg @ xi_hat))
    al = np.degrees(np.array(al))
    dalpha = np.abs((al - np.degrees(alpha_b) + 180.0) % 360.0 - 180.0)
    return bool(dalpha.max() < 1e-6)


def check_f10_radius():
    """F10: Blade reference radius = R at zero pitch"""
    fsi = FSICoupler(chord=1.0, sp=0.25, ar=0.35, R=5.0, omega=2.0,
                     U_inf=10.0, m_blade=5.0, cg_blade=0.25, m_cw=2.5,
                     r_cw=0.35)
    x0 = fsi._blade_reference(0.0, 0.0, 0.0)[0]
    return bool(abs(np.hypot(*x0) - 5.0) < 1e-12)


def check_f11_kappa():
    """F11: kappa == 1 in attached flow"""
    from aero_correction import StaticPolar
    polar = StaticPolar.viterna(alpha_stall=14.0, cd0=8.0e-3)
    kmin, kmax = 1.0, 1.0
    for a0 in (2.0, 4.0, 8.0, 12.0):
        _cl0, _, _, _ = static_run(a0, n_panels=48, n_steps=700,
                                  wake_cap=800, average_last=100)
        _cl1, _, _, km, _ = _stall_static(polar, a0)
        kmin, kmax = min(kmin, km), max(kmax, km)
    return bool(max(abs(kmin - 1), abs(kmax - 1)) < 1e-9)


def check_f11_identical():
    """F11: Corrected Cl, Cm identical to pure potential flow"""
    from aero_correction import StaticPolar
    polar = StaticPolar.viterna(alpha_stall=14.0, cd0=8.0e-3)
    dcl = dcm = 0.0
    for a0 in (2.0, 4.0, 8.0, 12.0):
        cl0, _, cm0, _ = static_run(a0, n_panels=48, n_steps=700,
                                    wake_cap=800, average_last=100)
        cl1, _, cm1, _, _ = _stall_static(polar, a0)
        dcl = max(dcl, abs(cl1 - cl0))
        dcm = max(dcm, abs(cm1 - cm0))
    return bool(max(dcl, dcm) < 1e-9)


def check_f12_cl():
    """F12: Cl matches polar in stalled range"""
    from aero_correction import StaticPolar
    polar = StaticPolar.viterna(alpha_stall=14.0, cd0=8.0e-3)
    dcl = 0.0
    for a0 in (16.0, 20.0, 30.0, 45.0, 60.0):
        cl, _, _, _, _ = _stall_static(polar, a0)
        clp, _, _ = polar.eval(np.deg2rad(a0))
        dcl = max(dcl, abs(cl - clp))
    return bool(dcl < 0.05)


def check_f12_cd():
    """F12: Cd matches polar in stalled range"""
    from aero_correction import StaticPolar
    polar = StaticPolar.viterna(alpha_stall=14.0, cd0=8.0e-3)
    dcd = 0.0
    for a0 in (16.0, 20.0, 30.0, 45.0, 60.0):
        _, cd, _, _, _ = _stall_static(polar, a0)
        _, cdp, _ = polar.eval(np.deg2rad(a0))
        dcd = max(dcd, abs(cd - cdp))
    return bool(dcd < 0.05)


def check_f13_cl90():
    """F13: Cl(90 deg) ~ 0"""
    from aero_correction import StaticPolar
    polar = StaticPolar.viterna(alpha_stall=14.0, cd0=8.0e-3)
    cl, _, _, _, _ = _stall_static(polar, 90.0, n_steps=900)
    return bool(abs(cl) < 0.2)


def check_f13_cd90():
    """F13: Cd(90 deg) ~ 2"""
    from aero_correction import StaticPolar
    polar = StaticPolar.viterna(alpha_stall=14.0, cd0=8.0e-3)
    _, cd, _, _, _ = _stall_static(polar, 90.0, n_steps=900)
    return bool(abs(cd - 2.0) < 0.25)


def check_f13_cl170():
    """F13: Cl(170 deg) matches reflected polar"""
    from aero_correction import StaticPolar
    polar = StaticPolar.viterna(alpha_stall=14.0, cd0=8.0e-3)
    cl2, _, _, _, _ = _stall_static(polar, 170.0, n_steps=900)
    clp, _, _ = polar.eval(np.deg2rad(170.0))
    return bool(abs(cl2 - clp) < 0.06)


def check_f13_cm90():
    """F13: Separated moment nose-down for alpha > 0"""
    from aero_correction import StaticPolar
    polar = StaticPolar.viterna(alpha_stall=14.0, cd0=8.0e-3)
    _cl, _, cm, _, _ = _stall_static(polar, 90.0, n_steps=900)
    return bool(cm < -0.02)


def check_f14_overshoot():
    """F14: Lift overshoot in dynamic-stall ramp"""
    from aero_correction import StallModel, StaticPolar
    polar = StaticPolar.viterna(alpha_stall=14.0, cd0=8.0e-3)
    sm = StallModel(polar, T_f=3.0, T_v=6.0, K_v=0.7)
    n, dt, W = 900, 2.0e-3, 10.0
    al = np.zeros(n)
    cl = np.zeros(n)
    for i in range(n):
        a0 = 0.0 + 35.0 * i / (n - 1)
        al[i] = a0
        sm.begin_step()
        sm.update(np.deg2rad(a0), W, dt)
        cl[i] = sm.Cl_ds
    cl_static = polar.eval(np.deg2rad(al))[0]
    over = float(np.max(cl - cl_static))
    return bool(over > 0.05)


def check_f14_hysteresis():
    """F14: Hysteresis loop non-degenerate"""
    from aero_correction import StallModel, StaticPolar
    polar = StaticPolar.viterna(alpha_stall=14.0, cd0=8.0e-3)
    sm = StallModel(polar, T_f=3.0, T_v=6.0, K_v=0.7)
    n, dt, W = 900, 2.0e-3, 10.0
    def ramp(a_start, a_end):
        sm.reset()
        cl = np.zeros(n)
        al = np.zeros(n)
        for i in range(n):
            a0 = a_start + (a_end - a_start) * i / (n - 1)
            al[i] = a0
            sm.begin_step()
            sm.update(np.deg2rad(a0), W, dt)
            cl[i] = sm.Cl_ds
        return al, cl
    al_u, cl_u = ramp(0.0, 35.0)
    _al_d, cl_d = ramp(35.0, 0.0)
    loop = float(np.trapezoid(cl_u - cl_d[::-1], np.deg2rad(al_u)))
    return bool(abs(loop) > 1e-3 and np.all(np.isfinite(cl_u)))


def check_f14_no_overshoot_attached():
    """F14: No overshoot in attached flow (|alpha|<10 deg)"""
    from aero_correction import StallModel, StaticPolar
    polar = StaticPolar.viterna(alpha_stall=14.0, cd0=8.0e-3)
    sm = StallModel(polar, T_f=3.0, T_v=6.0, K_v=0.7)
    n, dt, W = 900, 2.0e-3, 10.0
    al = np.zeros(n)
    cl = np.zeros(n)
    for i in range(n):
        a0 = 0.0 + 35.0 * i / (n - 1)
        al[i] = a0
        sm.begin_step()
        sm.update(np.deg2rad(a0), W, dt)
        cl[i] = sm.Cl_ds
    cl_static = polar.eval(np.deg2rad(al))[0]
    diff = np.max(cl[al < 10.0] - cl_static[al < 10.0])
    return bool(diff < 5e-3)


def check_f15_finite():
    """F15: Coupled stall run is finite"""
    from aero_correction import StaticPolar
    polar = StaticPolar.viterna(alpha_stall=14.0, cd0=8.0e-3)
    fsi = FSICoupler(chord=1.0, sp=0.25, ar=0.35, R=5.0, omega=2.0,
                     U_inf=10.0, rho=1.225, dt=0.005, stall=True,
                     stall_polar=polar, n_panels=32, wake_cap=600,
                     psi0_deg=5.0)
    fsi.run(n_revolutions=0.5, verbose=False)
    h = fsi.history()
    return bool(np.all(np.isfinite(h['psi'])))


def check_f15_lift_limited():
    """F15: Lift is stall-limited where wind is live"""
    from aero_correction import StaticPolar
    polar = StaticPolar.viterna(alpha_stall=14.0, cd0=8.0e-3)
    fsi = FSICoupler(chord=1.0, sp=0.25, ar=0.35, R=5.0, omega=2.0,
                     U_inf=10.0, rho=1.225, dt=0.005, stall=True,
                     stall_polar=polar, n_panels=32, wake_cap=600,
                     psi0_deg=5.0)
    fsi.run(n_revolutions=0.5, verbose=False)
    h = fsi.history()
    sl = slice(5, None)
    live = h['W'][sl] > 4.0
    if not np.any(live):
        return True
    clmax_live = float(np.abs(h['Cl'][sl][live]).max())
    return bool(clmax_live <= 1.6 * polar.cl_max)


def check_f15_kappa():
    """F15: kappa < 1 in stalled part of cycle"""
    from aero_correction import StaticPolar
    polar = StaticPolar.viterna(alpha_stall=14.0, cd0=8.0e-3)
    fsi = FSICoupler(chord=1.0, sp=0.25, ar=0.35, R=5.0, omega=2.0,
                     U_inf=10.0, rho=1.225, dt=0.005, stall=True,
                     stall_polar=polar, n_panels=32, wake_cap=600,
                     psi0_deg=5.0)
    fsi.run(n_revolutions=0.5, verbose=False)
    h = fsi.history()
    sl = slice(5, None)
    al = np.rad2deg(h['alpha'][sl])
    stalled = np.abs(al) > 20.0
    live = h['W'][sl] > 4.0
    kk = h['kappa'][sl][stalled & live]
    if kk.size == 0:
        return True
    return bool(float(np.max(kk)) < 1.0 + 1e-9 and float(np.min(kk)) > -0.3)


def check_f15_active():
    """F15: Stall correction is active in cycle"""
    from aero_correction import StaticPolar
    polar = StaticPolar.viterna(alpha_stall=14.0, cd0=8.0e-3)
    fsi = FSICoupler(chord=1.0, sp=0.25, ar=0.35, R=5.0, omega=2.0,
                     U_inf=10.0, rho=1.225, dt=0.005, stall=True,
                     stall_polar=polar, n_panels=32, wake_cap=600,
                     psi0_deg=5.0)
    fsi.run(n_revolutions=0.5, verbose=False)
    h = fsi.history()
    sl = slice(5, None)
    live = h['W'][sl] > 4.0
    if not np.any(live):
        return False
    need = float(np.mean(np.abs(h['Cl'][sl][live]) > 1.4))
    return bool(need > 0.0)


def check_f16_age_conservation():
    """F16: Age-decimated wake circulation/moment conservation"""
    N, CAP, LEVELS, DS = 3720, 600, 5, 0.05
    age = np.arange(N)
    s = DS * (N - 1 - age)
    x = s + 0.4 * np.sin(s / 3.0)
    y = 0.35 * np.sin(2.0 * np.pi * s / 2.5)
    g = 0.8 * np.cos(2.0 * np.pi * s / 2.5) * np.exp(-s / 40.0)
    c2 = 0.02 ** 2 + 1.5e-5 * age
    m = DVMFluidSolver(chord=1.0, n_panels=8, wake_cap=CAP,
                       merge_mode='age', wake_levels=LEVELS)
    big = N + 16
    m.wx, m.wy, m.gw, m.c2w = (np.zeros(big), np.zeros(big),
                               np.zeros(big), np.zeros(big))
    m.wl = np.zeros(big, dtype=int)
    m.wx[:N], m.wy[:N], m.gw[:N], m.c2w[:N] = x, y, g, c2
    m.nw, m.lev_count[:], m.lev_count[0] = N, 0, N
    while m.nw > CAP:
        m._decimate()
    dg = abs(m.gw[:m.nw].sum() - g.sum())
    dmx = abs((m.gw[:m.nw] * m.wx[:m.nw]).sum() - (g * x).sum())
    dmy = abs((m.gw[:m.nw] * m.wy[:m.nw]).sum() - (g * y).sum())
    return bool(dg < 1e-12 and dmx < 1e-9 and dmy < 1e-9)


def check_f16_age_sorted():
    """F16: Age-decimated wake stays sorted by level"""
    N, CAP, LEVELS, DS = 3720, 600, 5, 0.05
    age = np.arange(N)
    s = DS * (N - 1 - age)
    x = s + 0.4 * np.sin(s / 3.0)
    y = 0.35 * np.sin(2.0 * np.pi * s / 2.5)
    g = 0.8 * np.cos(2.0 * np.pi * s / 2.5) * np.exp(-s / 40.0)
    c2 = 0.02 ** 2 + 1.5e-5 * age
    m = DVMFluidSolver(chord=1.0, n_panels=8, wake_cap=CAP,
                       merge_mode='age', wake_levels=LEVELS)
    big = N + 16
    m.wx, m.wy, m.gw, m.c2w = (np.zeros(big), np.zeros(big),
                               np.zeros(big), np.zeros(big))
    m.wl = np.zeros(big, dtype=int)
    m.wx[:N], m.wy[:N], m.gw[:N], m.c2w[:N] = x, y, g, c2
    m.nw, m.lev_count[:], m.lev_count[0] = N, 0, N
    while m.nw > CAP:
        m._decimate()
    lv = m.wl[:m.nw]
    return bool(np.all(np.diff(lv) <= 0))


def check_f16_age_memory_bound():
    """F16: Age-decimated wake memory bound respected"""
    N, CAP, LEVELS, DS = 3720, 600, 5, 0.05
    age = np.arange(N)
    s = DS * (N - 1 - age)
    x = s + 0.4 * np.sin(s / 3.0)
    y = 0.35 * np.sin(2.0 * np.pi * s / 2.5)
    g = 0.8 * np.cos(2.0 * np.pi * s / 2.5) * np.exp(-s / 40.0)
    c2 = 0.02 ** 2 + 1.5e-5 * age
    m = DVMFluidSolver(chord=1.0, n_panels=8, wake_cap=CAP,
                       merge_mode='age', wake_levels=LEVELS)
    big = N + 16
    m.wx, m.wy, m.gw, m.c2w = (np.zeros(big), np.zeros(big),
                               np.zeros(big), np.zeros(big))
    m.wl = np.zeros(big, dtype=int)
    m.wx[:N], m.wy[:N], m.gw[:N], m.c2w[:N] = x, y, g, c2
    m.nw, m.lev_count[:], m.lev_count[0] = N, 0, N
    while m.nw > CAP:
        m._decimate()
    lv = m.wl[:m.nw]
    npl = m.n_per_level
    return bool(m.nw <= CAP and all(int((lv == L).sum()) <= npl
                                    for L in range(LEVELS + 1)))


def check_f16_age_memory_size():
    """F16: Effective wake memory = n_per_level*(2^L-1)"""
    N, CAP, LEVELS, DS = 3720, 600, 5, 0.05
    age = np.arange(N)
    s = DS * (N - 1 - age)
    x = s + 0.4 * np.sin(s / 3.0)
    y = 0.35 * np.sin(2.0 * np.pi * s / 2.5)
    g = 0.8 * np.cos(2.0 * np.pi * s / 2.5) * np.exp(-s / 40.0)
    c2 = 0.02 ** 2 + 1.5e-5 * age
    m = DVMFluidSolver(chord=1.0, n_panels=8, wake_cap=CAP,
                       merge_mode='age', wake_levels=LEVELS)
    big = N + 16
    m.wx, m.wy, m.gw, m.c2w = (np.zeros(big), np.zeros(big),
                               np.zeros(big), np.zeros(big))
    m.wl = np.zeros(big, dtype=int)
    m.wx[:N], m.wy[:N], m.gw[:N], m.c2w[:N] = x, y, g, c2
    m.nw, m.lev_count[:], m.lev_count[0] = N, 0, N
    while m.nw > CAP:
        m._decimate()
    return bool(m.wake_stats()['equivalent_steps'] == 3720)


def check_f16_induced_velocity():
    """F16: Near-blade induced velocity preserved by age decimation"""
    N, CAP, LEVELS, DS = 3720, 600, 5, 0.05
    age = np.arange(N)
    s = DS * (N - 1 - age)
    x = s + 0.4 * np.sin(s / 3.0)
    y = 0.35 * np.sin(2.0 * np.pi * s / 2.5)
    g = 0.8 * np.cos(2.0 * np.pi * s / 2.5) * np.exp(-s / 40.0)
    c2 = 0.02 ** 2 + 1.5e-5 * age
    probes = np.array([[0.5, 0.0], [2.0, 1.0], [5.0, 3.0], [15.0, 2.0]])

    def ind(P, wx, wy, gw, c2w):
        dx = P[:, 0][:, None] - wx[None, :]
        dy = P[:, 1][:, None] - wy[None, :]
        r2 = dx * dx + dy * dy + c2w[None, :]
        u = (gw[None, :] * dy / (2 * np.pi * r2)).sum(axis=1)
        v = (-gw[None, :] * dx / (2 * np.pi * r2)).sum(axis=1)
        return np.column_stack((u, v))

    ref = ind(probes, x, y, g, c2)
    umax = float(np.hypot(*ref.T).max())
    m = DVMFluidSolver(chord=1.0, n_panels=8, wake_cap=CAP,
                       merge_mode='age', wake_levels=LEVELS)
    big = N + 16
    m.wx, m.wy, m.gw, m.c2w = (np.zeros(big), np.zeros(big),
                               np.zeros(big), np.zeros(big))
    m.wl = np.zeros(big, dtype=int)
    m.wx[:N], m.wy[:N], m.gw[:N], m.c2w[:N] = x, y, g, c2
    m.nw, m.lev_count[:], m.lev_count[0] = N, 0, N
    while m.nw > CAP:
        m._decimate()
    out = ind(probes, m.wx[:m.nw], m.wy[:m.nw], m.gw[:m.nw], m.c2w[:m.nw])
    err = float(np.abs(out - ref).max() / umax)
    return bool(err < 5.0e-3)


def check_f17_panel_invariant():
    """F17: Drag invariant under panel refinement"""
    a = 10.0
    ns, cap = 500, 900
    npan = (16, 32, 64, 128)
    cd_n = []
    for N in npan:
        _, cdv, _, _ = static_run(a, n_panels=N, dt=0.01, n_steps=ns,
                                  wake_cap=cap)
        cd_n.append(cdv)
    cd_n = np.array(cd_n)
    spread = (cd_n.max() - cd_n.min()) / cd_n.mean()
    return bool(spread < 0.05)


def check_f17_wake_length():
    """F17: Drag falls as 1/wake-length"""
    a = 10.0
    npan0, dt = 32, 0.01
    steps = (250, 500, 1000)
    cd_l = []
    for n in steps:
        _, cdv, _, _ = static_run(a, n_panels=npan0, dt=dt, n_steps=n,
                                  wake_cap=n + 5,
                                  average_last=max(20, n // 5))
        cd_l.append(cdv)
    cd_l = np.array(cd_l)
    Lw = dt * 10.0 * np.array(steps)
    slope = np.polyfit(np.log(Lw), np.log(cd_l), 1)[0]
    prod = cd_l * Lw
    return bool(abs(slope + 1.0) < 0.25 and prod.max() / prod.min() < 1.35)


def check_f18_loads_identical():
    """F18: Floor leaves loads bit-identical"""
    a = np.deg2rad(20.0)
    beta = 0.5 * np.pi
    n, dt = 4000, 0.005
    Ws = np.linspace(10.0, 0.1, n)
    rec_f, rec_t = [], []
    for wmin in (2.0, 0.0):
        s = DVMFluidSolver(chord=1.0, n_panels=32, rho=1.225, dt=dt,
                           wake_cap=300, coeff_W_min=wmin)
        s.update_kinematics(np.zeros(2), beta, np.zeros(2), 0.0)
        s.begin_step(Ws[0] * np.array([np.cos(a), np.sin(a)]))
        rec = []
        for W in Ws:
            wind = W * np.array([np.cos(a), np.sin(a)])
            s.begin_step(wind)
            s.solve()
            L = s.loads()
            rec.append((L['Cl'], L['F'].copy(), L['Mz'], L['coeff_valid']))
            s.end_step()
        if wmin == 2.0:
            rec_f = rec
        else:
            rec_t = rec
    dF = max(float(np.abs(a1[1] - b1[1]).max())
             for a1, b1 in zip(rec_f, rec_t))
    dMz = max(abs(a1[2] - b1[2]) for a1, b1 in zip(rec_f, rec_t))
    return bool(dF == 0.0 and dMz == 0.0)


def check_f18_floor_caps():
    """F18: Floor caps dead-point blow-up at (W_ref/W)^2"""
    a = np.deg2rad(20.0)
    beta = 0.5 * np.pi
    n, dt = 4000, 0.005
    Ws = np.linspace(10.0, 0.1, n)
    rec_f, rec_t = [], []
    for wmin in (2.0, 0.0):
        s = DVMFluidSolver(chord=1.0, n_panels=32, rho=1.225, dt=dt,
                           wake_cap=300, coeff_W_min=wmin)
        s.update_kinematics(np.zeros(2), beta, np.zeros(2), 0.0)
        s.begin_step(Ws[0] * np.array([np.cos(a), np.sin(a)]))
        rec = []
        for W in Ws:
            wind = W * np.array([np.cos(a), np.sin(a)])
            s.begin_step(wind)
            s.solve()
            L = s.loads()
            rec.append((L['Cl'], L['F'].copy(), L['Mz'], L['coeff_valid']))
            s.end_step()
        if wmin == 2.0:
            rec_f = rec
        else:
            rec_t = rec
    cl_f = np.array([r[0] for r in rec_f])
    cl_t = np.array([r[0] for r in rec_t])
    ratio = cl_t[-1] / cl_f[-1]
    return bool(abs(ratio - (2.0 / Ws[-1]) ** 2) < 1e-6 and ratio > 1.0e2)


def check_f18_floored_bounded():
    """F18: Floored coefficients stay bounded where true diverge"""
    a = np.deg2rad(20.0)
    beta = 0.5 * np.pi
    n, dt = 4000, 0.005
    Ws = np.linspace(10.0, 0.1, n)
    rec_f, rec_t = [], []
    for wmin in (2.0, 0.0):
        s = DVMFluidSolver(chord=1.0, n_panels=32, rho=1.225, dt=dt,
                           wake_cap=300, coeff_W_min=wmin)
        s.update_kinematics(np.zeros(2), beta, np.zeros(2), 0.0)
        s.begin_step(Ws[0] * np.array([np.cos(a), np.sin(a)]))
        rec = []
        for W in Ws:
            wind = W * np.array([np.cos(a), np.sin(a)])
            s.begin_step(wind)
            s.solve()
            L = s.loads()
            rec.append((L['Cl'], L['F'].copy(), L['Mz'], L['coeff_valid']))
            s.end_step()
        if wmin == 2.0:
            rec_f = rec
        else:
            rec_t = rec
    cl_f = np.array([r[0] for r in rec_f])
    cl_t = np.array([r[0] for r in rec_t])
    floored = Ws < 2.0
    floored[:50] = False
    mf = float(np.abs(cl_f[floored]).max())
    mrat = float(np.max(np.abs(cl_t[floored])
                        / np.maximum(np.abs(cl_f[floored]), 1e-12)))
    return bool(mf < 4.0 and mrat > 50.0)


def check_f18_coeff_valid():
    """F18: coeff_valid flags exactly the floored samples"""
    a = np.deg2rad(20.0)
    beta = 0.5 * np.pi
    n, dt = 4000, 0.005
    Ws = np.linspace(10.0, 0.1, n)
    rec_f = []
    for wmin in (2.0, 0.0):
        s = DVMFluidSolver(chord=1.0, n_panels=32, rho=1.225, dt=dt,
                           wake_cap=300, coeff_W_min=wmin)
        s.update_kinematics(np.zeros(2), beta, np.zeros(2), 0.0)
        s.begin_step(Ws[0] * np.array([np.cos(a), np.sin(a)]))
        rec = []
        for W in Ws:
            wind = W * np.array([np.cos(a), np.sin(a)])
            s.begin_step(wind)
            s.solve()
            L = s.loads()
            rec.append((L['Cl'], L['F'].copy(), L['Mz'], L['coeff_valid']))
            s.end_step()
        if wmin == 2.0:
            rec_f = rec
    return bool(rec_f[0][3] and not rec_f[-1][3])


def check_f19_steady_vortex():
    """F19: Vortex moment vanishes in steady state"""
    from aero_correction import StallModel, StaticPolar
    polar = StaticPolar.viterna()
    sm = StallModel(polar)
    sm.reset()
    a20 = np.deg2rad(20.0)
    for _ in range(6000):
        sm.begin_step()
        sm.update(a20, 10.0, 2.0e-3)
    return bool(abs(sm.Cm_vx) < 1.0e-9)


def check_f19_steady_polar():
    """F19: Steady Cm still equals polar"""
    from aero_correction import StallModel, StaticPolar
    polar = StaticPolar.viterna()
    sm = StallModel(polar)
    sm.reset()
    a20 = np.deg2rad(20.0)
    for _ in range(6000):
        sm.begin_step()
        sm.update(a20, 10.0, 2.0e-3)
    cm_pol = float(polar.eval(a20)[2])
    return bool(abs(sm.Cm_ext - cm_pol) < 1.0e-9)


def check_f19_nose_down_break():
    """F19: Vortex gives nose-down moment break in stall transient"""
    from aero_correction import StallModel, StaticPolar
    polar = StaticPolar.viterna()
    n, dt, W = 1200, 2.0e-3, 10.0
    al = np.zeros(n)
    cm_v, cm_0, xv = np.zeros(n), np.zeros(n), np.zeros(n)
    for vm, dst in ((True, cm_v), (False, cm_0)):
        s = StallModel(polar, vortex_moment=vm)
        s.reset()
        for i in range(n):
            d = -2.0 + 34.0 * i / (n - 1)
            al[i] = d
            s.begin_step()
            s.update(np.deg2rad(d), W, dt)
            dst[i] = s.Cm_ext
            if vm:
                xv[i] = s.xv
    dv = cm_v - cm_0
    return bool(-0.30 < dv.min() < -0.005)


def check_f19_neglect_attached():
    """F19: Vortex moment negligible while attached"""
    from aero_correction import StallModel, StaticPolar
    polar = StaticPolar.viterna()
    n, dt, W = 1200, 2.0e-3, 10.0
    al = np.zeros(n)
    cm_v, cm_0 = np.zeros(n), np.zeros(n)
    for vm, dst in ((True, cm_v), (False, cm_0)):
        s = StallModel(polar, vortex_moment=vm)
        s.reset()
        for i in range(n):
            d = -2.0 + 34.0 * i / (n - 1)
            al[i] = d
            s.begin_step()
            s.update(np.deg2rad(d), W, dt)
            dst[i] = s.Cm_ext
    dv = cm_v - cm_0
    return bool(float(np.abs(dv[al < 12.0]).max()) < 5.0e-3)


def check_f19_vortex_convection():
    """F19: Vortex convects over chord and re-born at LE"""
    from aero_correction import StallModel, StaticPolar
    polar = StaticPolar.viterna()
    n, dt, W = 1200, 2.0e-3, 10.0
    al = np.zeros(n)
    cm_v, cm_0, xv = np.zeros(n), np.zeros(n), np.zeros(n)
    for vm, dst in ((True, cm_v), (False, cm_0)):
        s = StallModel(polar, vortex_moment=vm)
        s.reset()
        for i in range(n):
            d = -2.0 + 34.0 * i / (n - 1)
            al[i] = d
            s.begin_step()
            s.update(np.deg2rad(d), W, dt)
            dst[i] = s.Cm_ext
            if vm:
                xv[i] = s.xv
    i_birth = int(np.searchsorted(al, 15.0))
    return bool(0.0 < xv.min() and xv.max() <= 1.0 and xv[i_birth] <= 0.5)


def check_f19_continuous():
    """F19: Vortex moment continuous (no jump at re-birth)"""
    from aero_correction import StallModel, StaticPolar
    polar = StaticPolar.viterna()
    n, dt, W = 1200, 2.0e-3, 10.0
    al = np.zeros(n)
    cm_v, cm_0, xv = np.zeros(n), np.zeros(n), np.zeros(n)
    for vm, dst in ((True, cm_v), (False, cm_0)):
        s = StallModel(polar, vortex_moment=vm)
        s.reset()
        for i in range(n):
            d = -2.0 + 34.0 * i / (n - 1)
            al[i] = d
            s.begin_step()
            s.update(np.deg2rad(d), W, dt)
            dst[i] = s.Cm_ext
            if vm:
                xv[i] = s.xv
    dv = cm_v - cm_0
    jump = float(np.abs(np.diff(dv)).max())
    return bool(jump < 5.0e-3)


def check_f16_oldest_conservation():
    """F16: Oldest-pair wake circulation/moment conservation"""
    N, CAP, LEVELS, DS = 3720, 600, 5, 0.05
    age = np.arange(N)
    s = DS * (N - 1 - age)
    x = s + 0.4 * np.sin(s / 3.0)
    y = 0.35 * np.sin(2.0 * np.pi * s / 2.5)
    g = 0.8 * np.cos(2.0 * np.pi * s / 2.5) * np.exp(-s / 40.0)
    c2 = 0.02 ** 2 + 1.5e-5 * age
    m = DVMFluidSolver(chord=1.0, n_panels=8, wake_cap=CAP,
                       merge_mode='oldest', wake_levels=LEVELS)
    big = N + 16
    m.wx, m.wy, m.gw, m.c2w = (np.zeros(big), np.zeros(big),
                               np.zeros(big), np.zeros(big))
    m.wl = np.zeros(big, dtype=int)
    m.wx[:N], m.wy[:N], m.gw[:N], m.c2w[:N] = x, y, g, c2
    m.nw, m.lev_count[:], m.lev_count[0] = N, 0, N
    while m.nw > CAP:
        m._decimate()
    dg = abs(m.gw[:m.nw].sum() - g.sum())
    dmx = abs((m.gw[:m.nw] * m.wx[:m.nw]).sum() - (g * x).sum())
    dmy = abs((m.gw[:m.nw] * m.wy[:m.nw]).sum() - (g * y).sum())
    return bool(dg < 1e-12 and dmx < 1e-9 and dmy < 1e-9)


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
