"""
fsi_main.py
===========
Partitioned two-way FSI coupling for the 2D VAWT blade section: the DVM fluid
solver (fluid_solver.py) drives the single-DOF CPPC rocking-pitch model
(structural_solver.py).

Coupling scheme
---------------
Per time step:
  1. `begin_step` freezes the release position of the vortex that will be
     shed during the step (Katz-Plotkin placement) using the current state.
  2. Strong (Gauss-Seidel) coupling loop on the pitch kinematics: feed the
     fluid solver a pitch angle, solve the bound/shed vortex system, evaluate
     the aerodynamic pitch moment Q_aero = -Mz, advance the structural model
     one step with that moment, and under-relax the pitch used by the fluid.
     Iterate until |delta psi| < tol (2-3 iterations are typical).
  3. Once converged, re-solve with the accepted kinematics, then advance the
     free wake exactly once (`end_step`): RK2 convection of the existing wake
     plus append the newly shed vortex.

Sign/kinematics conventions (see fluid_solver.py docstring):
  * phi   : azimuth, CCW from the wind direction
  * psi   : rocking pitch, positive = nose-out
  * beta  : blade frame angle = phi - psi (given to the fluid solver)
  * omega_body = omega - psi_dot (blade inertial angular velocity)

Two different points are involved and must not be confused:

  * the STRUCTURAL pivot (the rocking joint), at radius Rp = R - ar*c;
  * the BLADE AERO REFERENCE, i.e. the point where the fluid solver places the
    chord line (chord coordinate x = 0, the spar / quarter chord).  It is
    carried on the rocking arm, so it sits at

        X_ref = X_pivot + ar*c*n_hat(beta),      n_hat = z_hat x xi_hat,

    i.e. at radius R at zero pitch -- exactly the arrangement of the CPPC in
    sharp_proj (its `stq/srq` quarter-chord offsets).  A check that the FSI
    kinematics reproduce sharp_proj's kernel formulas identically (u1,u2 and
    alpha) to 1e-15 is in /tmp/opencode/kin_check.py.

The fluid solver works entirely about X_ref.  The structural model needs the
moment about the pivot, so the coupler transfers it:

    Mz_pivot = Mz_ref + (X_ref - X_pivot) x F      (z component)
    Q_aero   = -Mz_pivot
"""

from __future__ import annotations

import time
import numpy as np

from fluid_solver import DVMFluidSolver
from structural_solver import CPPCParams, CPPCStructural


class FSICoupler:
    def __init__(
        self,
        # geometry / kinematics
        chord: float = 1.0,
        sp: float = 0.25,
        ar: float = 0.35,
        R: float = 5.0,
        omega: float = 2.0,
        U_inf: float = 10.0,
        rho: float = 1.225,
        dt: float = 0.005,
        induction: float = 0.0,
        # structural / CPPC
        m_blade: float = 5.0,
        cg_blade: float = 0.25,
        m_cw: float = 2.5,
        r_cw: float = 0.35,
        c_b: float = 0.15,
        mu_c: float = 0.0,
        win_deg: float = 25.0,
        wstop: float = 50.0,
        psi0_deg: float = 5.0,
        psi_dot0: float = 0.0,
        # fluid numerics
        n_panels: int = 32,
        wake_cap: int = 600,
        core_bound: float = 0.002,
        core_wake: float = 0.02,
        wake_diffusivity: float = 1.5e-5,
        wake_substeps: int = 1,
        viscous_cd0: float = 0.0,
        # aero correction (semi-empirical stall / drag layer)
        stall: bool = False,
        stall_polar=None,
        stall_alpha_stall: float = 14.0,
        stall_T_a: float = 0.0,
        stall_T_f: float = 3.0,
        stall_T_v: float = 6.0,
        stall_K_v: float = 0.7,
        stall_vortex_lift: bool = True,
        stall_cd_sep_coeff: float = 1.98,
        stall_cp_sep: float = 0.5,
        stall_cm_sep_const=None,
        stall_blend_state: bool = False,
        stall_vortex_moment: bool = True,
        stall_x_v0: float = 0.05,
        stall_conv_frac: float = 1.0 / 3.0,
        # coefficient normalisation
        coeff_W_min: float = 2.0,
        # coupling controls
        n_coupling_iter: int = 3,
        coupling_tol: float = 1e-4,
        relax: float = 0.7,
    ):
        self.dt = dt
        self.omega = omega
        self.U_inf = U_inf
        self.induction = induction
        self.n_iter = n_coupling_iter
        self.tol = coupling_tol
        self.relax = relax
        self.chord = chord
        self.R = R
        self.rho = rho

        # optional semi-empirical stall / separated-drag correction.
        # `stall=True` uses the self-contained Viterna polar; pass
        # `stall_polar=<aero_correction.StaticPolar>` to use a tabulated polar
        # (e.g. Sheldahl-Klimas NACA 0012) instead.
        self.stall_polar = None
        self.stall_model = None
        if stall or stall_polar is not None:
            from aero_correction import StaticPolar, StallModel
            if stall_polar is None:
                self.stall_polar = StaticPolar.viterna(
                    alpha_stall=stall_alpha_stall,
                    cd0=max(viscous_cd0, 8.0e-3))
            else:
                self.stall_polar = stall_polar
            self.stall_model = StallModel(
                self.stall_polar, T_a=stall_T_a, T_f=stall_T_f, T_v=stall_T_v,
                K_v=stall_K_v, vortex_lift=stall_vortex_lift,
                cd_sep_coeff=stall_cd_sep_coeff, cp_sep=stall_cp_sep,
                cm_sep_const=stall_cm_sep_const, chord=chord,
                blend_state=stall_blend_state,
                vortex_moment=stall_vortex_moment,
                x_v0=stall_x_v0, conv_frac=stall_conv_frac)

        self.fluid = DVMFluidSolver(
            chord=chord, n_panels=n_panels, rho=rho, dt=dt, sp=sp,
            wake_cap=wake_cap, core_bound=core_bound, core_wake=core_wake,
            wake_diffusivity=wake_diffusivity, viscous_cd0=viscous_cd0,
            wake_substeps=wake_substeps, stall_model=self.stall_model,
            coeff_W_min=coeff_W_min)
        self.ar_len = ar * chord
        self.Rp = R - self.ar_len
        self._lever = np.zeros(2)

        self.struct = CPPCStructural(CPPCParams(
            chord=chord, sp=sp, ar=ar, R=R, omega=omega, rho=rho,
            m_blade=m_blade, cg_blade=cg_blade, m_cw=m_cw, xi_cw=r_cw,
            c_b=c_b, mu_c=mu_c, win_deg=win_deg, wstop=wstop))
        self.struct.reset(np.deg2rad(psi0_deg), psi_dot0)

        self.psi = np.deg2rad(psi0_deg)
        self.psi_dot = psi_dot0
        self.t = 0.0
        self.phi = 0.0
        self.step_count = 0
        self.iter_counts = []

        self._init_history()

    # ------------------------------------------------------------ kinematics
    @property
    def U_eff(self):
        return (1.0 - self.induction) * self.U_inf

    def _pivot_position(self, phi):
        """Structural pitch pivot (radius Rp = R - ar*c)."""
        return self.Rp * np.array([np.cos(phi), np.sin(phi)])

    def _pivot_velocity(self, phi):
        return self.omega * self.Rp * np.array([-np.sin(phi), np.cos(phi)])

    def _wind(self):
        return np.array([self.U_eff, 0.0])

    def _blade_reference(self, phi, psi, psi_dot):
        """Blade aero reference point (quarter chord) and its velocity.

        The blade is carried on a rocking arm of length ar*c measured from the
        structural pivot along the local "nose-out" normal n_hat = z_hat x xi_hat
        (this is the arrangement of the CPPC in sharp_proj: the quarter chord
        sits at radius Rp + ar*c ~= R at psi = 0).  Since the arm+blade rotate
        rigidly about the pivot with angular velocity (omega - psi_dot), the
        velocity of the reference point follows from the rigid-body field.
        """
        beta = phi - psi
        n_hat = np.array([np.cos(beta), np.sin(beta)])      # z_hat x xi_hat
        om_body = self.omega - psi_dot
        x_ref = self._pivot_position(phi) + self.ar_len * n_hat
        v_ref = self._pivot_velocity(phi) + om_body * self.ar_len * np.array(
            [-n_hat[1], n_hat[0]])                          # z_hat x (ar*n_hat)
        return x_ref, v_ref, beta, om_body

    def _set_fluid_kinematics(self, phi, psi, psi_dot):
        x_ref, v_ref, beta, om_body = self._blade_reference(phi, psi, psi_dot)
        self._lever = x_ref - self._pivot_position(phi)
        self.fluid.update_kinematics(x_ref, beta, v_ref, om_body)

    def _moment_about_pivot(self, loads):
        """Transfer the aero torque from the blade reference to the pitch pivot."""
        F = loads['F']
        return loads['Mz'] + self._lever[0] * F[1] - self._lever[1] * F[0]

    # -------------------------------------------------------------- history
    HIST_KEYS = ('t', 'phi', 'alpha', 'psi', 'psi_dot', 'Cl', 'Cd', 'Cm',
                 'W', 'Q_aero', 'Mz', 'Gamma', 'Q_cent', 'Q_rotor', 'kappa',
                 'coeff_valid')

    def _init_history(self):
        self.max_steps = 20000
        for name in self.HIST_KEYS:
            setattr(self, 'h_' + name, np.zeros(self.max_steps))

    def _grow_history(self):
        for name in self.HIST_KEYS:
            arr = getattr(self, 'h_' + name)
            new = np.zeros(len(arr) * 2)
            new[:len(arr)] = arr
            setattr(self, 'h_' + name, new)
        self.max_steps *= 2

    def _record(self, alpha, Cl, Cd, Cm, W, Q_aero, Mz, Gamma, Q_cent,
                Q_rotor=0.0, kappa=1.0, coeff_valid=1.0):
        if self.step_count >= self.max_steps:
            self._grow_history()
        i = self.step_count
        self.h_t[i] = self.t
        self.h_phi[i] = self.phi
        self.h_alpha[i] = alpha
        self.h_psi[i] = self.psi
        self.h_psi_dot[i] = self.psi_dot
        self.h_Cl[i] = Cl
        self.h_Cd[i] = Cd
        self.h_Cm[i] = Cm
        self.h_W[i] = W
        self.h_Q_aero[i] = Q_aero
        self.h_Mz[i] = Mz
        self.h_Gamma[i] = Gamma
        self.h_Q_cent[i] = Q_cent
        self.h_Q_rotor[i] = Q_rotor
        self.h_kappa[i] = kappa
        self.h_coeff_valid[i] = coeff_valid

    # ------------------------------------------------------------ time step
    def step(self):
        phi = self.phi
        psi0, psd0 = self.struct.state()
        wind = self._wind()

        # freeze the shed-vortex release position for this step
        self._set_fluid_kinematics(phi, psi0, psd0)
        self.fluid.begin_step(wind)

        psi_g, psd_g = psi0, psd0
        loads = None
        n_it = 0
        for n_it in range(1, self.n_iter + 1):
            self._set_fluid_kinematics(phi, psi_g, psd_g)
            self.fluid.wind = wind
            self.fluid.solve()
            loads = self.fluid.loads()
            Q_aero = -self._moment_about_pivot(loads)
            self.struct.reset(psi0, psd0)
            psi_new, psd_new = self.struct.step(Q_aero, self.dt)
            dpsi = psi_new - psi_g
            if abs(dpsi) < self.tol:
                break
            psi_g = psi_g + self.relax * dpsi
            psd_g = psd_new
        self.iter_counts.append(n_it)

        # re-solve with the accepted kinematics so the stored fluid state and
        # loads correspond to the state we actually advance
        self._set_fluid_kinematics(phi, psi_new, psd_new)
        self.fluid.wind = wind
        self.fluid.solve()
        loads = self.fluid.loads()
        Mz_piv = self._moment_about_pivot(loads)
        Q_aero = -Mz_piv
        F = loads['F']
        x_ref = self.fluid.X_piv
        Q_rotor = x_ref[0] * F[1] - x_ref[1] * F[0] + loads['Mz']

        # advance the free wake exactly once
        self.fluid.end_step()

        self.psi, self.psi_dot = psi_new, psd_new
        self.struct.reset(psi_new, psd_new)

        self._record(loads['alpha'], loads['Cl'], loads['Cd'], loads['Cm'],
                     loads['W'], Q_aero, Mz_piv, loads['Gamma'],
                     self.struct.Q_cent(psi_new), Q_rotor,
                     loads.get('kappa', 1.0),
                     float(loads.get('coeff_valid', True)))

        self.t += self.dt
        self.phi = (self.phi + self.omega * self.dt) % (2.0 * np.pi)
        self.step_count += 1
        return self.state_dict()

    def state_dict(self):
        return {
            't': self.t, 'phi': self.phi, 'psi': self.psi,
            'psi_dot': self.psi_dot, 'alpha': self.fluid.alpha,
            'Cl': self.fluid.Cl, 'Cd': self.fluid.Cd, 'Cm': self.fluid.Cm,
            'W': self.fluid.W_rel,
        }

    def run(self, n_revolutions: float = 4.0, verbose: bool = True):
        T_rev = 2.0 * np.pi / self.omega
        n_steps = int(round(n_revolutions * T_rev / self.dt))
        t0 = time.perf_counter()
        if verbose:
            print(f"FSI run: {n_revolutions} revolutions, {n_steps} steps")
            print(f"  dt={self.dt}s  T_rev={T_rev:.3f}s  TSR="
                  f"{self.omega * self.R / self.U_inf:.2f}  "
                  f"induction={self.induction:.2f}")
            print(f"  psi_eq={np.rad2deg(self.struct.p.psi_eq):.2f} deg  "
                  f"fn_ratio={self.struct.p.fn_ratio:.2f}")
            if self.stall_model is not None:
                print(f"  stall correction ON: polar='{self.stall_polar.name}'"
                      f"  T_f={self.stall_model.T_f:g}"
                      f" T_v={self.stall_model.T_v:g}"
                      f" K_v={self.stall_model.K_v:g}")
            else:
                print("  stall correction OFF (pure potential flow)")
        for k in range(n_steps):
            self.step()
            if verbose and (k % max(1, n_steps // 10) == 0):
                print(f"  {100.0 * k / n_steps:5.1f}%  t={self.t:6.2f}s  "
                      f"phi={np.rad2deg(self.phi):7.1f}  "
                      f"alpha={np.rad2deg(self.fluid.alpha):6.2f}  "
                      f"psi={np.rad2deg(self.psi):6.2f}  "
                      f"Cl={self.fluid.Cl:6.3f}  wake={self.fluid.nw}")
        if verbose:
            el = time.perf_counter() - t0
            self.wall_time = el
            print(f"  done: {self.step_count} steps in {el:.1f}s "
                  f"({1e3 * el / max(1, self.step_count):.2f} ms/step, "
                  f"mean coupling iters {np.mean(self.iter_counts):.2f})")
        else:
            self.wall_time = time.perf_counter() - t0
        return self.history()

    def history(self):
        n = self.step_count
        return {name: getattr(self, 'h_' + name)[:n].copy()
                for name in self.HIST_KEYS}

    def save_history(self, path='fsi_validation_history.txt'):
        h = self.history()
        cols = list(self.HIST_KEYS)
        header = ' '.join(cols)
        data = np.column_stack([h[c] for c in cols])
        np.savetxt(path, data, header=header, fmt='%.8e')
        print(f"history saved to {path} ({len(data)} rows)")


def create_fsi_coupler(config=None):
    cfg = dict(
        chord=1.0, sp=0.25, ar=0.35, R=5.0, omega=2.0, U_inf=10.0,
        rho=1.225, dt=0.005, induction=0.0,
        m_blade=5.0, cg_blade=0.25, m_cw=2.5, r_cw=0.35,
        c_b=0.15, mu_c=0.0, win_deg=25.0, wstop=50.0,
        psi0_deg=5.0, n_panels=32, wake_cap=600,
        core_bound=0.002, core_wake=0.02, wake_substeps=1,
        viscous_cd0=0.0, n_coupling_iter=3, coupling_tol=1e-4, relax=0.7,
        stall=False, stall_polar=None, stall_alpha_stall=14.0,
        stall_T_a=0.0, stall_T_f=3.0, stall_T_v=6.0, stall_K_v=0.7,
        stall_vortex_lift=True, stall_cd_sep_coeff=1.98, stall_cp_sep=0.5,
        stall_cm_sep_const=None, stall_blend_state=False,
        stall_vortex_moment=True, stall_x_v0=0.05, stall_conv_frac=1.0 / 3.0,
        coeff_W_min=2.0)
    if config:
        cfg.update(config)
    return FSICoupler(**cfg)


if __name__ == "__main__":
    fsi = create_fsi_coupler(dict(dt=0.005, n_panels=24, wake_cap=400))
    h = fsi.run(n_revolutions=0.5, verbose=True)
    print('last state:', fsi.state_dict())
