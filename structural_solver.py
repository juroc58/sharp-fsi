"""
structural_solver.py
====================
1-DOF rotational (rocking-pitch) structural model with Centrifugal Pendulum
Pitch Control (CPPC), for the sharp cycloturbine blade layout.  Pure NumPy.

GEOMETRY (identical to sharp_proj's Geometry/MassProps so that the two codes
can be compared directly)
--------------------------------------------------------------------------
                       blade chord line
        LE --- x=(-sp*c) --- pitch axis (x=0) --- TE  x=(1-sp)*c
                              ^
                              |  rocking arm, length ar*c  (perpendicular to
                              |  the chord at zero pitch)
                          rocking pivot  (radius Rp = R - ar*c from shaft)

* The rotor spins about the shaft at constant angular speed omega (the FSI
  targets fixed-rpm operation, exactly like sharp_proj's free=False mode);
  azimuth phi is measured CCW from the wind direction.
* psi is the rocking (pitch) angle about the pivot, positive = nose-out.
  The blade frame angle (used by the fluid solver) is  beta = phi - psi, and
  the blade's inertial angular velocity is omega_body = omega - psi_dot.
* The blade radius R is the radius of the pitch axis (spar) at zero pitch;
  the pivot sits at radius Rp = R - ar*c.
* Blade CG at chord coordinate xi_b = (cg_blade - sp)*c; counterweight mass
  m_cw at chord coordinate xi_cw (m, measured from the pitch axis).  Both are
  rigidly mounted on the rocking arm, hence both carry an (ar*c)^2 inertia
  term -- same convention as sharp_proj's Ip formula.

DYNAMICS (constant rotor speed)
-------------------------------
    I_p * psi_ddot = Q_cent(psi) + Q_aero + Q_damp + Q_fric + Q_stop

with the total CG at

    xi_g   = (m_b*xi_b + m_cw*xi_cw) / (m_b + m_cw)
    s_t    = -xi_g*cos(psi) - ar*c*sin(psi)          (tangential CG offset
                                                      from the pivot; s_t = 0
                                                      defines the mechanical
                                                      equilibrium)
    Q_cent = M * omega^2 * Rp * s_t                  (centrifugal restoring
                                                      moment; equals
                                                      sharp_proj's  X - Qp)
    Q_aero = -Mz   (Mz = aero torque about the pitch axis from the fluid
                    solver, so Q_aero is the generalised moment conjugate to
                    psi)
    Q_damp = -c_b*psi_dot
    Q_fric = -mu_c*F_cen*r_b*tanh(psi_dot/0.5),  F_cen = M*omega^2*(Rp+s_r),
             s_r = -xi_g*sin(psi) + ar*c*cos(psi)
    Q_stop = -k_s*(psi -+ win) - c_s*psi_dot  outside the +/-win limits,
             k_s = I_p*wstop^2, c_s = 2*I_p*wstop  (sharp_proj defaults).

All signs have been checked against the physical pendulum picture (see
verify_solver.py structural tests): with the CG ahead of the pivot
(xi_g < 0) the equilibrium is nose-out, psi_eq = atan2(-xi_g, ar*c), and the
natural frequency is omega_n = sqrt(M*omega^2*Rp*(ar*c)/I_p) to leading order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class CPPCParams:
    # geometry
    chord: float = 1.0
    sp: float = 0.25            # pitch axis (spar) from LE / chord
    ar: float = 0.35            # rocking arm / chord
    R: float = 5.0              # radius of the pitch axis at zero pitch [m]
    omega: float = 2.0          # rotor angular speed [rad/s]
    rho: float = 1.225
    # mass properties
    m_blade: float = 5.0
    cg_blade: float = 0.20      # blade CG from LE / chord
    m_cw: float = 2.5
    xi_cw: float = 0.35         # counterweight chord coordinate from the
                                # pitch axis [m] (positive = towards TE)
    I_add: float = 0.0          # optional extra polar inertia [kg m^2]
    # pivot dissipation
    c_b: float = 0.15           # viscous pivot damping [N m s/rad]
    mu_c: float = 0.0           # Coulomb friction coefficient (x F_cen*r_b)
    r_bearing: float = 0.002
    # mechanical stops
    win_deg: float = 25.0       # pitch limit [deg]
    wstop: float = 50.0         # stop natural frequency [rad/s]
    # gravity (off by default, matches sharp_proj)
    g: float = 0.0
    xi_cg_gravity_arm: float = 0.0   # unused unless g != 0

    # --- derived (filled by __post_init__) ---
    Rp: float = field(init=False)
    ar_len: float = field(init=False)
    xi_b: float = field(init=False)
    xi_g: float = field(init=False)
    M: float = field(init=False)
    I_p: float = field(init=False)
    win: float = field(init=False)
    k_s: float = field(init=False)
    c_s: float = field(init=False)
    psi_eq: float = field(init=False)
    omega_n: float = field(init=False)
    fn_ratio: float = field(init=False)

    def __post_init__(self):
        self.ar_len = self.ar * self.chord
        self.Rp = self.R - self.ar_len
        self.xi_b = (self.cg_blade - self.sp) * self.chord
        self.M = self.m_blade + self.m_cw
        self.xi_g = (self.m_blade * self.xi_b
                     + self.m_cw * self.xi_cw) / self.M
        # polar inertia about the pivot (same structure as sharp_proj)
        self.I_p = (self.m_blade * (self.chord ** 2 / 12.0
                                    + self.xi_b ** 2 + self.ar_len ** 2)
                    + self.m_cw * (self.xi_cw ** 2 + self.ar_len ** 2)
                    + self.I_add)
        self.win = np.deg2rad(self.win_deg)
        self.k_s = self.I_p * self.wstop ** 2
        self.c_s = 2.0 * self.I_p * self.wstop
        self.psi_eq = np.arctan2(-self.xi_g, self.ar_len)
        if self.I_p > 0:
            self.omega_n = np.sqrt(self.M * self.omega ** 2 * self.Rp
                                   * self.ar_len / self.I_p)
            self.fn_ratio = self.omega_n / self.omega
        else:
            self.omega_n = 0.0
            self.fn_ratio = 0.0


class CPPCStructural:
    """State-integrated rocking-pitch model (RK4)."""

    def __init__(self, params: CPPCParams | None = None):
        self.p = params or CPPCParams()
        self.psi = 0.0
        self.psi_dot = 0.0
        # diagnostics
        self.last_Q = {}

    # ------------------------------------------------------------- moments
    def s_t(self, psi):
        """Tangential CG offset from the pivot (0 at mechanical equilibrium)."""
        return -self.p.xi_g * np.cos(psi) - self.p.ar_len * np.sin(psi)

    def s_r(self, psi):
        """Radial CG offset from the pivot (used for the bearing load)."""
        return -self.p.xi_g * np.sin(psi) + self.p.ar_len * np.cos(psi)

    def Q_cent(self, psi):
        return self.p.M * self.p.omega ** 2 * self.p.Rp * self.s_t(psi)

    def Q_gravity(self, psi):
        if self.p.g == 0.0:
            return 0.0
        # gravity acts along -y in the world; projected on the rocking DOF
        # (only meaningful when the caller knows the azimuth -- kept simple)
        return 0.0

    def Q_fric(self, psi, psi_dot):
        if self.p.mu_c == 0.0:
            return 0.0
        F_cen = self.p.M * self.p.omega ** 2 * (self.p.Rp + self.s_r(psi))
        return -self.p.mu_c * F_cen * self.p.r_bearing * np.tanh(psi_dot / 0.5)

    def Q_stop(self, psi, psi_dot):
        win = self.p.win
        if psi > win:
            return -self.p.k_s * (psi - win) - self.p.c_s * psi_dot
        if psi < -win:
            return -self.p.k_s * (psi + win) - self.p.c_s * psi_dot
        return 0.0

    def total_Q(self, psi, psi_dot, Q_aero):
        return (self.Q_cent(psi) + Q_aero
                - self.p.c_b * psi_dot
                + self.Q_fric(psi, psi_dot)
                + self.Q_stop(psi, psi_dot))

    # -------------------------------------------------------------- stepping
    def _deriv(self, psi, psi_dot, Q_aero):
        return psi_dot, self.total_Q(psi, psi_dot, Q_aero) / self.p.I_p

    def step(self, Q_aero: float, dt: float, n_sub: int = 1):
        """Advance the pitch state by dt with the aero moment held constant."""
        h = dt / n_sub
        psi, psd = self.psi, self.psi_dot
        for _ in range(n_sub):
            k1 = self._deriv(psi, psd, Q_aero)
            k2 = self._deriv(psi + 0.5 * h * k1[0], psd + 0.5 * h * k1[1], Q_aero)
            k3 = self._deriv(psi + 0.5 * h * k2[0], psd + 0.5 * h * k2[1], Q_aero)
            k4 = self._deriv(psi + h * k3[0], psd + h * k3[1], Q_aero)
            psi += h / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            psd += h / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        self.psi, self.psi_dot = psi, psd
        self.last_Q = {
            'Q_aero': Q_aero,
            'Q_cent': self.Q_cent(psi),
            'Q_damp': -self.p.c_b * psd,
            'Q_fric': self.Q_fric(psi, psd),
            'Q_stop': self.Q_stop(psi, psd),
        }
        return psi, psd

    def reset(self, psi0=0.0, psi_dot0=0.0):
        self.psi = float(psi0)
        self.psi_dot = float(psi_dot0)

    def state(self):
        return self.psi, self.psi_dot

    # ------------------------------------------------------------ diagnostics
    def energy(self):
        """Kinetic + effective centrifugal potential of the rocking DOF."""
        psi, psd = self.psi, self.psi_dot
        KE = 0.5 * self.p.I_p * psd ** 2
        # potential of Q_cent = -dU/dpsi  ->  U = M omega^2 Rp * (xi_g sin psi
        # - ar*c cos psi) up to a constant
        U = self.p.M * self.p.omega ** 2 * self.p.Rp * (
            self.p.xi_g * np.sin(psi) - self.p.ar_len * np.cos(psi))
        return KE + U


# ---------------------------------------------------------------------------
# Backwards-compatible convenience wrapper: counterweight specified by a
# radial offset from the pitch axis (as in the original script).
# ---------------------------------------------------------------------------
def make_cppc(m_blade=5.0, cg_blade=0.25, chord=1.0,
              m_cw=2.5, r_cw=0.35, ar=0.35, sp=0.25, R=5.0, omega=2.0,
              c_b=0.15, mu_c=0.0, win_deg=25.0, wstop=50.0, rho=1.225):
    """Create a CPPCStructural with the counterweight at chord coordinate
    xi_cw = r_cw (positive = aft of the pitch axis)."""
    p = CPPCParams(chord=chord, sp=sp, ar=ar, R=R, omega=omega, rho=rho,
                   m_blade=m_blade, cg_blade=cg_blade, m_cw=m_cw,
                   xi_cw=r_cw, c_b=c_b, mu_c=mu_c, win_deg=win_deg,
                   wstop=wstop)
    return CPPCStructural(p)


if __name__ == "__main__":
    print("Structural solver smoke test")
    s = make_cppc()
    p = s.p
    print(f"  Rp={p.Rp:.3f} m  xi_g={p.xi_g:.4f} m  I_p={p.I_p:.4f} kg m^2")
    print(f"  psi_eq={np.rad2deg(p.psi_eq):.2f} deg  "
          f"omega_n={p.omega_n:.3f} rad/s  fn_ratio={p.fn_ratio:.3f}")
    # free oscillation from rest at psi=0 with no aero moment
    s.reset(0.0, 0.0)
    dt = 1e-4
    T = 6.0
    n = int(T / dt)
    hist = np.zeros(n)
    for i in range(n):
        s.step(0.0, dt)
        hist[i] = s.psi
    # estimate period from zero crossings of (psi - psi_eq)
    x = hist - p.psi_eq
    zc = np.where(np.diff(np.sign(x)))[0]
    if len(zc) > 2:
        per = 2 * np.mean(np.diff(zc)) * dt
        print(f"  measured period={per:.4f}s -> omega={2*np.pi/per:.3f} "
              f"rad/s (analytic {p.omega_n:.3f})")
    print(f"  final psi={np.rad2deg(s.psi):.3f} deg (eq "
          f"{np.rad2deg(p.psi_eq):.3f} deg)")
