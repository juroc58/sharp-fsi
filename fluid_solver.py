"""
fluid_solver.py
===============
2D unsteady potential-flow solver (Discrete Vortex Method / lumped-vortex
lattice) for a thin symmetric aerofoil (NACA 0012 camber line => flat plate)
with a free vortex wake.  Pure NumPy, fully vectorised, bounded memory.

This is the fluid half of the sharpFSI 2D VAWT FSI suite.  The structural
half lives in structural_solver.py and the coupling in fsi_main.py.

MODEL
-----
* Bound vorticity: N cosine-spaced panels along the camber line (chord line);
  one lumped vortex Gamma_j at the 1/4 point of each panel, flow tangency
  enforced at the 3/4 point of each panel.
* Unsteady Kutta condition (Katz & Plotkin style): the linear system solved
  each step has N+1 unknowns -- the N bound vortices and the strength of the
  single new wake vortex shed during that step.  The extra equation is
  Kelvin's circulation theorem,

        sum_j Gamma_j^(n+1) + Gamma_shed = sum_j Gamma_j^(n)

  so the total circulation of (bound + wake) is conserved exactly.
* The new wake vortex is released at  X_shed = X_TE + f*dt*V_local(X_TE)
  (f = shed_factor, default 0.5 = the centroid of the vorticity shed during
  the step) and starts convecting freely on the following step.
* Free wake: wake vortices convect with the local fluid velocity
  (freestream + all bound + all wake induction) integrated with RK2.
* Wake management: the wake is thinned to respect the ``wake_cap`` memory
  bound.  The default scheme is multiresolution *age decimation*
  (``merge_mode='age'``): the newest ``n_per_level`` vortices are kept at full
  resolution, and older ones are merged pairwise into successively coarser
  levels, so the wake represents ``n_per_level*(2**wake_levels - 1)`` shed
  vortices (~4-6 revolutions at the default settings) for the cost of
  ``n_per_level*(wake_levels + 1)``.  Merging conserves total circulation and
  the circulation-weighted centroid.  Every level -- including the deepest --
  is thinned to ``n_per_level``, so the population, and hence the storage, is
  bounded by ``n_per_level*(wake_levels + 1)`` (about ``1.2*wake_cap``).
  ``merge_mode='oldest'`` restores the simpler scheme of repeatedly merging the
  two oldest vortices.  In both cases the cost stays bounded (O(N_wake^2) per
  step with a fixed cap).  Wake cores grow diffusively: core^2 += 4*nu*dt (nu
  default 1.5e-5 m^2/s, air).
* Loads: two contributions per panel (see the derivation in `loads()`):
  (a) the quasi-steady lumped-vortex Kutta-Joukowski

        dF_j = rho * Gamma_j * (W_j x z_hat)

      with W_j the local fluid velocity relative to the (rigidly moving) body
      at the 1/4 point of panel j (the vortex's own induction is exactly zero
      there), which reproduces Cl = 2*pi*sin(alpha) with the leading-edge
      suction implicit in the induced velocities and the centre of pressure at
      the quarter chord;
  (b) the unsteady (apparent-mass) term from Bernoulli's equation for the
      potential jump across the sheet,

        dF_j += -rho * d/dt[ sum_{k<j} Gamma_k ] * dl_j * n_hat_j .

      It vanishes in the steady state and supplies the fast initial response
      (Wagner), including the correct added-mass impulse direction.
  A constant `viscous_cd0` (profile drag, e.g. 0.008 for NACA 0012) can be
  added to the *reported* Cd only; it is applied at the pitch axis and so
  contributes no pitching moment.

OPTIONAL STALL / DRAG CORRECTION
-------------------------------
Passing ``stall_model=<aero_correction.StallModel>`` layers a semi-empirical
static + dynamic stall and separated-drag correction on top of the (exact)
attached potential solution.  Only the circulatory load is scaled, by
``kappa = Cl_ds(alpha)/[2*pi*sin(alpha)]``; with the default polar ``kappa`` is
exactly 1 while the flow is attached, so all the verified results below are
unchanged wherever the flow is attached.  The extra drag is applied at the
quarter chord and the separated pitching moment as a pure couple.  See
``aero_correction.py`` for the model and ``verify_solver.py`` F11-F14.

Numerical validation of all of the above is in verify_solver.py (static polar,
d'Alembert drag, quarter-chord moment, pivot-shift law, Wagner impulsive start,
panel/timestep convergence, Kelvin budget).

CONVENTIONS (fixed -- re-run verify_solver.py if you touch them)
----------------------------------------------------------------
* Vortices are positive counter-clockwise (CCW):

        u(P) = Gamma/(2*pi) * (-dy, dx) / (dx^2 + dy^2 + core^2),
        (dx, dy) = P - X_vortex.

* Blade kinematics are supplied by the caller (see fsi_main.py):
    X_pivot    : world position of the pitch axis
    beta       : blade angle; chord direction (LE -> TE) is
                 xi_hat = (sin(beta), -cos(beta)),
                 n_hat  = z_hat x xi_hat = (cos(beta), sin(beta))  (+lift side)
    v_pivot    : world velocity of the pitch axis
    omega_body : angular velocity of the blade about the pivot (CCW +)
  World position of a body point with chord coordinate x (x = 0 at the pitch
  axis, LE at -sp*c, TE at +(1-sp)*c):

        X = X_pivot + x * xi_hat

  and the rigid-body velocity field is

        V_body(X) = v_pivot + omega_body * z_hat x (X - X_pivot).

* Reported coefficients use the geometric relative wind at the pitch axis,
  W_geo = V_wind - v_pivot, and q = 0.5*rho*|W_geo|^2, S = c:
    alpha = atan2(W_geo . n_hat, W_geo . xi_hat)        (local AoA)
    Cl    = F . (z_hat x W_geo_hat) / (q*c)
    Cd    = F . W_geo_hat           / (q*c)   (+ viscous_cd0)
    Cm    = -Mz / (q*c^2),  Mz = sum_j (X_j - X_pivot) x dF_j   (z component)

  The normalising q is floored at `coeff_W_min` (default 2 m/s) so the reported
  coefficients stay bounded and continuous through the zero-relative-wind dead
  point (where q -> 0 but the force does not); `coeff_valid` flags the samples
  where the floor is active.  The *loads* always use the true q.
  Cm > 0 means nose-up: the moment tends to increase the local angle of
  attack.  The lift acts near the quarter chord, so the exact static result is
  Cm = Cl * (sp - 0.25) (verified in verify_solver.py F3): zero with the pitch
  axis at 0.25c, ~+0.1*Cl with the axis at 0.40c (statically unstable, CP aft
  of the axis) and ~-0.1*Cl with the axis at 0.15c.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


TWO_PI = 2.0 * np.pi


def _induced_by_blobs(P, Xs, gammas, core2):
    """Biot-Savart velocity at points P (K,2) from 2-D vortex blobs.

    Uses the identity

        u = sum_j -(Py - Yj) Gj ,   v = sum_j (Px - Xj) Gj     (Gj = g_j/(2 pi r_j^2))

    written as  u = -Py*sum_j Gj + (G @ Y) ,  v = +Px*sum_j Gj - (G @ X) .
    This avoids materialising the (K,N) differences dx, dy and reduces the
    operation to one distance matrix (scipy's C implementation) plus one BLAS
    matrix-vector product; on the target hardware it is ~3x faster than the
    plain broadcasting form while giving identical results to ~1e-13.

    `core2` may be a scalar (uniform core) or an (N,) array (per-vortex core^2).
    """
    P = np.ascontiguousarray(P, dtype=float)
    Xs = np.ascontiguousarray(Xs, dtype=float)
    r2 = cdist(P, Xs, 'sqeuclidean')
    r2 += core2
    g = gammas[None, :] / (TWO_PI * r2)
    s = g.sum(axis=1)
    u = g @ Xs[:, 1] - P[:, 1] * s
    v = P[:, 0] * s - g @ Xs[:, 0]
    return np.column_stack((u, v))


class DVMFluidSolver:
    """Unsteady thin-aerofoil DVM with free wake (one lifting surface)."""

    # ------------------------------------------------------------------ setup
    def __init__(
        self,
        chord: float = 1.0,
        n_panels: int = 32,
        rho: float = 1.225,
        dt: float = 0.01,
        sp: float = 0.25,
        wake_cap: int = 600,
        core_bound: float = 0.002,
        core_wake: float = 0.02,
        wake_diffusivity: float = 1.5e-5,
        viscous_cd0: float = 0.0,
        wake_substeps: int = 1,
        spacing: str = 'cosine',
        shed_factor: float = 0.5,
        unsteady_force: bool = True,
        stall_model=None,
        wake_levels: int = 5,
        merge_mode: str = 'age',
        coeff_W_min: float = 2.0,
    ):
        self.c = float(chord)
        self.N = int(n_panels)
        self.rho = float(rho)
        self.dt = float(dt)
        self.sp = float(sp)
        self.wake_cap = int(wake_cap)
        self.core_b2 = (core_bound * self.c) ** 2
        self.core_w2_0 = (core_wake * self.c) ** 2
        self.nu = float(wake_diffusivity)
        self.viscous_cd0 = float(viscous_cd0)
        self.wake_substeps = max(1, int(wake_substeps))
        self.spacing = spacing
        self.shed_factor = float(shed_factor)
        self.unsteady_force = bool(unsteady_force)
        # floor on |W| (m/s) used *only* to normalise the reported
        # coefficients, so that they stay bounded (and continuous) through the
        # zero-relative-wind dead point where the true dynamic pressure
        # vanishes.  The physical loads always use the true q = 0.5*rho*W^2.
        self.coeff_W_min = float(coeff_W_min)
        # optional semi-empirical stall/drag correction (aero_correction.StallModel)
        self.stall = stall_model
        self.stall_info = None
        if self.stall is not None:
            self.stall.chord = self.c
        # wake thinning: 'age' = multiresolution age decimation (default),
        # 'oldest' = repeatedly merge the two oldest vortices
        self.wake_levels = max(1, int(wake_levels))
        self.merge_mode = str(merge_mode)
        self.n_per_level = max(2, self.wake_cap // self.wake_levels)

        self._setup_geometry()

        # --- free-wake storage (fixed capacity, age ordered: index 0 oldest)
        # Every merge level is thinned to n_per_level (see _decimate), so the
        # worst-case population is n_per_level*(wake_levels+1) plus the block
        # being merged; size the arrays for that plus a small slack so that the
        # append-before-thin order in end_step can never write past the end.
        self.wake_storage = self.n_per_level * (self.wake_levels + 1) + 8
        cap = self.wake_storage
        self.wx = np.zeros(cap)
        self.wy = np.zeros(cap)
        self.gw = np.zeros(cap)
        self.c2w = np.zeros(cap)
        self.wl = np.zeros(cap, dtype=int)          # merge level per vortex
        self.lev_count = np.zeros(self.wake_levels + 2, dtype=int)
        self.nw = 0
        self.merge_count = 0
        self.step_index = 0

        # --- state
        self.gamma_b = np.zeros(self.N)
        self.gamma_shed = 0.0
        self.gamma_prev = 0.0
        self.gamma_total_shed = 0.0   # diagnostic
        self.pref_prev = np.zeros(self.N)  # prefix circulation sums (unsteady force)

        # kinematics (set by update_kinematics)
        self.X_piv = np.zeros(2)
        self.v_piv = np.zeros(2)
        self.beta = 0.0
        self.omega_body = 0.0
        self.xi_hat = np.array([0.0, -1.0])
        self.n_hat = np.array([1.0, 0.0])
        self.Xv = np.zeros((self.N, 2))
        self.Xc = np.zeros((self.N, 2))
        self.Xte = np.zeros(2)
        self.Vb_ctrl = np.zeros((self.N, 2))
        self.Vb_vort = np.zeros((self.N, 2))
        self.wind = np.zeros(2)

        # last loads / coefficients
        self.Cl = self.Cd = self.Cm = 0.0
        self.kappa = 1.0
        self.alpha = 0.0
        self.W_rel = 0.0
        self.Fx = self.Fy = 0.0
        self.Mz = 0.0
        self.x_new = None

    # --------------------------------------------------------------- geometry
    def _setup_geometry(self):
        """Panel layout on the camber line, in body coordinates.

        'cosine' clusters panels towards both ends (classic vortex-lattice
        spacing); 'uniform' gives equal panels.
        """
        c, N, sp = self.c, self.N, self.sp
        if self.spacing == 'uniform':
            x_ep = np.linspace(-sp * c, (1.0 - sp) * c, N + 1)
        else:
            beta = np.linspace(0.0, np.pi, N + 1)
            x_ep = -sp * c + 0.5 * c * (1.0 - np.cos(beta))
        self.x_ep = x_ep
        self.x_vort = 0.75 * x_ep[:-1] + 0.25 * x_ep[1:]
        self.x_ctrl = 0.25 * x_ep[:-1] + 0.75 * x_ep[1:]
        self.panel_len = np.diff(x_ep)
        self.x_le = float(x_ep[0])
        self.x_te = float(x_ep[-1])

    # ------------------------------------------------------------ kinematics
    def update_kinematics(self, X_pivot, beta, v_pivot, omega_body):
        """Set the current body position, angle and velocity."""
        self.X_piv = np.asarray(X_pivot, dtype=float)
        self.beta = float(beta)
        self.v_piv = np.asarray(v_pivot, dtype=float)
        self.omega_body = float(omega_body)

        sb, cb = np.sin(beta), np.cos(beta)
        self.xi_hat = np.array([sb, -cb])          # LE -> TE
        self.n_hat = np.array([cb, sb])            # z_hat x xi_hat

        xi = self.xi_hat
        self.Xv = self.X_piv + np.outer(self.x_vort, xi)
        self.Xc = self.X_piv + np.outer(self.x_ctrl, xi)
        self.Xte = self.X_piv + self.x_te * xi

        self.Vb_ctrl = self._body_velocity(self.Xc)
        self.Vb_vort = self._body_velocity(self.Xv)

    def _body_velocity(self, X):
        """Rigid body velocity field V = v_pivot + omega (z x (X - X_pivot))."""
        r = X - self.X_piv
        return self.v_piv + self.omega_body * np.column_stack((-r[:, 1], r[:, 0]))

    # ---------------------------------------------------- induced velocities
    def _bound_vel_at_many(self, P):
        """Velocity at points P (K,2) induced by all bound vortices."""
        if self.N == 0:
            return np.zeros((len(P), 2))
        return _induced_by_blobs(P, self.Xv, self.gamma_b, self.core_b2)

    def _wake_vel_at_many(self, P):
        """Velocity at points P (K,2) induced by all wake vortices."""
        m = self.nw
        if m == 0:
            return np.zeros((len(P), 2))
        return _induced_by_blobs(P, np.column_stack((self.wx[:m], self.wy[:m])),
                                 self.gw[:m], self.c2w[:m])

    def _vel_all(self, P):
        """Freestream + bound + wake velocity at points P (K,2)."""
        return (self.wind[None, :]
                + self._bound_vel_at_many(P)
                + self._wake_vel_at_many(P))

    def _wake_self_vel(self, P):
        """Velocity at wake points P (m,2) due to the wake itself (no freestream/bound)."""
        m = len(P)
        if m <= 1:
            return np.zeros((m, 2))
        c2 = self.c2w[:m]
        r2 = cdist(np.ascontiguousarray(P, dtype=float), P, 'sqeuclidean')
        r2 += c2[:, None]
        r2 += c2[None, :]
        g = self.gw[None, :m] / (TWO_PI * r2)
        s = g.sum(axis=1)
        u = g @ P[:, 1] - P[:, 1] * s
        v = P[:, 0] * s - g @ P[:, 0]
        return np.column_stack((u, v))

    # -------------------------------------------------------------- one step
    def begin_step(self, wind):
        """Freeze the release position of the vortex about to be shed.

        Must be called once per time step, after `update_kinematics` with the
        kinematics at the start of the step and before the (possibly
        iterated) solves.
        """
        self.wind = np.asarray(wind, dtype=float)
        if self.stall is not None:
            self.stall.begin_step()
        # local fluid velocity at the trailing edge, using the current state
        v_te = self.wind + self._bound_vel_at_many(self.Xte[None, :])[0] \
            + self._wake_vel_at_many(self.Xte[None, :])[0]
        self.x_new = self.Xte + self.shed_factor * self.dt * v_te

    def solve(self):
        """Solve the (N+1) bound/shed vortex system -> self.gamma_b, gamma_shed.

        The strength of the vortex shed during this step is an unknown of the
        linear system; the (N+1)-th equation is Kelvin's circulation theorem
        (the total bound+wake circulation is conserved exactly).  This keeps
        the Kutta condition in its standard implicit lumped-vortex (1/4-3/4)
        form while the shed vortex satisfies the conservation law.
        """
        N = self.N
        nx, ny = self.n_hat

        # bound-bound influence (normal velocity at control points)
        dx = self.Xc[:, 0][:, None] - self.Xv[None, :, 0]
        dy = self.Xc[:, 1][:, None] - self.Xv[None, :, 1]
        r2 = dx * dx + dy * dy + self.core_b2
        A = (-dy * nx + dx * ny) / (TWO_PI * r2)

        # influence of the vortex shed during this step
        dxn = self.Xc[:, 0] - self.x_new[0]
        dyn = self.Xc[:, 1] - self.x_new[1]
        r2n = dxn * dxn + dyn * dyn + self.core_w2_0
        a_new = (-dyn * nx + dxn * ny) / (TWO_PI * r2n)

        # right hand side: n . (V_body - V_wind - u_wake)
        vw = self._wake_vel_at_many(self.Xc)
        rhs = (nx * (self.Vb_ctrl[:, 0] - self.wind[0] - vw[:, 0])
               + ny * (self.Vb_ctrl[:, 1] - self.wind[1] - vw[:, 1]))

        M = np.empty((N + 1, N + 1))
        M[:N, :N] = A
        M[:N, N] = a_new
        M[N, :N] = 1.0
        M[N, N] = 1.0

        rhs_full = np.empty(N + 1)
        rhs_full[:N] = rhs
        rhs_full[N] = self.gamma_prev

        sol = np.linalg.solve(M, rhs_full)
        self.gamma_b = sol[:N]
        self.gamma_shed = float(sol[N])
        return self.gamma_b

    def loads(self):
        """Per-panel loads and coefficients.

        Two contributions (both per unit span):

        * quasi-steady Kutta-Joukowski from the local flow velocity relative
          to the body at the 1/4 point of each panel,
        * the unsteady (apparent-mass / non-circulatory) term.

        The latter is derived from Bernoulli's equation applied to the two
        sides of the sheet.  With gamma = u_lower - u_upper and J(x) = phi_low
        - phi_up the potential jump (J(0) = 0 at the leading edge, J' = gamma),

            dp(x) = p_low - p_up = -rho[ ubar(x) gamma(x) + d/dt J(x) ],
            J(x)  = int_0^x gamma(xi) dxi          (a *prefix* integral)

        so the discrete panel term is

            dF_j = -rho * d/dt[ sum_{k<j} Gamma_k ] * dl_j * n_hat_j .

        This vanishes in the steady state, gives the correct (classical)
        added-mass direction for an impulsive start (+n_hat when the normal
        velocity of the incoming flow grows), and reproduces Wagner's
        function (see verify_solver.py F4).
        """
        ub = self._bound_vel_at_many(self.Xv)
        uw = self._wake_vel_at_many(self.Xv)
        Wj = self.wind[None, :] + ub + uw - self.Vb_vort          # (N,2)
        # Unsteady Kutta-Joukowski with CCW-positive circulation:
        #   dF = rho * Gamma * (W x z_hat), i.e. -rho*Gamma*(z_hat x W)
        # (equivalently F = -rho*Gamma*(-Wy, Wx) = rho*Gamma*(Wy, -Wx)).
        # With W = (+W, 0) this gives lift DOWN for Gamma > 0, which is the
        # correct Magnus sense: suction-on-top requires CW (negative) bound
        # circulation.  See verify_solver.py static tests.
        Fc = self.rho * self.gamma_b[:, None] * np.column_stack((Wj[:, 1], -Wj[:, 0]))

        # exclusive prefix sums of the bound circulation, and their rate
        pref = np.cumsum(self.gamma_b) - self.gamma_b
        self.pref_curr = pref
        if self.unsteady_force:
            dprefdt = (pref - self.pref_prev) / self.dt
            Fu = -self.rho * (dprefdt * self.panel_len)[:, None] * self.n_hat[None, :]
        else:
            Fu = np.zeros_like(Fc)

        # coefficients are based on the geometric relative wind at the
        # reference (pitch) axis; the stall model needs the same alpha/W.
        Wgeo = self.wind - self.v_piv
        W = float(np.hypot(Wgeo[0], Wgeo[1]))
        if W < 1e-9:
            W = 1e-9
        q = 0.5 * self.rho * W * W
        w_hat = Wgeo / W
        lift_dir = np.array([-w_hat[1], w_hat[0]])                # z_hat x w_hat
        alpha = float(np.arctan2(Wgeo @ self.n_hat, Wgeo @ self.xi_hat))

        # Reference dynamic pressure for the *reported coefficients*.  At the
        # zero-relative-wind dead point q -> 0 while the force stays finite, so
        # normalising by q makes Cl/Cd/Cm diverge (up to ~1e4).  Flooring the
        # reference speed at coeff_W_min keeps them bounded and continuous;
        # `coeff_valid` flags the samples where the floor is active (there the
        # coefficients are force/q_ref, not the true section coefficients).
        W_ref = max(W, self.coeff_W_min)
        q_ref = 0.5 * self.rho * W_ref * W_ref
        coeff_valid = W >= self.coeff_W_min

        # --- optional semi-empirical stall / profile-drag correction --------
        # In the attached/near-attached regime only the *circulatory* part is
        # scaled (the unsteady/added-mass term and the Wagner lag carried by
        # the DVM solution are left untouched); with kappa == 1 the corrected
        # loads are the verified potential-flow loads.  Deep stall
        # (|alpha| > ~90 deg) blends over to *replacing* the circulatory load
        # by the model load, because there the attached Kutta-condition
        # solution is on the leading-edge-separation branch.
        kappa, dCd, Cm_ext, wrep, Cl_ds = 1.0, 0.0, 0.0, 0.0, 0.0
        Fc_tot = Fc.sum(axis=0)
        r = self.Xv - self.X_piv
        Mc = float(np.sum(r[:, 0] * Fc[:, 1] - r[:, 1] * Fc[:, 0]))
        Fu_tot = Fu.sum(axis=0)
        Mu = float(np.sum(r[:, 0] * Fu[:, 1] - r[:, 1] * Fu[:, 0]))

        if self.stall is not None:
            kappa, Cd_ds, Cm_ds, info = self.stall.update(
                alpha, W, self.dt, step=self.step_index)
            self.stall_info = info
            dCd = Cd_ds - self.viscous_cd0        # cd0 is re-added just below
            Cm_ext = Cm_ds
            wrep = info['replace_blend']
            Cl_ds = info['Cl_ds']

            Xqc = self.X_piv + (0.25 - self.sp) * self.c * self.xi_hat
            rq = Xqc - self.X_piv

            # (A) scale the whole aerodynamic load (circulatory + unsteady:
            #     kappa == 1 while attached, and a reduced added mass when
            #     separated), add the extra drag at the quarter chord
            Fdrag = q * self.c * dCd * w_hat
            Fa = kappa * (Fc_tot + Fu_tot) + Fdrag
            Ma = (kappa * (Mc + Mu)
                  + float(rq[0] * Fdrag[1] - rq[1] * Fdrag[0]))

            # (B) replace the aerodynamic load entirely by the model load
            #     (the DVM's attached Kutta-condition solution is on the
            #     leading-edge-separation branch beyond ~90 deg and is neither
            #     reliable nor relevant there)
            Fmodel = q * self.c * (Cl_ds * lift_dir + dCd * w_hat)
            Fb = Fmodel
            Mb = float(rq[0] * Fmodel[1] - rq[1] * Fmodel[0])

            Ftot = (1.0 - wrep) * Fa + wrep * Fb
            Mz = (1.0 - wrep) * Ma + wrep * Mb
            # polar Cm is in the standard (nose-down negative) convention,
            # while Mz is CCW-positive and Cm_reported = -Mz/(q c^2)
            Mz += -q * self.c * self.c * Cm_ext
        else:
            Ftot = Fc_tot + Fu_tot
            Mz = Mc + Mu

        Cl = float(Ftot @ lift_dir) / (q_ref * self.c)
        Cd = float(Ftot @ w_hat) / (q_ref * self.c)
        Cd += self.viscous_cd0
        Cm = -Mz / (q_ref * self.c * self.c)

        self.Fx, self.Fy = float(Ftot[0]), float(Ftot[1])
        self.Mz = Mz
        self.Cl, self.Cd, self.Cm = Cl, Cd, Cm
        self.alpha = alpha
        self.W_rel = W
        self.kappa = kappa
        return {
            'Cl': Cl, 'Cd': Cd, 'Cm': Cm, 'alpha': alpha, 'W': W,
            'q': q, 'q_ref': q_ref, 'coeff_valid': coeff_valid,
            'Fx': self.Fx, 'Fy': self.Fy, 'Mz': Mz, 'kappa': kappa,
            'Gamma': float(self.gamma_b.sum()), 'F': Ftot,
        }

    def end_step(self):
        """Convect the existing wake (RK2) and append the new shed vortex."""
        dt = self.dt
        n = self.nw
        if n > 0:
            P = np.column_stack((self.wx[:n], self.wy[:n]))
            for _ in range(self.wake_substeps):
                dts = dt / self.wake_substeps
                v1 = self.wind[None, :] + self._bound_vel_at_many(P) + self._wake_self_vel(P)
                Pm = P + 0.5 * dts * v1
                v2 = self.wind[None, :] + self._bound_vel_at_many(Pm) + self._wake_self_vel(Pm)
                P = P + dts * v2
            self.wx[:n] = P[:, 0]
            self.wy[:n] = P[:, 1]
            self.c2w[:n] += 4.0 * self.nu * dt

        # keep the wake bounded (see _decimate / the module docstring)
        # append the vortex that was shed during this step first, then thin
        i = self.nw
        self.wx[i] = self.x_new[0]
        self.wy[i] = self.x_new[1]
        self.gw[i] = self.gamma_shed
        self.c2w[i] = self.core_w2_0
        self.wl[i] = 0
        self.nw = i + 1
        self.lev_count[0] += 1
        self._decimate()

        self.gamma_prev = float(self.gamma_b.sum())
        self.gamma_total_shed += self.gamma_shed
        self.pref_prev = (np.cumsum(self.gamma_b) - self.gamma_b).copy()
        self.step_index += 1

    # ------------------------------------------------------------ wake thinning
    def _level_start(self, lev):
        """Index of the first vortex of merge level `lev` (levels are ordered
        oldest-first, so the level-`lev` block starts after all higher levels).
        """
        return int(self.lev_count[lev + 1:].sum())

    def _decimate(self):
        """Thin the wake to honour the memory bound ``wake_cap``.

        Two schemes are available:

        * ``merge_mode='oldest'``: repeatedly merge the two oldest vortices.
          Simple, but the far wake ends up represented by a single blob whose
          circulation is the sum of everything shed long ago.
        * ``merge_mode='age'`` (default): multiresolution age decimation.  The
          wake is held in age order and each vortex carries a merge level.  The
          most recent ``n_per_level`` vortices are kept at full resolution;
          beyond that, whenever a level holds more than ``n_per_level``
          vortices the two oldest of that level are merged into one at the
          next level (circulation and circulation-weighted centroid are
          conserved, core^2 grows by the usual ``+sep^2/4`` term).  Every level
          is thinned this way, including the deepest, so the result represents
          ``n_per_level*(2**wake_levels - 1)`` shed vortices at the cost of at
          most ``n_per_level*(wake_levels + 1)``, i.e. a much longer wake memory
          -- and hence a much better far-field/induced-velocity description --
          for the same CPU cost as the oldest-pair scheme.
        """
        if self.merge_mode != 'age':
            while self.nw >= self.wake_cap:
                self._merge_oldest()
            return
        npl = self.n_per_level
        lev = 0
        # thin *every* level, including the deepest, so that the wake population
        # is bounded by n_per_level*(wake_levels+1) and can never overrun the
        # storage (the deepest level used to be open-ended)
        while lev <= self.wake_levels and self.lev_count[lev] > npl:
            self._merge_pair(self._level_start(lev), lev)
            lev += 1

    def _merge_pair(self, i, lev):
        """Merge wake vortices ``i`` and ``i+1`` into ``i`` at level ``lev+1``.

        Both must be in the same level block; the merged vortex is the
        youngest of the next level, which is exactly index ``i`` in the
        age-ordered layout, so no additional reordering is needed.
        """
        g0, g1 = self.gw[i], self.gw[i + 1]
        gs = g0 + g1
        x0, y0 = self.wx[i], self.wy[i]
        x1, y1 = self.wx[i + 1], self.wy[i + 1]
        if abs(gs) > 1e-14:
            self.wx[i] = (g0 * x0 + g1 * x1) / gs
            self.wy[i] = (g0 * y0 + g1 * y1) / gs
        else:
            self.wx[i] = 0.5 * (x0 + x1)
            self.wy[i] = 0.5 * (y0 + y1)
        self.gw[i] = gs
        self.c2w[i] = (self.c2w[i] + self.c2w[i + 1]
                       + 0.25 * ((x1 - x0) ** 2 + (y1 - y0) ** 2))
        self.wl[i] = lev + 1
        self.lev_count[lev] -= 2            # two vortices merged ...
        self.lev_count[lev + 1] += 1        # ... into one at the next level
        n = self.nw
        if i + 2 < n:
            self.wx[i + 1:n - 1] = self.wx[i + 2:n].copy()
            self.wy[i + 1:n - 1] = self.wy[i + 2:n].copy()
            self.gw[i + 1:n - 1] = self.gw[i + 2:n].copy()
            self.c2w[i + 1:n - 1] = self.c2w[i + 2:n].copy()
            self.wl[i + 1:n - 1] = self.wl[i + 2:n].copy()
        self.nw = n - 1
        self.merge_count += 1

    def _merge_oldest(self):
        """Merge wake vortices 0 and 1 (oldest pair), conserving circulation."""
        g0, g1 = self.gw[0], self.gw[1]
        gs = g0 + g1
        x0, y0, x1, y1 = self.wx[0], self.wy[0], self.wx[1], self.wy[1]
        if abs(gs) > 1e-14:
            self.wx[0] = (g0 * x0 + g1 * x1) / gs
            self.wy[0] = (g0 * y0 + g1 * y1) / gs
        else:
            self.wx[0] = 0.5 * (x0 + x1)
            self.wy[0] = 0.5 * (y0 + y1)
        self.gw[0] = gs
        sep2 = (x1 - x0) ** 2 + (y1 - y0) ** 2
        self.c2w[0] = self.c2w[0] + self.c2w[1] + 0.25 * sep2

        n = self.nw
        self.wx[1:n - 1] = self.wx[2:n].copy()
        self.wy[1:n - 1] = self.wy[2:n].copy()
        self.gw[1:n - 1] = self.gw[2:n].copy()
        self.c2w[1:n - 1] = self.c2w[2:n].copy()
        self.wl[1:n - 1] = self.wl[2:n].copy()
        self.nw = n - 1
        self.lev_count[0] -= 1          # keep the level bookkeeping valid
        self.merge_count += 1

    # ------------------------------------------------------------- utilities
    def reset_wake(self):
        self.nw = 0
        self.merge_count = 0
        self.lev_count[:] = 0
        self.wl[:] = 0
        self.gamma_prev = 0.0
        self.gamma_total_shed = 0.0
        self.gamma_b[:] = 0.0
        self.pref_prev[:] = 0.0
        if self.stall is not None:
            self.stall.reset()

    def wake_positions(self):
        return self.wx[:self.nw].copy(), self.wy[:self.nw].copy()

    def wake_stats(self):
        """Diagnostics for the wake thinning scheme."""
        cnt = [int((self.wl[:self.nw] == lv).sum())
               for lv in range(self.wake_levels + 2)]
        return {
            'n': self.nw,
            'merges': self.merge_count,
            'levels': cnt,
            'per_level': self.n_per_level,
            'equivalent_steps': int(self.n_per_level
                                    * (2 ** self.wake_levels - 1)),
            'mode': self.merge_mode,
        }

    def circulation_budget(self):
        """Total bound + wake circulation (should stay ~constant ~0)."""
        return float(self.gamma_b.sum() + self.gw[:self.nw].sum())


# ---------------------------------------------------------------------------
# Thin wrapper adding the VAWT blade kinematics (kept deliberately thin: the
# physics lives in DVMFluidSolver; geometry conventions live here).
# ---------------------------------------------------------------------------
class VAWTFluidSolver:
    """DVM section solver driven by rotor azimuth and pitch angle.

    Geometry (matches the Sharp cycloturbine layout used in sharp-vawt-lab):
      * rotor radius R is the radius of the blade pitch axis (spar) at zero
        pitch, i.e. the rocking-arm tip;
      * the pitch axis / rocking pivot sits at radius Rp = R - ar*c;
      * blade chord coordinate x runs from the pitch axis, LE at -sp*c and
        TE at +(1-sp)*c;
      * azimuth phi measured CCW from the wind direction (+x);
      * psi is the rocking angle, positive = nose-out; the blade frame angle
        is beta = phi - psi.
    """

    def __init__(self, chord=1.0, n_panels=32, rho=1.225, dt=0.01,
                 sp=0.25, ar=0.35, R=5.0, omega=2.0,
                 wake_cap=600, core_bound=0.002, core_wake=0.02,
                 wake_diffusivity=1.5e-5, viscous_cd0=0.0,
                 wake_substeps=1, U_inf=10.0, stall_model=None,
                 coeff_W_min=2.0):
        self.dvm = DVMFluidSolver(
            chord=chord, n_panels=n_panels, rho=rho, dt=dt, sp=sp,
            wake_cap=wake_cap, core_bound=core_bound, core_wake=core_wake,
            wake_diffusivity=wake_diffusivity, viscous_cd0=viscous_cd0,
            wake_substeps=wake_substeps, stall_model=stall_model,
            coeff_W_min=coeff_W_min)
        self.c = chord
        self.sp = sp
        self.ar = ar
        self.R = R
        self.omega = omega
        self.U_inf = U_inf
        self.Rp = R - ar * chord
        self.azimuth = 0.0

    # geometry helpers ----------------------------------------------------
    def pivot_position(self, phi):
        return self.Rp * np.array([np.cos(phi), np.sin(phi)])

    def pivot_velocity(self, phi):
        return self.omega * self.Rp * np.array([-np.sin(phi), np.cos(phi)])

    def blade_angle(self, phi, psi):
        return phi - psi

    def blade_omega(self, psi_dot):
        return self.omega - psi_dot

    def wind(self, U=None, gust=0.0):
        u = self.U_inf if U is None else U
        return np.array([u + gust, 0.0])

    # one fluid sub-step (called by the coupler) --------------------------
    def update(self, phi, psi, psi_dot, U=None, gust=0.0):
        beta = self.blade_angle(phi, psi)
        self.dvm.update_kinematics(
            self.pivot_position(phi), beta,
            self.pivot_velocity(phi), self.blade_omega(psi_dot))
        self.azimuth = phi
        self.dvm.wind = self.wind(U, gust)
        return beta

    def begin_step(self, phi, psi, psi_dot, U=None, gust=0.0):
        self.update(phi, psi, psi_dot, U, gust)
        self.dvm.begin_step(self.dvm.wind)

    def solve(self):
        return self.dvm.solve()

    def loads(self):
        return self.dvm.loads()

    def end_step(self):
        return self.dvm.end_step()

    def reset_wake(self):
        self.dvm.reset_wake()

    # convenience: coefficient access ------------------------------------
    @property
    def alpha(self):
        return self.dvm.alpha

    @property
    def Cl(self):
        return self.dvm.Cl

    @property
    def Cd(self):
        return self.dvm.Cd

    @property
    def Cm(self):
        return self.dvm.Cm


if __name__ == "__main__":
    # tiny smoke test: static plate, check Cl ~ 2*pi*sin(alpha)
    print("DVMFluidSolver smoke test (static plate) ...")
    s = DVMFluidSolver(n_panels=32, dt=0.01)
    for a_deg in (5.0, 10.0):
        a = np.deg2rad(a_deg)
        s.reset_wake()
        beta = 0.5 * np.pi            # xi_hat = (1, 0): chord along +x
        wind = 10.0 * np.array([np.cos(a), np.sin(a)])
        s.update_kinematics(np.zeros(2), beta, np.zeros(2), 0.0)
        s.begin_step(wind)
        for _ in range(600):
            s.solve()
            s.loads()
            s.end_step()
        print(f"  alpha={a_deg:5.1f} deg  Cl={s.Cl:7.4f}  "
              f"(2*pi*sin={2*np.pi*np.sin(a):7.4f})  Cd={s.Cd:8.5f}  Cm={s.Cm:8.5f}")
    print("done.")
