"""
aero_correction.py
==================
Semi-empirical static / dynamic-stall and profile-drag corrections layered on
top of the pure potential-flow DVM in ``fluid_solver.py``.

WHY THIS EXISTS
---------------
The DVM core solves the *attached* potential flow exactly: thin-flat-plate lift
``Cl = 2*pi*sin(alpha)``, the Wagner impulsive-start response, the exact
added-mass impulse and the exact d'Alembert (zero) drag.  Those properties are
what makes the core worth having, and they are verified in ``verify_solver.py``
(F1-F4, F9).  A real aerofoil, however, separates: beyond ``alpha_stall`` the
lift falls away, the drag climbs steeply and the pitching moment turns
nose-down.  These effects dominate a cycloturbine blade over most of a
revolution, and without them the predicted torque is far too high.

Modelling separation *inside* the vortex sheet is a research problem.  Instead
this module keeps the DVM flow solution (so the wake, the unsteady pressures
and every verified attached-flow behaviour are untouched) and corrects the
LOADS:

    F        = kappa * F_circulatory^DVM + F_unsteady^DVM + q*c*dCd*w_hat
    Mz_ref   = kappa * Mz_ref^DVM + q*c^2*Cm_ds
               + moment of the extra drag about the reference point

with

    kappa = Cl_ds(alpha) / (2*pi*sin(alpha))

the ratio of the dynamic-stall lift to the attached potential lift.  Where the
flow is attached ``Cl_ds = 2*pi*sin(alpha)`` exactly, so ``kappa = 1`` and the
corrected loads *are* the verified potential-flow loads (see F11 in
``verify_solver.py``).  Note the correction scales only the circulatory part:
the DVM's unsteady (added-mass) term and the Wagner lag in the circulatory part
are preserved as computed by the DVM.

DYNAMIC-STALL STATE MODEL
-------------------------
Beddoes-Leishman style state-space model advanced with the semi-chord
convective time ``tau = c/(2*W)``:

    ae  : lagged effective angle of attack      (attached-flow lag  T_a)
    f   : lagged separation parameter           (separation lag     T_f)
    vx  : vortex-lift overshoot                 (vortex decay time  T_v)

Static closure at the (lagged) angle of attack:

    Cl_s(ae), Cd_s(ae)      prescribed static polar (see StaticPolar)
    Cl_att = 2*pi*sin(ae)   attached potential lift
    Cl_sep = 2*sin(ae)*cos(ae)               fully separated flat plate
    Cd_sep = 1.98*sin(ae)^2                  separated (pressure) drag
    f_s(ae) = (Cl_s - Cl_sep) / (Cl_att - Cl_sep)      clipped to [0, 1]

The lift is written in *increment* form so that the steady state reproduces the
prescribed static polar exactly:

    Cl_ds = Cl_s(ae) + (f - f_s(ae))*(Cl_att - Cl_sep) + vx
    Cd_ds = Cd_s(ae) + (f_s(ae) - f)*(Cd_sep - Cd_s(ae))
    Cm_ds = Cm_s(ae) + (f_s(ae) - f)*(Cm_sep - Cm_s(ae))

The separated pitching moment is obtained from a centre-of-pressure shift:

    Cn_sep = sin(ae)*(Cd_sep + 2*cos(ae)^2)         (separated normal force)
    Cm_sep = -(cp_sep - 0.25)*Cn_sep                (cp_sep = 0.5 by default)

Vortex lift (the dynamic-stall lift overshoot, built only while alpha grows):

    vx_target = K_v*max(0, f - f_s(ae))*(Cl_att - Cl_sep)   (zero if alpha_dot<0)

STATIC POLAR
------------
``StaticPolar`` wraps an ``(alpha, Cl, Cd, Cm)`` table on a uniform alpha grid.

* ``StaticPolar.viterna(...)`` builds a self-contained polar: the attached
  branch is the thin-flat-plate lift up to ``alpha_stall`` (so the correction
  is exactly transparent there), the post-stall branch is the
  Viterna-Corrigan extension, and the region beyond 90 deg is obtained from the
  physical reflection ``Cl(180-|a|) = -Cl(|a|)``, ``Cd(180-|a|) = Cd(|a|)``.
* ``StaticPolar.from_csv(path)`` reads an ``alpha_deg, Cl, Cd[, Cm]`` table
  (e.g. the Sheldahl-Klimas NACA 0012 data in ``sharp-vawt-lab/polars``).

Everything is SI / radians internally.
"""

from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi
D2R = np.pi / 180.0


def wrap_pi(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(a, dtype=float) + np.pi) % TWO_PI - np.pi


def _clip(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


# ---------------------------------------------------------------------------
# static polar
# ---------------------------------------------------------------------------
class StaticPolar:
    """Cl, Cd, Cm as a function of alpha (radians) on a uniform grid.

    Parameters
    ----------
    alpha, cl, cd : array_like
        Tabulated polar; alpha in radians, monotonically increasing, covering
        at least [-pi, pi] (values outside are clamped to the ends).
    cm : array_like, optional
        Quarter-chord pitching moment (default 0, a symmetric thin section).
    cla : float
        Attached lift-curve slope used by the attached-flow closure of the
        dynamic model (rad^-1).  For the potential-flow DVM this must be 2*pi.
    """

    def __init__(self, alpha, cl, cd, cm=None, cla=TWO_PI, name='polar'):
        self.alpha = np.asarray(alpha, dtype=float)
        self.cl = np.asarray(cl, dtype=float)
        self.cd = np.asarray(cd, dtype=float)
        self.cm = (np.zeros_like(self.cl) if cm is None
                   else np.asarray(cm, dtype=float))
        self.cla = float(cla)
        self.name = str(name)
        self.cl_max = float(np.max(np.abs(self.cl)))
        self.cl_min = float(np.min(self.cl))

    # ------------------------------------------------------------------ eval
    def eval(self, alpha):
        """Linear interpolation; returns (Cl, Cd, Cm). Preserves array shape."""
        a = np.asarray(alpha, dtype=float)
        cl = np.interp(a, self.alpha, self.cl)
        cd = np.interp(a, self.alpha, self.cd)
        cm = np.interp(a, self.alpha, self.cm)
        if a.ndim == 0:
            return float(cl), float(cd), float(cm)
        return cl, cd, cm

    # ------------------------------------------------------------- builders
    @classmethod
    def viterna(cls, alpha_stall=14.0, cla=TWO_PI, cd0=8.0e-3, cd_stall=None,
                cd_max=2.0, cp_sep=0.5, cd_sep_coeff=1.98, n=2881, name=None):
        """Self-contained flat-plate/NACA-0012-like polar.

        The attached branch (|alpha| <= alpha_stall) is exactly
        ``Cl = cla*sin(alpha)`` and ``Cd = cd0``.  The post-stall branch is the
        Viterna-Corrigan extension

            Cl = A1*sin(2a) + A2*cos(a)^2/sin(a)
            Cd = B1*sin(a)^2 + B2*cos(a)

        with ``A1 = cd_max/2``, ``B1 = cd_max``, and A2, B2 fixed by matching
        ``(Cl_stall, Cd_stall)`` at the stall angle.  Beyond 90 deg the polar
        is continued by the reflection
        ``Cl(180-|a|) = -Cl(|a|),  Cd(180-|a|) = Cd(|a|)``.

        Parameters
        ----------
        alpha_stall : float
            Lift-stall angle in degrees.
        cla : float
            Attached lift-curve slope (default 2*pi; keep it at 2*pi for the
            correction to be transparent in attached flow).
        cd0 : float
            Attached profile drag.
        cd_stall : float, optional
            Drag at the stall angle (default ``max(cd0, 0.02)``).
        cd_max : float
            Two-dimensional drag maximum at 90 deg (2.0 for a flat plate).
        cp_sep : float
            Centre of pressure of the separated flow (fraction of chord) used
            for the separated pitching moment.
        cd_sep_coeff : float
            Coefficient in the separated drag ``cd_sep = c*sin(alpha)^2``.
        """
        a = np.linspace(-np.pi, np.pi, int(n))
        ad = np.abs(a) / D2R                       # |alpha| in degrees
        as_ = float(alpha_stall)
        Cds = max(cd0, 2.0e-2) if cd_stall is None else float(cd_stall)

        sa, ca = np.sin(as_ * D2R), np.cos(as_ * D2R)
        Cls = cla * sa
        A1 = 0.5 * cd_max
        B1 = cd_max
        A2 = (Cls - cd_max * sa * ca) * sa / ca**2
        B2 = (Cds - cd_max * sa**2) / ca

        def viterna_branch(deg):
            r = deg * D2R
            return (A1 * np.sin(2.0 * r) + A2 * np.cos(r) ** 2 / np.sin(r),
                    B1 * np.sin(r) ** 2 + B2 * np.cos(r))

        cl = np.empty_like(a)
        cd = np.empty_like(a)

        # --- "lower" branch on |alpha| in [0, 90] (attached + Viterna)
        def low(deg):
            deg = np.asarray(deg, dtype=float)
            cv, dv = viterna_branch(np.maximum(deg, as_ + 1.0e-6))
            m = deg > as_
            return (np.where(m, cv, cla * np.sin(deg * D2R)),
                    np.where(m, dv, cd0))

        lo = ad <= 90.0
        # Evaluate the lower branch at the reduced angle, then apply the exact
        # symmetry of a symmetric section (a flat plate):
        #   |a| <= 90 :  Cl(a) = +sign(a)*Cl_lo(|a|)
        #   |a| >  90 :  Cl(a) = -sign(a)*Cl_lo(180-|a|)
        #   Cd is even and uses Cl_lo's drag at the reduced angle.
        deg_eff = np.where(lo, ad, 180.0 - ad)
        cv, dv = low(deg_eff)
        sgn = np.sign(a)
        cl = np.where(lo, sgn * cv, -sgn * cv)
        cd = dv

        nm = name or f'viterna(as={as_:g}deg, cd0={cd0:g}, cdmax={cd_max:g})'
        # --- separated pitching moment about the quarter chord ------------
        # Fraction of the lift that is still attached (same normal-force
        # reference pair as the dynamic model, so the increment is bounded):
        Cl_att = cla * np.sin(a) * np.cos(a)
        Cl_sep = 2.0 * np.sin(a) * np.cos(a)
        den = Cl_att - Cl_sep
        good = np.abs(den) > 1e-9
        f_s = np.where(good, (cl - Cl_sep) / np.where(good, den, 1.0), 1.0)
        f_s = np.where(np.abs(np.cos(a)) < 1e-6, 0.0, f_s)
        f_s = np.clip(f_s, 0.0, 1.0)
        # separated normal force and the resulting CP-shift moment
        s, cth = np.sin(a), np.cos(a)
        Cn_sep = s * (cd_sep_coeff * s * s + 2.0 * cth * cth)
        cm = -(cp_sep - 0.25) * Cn_sep * (1.0 - f_s)
        return cls(a, cl, cd, cm, cla=cla, name=nm)

    @classmethod
    def from_csv(cls, path, cla=TWO_PI, name=None, interpolate_to=None):
        """Read ``alpha_deg, Cl, Cd[, Cm]`` (``#`` comments allowed)."""
        dat = np.loadtxt(path, delimiter=',', comments='#')
        if dat.ndim == 1:
            dat = dat[None, :]
        ad, cl, cd = dat[:, 0], dat[:, 1], dat[:, 2]
        cm = dat[:, 3] if dat.shape[1] > 3 else np.zeros_like(cl)
        order = np.argsort(ad)
        a = ad[order] * D2R
        cl, cd, cm = cl[order], cd[order], cm[order]
        if interpolate_to:
            grid = np.linspace(-np.pi, np.pi, int(interpolate_to))
            cl = np.interp(grid, a, cl)
            cd = np.interp(grid, a, cd)
            cm = np.interp(grid, a, cm)
            a = grid
        nm = name or str(path).split('/')[-1]
        return cls(a, cl, cd, cm, cla=cla, name=nm)

    @classmethod
    def naca0012(cls, re=1.0e6, root=None, cla=TWO_PI, verbose=True):
        """Load a tabulated NACA 0012 polar (Sheldahl-Klimas data).

        Searches ``root`` and a few standard locations for
        ``polars_naca0012_Re_<re>.csv``.  This is optional: the built-in
        ``viterna`` polar keeps the suite self-contained.
        """
        import glob
        import os
        import sys

        cands = []
        if root:
            cands.append(root)
        # The reference solver ships these tables (see ref_solver.py).  Looked up
        # lazily so this module stays usable on its own.
        try:
            _here = os.path.dirname(os.path.abspath(__file__))
            if _here not in sys.path:
                sys.path.insert(0, _here)
            import ref_solver
            _ref = ref_solver.polar_dir()
        except Exception:                                   # pragma: no cover
            _ref = None
        if _ref:
            cands.append(_ref)
        cands += ['./airfoil_polars/naca0012', './polars/naca0012']
        tag = f'{re:.0e}'.replace('e+0', 'e+').replace('e-0', 'e-')
        for d in cands:
            for pat in (f'*Re_{tag}.csv', f'*Re_{int(re)}.csv',
                        '*Re_*.csv'):
                hits = sorted(glob.glob(os.path.join(d, pat)))
                if hits:
                    if verbose:
                        print(f"StaticPolar.naca0012: using {hits[0]}")
                    return cls.from_csv(hits[0], cla=cla,
                                        name=f'naca0012:{os.path.basename(hits[0])}',
                                        interpolate_to=2881)
        raise FileNotFoundError(
            'no NACA 0012 polar CSV found (searched: '
            + ', '.join(cands) + ')')

    def __repr__(self):
        return (f"<StaticPolar '{self.name}': {len(self.alpha)} pts, "
                f"Clmax={self.cl_max:.3f}, Clmin={self.cl_min:.3f}>")


# ---------------------------------------------------------------------------
# dynamic-stall / drag correction
# ---------------------------------------------------------------------------
class StallModel:
    """Beddoes-Leishman style dynamic-stall + profile-drag load correction.

    The instance is *stateful*: it must be advanced exactly once per time step
    (``begin_step`` then the first ``update`` call); further ``update`` calls
    within the same step reuse the advanced state so that coupling iterations
    do not over-integrate it.

    Parameters
    ----------
    polar : StaticPolar
        Target static polar.
    T_a, T_f, T_v : float
        Attached-flow, separation and vortex time constants in semi-chords
        (state times are ``T * c/(2W)``).  ``T_a`` defaults to **0**: the DVM
        already reproduces the potential-flow (Wagner) attached response, so a
        Beddoes-Leishman attached lag here would apply it twice.
    K_v : float
        Vortex-lift coefficient.
    vortex_lift : bool
        Enable the vortex-lift overshoot term.
    cd_sep_coeff : float
        Coefficient in ``Cd_sep = cd_sep_coeff*sin(ae)^2`` (1.98 = flat plate).
    cp_sep : float
        Centre of pressure of the separated flow as a fraction of the chord
        (0.5 = mid chord), used for the separated pitching moment.
    cm_sep_const : float, optional
        If given, use a constant |Cm_sep| of this value instead of the
        centre-of-pressure model (sign flips with alpha).
    w_floor : float
        Floor on |W| (m/s) in the time scale so the state freezes, rather than
        blowing up, at the zero-relative-wind dead point.
    kappa_max : float
        Safety clamp on the lift scaling factor.
    cl_floor : float
        Lift floor used only to regularise the ``kappa`` ratio as alpha -> 0.
    blend : (float, float)
        Azimuth-angle range (deg, absolute alpha) over which the correction
        blends from scaling the DVM circulatory load to *replacing* it by the
        model load.  The replacement is needed at |alpha| > ~90 deg, where the
        attached Kutta-condition solution is on the leading-edge-separation
        branch and no longer equals ``2*pi*sin(alpha)``.
    blend_state : bool
        If True the replacement weight is additionally driven by the
        *separation progress* ``1 - f``, so the model load takes over as soon as
        the flow is dynamically separated, rather than only inside the fixed
        ``blend`` window.  This suppresses the DVM's leading-edge-branch content
        in deep stall and moves the coupled pitch amplitude toward the BEM's,
        but it is largely redundant with the ``kappa`` scaling (which already
        reduces the lift to the polar value under separation), and it degrades
        the torque and pointwise agreement in the matched TSR = 1 comparison.
        It is therefore **off by default** -- see COMPARISON_REPORT.md section 6.
    vortex_moment : bool
        If True (default) the leading-edge vortex contributes to the pitching
        moment, acting at its (convecting) chordwise position ``xv``:
        ``Cm_vx = -(xv - 0.25)*vx``.  The vortex is born near the leading edge
        (nose-up contribution) and is convected aft at ``conv_frac*|W|``
        (nose-down as it passes the quarter chord), which reproduces the
        characteristic dynamic-stall moment excursion.  ``conv_frac`` and
        ``x_v0`` set the convection speed (in units of |W|) and the birth
        position (fraction of chord from the leading edge); the default
        ``conv_frac = 2/T_v`` makes the chord passage take exactly the vortex
        lift time ``T_v*tau``.
    """

    def __init__(self, polar, T_a=0.0, T_f=3.0, T_v=6.0, K_v=0.7,
                 vortex_lift=True, cd_sep_coeff=1.98, cp_sep=0.5,
                 cm_sep_const=None, w_floor=1.0e-3, kappa_max=25.0,
                 chord=1.0, cl_floor=0.02, blend=(85.0, 95.0),
                 blend_state=False, vortex_moment=True, conv_frac=1.0 / 3.0,
                 x_v0=0.05, name='stall'):
        self.polar = polar
        self.T_a = float(T_a)
        self.T_f = float(T_f)
        self.T_v = float(T_v)
        self.K_v = float(K_v)
        self.vortex_lift = bool(vortex_lift)
        self.vortex_moment = bool(vortex_moment)
        self.blend_state = bool(blend_state)
        self.conv_frac = float(conv_frac)
        self.x_v0 = float(x_v0)
        self.cd_sep_coeff = float(cd_sep_coeff)
        self.cp_sep = float(cp_sep)
        self.cm_sep_const = cm_sep_const
        self.w_floor = float(w_floor)
        self.kappa_max = float(kappa_max)
        self.chord = float(chord)
        self.cl_floor = float(cl_floor)
        self.blend_lo, self.blend_hi = float(blend[0]), float(blend[1])
        self.name = str(name)
        self.reset()

    # ------------------------------------------------------------- lifecycle
    def reset(self):
        """Zero the lag states (call on any restart / wake reset)."""
        self.ae = 0.0                 # lagged effective angle of attack
        self.f = 1.0                  # lagged separation parameter
        self.vx = 0.0                 # vortex-lift state
        self.xv = self.x_v0           # vortex chordwise position (x/c from LE)
        self.vx_birth = 0.0           # |vortex lift| at birth (for its lifetime)
        self.Cm_vx = 0.0              # vortex contribution to Cm
        self.alpha_prev = 0.0
        self.kappa = 1.0
        self.dCd = 0.0
        self.Cm_ext = 0.0
        self.Cl_ds = 0.0
        self.Cd_ds = 0.0
        self.f_s = 1.0
        self.replace_blend = 0.0
        self._stepped = False
        self._last_step = None
        self.n_updates = 0

    def begin_step(self):
        """Force the next ``update`` to advance the states (optional)."""
        self._last_step = None

    # ---------------------------------------------------------------- closure
    def static_states(self, ae):
        """Static closure quantities at a (lagged) angle of attack.

        The separation-progress parameter is defined with the classical
        *normal-force* reference pair (both proportional to sin(a)cos(a), so
        the increment stays bounded through a = 90 deg):

            Cl_att_ref = cla*sin(a)*cos(a)     (attached reference)
            Cl_sep_ref = 2*sin(a)*cos(a)       (fully separated reference)
            f_s = (Cl_s - Cl_sep_ref)/(Cl_att_ref - Cl_sep_ref)
        """
        p = self.polar
        Cl_s, Cd_s, Cm_s = p.eval(ae)
        s, cth = np.sin(ae), np.cos(ae)
        Cl_att = p.cla * s * cth
        Cl_sep = 2.0 * s * cth
        den = Cl_att - Cl_sep
        if abs(den) > 1.0e-9:
            f_s = (Cl_s - Cl_sep) / den
        elif abs(cth) < 1.0e-6:                   # alpha ~ +-90 deg
            f_s = 0.0                             # fully separated
        else:                                     # alpha ~ 0 or 180 deg
            f_s = 1.0
        f_s = _clip(f_s, 0.0, 1.0)
        Cd_sep = self.cd_sep_coeff * s * s
        Cn_sep = s * (Cd_sep + 2.0 * cth * cth)
        if self.cm_sep_const is not None:
            Cm_sep = -np.copysign(abs(self.cm_sep_const), ae)
        else:
            Cm_sep = -(self.cp_sep - 0.25) * Cn_sep
        return Cl_s, Cd_s, Cm_s, Cl_att, Cl_sep, den, f_s, Cd_sep, Cm_sep

    def update(self, alpha, W, dt, step=None):
        """Advance (once per step) and return ``(kappa, dCd, Cm_ext, info)``.

        Parameters
        ----------
        alpha : float
            Geometric angle of attack of the relative wind at the reference
            point (radians), i.e. the same alpha used for the coefficients.
        W : float
            Relative wind speed magnitude (m/s); sets the convective time.
        dt : float
            Time step (s).
        step : int, optional
            Solver step index.  The states are advanced only when it changes,
            so that the (possibly iterated) load evaluations within one step
            do not over-integrate the model.  If omitted the states advance on
            every call.
        """
        self.n_updates += 1
        new_step = (step is None) or (step != self._last_step)
        if new_step:
            self._last_step = step
            p = self.polar
            tau = self.chord / (2.0 * max(abs(W), self.w_floor))
            a = float(wrap_pi(alpha))
            # --- attached-flow lag on alpha.  NOTE the default T_a = 0: the
            #     DVM already carries the *potential* (Wagner) attached lag in
            #     its circulation, so a Beddoes-Leishman attached lag here
            #     would double-count it.
            if self.T_a > 0.0:
                self.ae = float(wrap_pi(self.ae + wrap_pi(a - self.ae)
                                        * min(1.0, dt / (self.T_a * tau))))
            else:
                self.ae = a
            ae = self.ae
            (Cl_s, Cd_s, Cm_s, Cl_att, Cl_sep, den, f_s,
             Cd_sep, Cm_sep) = self.static_states(ae)
            # --- separation lag
            self.f += (f_s - self.f) * min(1.0, dt / (self.T_f * tau))
            # --- vortex lift (builds only while alpha grows)
            if self.vortex_lift:
                adot = (a - self.alpha_prev) / max(dt, 1e-12)
                vt = self.K_v * max(0.0, self.f - f_s) * den
                if adot < 0.0:
                    vt = 0.0
                vt = _clip(vt, -p.cl_max, p.cl_max)
                vx_old = self.vx
                self.vx += (vt - self.vx) * min(1.0, dt / (self.T_v * tau))
                self.vx = _clip(self.vx, -1.5 * p.cl_max, 1.5 * p.cl_max)
                # a new leading-edge vortex is born (the state has just started
                # to grow): remember its strength and put it at the leading edge
                if abs(vx_old) <= 1e-12 < abs(self.vx):
                    self.vx_birth = abs(self.vx)
                    self.xv = self.x_v0
                if self.vortex_moment:
                    # convect aft; a vortex is a coherent structure with a
                    # lifetime, so once it has decayed it is replaced by the
                    # next one (this keeps the moment contribution continuous)
                    self.xv = min(1.0, self.xv
                                  + self.conv_frac * max(abs(W), self.w_floor)
                                  * dt / self.chord)
                    if abs(self.vx) < 0.05 * self.vx_birth:
                        self.xv = self.x_v0
            self.alpha_prev = a
            self.f_s = float(f_s)
            self._stepped = True

        # --- load reduction (recomputed from the frozen states; independent
        #     of the instantaneous alpha except through the kappa ratio)
        a = float(wrap_pi(alpha))
        (Cl_s, Cd_s, Cm_s, Cl_att, Cl_sep, den, f_s,
         Cd_sep, Cm_sep) = self.static_states(self.ae)
        Cl = Cl_s + (self.f - f_s) * den + self.vx
        Cd = Cd_s + (f_s - self.f) * (Cd_sep - Cd_s)
        Cm = Cm_s + (f_s - self.f) * (Cm_sep - Cm_s)
        # leading-edge-vortex pitching moment: the vortex lift acts at its
        # (convecting) chordwise position, so it is nose-up while the vortex is
        # still near the leading edge and nose-down once it passes the quarter
        # chord.  The arm is bounded to +/-0.25c -- the same centre-of-pressure
        # range the static separated model uses (cp in [0, 0.5]) -- so the
        # transient moment stays on the scale of the static separated moment
        # instead of growing with the whole chord.  It vanishes in the steady
        # state (vx -> 0 there), so it does not disturb F12/F13.
        arm = _clip(self.xv - 0.25, -0.25, 0.25) if self.vortex_moment else 0.0
        Cm_vx = -arm * self.vx
        Cm += Cm_vx
        self.Cm_vx = float(Cm_vx)
        # physical bound on the dynamic-stall overshoot
        lim = 1.6 * self.polar.cl_max
        Cl = _clip(Cl, -lim, lim)
        Cd = max(Cd, 0.0)

        Cl_pot = self.polar.cla * np.sin(a)
        if abs(Cl_pot) >= self.cl_floor:
            kappa = Cl / Cl_pot
        else:
            # alpha -> 0: the ratio is ill-conditioned (both vanish); the floor
            # keeps it finite and leaves the force unchanged there because the
            # potential lift itself vanishes proportionally.
            sg = np.copysign(1.0, Cl_pot if Cl_pot != 0.0
                             else (Cl if Cl != 0.0 else 1.0))
            kappa = Cl / (sg * self.cl_floor)
        kappa = _clip(kappa, -self.kappa_max, self.kappa_max)

        # Beyond ~90 deg the attached Kutta-condition solution is on the
        # leading-edge-separation branch (it no longer satisfies
        # Cl = 2*pi*sin(alpha)) and the real flow is massively separated, so
        # blend over to *replacing* the circulatory load by the model load.
        adeg = abs(float(np.rad2deg(a)))
        if adeg <= self.blend_lo:
            wrep = 0.0
        elif adeg >= self.blend_hi:
            wrep = 1.0
        else:
            t = (adeg - self.blend_lo) / (self.blend_hi - self.blend_lo)
            wrep = float(t * t * (3.0 - 2.0 * t))
        if self.blend_state:
            # Take the replacement weight to be at least the separation
            # progress 1 - f.  When the flow is dynamically separated the DVM's
            # attached Kutta-condition solution is no longer the right load and
            # its leading-edge branch produces spurious spikes, so hand over to
            # the bounded model load.  The steady state (f == f_s, and both
            # branches then give exactly the polar load) is unaffected, so this
            # only changes transients and deep stall.
            wrep = max(wrep, 1.0 - _clip(self.f, 0.0, 1.0))

        self.kappa = float(kappa)
        self.dCd = float(Cd)
        self.Cm_ext = float(Cm)
        self.Cl_ds = float(Cl)
        self.Cd_ds = float(Cd)
        self.replace_blend = float(wrep)
        info = {'Cl_ds': float(Cl), 'Cd_ds': float(Cd), 'Cm_ds': float(Cm),
                'f': float(self.f), 'f_s': float(f_s), 'ae': float(self.ae),
                'vx': float(self.vx), 'kappa': float(kappa),
                'xv': float(self.xv), 'Cm_vx': float(Cm_vx),
                'replace_blend': float(wrep)}
        return float(kappa), float(Cd), float(Cm), info


# ---------------------------------------------------------------------------
# self test
# ---------------------------------------------------------------------------
def _selftest():
    print('StaticPolar.viterna self-test')
    p = StaticPolar.viterna()
    print(' ', p)
    for deg in (0, 5, 14, 20, 30, 45, 60, 90, 135, 170, 180, -20, -135):
        cl, cd, cm = p.eval(deg * D2R)
        print(f'   alpha={deg:6.1f} deg  Cl={cl:+.4f}  Cd={cd:.4f}')
    # symmetry checks
    for deg in (20.0, 45.0, 135.0, 170.0):
        c1 = p.eval(deg * D2R)[0]
        c2 = p.eval(-deg * D2R)[0]
        assert abs(c1 + c2) < 1e-12 and abs(p.eval(deg * D2R)[1]
                                            - p.eval(-deg * D2R)[1]) < 1e-12, deg
    print('  odd Cl / even Cd symmetry OK')
    # attached transparency
    a = np.linspace(-13, 13, 27) * D2R
    cl, cd, cm = p.eval(a)
    err = np.max(np.abs(cl - TWO_PI * np.sin(a)))
    print(f'  attached |Cl - 2pi sin(a)| max = {err:.3e}   (should be ~0)')
    cl90, cd90, _ = p.eval(0.5 * np.pi)
    print(f'  Cl(90)={cl90:+.4f} (0), Cd(90)={cd90:.4f} (2), '
          f'Cl(180)={p.eval(np.pi)[0]:+.4f} (0)')

    print('StallModel self-test (steady state must equal the polar)')
    sm = StallModel(p, T_a=0.75, T_f=3.0, T_v=6.0)
    for deg in (5, 14, 20, 30, 60, 89):
        sm.reset()
        for _ in range(4000):
            sm.begin_step()
            sm.update(deg * D2R, 10.0, 2.0e-3)
        cl, cd, cm = p.eval(deg * D2R)
        print(f'   alpha={deg:5.1f}  Cl={sm.Cl_ds:+.4f} (static {cl:+.4f})  '
              f'Cd={sm.Cd_ds:.4f} (static {cd:.4f})  kappa={sm.kappa:+.4f}')

    print('  hysteresis check: ramp up/down through stall')
    sm = StallModel(p)
    sm.reset()
    cl_up = []
    cm_up = []
    n = 800
    for i in range(n):
        deg = -5.0 + 40.0 * i / (n - 1)
        sm.begin_step()
        sm.update(deg * D2R, 10.0, 2.0e-3)
        cl_up.append(sm.Cl_ds)
        cm_up.append(sm.Cm_ext)
    cl_dn = []
    for i in range(n):
        deg = 35.0 - 40.0 * i / (n - 1)
        sm.begin_step()
        sm.update(deg * D2R, 10.0, 2.0e-3)
        cl_dn.append(sm.Cl_ds)
    cl_up, cl_dn, cm_up = np.array(cl_up), np.array(cl_dn), np.array(cm_up)
    print(f'   Clmax(up)={cl_up.max():.3f}  Clmax(down)={cl_dn.max():.3f}  '
          f'overshoot={cl_up.max() - cl_dn.max():+.3f}')
    # the leading-edge vortex must produce a nose-down moment break during the
    # ramp and leave the steady state untouched
    sm0 = StallModel(p, vortex_moment=False)
    sm0.reset()
    cm0 = []
    for i in range(n):
        deg = -5.0 + 40.0 * i / (n - 1)
        sm0.begin_step()
        sm0.update(deg * D2R, 10.0, 2.0e-3)
        cm0.append(sm0.Cm_ext)
    dv = cm_up - np.array(cm0)
    print(f'   LE-vortex moment break: min(Cm_vortex)={dv.min():+.4f} '
          f'(nose-down), max={dv.max():+.4f}')
    print('OK')


if __name__ == '__main__':
    _selftest()
