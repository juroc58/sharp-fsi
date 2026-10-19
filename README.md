# sharpFSI — a self-contained 2-D VLM/DVM + CPPC FSI suite for a cycloturbine blade section

`sharpFSI` is a from-scratch, dependency-light fluid–structure-interaction code for a
**single cycloturbine (VAWT) blade section**:

* **fluid** — an unsteady 2-D potential-flow solver (discrete vortex method, DVM)
  with a free vortex wake, giving the exact thin-flat-plate lift, the Wagner
  impulsive-start response, the exact added-mass impulse and exactly zero
  d'Alembert drag;
* **structure** — a 1-DOF rocking-pitch model with centrifugal pendulum pitch
  control (CPPC) and elastic pitch stops, matching the geometry, mass properties
  and pitch-stop model of the independent BEM code
  [`juroc58/sharp-vawt-lab`](https://github.com/juroc58/sharp-vawt-lab)
  (`vawt_core.py`, referred to below as **`sharp_proj`**, its local directory name);
* **coupling** — a partitioned two-way implicit (fixed-point, under-relaxed)
  scheme, two to three coupling iterations per time step;
* **aerodynamics of a real blade** — an optional semi-empirical static /
  dynamic-stall and separated-drag correction layer that is *exactly transparent*
  in attached flow, so none of the verified potential-flow behaviour is altered;
* **validation** — a 54-check verification suite (F1–F19) and a head-to-head
  comparison against the independent BEM solver `sharp_proj`, using four
  progressively better-matched BEM aerodynamic models.

### Relationship to sharp-vawt-lab

This is the companion validation code to
[`juroc58/sharp-vawt-lab`](https://github.com/juroc58/sharp-vawt-lab), an
independent 2-D BEM solver for the same turbine. The two are deliberately
separate: `sharp-vawt-lab` is the production design/optimisation tool, while this
repository is a from-scratch second solver whose job is to *independently check*
it — different numerics (vortex method vs blade-element/streamtube), different
wake treatment (free vortex wake vs induction factors), same geometry, mass
properties and pitch equation.

`compare_with_bem.py`, the optional tabulated NACA 0012 polar and the F10
cross-code check need `sharp-vawt-lab` cloned alongside (default
`~/Documents/sharp_proj`, override with `SHARP_PROJ`). Everything else —
`verify_solver.py` except F10, `run_validation.py`, `design_study.py` — is
self-contained.

---

## 1. Requirements and how to run

Python 3 with **NumPy, SciPy and Matplotlib only**. There are no
compiled extensions, no OpenFOAM, no Numba and no AVX-dependent wheels — the code
is pure NumPy and vectorised (SciPy supplies the `cdist` distance matrix in the
wake–wake induction kernel), so it runs on an old CPU (developed and tested on an
AMD A8-6600K, 4 threads, 8 GB DDR3). Use `python3` (there is no `python` alias on
the development box).

```bash
python3 verify_solver.py                       # 54-check verification suite (F1-F19)
python3 run_validation.py                      # matched TSR=1 run (canonical config)
python3 run_validation.py 4 0.005 3.0 stall    # 4 revs, dt=5 ms, TSR=3, stall ON
python3 compare_with_bem.py 1.0 4 potential    # 4-rev TSR=1 comparison vs sharp_proj
python3 compare_with_bem.py 1.0 4 stall        # same, with the stall layer on
python3 design_study.py both stall 2.0         # TSR sweep + counterweight map
python3 pitch_moment_check.py 1.0 stall        # pitch-axis moment diagnostic vs BEM
python3 moment_model_check.py                   # separated-moment model comparison (closed form)
```

Command-line arguments are positional and documented in each file's docstring.
Output files are tagged with the TSR and the aero model so runs never overwrite
each other.

**External dependency.** The FSI suite itself is self-contained, but the
head-to-head comparison (`compare_with_bem.py`) and the optional tabulated NACA
0012 polar need the independent BEM code: it is imported from
`~/Documents/sharp_proj` (override with the `SHARP_PROJ` environment variable).
`verify_solver.py` detects its absence and skips the cross-code checks (F10) with
a clear message; everything else runs without it.

---

## 2. File map

| file | role |
| --- | --- |
| `fluid_solver.py` | `DVMFluidSolver` / `VAWTFluidSolver` — vortex lattice (bound panels + free wake), unsteady Kutta condition, Kelvin system, wake management, load integration |
| `structural_solver.py` | `CPPCParams`, `CPPCStructural` — 1-DOF rocking pitch with CPPC counterweight, pivot damping, elastic stops |
| `aero_correction.py` | `StaticPolar` (Viterna–Corrigan, CSV tables, NACA 0012) and `StallModel` (Beddoes–Leishman-style dynamic stall + separated drag/moment) |
| `fsi_main.py` | `FSICoupler` — partitioned two-way coupling, arm-corrected kinematics, history recording |
| `verify_solver.py` | verification suite F1–F19 + figures (`verify_*.png`) |
| `run_validation.py` | production validation run (history + phase portrait + time histories) |
| `compare_with_bem.py` | head-to-head comparison against `sharp_proj` (four BEM aero models) |
| `pitch_moment_check.py` | pitch-axis aerodynamic-moment diagnostic (reconstructs the BEM's moment by inverting the pitch dynamics) |
| `moment_model_check.py` | closed-form comparison of the separated pitching-moment models + load-integration check (`moment_model_check.png`) |
| `design_study.py` | TSR sweep and counterweight map (`design_*.csv`, `design_study_*.png`) |

---

## 3. Physics

### 3.1 Fluid: discrete vortex method

* **Bound vorticity** — `n_panels` cosine-spaced panels along the camber line
  (the aerofoil is a flat plate: thickness is not modelled, only the camber line),
  one lumped vortex `Γ_j` at the ¼ point of each panel, flow tangency enforced at
  the ¾ point (Katz & Plotkin).
* **Unsteady Kutta condition** — the linear system solved each step has `N+1`
  unknowns: the `N` bound vortex strengths and the strength of the single new
  vortex shed during the step. The extra equation is Kelvin's circulation
  theorem, `Σ Γ_j^(n+1) + Γ_shed = Σ Γ_j^(n)`, so the total circulation of
  (bound + wake) is conserved to machine precision (the `Kelvin budget` printed by
  every run is `~1e-13`).
* **Free wake** — the new vortex is released at
  `X_shed = X_TE + f·dt·V_local(X_TE)`, `f = 0.5` (the centroid of the vorticity
  shed during the step), and convects freely with the local velocity
  (freestream + all bound + all wake induction) integrated with RK2. Wake cores
  grow diffusively, `core² += 4ν dt` (ν = 1.5·10⁻⁵ m²/s, air).
* **Loads** — per panel,
  `dF_j = ρ Γ_j (W_j × ẑ)` (Kutta–Joukowski, giving `Cl = 2π sin α`, the leading-edge
  suction implicitly, and the centre of pressure at the quarter chord) plus the
  unsteady apparent-mass term from Bernoulli for the potential jump across the
  sheet, `dF_j += −ρ d/dt[Σ_{k<j} Γ_k] dl_j n̂_j`, which vanishes in the steady
  state and supplies the Wagner initial response.
* **Reported coefficients** — `Cl/Cd/Cm` are the loads normalised by the dynamic
  pressure of the geometric relative wind at the pitch axis, `q = ½ρ|W_geo|²`.
  At the **zero-relative-wind dead point** `q → 0` while the force stays finite,
  so that normalisation diverges (`|Cl|` up to ~10⁴). The reference speed is
  therefore floored at `coeff_W_min` (default **2 m/s**), which keeps the
  coefficients bounded *and continuous* there. The *loads* always use the true
  `q` — the floor is a pure reporting normalisation and leaves every force and
  moment bit-identical (F18). The `coeff_valid` history column flags the samples
  where the floor is active (`|W| < coeff_W_min`); in those samples `Cl` is the
  force per unit `q_ref`, not the true section coefficient.

### 3.2 Wake management (bounded memory)

The wake is thinned to respect the `wake_cap`. The default scheme is
**multiresolution age decimation** (`merge_mode='age'`): the newest `n_per_level`
vortices are kept at full resolution and older ones are merged pairwise into
successively coarser levels, so the wake represents
`n_per_level·(2^wake_levels − 1)` shed vortices for the cost of
`n_per_level·wake_levels`. At the default `wake_cap=600, wake_levels=5` that is
**3720 shed vortices (≈6 revolutions at dt = 5 ms) represented by 600 vortices**.
Every level — including the deepest — is thinned to `n_per_level`, so the wake
population is bounded by `n_per_level·(wake_levels+1)` (≈1.2·`wake_cap`) and the
storage is sized for exactly that worst case (the deepest level used to be
open-ended, which could overrun the array on a long low-wind run).
Merging conserves total circulation *and* the circulation-weighted centroid
exactly, and the core² grows by `sep²/4` to preserve the induced velocity at
distance. `merge_mode='oldest'` restores the simpler scheme (repeatedly merge the
two oldest vortices); it is kept for comparison and is measurably worse in the
near/mid field (F16).

### 3.3 Stall / drag correction layer

Modelling separation *inside* the vortex sheet is a research problem. Instead this
module keeps the DVM flow solution — so the wake and every verified attached-flow
behaviour are untouched — and corrects the loads:

```
F      = κ · F_circulatory^DVM + F_unsteady^DVM + q c dCd ŵ
Mz_ref = κ · Mz_ref^DVM + q c² Cm_ds + (moment of the extra drag about the ref. point)
κ      = Cl_ds(α) / (2π sin α)
```

Where the flow is attached `Cl_ds = 2π sin α` exactly, so `κ = 1` and the corrected
loads *are* the verified potential-flow loads (F11: `max|ΔCl| = 4.4e-16`).

The dynamic stall model is Beddoes–Leishman style, advanced with the semi-chord
convective time `τ = c/(2W)`:

| state | meaning | time constant |
| --- | --- | --- |
| `ae` | lagged effective angle of attack | `T_a` (default **0**: the DVM already carries the Wagner/potential lag, so a separate attached-flow lag would double-count it) |
| `f` | lagged separation parameter | `T_f = 3` semi-chords |
| `vx` | vortex-lift overshoot | `T_v = 6`, strength `K_v = 0.7` |
| `xv` | chordwise position of the leading-edge vortex | convected aft at `2/T_v·W`, i.e. the chord passage takes exactly `T_v·τ` |

`T_f`, `T_v` and `K_v` are the standard Leishman–Beddoes semi-chord values; the
only departure is `T_a = 0` (justified above). Everything else is a modeller's
choice, documented here rather than tuned to any dataset.

Static closure at the lagged angle (the classical normal-force reference pair, so
it stays bounded through 90°):

```
Cl_att = 2π sin(ae)          Cl_sep = 2 sin(ae) cos(ae)      Cd_sep = 1.98 sin²(ae)
f_s(ae) = (Cl_s − Cl_sep)/(Cl_att − Cl_sep)                  clipped to [0,1]
Cl_ds = Cl_s(ae) + (f − f_s)(Cl_att − Cl_sep) + vx
Cd_ds = Cd_s(ae) + (f_s − f)(Cd_sep − Cd_s(ae))
Cm_ds = Cm_s(ae) + (f_s − f)(Cm_sep − Cm_s(ae)) − arm·vx,  arm = clip(xv − 0.25, −0.25, 0.25)
```

The model is written in *increment* form so the steady state reproduces the
prescribed static polar exactly (F12); the vortex term vanishes there (`vx → 0`
in the steady state, F19), so it does not disturb it.

**Leading-edge-vortex pitching moment.** The vortex lift acts at its *convecting*
chordwise position `xv` (born at `x_v0 = 0.05c`, convected aft at `2/T_v·W`, i.e.
crossing the chord in exactly `T_v·τ`), so its contribution `−arm·vx` is weakly
nose-up while the vortex is forward of the pivot and **nose-down** once it passes
it. The arm is bounded to `±0.25c` — the same centre-of-pressure range the static
separated model uses (`cp ∈ [0, 0.5]`) — so the transient moment stays on the
scale of the static separated moment rather than growing with the whole chord.
This is a *correction* to the separated moment: the dominant dynamic-stall moment
signature remains the separation-lag term `(f_s − f)(Cm_sep − Cm_s)`. In a pitch
ramp through stall the vortex term adds a nose-down excursion of ≈0.026 in `Cm`
(≈5 % of the static range) at the lift maximum, with no jump when the next vortex
is born (F19). Physically it stands in for the aft movement of the suction peak;
it is a compact representation, not a resolved vortex.

**Separation-driven deep-stall hand-over (option, off by default).** Beyond
≈85–95° the DVM's attached Kutta branch *is* the leading-edge-separation branch
and is unreliable, so the whole aerodynamic load is replaced by the model load
there (`smoothstep(85°, 95°)`). An optional extension,
`stall_blend_state=True`, additionally drives that weight with the *separation
progress* `1 − f`, so the bounded model load takes over as soon as the flow is
dynamically separated rather than only inside the fixed angle window. It is
**off by default**: it is largely redundant with the `κ` scaling (which already
reduces the lift to the polar value under separation) and it degrades the
matched TSR = 1 torque and pointwise agreement, while moving the pitch amplitude
toward the BEM (see `COMPARISON_REPORT.md` §6). Because *both* branches return
exactly the prescribed polar load in the steady state (`f = f_s`), neither
setting disturbs the steady state — and hence F12. Lift is clamped to
`±1.6·Cl_max`, `Cd_ds ≥ 0`.

`StaticPolar.viterna(...)` builds a self-contained polar: the attached branch is
the thin-flat-plate lift up to `alpha_stall` (so the correction is exactly
transparent there), the post-stall branch is the Viterna–Corrigan extension, and
angles beyond 90° are obtained from the physical reflection
`Cl(180−|α|) = −Cl(|α|)`, `Cd(180−|α|) = Cd(|α|)`. `StaticPolar.from_csv` reads
`alpha_deg, Cl, Cd[, Cm]` tables (e.g. the Sheldahl–Klimas NACA 0012 data), and
`StaticPolar.naca0012(re=…)` searches `sharp_proj/airfoil_polars/naca0012/`
(and the older `sharp_proj/polars/naca0012/` layout).

### 3.4 Structure (CPPC rocking pitch)

Geometry, mass properties and the pitch-stop model are identical to
`sharp_proj`'s `Geometry`/`MassProps`: the pitch axis sits at chord coordinate
`sp·c`, the blade CG at `cg_blade·c`, and the counterweight of mass `m_cw` at
chord coordinate `ξ_cw` measured from the pitch axis. The single DOF is the pitch
`ψ` about the rocking-arm pivot at `Rp = R − ar·c`, with polar inertia

```
I_p = m_b(c²/12 + ξ_b² + ar_len²) + m_cw(ξ_cw² + ar_len²) + I_add ,
ξ_b = (cg_blade − sp)c ,     ar_len = ar·c
```

pivot viscous damping `c_b`, optional Coulomb friction, and **elastic stops**:
beyond `±win` the restoring moment is `k_s(ψ ∓ win)` with `k_s = I_p ω_stop²`
(so the stop is soft and the pitch can overshoot by `Q/k_s`).

The static pitch bias is `ψ_eq = atan2(−ξ_g, ar_len)` with
`ξ_g = (m_b ξ_b + m_cw ξ_cw)/(m_b + m_cw)`. **This single number controls the
performance of the rotor** (§6). The natural pitch frequency is
`ω_n = sqrt(M ω² Rp ar_len / I_p)` and `fn_ratio = ω_n/ω` sets how strong the
pitch response is (`fn_ratio ≈ 2.7–3` for the mass scale used here, i.e. a stiff,
inertia-dominated pitch).

### 3.5 Coupling

Per time step: `begin_step` freezes the release position of the vortex that will
be shed, using the current state (Katz–Plotkin placement); then the fluid and
structure are sub-iterated to convergence (relative tolerance `coupling_tol`,
under-relaxation `relax = 0.7`, up to `n_coupling_iter = 3`), and `end_step`
advances the wake and increments the step counter. The default case converges in
**2.5–2.9 iterations per step**. Because the state is fully overwritten on both
sides, the scheme is stable without sub-stepping for the dt used here.

### 3.6 Kinematics

The aero reference point is carried on the rocking arm, and the pivot is at
`Rp = R − ar·c`; the solver's `x = 0` reference is the quarter chord when
`sp = 0.25`. The kinematics (chord direction, normal, rigid-body velocity field
and the effective angle of attack) reproduce `sharp_proj`'s BEM kernel to
**4·10⁻¹² degrees** over a full azimuth sweep (F10), and the blade reference
radius is exactly `R` at zero pitch.

---

## 4. Verification (F1–F19 — **54/54 checks pass**)

Run `python3 verify_solver.py` (≈8 min single-threaded on the A8-6600K). Figures:
`verify_solver_plots.png`, `verify_static.png`, `verify_wagner.png`,
`verify_convergence.png`, `verify_structural.png`, `verify_stall.png`,
`verify_kinematics.png`, `verify_wake_drag.png`, `verify_deadpoint.png`,
`verify_vortex_moment.png`.

| group | check | result |
| --- | --- | --- |
| **F1** | static lift = `2π sin α` | max‖ΔCl‖ = 0.0173 (0.80 % of max) |
| **F2** | potential (d'Alembert) drag ≈ 0 | max‖Cd‖ = 0.0116, residual `Cd ≈ K·Cl²` with `K ∝ 1/L_wake` (F17: a finite-wake truncation effect, **not** a lattice artefact) |
| **F3** | moment about the ¼ chord ≈ 0 | max‖Cm‖ = 0.0035 |
| **F3** | pivot-shift law (`sp = 0.15 / 0.4`) | −0.0636 vs −0.0651 and +0.0983 vs +0.0977 |
| **F4** | Wagner impulsive start matches Jones | RMS ‖ΔCl‖/Cl_ss = 0.0054 (τ>0.5); `Cl(τ=40) = 0.5447` vs Jones 0.5452 |
| **F4** | added-mass impulse ≈ `ρ(πc²/4)ΔW_n` | 0.9561 vs 0.8385 N·s (ratio 1.14); first-step Cl = +3.12 > 0 (correct direction) |
| **F5** | panel/timestep convergence | max error 1.07 % (< 3 %) |
| **F6** | pitch oscillates about `ψ_eq` | centre −13.274° vs `ψ_eq` −13.392° |
| **F6** | pitch frequency = `sqrt(Mω²Rp·ar·c/I_p)` | 5.526 vs 5.433 rad/s (+1.7 %) |
| **F7** | positive aero moment increases ψ | Δψ = +0.0173° |
| **F7** | elastic stop law `ψ = −(win + Q/k_s)` | −25.55° vs model −25.69° |
| **F8** | coupled run finite, wake bounded | γ-budget −2.1e-14, `nw+merges = steps`, 1 vortex shed per step |
| **F9** | unsteady term vanishes at steady state | ‖ΔCl‖ = 9.1e-04 |
| **F10** | kinematics vs `sharp_proj` BEM kernel | max‖Δα‖ = 4.0e-12° |
| **F11** | stall layer transparent while attached | `κ ≡ 1`, max‖ΔCl‖ = 4.4e-16, max‖ΔCm‖ = 0 |
| **F12** | corrected loads recover the static polar | max‖ΔCl‖ = 0.0038, max‖ΔCd‖ = 0.0070 |
| **F13** | deep-stall limits | `Cl(90°) = 0`, `Cd(90°) = 2`, `Cl(170°) = −1.0911` (= reflected polar), `Cm(90°) = −0.495` nose-down |
| **F14** | dynamic-stall hysteresis | overshoot +0.349, non-degenerate loop, none in attached flow (2e-9) |
| **F15** | coupled FSI run with stall | finite; `κ ∈ [0.36, 0.82]` when `|α|>20°`; max‖Cl‖ = 1.755 = +15 % over static `Cl_max` |
| **F16** | multiresolution wake thinning | circulation/moments conserved to 1e-13/1e-14; level invariants hold; near-blade induced velocity preserved to 1.8e-3 (vs 3.8e-3 for the oldest-pair scheme) |
| **F17** | residual-drag scaling | panel-invariant (spread 2.07 % over N = 16…128) but falls as `1/L_wake` (log-log slope −1.07); `Cd·L_wake` constant to ±5 % ⇒ finite-wake truncation, not discretisation |
| **F18** | dead-point coefficient normalisation | the `coeff_W_min` floor leaves the loads **bit-identical** (`max|ΔF| = 0`, `max|ΔMz| = 0`) while capping the coefficient blow-up exactly at `(W_ref/W)²` — a factor 400 at 0.1 m/s; the floored values stay in the physical range and `coeff_valid` flags the floored samples |
| **F19** | leading-edge-vortex pitching moment | vanishes in the steady state (`Cm_vx = 8e-16`, steady `Cm` still equals the polar to 1e-9); a pitch ramp gives a nose-down break of −0.026 at α = 23° (negligible while attached, 2.7e-12); the vortex convects over the chord and the contribution is continuous (max step jump 3.2e-4) |

---

## 5. Validation against the BEM solver `sharp_proj`

`compare_with_bem.py` runs the *same* rotor and structural data through both codes
(R = 5 m, c = 1 m, N = 1 blade, ar = 0.35, sp = 0.25, m_b = 5 kg with CG at 0.25c,
m_cw = 2.5 kg at ξ_cw = 0.35 m, pitch limits ±25°, stop ω = 50 rad/s, c_b = 0.15,
U = 10 m/s). The BEM is run as a strictly **two-dimensional** model (H = 1 m) so
that forces, moments and torque are per unit span; tip loss, flow-curvature
corrections, dynamic inflow and arm drag are switched **off** in every case
because they are 3-D/empirical and have no counterpart in the 2-D DVM.

Four BEM aerodynamic models are compared:

| case | aero model | purpose |
| --- | --- | --- |
| **A** `quasi` | `Cl = 2π sin α`, `Cd = 0`, dynamic stall OFF | isolates the numerical + structural implementation |
| **B** `unsteady` | A + apparent-mass lift `Cl_nc = (π/2)(c/W) α̇` | also tests the unsteady implementation |
| **D** `viterna` | **the same** `StaticPolar.viterna` table the FSI stall layer uses, plus the BEM's Noll–Ham dynamic stall with the time constants translated to the FSI convention (`ds_Tf = T_f/2`, `ds_Tv = T_v/2`, because the BEM's `τ = c/W` is twice the FSI's `c/2W`), matched `K_v` and separated moment | apples-to-apples **stalled** comparison — isolates the *flow/coupling* difference |
| **C** `real` | NACA 0012 polar tables + full dynamic stall + `a_fix = 0.2` | the BEM's "real aero" reference |

### 5.1 Matched TSR = 1, potential flow (the canonical case)

Last-revolution statistics (`compare_log_tsr1.txt`):

| run | ⟨ψ⟩ (deg) | ⟨α⟩ (deg) | \|α\|max (deg) | stop (%) | ⟨T⟩ (N·m) | Cp |
| --- | --- | --- | --- | --- | --- | --- |
| **FSI (DVM, potential)** | **−21.07** | −19.01 | 168 | 50.4 | **+787.3** | +0.2571 |
| BEM A quasi-steady | −25.42 | −23.65 | 180 | 77.9 | +903.7 | +0.2975 |
| BEM B +apparent mass | −24.16 | −21.85 | 180 | 70.9 | +892.6 | +0.2933 |
| BEM D matched stall | −11.93 | −8.41 | 80 | 42.6 | +90.4 | +0.0282 |
| BEM C real aero | −13.27 | −12.99 | 77 | 34.7 | +61.0 | +0.0186 |

Pointwise, on the common last-revolution grid, versus the FSI:

| comparison | rms Δψ | max\|Δψ\| | rms Δα | ⟨ΔT⟩ |
| --- | --- | --- | --- | --- |
| **FSI vs A (quasi)** | **8.89°** | 24.8° | 8.99° | +116 N·m |
| **FSI vs B (unsteady)** | **6.83°** | 25.2° | 6.95° | +105 N·m |
| FSI vs D (stalled) | 24.08° | 53.3° | 29.4° | −697 N·m |
| FSI vs C (real) | 19.70° | 54.6° | 31.4° | −726 N·m |

**Reading of this table.** Against the two *matched* models the agreement is good
(⟨ψ⟩ within 3–4°, phase-averaged Δψ rms 7–9°, ⟨T⟩ within 13 %, stopping fraction
within 20 %) and the residual is dominated by the missing stall physics: at TSR = 1
the blade spends most of the cycle at large |α| (72 % of steps have |α| > 15°),
which is exactly where the *stalled* BEM models (D, C) predict 8–13× less torque.
The pure potential DVM therefore sits between the potential and the stalled BEM
models — it is correct for what it models, and it is the reason the stall layer
exists.

**Where the matched-potential check stops being informative.** Running the same
matched case at the design TSR (`compare_log_tsr3.txt`) gives the same excellent
*kinematic* agreement — ψ, α and the stopping fraction match to 5.0°, 4.7° rms and
3 points — but the rotor torque does **not**: 734 N·m (FSI, Cp +0.719) vs 1223 N·m
(BEM A, Cp +1.199). The reason is the ±25° pitch stop. The TSR = 1 counterweight
leaves `ψ_eq = −18.4°`, so both codes sit pinned against the stop for ≈2/3 of the
cycle, and *while pinned the rotor torque is a stop reaction that scales with the
aerodynamic moment about the pivot* — exactly the quantity whose convention differs
between the codes (§5.4) — and it is only meaningful once the stall layer is active
and the counterweight is trimmed (§6). In other words the matched-potential
comparison is a numerical and kinematic consistency check (and it is a strong one:
the two independent codes reproduce each other's pitch and incidence histories to a
few degrees with a 1e-13 circulation budget, and the attached-flow loads agree to
0.8 %, F1/F10), not a torque validation at TSR = 3.

### 5.2 Matched TSR = 1 with the stall layer ON

| run | ⟨ψ⟩ (deg) | \|α\|max (deg) | stop (%) | ⟨T⟩ (N·m) | Cp |
| --- | --- | --- | --- | --- | --- |
| **FSI + Viterna/dynamic stall** | **−10.58** | 91.1 | 15.3 | **+102.1** | **+0.0333** |
| BEM D matched stall | −11.93 | 80.2 | 42.6 | +90.4 | +0.0282 |
| BEM C real aero | −13.27 | 77.3 | 34.7 | +61.0 | +0.0186 |

Turning the stall layer on drops the predicted torque by a factor of **7.7**
(+787 → +102 N·m) and lands **within 13 %** of the matched BEM case D
(+90.4 N·m) and within 67 % of the "real aero" case C. This is the single most
important validation result of the suite: the DVM alone is quantitatively useless
in the stalled part of the cycle, and the correction layer fixes it. rms Δψ
against case D falls from 24.1° (potential) to 16.9°.

### 5.3 TSR = 3 (design point) with the stall layer ON

`compare_log_tsr3_stall.txt`:

| run | ⟨ψ⟩ (deg) | \|α\|max (deg) | stop (%) | ⟨T⟩ (N·m) | Cp |
| --- | --- | --- | --- | --- | --- |
| **FSI + stall** | **−16.40** | 35.4 | 10.5 | **−288.3** | **−0.2824** |
| BEM A quasi | −26.70 | 52.0 | 71.0 | +1222.8 | +1.199 (unphysical: no stall) |
| BEM B unsteady | −26.32 | 52.1 | 64.8 | +1243.3 | +1.219 |
| BEM D matched stall | −15.64 | 44.0 | 25.7 | −189.5 | −0.2000 |
| BEM C real aero | −14.29 | 41.4 | 20.5 | **−274.8** | −0.2434 |

Here the FSI agrees with the BEM's *real-aero* torque to **5 %** (−288.3 vs
−274.8 N·m) — but the torque is **negative in both codes**: with the counterweight
tuned for TSR = 1 the pitch bias (ψ_eq = −18.4°) is far too nose-down for TSR = 3,
so the blade brakes. This is a *configuration* conclusion, independently confirmed
by the BEM, and it motivates the counterweight retune in §6.

The remaining systematic difference is the **pitch amplitude**: the FSI swings
±10° where the BEM swings ±17° (fn_ratio = 2.7 is the same in both). The torque
agrees to a few percent while the pitch amplitude does not, so the difference is
in the *pitching moment* about the pivot. `pitch_moment_check.py` diagnoses it
(§5.4 and `COMPARISON_REPORT.md` §6); `moment_model_check.py` then settles it in
closed form. Outcome: the pitching-moment *model* is validated in matched
potential flow (13 % rms), its end-to-end load integration reproduces the
prescribed polar to 7 %, and the residual amplitude difference is a
separated-flow model variant in which the FSI is the more physical of the two
(it reproduces the exact flat-plate limit at α = 90°, where the BEM's variant
loses the moment entirely). No change to the FSI is warranted.

### 5.4 Pitch-axis moment diagnostic

`pitch_moment_check.py` reconstructs the BEM's aerodynamic pitch moment from its
own pitch history by inverting the rocking-pitch equation (validated by
self-inversion on the FSI to 2–3 % of the moment's peak-to-peak). Summary:

| | FSI | BEM A quasi | BEM B unsteady | BEM D matched stall | BEM C real |
| --- | --- | --- | --- | --- | --- |
| TSR = 1, potential (rms, N·m) | **118** | **134** | 132 | 31 | 19 |
| TSR = 1, stalled (rms, N·m) | 12.2 | 134 | 132 | **31** | 19 |
| TSR = 3, stalled (rms, N·m) | 43 | 466 | 465 | **68** | 59 |

(The stalled rows must be read with the model difference below in mind: the BEM's
deep-stall moment is larger because it shifts the *lift*, and that inflates its
rms in the pinned/deep-stall phases. In the matched *potential* row — where the
models are the same — the two codes agree to 13 %.)

* Matched potential flow: the moments agree to **13 %**.
* Stalled, both codes: the moment collapses by 3–4× and the cycle-mean moment
  tracks α with the *same* monotonic shape in both codes (within a factor ≈ 2 per
  α bin) — the stall layer's moment model is qualitatively validated.
* The residual is a separated-flow **model** difference, quantified in closed form
  by `moment_model_check.py`: both codes move the load aft by `0.25c(1−f)`, but the
  FSI moves the **normal force** `Cn_sep` while the BEM moves the **lift** `Cl`.
  The FSI's formulation is reproduced exactly by a force at `cp_sep = 0.5c`
  (mean |ΔCm| = 0.0000 over 0–90°); against the BEM's lift-arm variant the mean
  |ΔCm| is 0.149 (peak 0.495). **At α = 90° the FSI is right**: a flat plate
  normal to the flow has `Cn = 2` at mid-chord ⇒ `Cm(¼c) = −0.5` (FSI: −0.495),
  whereas the BEM's moment vanishes because `Cl → 0`. The BEM's shift should be
  applied to `Cn`, not `Cl`.
* The FSI's end-to-end load integration reproduces its prescribed polar moment to
  a mean |ΔCm| of **0.034 on a peak of 0.470 (7 %)** in a stalled run.
* The amplitude discrepancy is amplified by stop-pinning: the BEM's larger
  deep-stall moment pushes its pitch into the ±25° stop, after which the amplitude
  is set by the stop, not the aerodynamics. No change to the FSI is warranted.

---

## 6. Design study and counterweight retune

`design_study.py` runs short (2-revolution) coupled runs and reports
last-revolution statistics. It first maps the counterweight at the design TSR,
then sweeps TSR with the best counterweight found.

**Counterweight map at TSR = 3** (`design_cw_sweep.csv`, `design_study_cw.png`):

| m_cw (kg) \ ξ_cw (m) | 0.05 | 0.15 | 0.25 | 0.35 | 0.50 |
| --- | --- | --- | --- | --- | --- |
| **0.5** | **+0.309** | +0.289 | +0.260 | +0.226 | +0.180 |
| **1.0** | +0.300 | +0.246 | +0.167 | +0.128 | +0.039 |
| **1.5** | +0.293 | +0.196 | +0.131 | +0.038 | −0.212 |
| **2.5** | +0.276 | +0.142 | +0.007 | −0.290 | −0.712 |
| **3.5** | +0.261 | +0.115 | −0.163 | −0.579 | −0.867 |

(the baseline TSR = 1 counterweight is m_cw = 2.5 kg, ξ_cw = 0.35 m → Cp = −0.290)

**The map is almost one-dimensional: Cp is a function of the pitch bias
`ψ_eq = atan2(−ξ_g, ar_len)` alone.** Every entry with ψ_eq in the range
0 to −5° has Cp ≈ +0.26 to +0.31 regardless of the counterweight mass; every entry
beyond ψ_eq ≈ −10° collapses toward zero and then negative. Physically: the
counterweight's only useful job is to trim the *mean* pitch; its mass sets the
pitch stiffness (I_p), but over this range the pitch response is
inertia-dominated (fn_ratio ≈ 2.5–3.1) so the response hardly changes.

**Recommended retune for TSR = 3:** a *small* counterweight, e.g.
`m_cw = 0.5 kg, ξ_cw = 0.05 m` → ψ_eq = −0.74°, giving **Cp = +0.309** with the
stops never engaged (stop fraction 0 %), versus **Cp = −0.290** for the
TSR = 1 baseline — a swing of 0.6 in Cp. A conservative choice that keeps a
non-degenerate pitch bias and some trim authority is
`m_cw = 1.0 kg, ξ_cw = 0.05 m` (ψ_eq = −1.36°, Cp = +0.300).

**TSR sweep with the retuned counterweight** (`design_tsr_sweep.csv`,
`design_study_tsr.png`):

| TSR | ψ_eq (deg) | ⟨ψ⟩ (deg) | α_rms (deg) | stop (%) | ⟨T⟩ (N·m) | Cp |
| --- | --- | --- | --- | --- | --- | --- |
| 1.00 | −0.74 | +0.08 | 36.9 | 11.6 | +161.6 | +0.053 |
| 1.50 | −0.74 | −1.38 | 21.6 | 1.7 | +176.4 | +0.086 |
| 2.00 | −0.74 | −0.19 | 17.3 | 0.0 | +229.4 | +0.150 |
| 2.50 | −0.74 | +0.59 | 15.0 | 0.0 | +275.5 | +0.225 |
| **3.00** | −0.74 | +1.03 | 13.4 | 0.0 | +315.0 | **+0.309** |
| 3.50 | −0.74 | +1.20 | 12.1 | 0.0 | +351.0 | +0.401 |
| 4.00 | −0.74 | +1.63 | 10.7 | 0.0 | +385.7 | +0.504 |

Two things stand out:

* **The CPPC works as intended.** With a correctly trimmed counterweight the rotor
  is always extractive (Cp > 0 at every TSR) and the pitch control *self-limits*
  the angle of attack: α_rms falls from 37° at TSR = 1 to 11° at TSR = 4 as the
  rotor speeds up, so the blade stays out of deep stall without any prescribed
  pitch law. The stops are never engaged above TSR = 1.5.
* **Cp rises monotonically over the whole sweep** and reaches **+0.504 at
  TSR = 4**. For comparison, a real 3-D VAWT peaks at Cp ≈ 0.35–0.45. This is a
  *two-dimensional, per-unit-span* model with no tip loss, no finite aspect ratio
  and only the small profile/separated drag of the stall layer, so a high Cp is
  expected; the monotonic rise is the signature of the missing 3-D losses
  (which grow with TSR), not a numerical artefact. The design point should
  therefore be read relative to a 3-D rotor as "Cp keeps improving with TSR until
  3-D losses take over".

The counterweight conclusion is robust across both sweeps: **trim the pitch bias
ψ_eq to within ≈ 0 to −4°, and keep the counterweight small**
(m_cw ≈ 0.5–1.0 kg at this blade scale). The TSR = 1 counterweight (2.5 kg at
0.35 m) is badly mistuned for any TSR other than 1.

---

## 7. Performance

Measured single-threaded on the AMD A8-6600K (contended: the box was also running
another job):

| quantity | value |
| --- | --- |
| `solve()` per step (32 panels, ~360 wake vortices) | ~1.6 ms |
| `end_step()` per step (RK2 wake convection + merge) | ~71 ms |
| full coupled step (FSI, dt = 5 ms) | ~14 ms/step at TSR = 3, ~30 ms/step at TSR = 1 (larger wake, more coupling iterations) |
| 4 revolutions at TSR = 3 (838 steps) | 11.8 s |
| 4 revolutions at TSR = 1 (2513 steps) | ~1 min |
| whole verification suite F1–F19 | ~8 min |

The cost is dominated by the O(N_wake²) wake–wake convection, which is *bounded*
by the wake cap; memory is bounded by the same cap. Optimisations used: fully
vectorised `cdist`-style induced-velocity sums, BLAS-backed least squares,
deserialised wake cores, and the age-decimation wake.

---

## 8. Known limitations and caveats

1. **2-D, single section.** One blade section, per-unit-span forces; no blade
   thickness, no 3-D effects, no tip loss, no induction tower/streamtube
   coupling (`induction = 0` in the matched comparison).
2. **Dead-point coefficient blow-up — fixed (F18).** Near the azimuth where
   `|W| → 0` the *coefficients* used to diverge (the forces were always
   bounded). The reference dynamic pressure is now floored at `coeff_W_min`
   (2 m/s), which bounds the coefficients and makes them continuous, leaves the
   loads bit-identical, and can be identified in the output via the
   `coeff_valid` history column. Validity indicators still use `|W| < 2 m/s` for
   the *relative wind*, and in that band `Cl` is read as force per unit `q_ref`
   rather than as a section coefficient.
3. **Residual potential drag is a finite-wake effect, not a lattice artefact**
   (F17). d'Alembert's paradox says the steady drag must be zero, yet the DVM
   returns `Cd ≈ K·Cl²`. Panel refinement leaves it unchanged (2.07 % spread from
   N = 16 to 128 — so it is *not* the lumped-vortex discretisation), but it falls
   as `1/L_wake` (log-log slope −1.07; `Cd·L_wake` constant to ±5 %): the
   truncated wake's free end acts like a starting vortex and induces a small
   chordwise downwash. `K ≈ 0.11·c/L_wake`, so it is ≈ 3e-3 at the production
   wake length and shrinks linearly with `wake_cap`. It is small compared with the
   separated drag the stall layer adds, and it is *not* tuned away.
4. **Deep-stall load replacement.** Beyond ≈85–95° the DVM's attached branch is
   replaced wholesale by the static-model load (optionally extended to the whole
   dynamically separated range by `stall_blend_state=True`, off by default). This
   is a deliberate modelling choice and the main reason the stall layer is *not*
   transparent at very large |α|. Note that the very large `|Cl|` values once
   reported at TSR = 1 were the dead-point normalisation artefact (item 2, now
   fixed), not a force.
5. **Pitch-amplitude discrepancy vs BEM — resolved** (`COMPARISON_REPORT.md` §6):
   torque agrees to a few percent but the FSI pitch swing is ~60–75 % of the BEM's.
   `pitch_moment_check.py` reconstructs the BEM's pitch-axis aerodynamic moment by
   inverting the (identical) rocking-pitch equation from its recorded ψ(t)/ψ̇(t),
   validated by self-inversion on the FSI (2–3 % of peak-to-peak). In *matched
   potential flow* the moments agree to 13 % (118 vs 134 N·m rms at TSR = 1); with
   the stall model both codes' moments collapse by 3–4× and show the same
   monotonic dependence on α. `moment_model_check.py` settles the residual in
   closed form (case D feeds the BEM the FSI's own polar and both codes use the
   identical `_fstat` closure): the FSI's `Cm(¼c) = −(cp_sep−0.25)·Cn_sep·(1−f_s)`
   is *exactly* "place the separated normal force at `cp_sep = 0.5c`"
   (mean |ΔCm| = 0.0000 vs that construction over 0–90°), while the BEM shifts the
   **lift** instead (`−0.25(1−f)·Cl`). The two differ by a mean |ΔCm| of 0.149
   (peak 0.495) with sign-changing discrepancy, and **at α = 90° the FSI is the
   correct one** — a flat plate normal to the flow has `Cn = 2` at mid-chord, i.e.
   `Cm(¼c) = −0.5`, which the FSI reproduces (−0.495) and the BEM loses (Cl → 0
   ⇒ moment → 0). The FSI's end-to-end load integration was separately validated:
   binning the recorded `Cm` of a stalled run against the signed effective α
   reproduces the prescribed polar to a mean |ΔCm| of 0.034 on a peak of 0.470
   (7 %). The amplitude gap is amplified by stop-pinning (the BEM's larger
   deep-stall moment drives its pitch into the ±25° stop, after which the stop, not
   the aerodynamics, sets the amplitude). Conclusion: a model variant in which the
   FSI's formulation reproduces the flat-plate limit and the reference's does not;
   applying the same shift to `Cn` rather than `Cl` would remove the difference.
   No FSI change warranted.
6. **Stall closure is semi-empirical (improved, but not CFD-grade).** The
   dynamic-stall model is a Beddoes–Leishman-style state model: a lagged
   separation parameter, a vortex-lift overshoot and a bounded convecting
   leading-edge-vortex pitching moment, with the separation-lag / vortex time
   constants and `K_v` at their standard Leishman–Beddoes values (`T_a = 0` is
   the one deliberate departure). It reproduces the prescribed static polar
   exactly, gives a non-degenerate hysteresis loop with a large overshoot and a
   nose-down moment break (F14, F19). It remains a *semi-empirical* model: no
   calibration to a specific dynamic-stall dataset, no reduced-frequency-dependent
   constants, no description of the shed-vortex position beyond the single
   convecting state, and no compressibility. Treat dynamic-stall absolute loads as
   indicative rather than predictive.
7. **Non-git directory.** `sharpFSI/` is not a repository; there are no commits or
   branches.

---

## 9. References

* J. Katz & A. Plotkin, *Low-Speed Aerodynamics* (unsteady Kutta condition,
  lumped-vortex method, added-mass term).
* R. T. Jones, "The unsteady lift of a wing of finite aspect ratio" (Wagner
  function approximation used in F4).
* F. J. Viterna & R. D. Corrigan, "Fixed pitch rotor performance of large
  Darrieus turbines" (post-stall polar extension).
* Sheldahl & Klimas, NACA 0012 polars (used by `StaticPolar.naca0012`).
* J. G. Leishman & T. S. Beddoes, dynamic stall (separation lag, vortex lift).
* The independent BEM/CPPC solver in `~/Documents/sharp_proj`
  (`vawt_core.py`), used as the cross-validation reference.
