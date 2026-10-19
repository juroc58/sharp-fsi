# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-19

Initial public release.

### Added

- **Fluid solver** (`fluid_solver.py`): unsteady 2-D discrete vortex method
  (1/4–3/4 lumped vortex lattice, `N+1` Katz–Plotkin system closed by the Kelvin
  condition, RK2 free-wake convection, desingularised cores, diffusive core
  growth). Quasi-steady Katz–Plotkin loads plus a Wagner/impulsive apparent-mass
  term. Eulerian-hill and cycloturbine front-ends (`DVMFluidSolver`,
  `VAWTFluidSolver`).
- **Structure** (`structural_solver.py`): 1-DOF rocking-pitch CPPC model with
  centrifugal pendulum, elastic pitch stops and matching geometry/mass
  conventions (`CPPCParams`, `CPPCStructural`).
- **Coupling** (`fsi_main.py`): partitioned implicit (fixed-point, under-relaxed)
  two-way FSI, 2–3 coupling iterations per step.
- **Aero correction layer** (`aero_correction.py`): `StaticPolar` (self-contained
  Viterna–Corrigan extension with physical reflection beyond 90°, plus
  `from_csv`/`naca0012` loaders) and `StallModel` (Beddoes–Leishman-style lagged
  incidence, lagged separation and vortex-lift states, with a convecting
  leading-edge-vortex pitching moment). Exactly transparent in attached flow.
- **Verification** (`verify_solver.py`): 54-check suite (F1–F19) covering the
  circulation budget, Wagner/impulsive response, added mass, drag floor,
  convergence, structural parameters, pitch-stop model, load integration,
  dynamic-stall behaviour and the two fixes below. Writes `verify_log.txt` and
  `verify_*.png`.
- **Cross-code comparison** (`compare_with_bem.py`): head-to-head against the
  independent BEM solver
  [`juroc58/sharp-vawt-lab`](https://github.com/juroc58/sharp-vawt-lab) using four
  progressively better-matched BEM aero models (quasi-steady potential, unsteady
  potential, Viterna + dynamic stall, real NACA 0012 polars + full dynamic stall).
  See `COMPARISON_REPORT.md`.
- **Validation runs and design study**: `run_validation.py` (single run),
  `design_study.py` (counterweight map + TSR sweep), `pitch_moment_check.py` and
  `moment_model_check.py` (pitching-moment diagnostics).

### Changed

- **Dead-point coefficient blow-up fixed.** At TSR = 1 the reference wind speed
  `|W|` passes through zero twice per revolution, so the reported `Cl`, `Cd` and
  `Cm` divided by an arbitrarily small dynamic pressure and blew up (a
  *normalisation* artefact only — the forces were always bounded). Reported
  coefficients now use a floored reference pressure
  (`q_ref = ½ρ·max(|W|, 2.0 m/s)²`) and a `coeff_valid` flag marks the affected
  samples. Physical loads are bit-identical to the un-floored solver (check F18).
- **Stall closure improved.** Added a convecting leading-edge-vortex pitching
  moment (arm bounded to ±0.25 c) and an optional separation-driven deep-stall
  hand-over (`stall_blend_state`, **off by default** — it moves the pitch
  amplitude towards the BEM but worsens `rms Δψ` and torque, as documented in
  README §8). Constants are standard Leishman–Beddoes values; no calibration to
  any dynamic-stall dataset.

### Fixed

- **Wake-capacity overflow**: the deepest multiresolution merge level was never
  thinned, so a long low-wind run could overrun the wake arrays. All levels are
  now decimated and storage is sized accordingly.
- **Artifact-name collisions**: potential and stalled runs at the same TSR
  silently overwrote each other's history and figure. All outputs are now tagged
  with the aero model.
- **Upstream path drift**: `sharp-vawt-lab` moved its polar tables from
  `polars/naca0012` to `airfoil_polars/naca0012`. The comparison now accepts
  either layout and **fails loudly** if the tables are missing rather than
  silently degrading the "real aero" case to a surrogate.
- **No machine-specific paths**: the location of the reference solver is resolved
  by a single new module, `ref_solver.py` (the `SHARP_VAWT_LAB` environment
  variable, else conventional sibling/child clone locations). Previously three
  separate modules hard-coded a directory in the author's home folder.
