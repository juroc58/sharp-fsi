# sharpFSI — a self-contained 2-D DVM + CPPC fluid–structure-interaction solver for a cycloturbine blade section

`sharpFSI` is a from-scratch, dependency-light fluid–structure-interaction code for a
**single cycloturbine (variable-pitch VAWT) blade section**. It is a cross-check
artefact for the independent BEM solver `sharp-vawt-lab` — not an independent
validation, and not a design tool for real turbines.

## 1. What this code does (first screen)

| Question | Answer |
|---|---|
| **What does it simulate?** | One 2-D blade section of a passive variable-pitch (CPPC) cycloturbine in incompressible flow. The fluid is an unsteady discrete-vortex method (DVM) with a free wake; the structure is a 1-DOF rocking-pitch CPPC model with elastic stops; the two are coupled two-way per time step. |
| **What is verified?** | Internal numerical behaviour: 54 checks (F1–F19) covering static lift `Cl = 2π sin α`, d'Alembert drag, quarter-chord moment, Wagner impulsive start, panel/timestep convergence, structural frequency, pitch-stop law, coupled Kelvin circulation budget (~1e-13), added-mass term, stall-layer transparency in attached flow, static-polar reproduction, dynamic-stall hysteresis, multiresolution wake conservation, dead-point coefficient flooring, and leading-edge-vortex moment. Run `python3 scripts/verify_solver.py`. |
| **What is cross-checked?** | Against the separate BEM/CPPC solver `sharp-vawt-lab` (`juroc58/sharp-vawt-lab`, `vawt_core.py`). The two codes share geometry, mass properties and the pitch equation — so agreement is a **cross-check**, not independent validation. Four progressively better-matched BEM aero models are compared at matched TSR = 1 and design TSR = 3. Run `python3 scripts/compare_with_bem.py`. |
| **What is NOT validated?** | **Nothing.** There is no comparison against independent measured data (wind-tunnel, prototype, flight). Dynamic-stall absolute loads are indicative, not predictive. Do not claim experimental validation. |
| **Main numerical limitations** | 2-D single section only (no blade thickness, no tip loss, no induction, no 3-D losses); residual potential drag is a finite-wake effect `Cd ≈ K·Cl² ∝ 1/L_wake`; deep-stall load replacement beyond ~85–95° is semi-empirical and uncalibrated; coefficients are meaningless at the zero-relative-wind dead point (`|W| → 0`, floored at `coeff_W_min`, flagged by `coeff_valid`). |
| **Production defaults** | `n_panels=32`, `dt=0.005 s`, `wake_cap=600`, `wake_substeps=1`, `n_coupling_iter=3`, `coupling_tol=1e-4`, `relax=0.7`, stall layer ON (Viterna polar, `T_f=3`, `T_v=6`, `K_v=0.7`, vortex moment ON, blend_state OFF), `coeff_W_min=2.0 m/s`. See `scripts/convergence_study.py` for the sensitivity evidence behind these values. |
| **dt sensitivity** | Design-point Cp at TSR ≈ 3 is dt-sensitive: halving dt from 5 ms to 2.5 ms changes Cp by ~5 %. Use `dt = 2.5 ms` for quantitative design-point work; `dt = 5 ms` is adequate away from stall onset. |
| **How to reproduce the main result** | `python3 scripts/compare_with_bem.py 1.0 4 stall` → matched stalled case at TSR = 1 (FSI torque +102.1 N·m vs matched BEM case D +90.4 N·m, within 13 %). Requires `sharp-vawt-lab` cloned alongside (`SHARP_VAWT_LAB` env var or sibling clone). |
| **What must I never claim from this code?** | See the “Do not claim” section below. In short: no experimental validation, no real turbine performance, no BEM validation-by-agreement, no counter-rotating gain predictions, no deep-stall quantitative reliability, no converged result without showing the convergence study. |

## 2. Requirements and how to run

Python 3 with **NumPy, SciPy and Matplotlib only**. No compiled extensions, no OpenFOAM,
no Numba, no AVX-dependent wheels — pure NumPy, vectorised. Developed and tested on an
AMD A8-6600K (4 threads, 8 GB DDR3). Invoke as `python3`.

```bash
python3 scripts/ref_solver.py                          # where the reference solver was found
python3 scripts/verify_solver.py                       # 54-check verification suite (F1-F19), ~8 min
python3 scripts/run_validation.py                      # single matched TSR=1 run (canonical config)
python3 scripts/run_validation.py 4 0.005 3.0 stall    # 4 revs, dt=5 ms, TSR=3, stall ON
python3 scripts/compare_with_bem.py 1.0 4 potential    # head-to-head vs sharp-vawt-lab, TSR=1
python3 scripts/compare_with_bem.py 1.0 4 stall        # same, stall layer ON
python3 scripts/compare_with_bem.py 3.0 4 stall        # design TSR=3, stall layer ON
python3 scripts/convergence_study.py           # full numerical sensitivity study (~30-40 min)
python3 scripts/convergence_study.py --quick   # fast smoke run (~2 min)
python3 scripts/design_study.py both stall 2.0         # TSR sweep + counterweight map
python3 scripts/pitch_moment_check.py 1.0 stall        # pitch-axis moment diagnostic vs BEM
python3 scripts/moment_model_check.py                  # separated-moment model comparison
```

Command-line arguments are positional and documented in each file's docstring.
Output files are tagged with the TSR and aero model so runs never overwrite each other.

## 3. External dependency (cross-check only)

The FSI suite itself is self-contained and runs without anything else. But the
head-to-head comparison (`scripts/compare_with_bem.py`), the optional tabulated NACA 0012
polar, and check F10 need the independent BEM solver
[`sharp-vawt-lab`](https://github.com/juroc58/sharp-vawt-lab) — **not vendored here**.
Clone it and point `SHARP_VAWT_LAB` at the clone root, or clone it as a sibling
directory called `sharp-vawt-lab`; `scripts/ref_solver.py` resolves and prints every path it
tried. `scripts/verify_solver.py` skips F10 gracefully when the reference is absent; everything
else runs without it.

## 4. Verification / cross-check / validation — precise terminology

This project is careful to distinguish three things that are often conflated:

| Term | Meaning in this project | Status |
|---|---|---|
| **Verification** | Internal mathematical/numerical checks against analytical benchmarks (F1–F19) | 54/54 pass |
| **Cross-check** | Comparison against the independent BEM solver `sharp-vawt-lab` (same geometry/mass/pitch — so **not** independent) | Matched potential: kinematics to 4e-12°, torque within 13 %; matched stall: torque within 13 % at TSR = 1, 5 % vs real-aero case at TSR = 3 |
| **Validation** | Comparison against independent **measured** data | **None — not yet available** |

The directory contains no experimental data, no wind-tunnel measurements, and no
prototype results. All performance numbers (Cp, torque, pitch) are outputs of
numerical models, not measurements.

## 5. Do not claim

Do not claim any of the following from this code:

* **Experimental validation.** There is no independent measured data.
* **Real turbine performance.** This is a 2-D, per-unit-span, single-section model
  with no tip loss, no finite aspect ratio, no 3-D losses. High 2-D Cp values
  (e.g. +0.504 at TSR = 4 in the design study) are upper bounds / scenario-model
  outputs, not predictions of a real rotor.
* **BEM validation by agreement.** Agreement between this DVM and `sharp-vawt-lab`
  is a cross-check; both codes share geometry, mass properties and the pitch
  equation, so they are not independent.
* **Counter-rotating gains as predictions.** Any such figure would be a scenario
  model, not a prediction.
* **Deep-stall quantitative reliability.** The dynamic-stall layer is
  semi-empirical, uncalibrated, and unvalidated against dynamic-stall data.
* **A result is converged** without showing the convergence study
  (`scripts/convergence_study.py`).

## 6. File map

| File | Role |
|---|---|
| `fluid_solver.py` | `DVMFluidSolver` / `VAWTFluidSolver` — 2-D discrete vortex method, free wake, Kelvin system, loads |
| `structural_solver.py` | `CPPCParams`, `CPPCStructural` — 1-DOF rocking-pitch CPPC, counterweight, stops |
| `fsi_main.py` | `FSICoupler` — partitioned two-way coupling, history recording |
| `aero_correction.py` | `StaticPolar`, `StallModel` — Viterna polar + Beddoes–Leishman-style stall layer |
| `scripts/ref_solver.py` | Locates the `sharp-vawt-lab` clone (no machine-specific paths) |
| `scripts/verify_solver.py` | Verification suite F1–F19 + figures |
| `scripts/compare_with_bem.py` | Four-case head-to-head cross-check vs `sharp-vawt-lab` |
| `scripts/run_validation.py` | Single coupled run + figures (a run, **not** a validation) |
| `scripts/design_study.py` | TSR sweep + counterweight map (scenario model) |
| `scripts/pitch_moment_check.py` | Pitch-axis moment diagnostic vs BEM |
| `scripts/moment_model_check.py` | Separated-moment model comparison |
| `scripts/convergence_study.py` | Numerical sensitivity study (CSV/JSON + plots) |
| `AUDIT_PLAN.md` | Senior-reviewer audit plan (local, not committed) |

## 7. Performance

Measured single-threaded on the AMD A8-6600K:

| quantity | value |
|---|---|
| coupled step, dt = 5 ms | ~14 ms/step at TSR = 3, ~30 ms/step at TSR = 1 |
| 4 revolutions at TSR = 3 (838 steps) | ~12 s |
| 4 revolutions at TSR = 1 (2513 steps) | ~1 min |
| whole verification suite F1–F19 | ~8 min |

Cost is dominated by the bounded O(N_wake²) wake–wake convection.

## 8. Known limitations and caveats

1. **2-D, single section.** One blade section, per-unit-span forces; no blade
   thickness, no 3-D effects, no tip loss, no induction/streamtube coupling.
2. **Residual potential drag is a finite-wake effect** (F17), panel-invariant but
   falling as `1/L_wake`; not a lattice artefact, not tuned away.
3. **Dead-point coefficient flooring** (F18): coefficients are floored at
   `coeff_W_min = 2.0 m/s` so they stay bounded where `|W| → 0`; loads are
   bit-identical, `coeff_valid` flags the affected samples.
4. **Deep-stall load replacement** beyond ~85–95° is a deliberate modelling choice,
   not CFD-grade; `stall_blend_state=True` is available but off by default.
5. **Pitch-amplitude discrepancy vs BEM** is a separated-flow model difference
   (FSI shifts the normal force `Cn`; BEM shifts the lift `Cl`), amplified by
   stop-pinning; no FSI change warranted (see COMPARISON_REPORT.md §6).
6. **Stall closure is semi-empirical** (Leishman–Beddoes style, uncalibrated);
   treat absolute dynamic-stall loads as indicative, not predictive.

## 9. Version and licensing

* Version 1.0.0 (2026-10-19). See `CHANGELOG.md`.
* MIT License. See `LICENSE`.
* Citation: see `CITATION.cff`.

## 10. References

* J. Katz & A. Plotkin, *Low-Speed Aerodynamics* — unsteady Kutta condition,
  lumped-vortex method, added-mass term.
* R. T. Jones, "The unsteady lift of a wing of finite aspect ratio" — Wagner
  function (F4).
* F. J. Viterna & R. D. Corrigan — post-stall polar extension.
* Sheldahl & Klimas — NACA 0012 polars (`StaticPolar.naca0012`).
* J. G. Leishman & T. S. Beddoes — dynamic stall (separation lag, vortex lift).
* [`juroc58/sharp-vawt-lab`](https://github.com/juroc58/sharp-vawt-lab) — the
  independent BEM/CPPC solver used as the cross-check reference.
