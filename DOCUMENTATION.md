# GPT_tools — Detailed Documentation

`GPT_tools` is a Python toolkit built around **LUME-GPT** (the Python wrapper for the
[General Particle Tracer](http://www.pulsar.nl/gpt/), GPT) and the
[openPMD-beamphysics](https://github.com/ChristopherMayes/openPMD-beamphysics) `ParticleGroup`
object. It provides:

- A higher-level run/evaluation layer over LUME-GPT (auto-phasing, restarts, multithreading,
  merit functions, optimization hooks for [Xopt](https://github.com/ChristopherMayes/Xopt)).
- An extended `ParticleGroup` subclass with many derived beam-physics quantities (slice/core/4D
  emittance, action variables, transverse energy, …).
- Interactive Jupyter (ipywidgets + matplotlib) GUIs for exploring GPT output and Xopt
  optimization fronts.
- A library of beam-/cathode-/photoemission physics models (image-charge barrier, metal &
  semiconductor energy distributions, field-emission tips, THz streaking, inverse Compton
  scattering, boundary-element field solving, aperture-scan emittance reconstruction).

This document describes every module and the function/class signatures within each, including
inputs and outputs. It is organized roughly from the lowest-level utilities up to the GUIs and
specialized physics applications.

> **Conventions used throughout the codebase**
> - Momenta are stored in **eV/c**, positions in **meters**, time in **seconds**, energies in **eV**
>   (the openPMD-beamphysics convention). `ParticleGroup` weights are charges in **Coulombs**.
> - The electron rest energy is hard-coded in several places as `510998.95` eV (or `510998950.0`
>   meV / `5.11e5`). The factor `1010.93912` = `sqrt(2·mₑ·1 eV)` converts √(energy in eV) → eV/c.
>   The factor `586.679206` converts momentum in eV/c → velocity in m/s (non-relativistic).
> - A "screen" is a fixed-z output plane (all particles at the same z, varying t); a "tout" is a
>   fixed-t output (time snapshot). Many functions distinguish the two.
> - Functions taking `**params` generally forward them to `postprocess_screen` and the plotting
>   helpers; the recognized keys are documented under `postprocess.py`.

---

## Package layout

```
GPT_tools/
├── __init__.py                  # empty package marker
├── nicer_units.py               # SI prefix / auto-scaling helpers
├── tools.py                     # plotting + screen-selection + unit utilities
├── ParticleGroupExtension.py    # ParticleGroup subclass with derived quantities
├── postprocess.py               # particle-distribution filtering / clipping
├── GPTExtension.py              # core LUME-GPT run/evaluate/multithread layer
├── gpt_plot.py                  # trend / 1D / 2D distribution / trajectory plots
├── gpt_plot_gui.py              # ipywidgets GUI wrapping gpt_plot
├── front_tools.py               # Xopt population (Pareto front) loading/plotting
├── front_gui.py                 # ipywidgets GUI for exploring Xopt fronts
├── SnappingCursor.py            # matplotlib cursor that snaps to data points
├── cathode_particlegroup.py     # DistGen cathode/core-shield beam generation
├── image_charge.py              # image-charge barrier + metal/semiconductor emission
├── THz_functions.py             # THz streaking "lump element" + analytic models
├── compton.py                   # inverse Compton scattering generator
├── tip_emission.py              # spheroidal field-emitter tip geometry/emission
├── conical_tip_emission.py      # rounded-cone field-emitter tip geometry/emission
├── boundary_element_solver.py   # axisymmetric BEM electrostatic field solver + RK4 tracker
├── aperture_scan.py             # simulated aperture-scan + 4D phase-space/MTE reconstruction
└── emittance_vs_fraction.py     # emittance-vs-fraction and core-emittance curves
```

The `examples/` directory holds runnable references: `example.ipynb`, `example_population.ipynb`,
`gpt.in` (a GPT input deck), `distgen.in.yaml` (a DistGen beam definition), `xopt.in.yaml` (an Xopt
optimization config), and `eindhoven_rf_4mm_center.gdf` (a field map).

---

## `nicer_units.py` — SI prefix scaling

Utilities for choosing human-friendly units when plotting (e.g. turning `2e-10 m` into `200 pm`).

| Function | Inputs | Outputs |
|---|---|---|
| `nicer_scale_prefix(scale, mm_cutoff=0.1)` | `scale`: scalar or array of representative magnitudes; `mm_cutoff`: threshold tuning factor | `(f, prefix)` — `f` is a power-of-1000 scale factor (float), `prefix` is the matching short SI prefix string (e.g. `'p'`, `'n'`, `'u'`, `'k'`). Returns `(1, '')` if all-NaN. |
| `nicer_array(a, mm_cutoff=0.3)` | `a`: scalar or array | `(a/fac, fac, prefix)` — the array rescaled into the chosen unit, the scale factor, and the prefix. |

Module-level dicts: `PREFIX_FACTOR` / `PREFIX` (long names ↔ factors) and
`SHORT_PREFIX_FACTOR` / `SHORT_PREFIX` (short symbols ↔ factors, e.g. `'k'→1e3`). `SHORT_PREFIX`
is the factor→symbol inverse used by `nicer_scale_prefix`.

```python
from GPT_tools.nicer_units import nicer_array, nicer_scale_prefix
import numpy as np

# A beam size of ~2e-4 m is more readable in microns:
scaled, factor, prefix = nicer_array(np.array([2.1e-4, 1.9e-4, 2.3e-4]))
# scaled ≈ [210., 190., 230.], factor == 1e-6, prefix == 'u'
print(f"sigma_x = {scaled[0]:.0f} {prefix}m")     # "sigma_x = 210 um"

f, p = nicer_scale_prefix(2e-10)                  # (1e-12, 'p')  -> picometers
```

---

## `tools.py` — plotting, screen selection, and unit utilities

Shared helpers used by the plotting routines.

**Figure / label helpers**
- `make_default_plot(plot_width=700, plot_height=400, dpi=120, is_table=False, **params)` → `(fig, ax)`.
  Creates a non-interactive matplotlib figure sized in pixels with toolbar/header/footer hidden
  (intended for embedding in ipywidgets).
- `format_label(s, latex=True, use_base=False, add_underscore=True)` → formatted string.
  Turns internal variable names (`norm_emit_x`, `sigma_t`, `kinetic_energy`, `ptrans`, …) into
  LaTeX or unicode axis labels. `use_base` strips `mean_`/`sigma_`/`norm_` prefixes.
- `get_y_label(var)` → a y-axis category string (`'Emittance'`, `'Beam Size'`, `'Energy'`, …) given
  a list of variable names.
- `check_mu(str)` → replaces the `'u'` micro prefix with the LaTeX `$\mu$`.

**Weighted statistics**
- `mean_weights(x, w)` → weighted mean.
- `std_weights(x, w)` → weighted standard deviation.
- `corr_weights(x, y, w)` → weighted covariance ⟨(x−x̄)(y−ȳ)⟩.

**Histogram / data shaping**
- `duplicate_points_for_hist_plot(edges, hist)` → `(edges_plt, hist_plt)` for drawing a histogram as
  a step plot.
- `pad_data_with_zeros(edges_plt, hist_plt, sides=[True,True])` → pads both ends of a step histogram
  with zero bins so the curve closes to baseline.
- `map_hist(x, y, h, bins)` → for each point `(x,y)`, the value of 2D histogram `h` in its bin
  (NaN outside range). Used to color scatter points by local density.

**Screen selection**
- `special_screens(z_input, decimals=6, min_length=10)` → list of indices of "special" screens — the
  irregularly-spaced screens within an otherwise evenly-spaced z list (e.g. user-placed markers).
  Always forces the last screen to be included.
- `get_screen_data(gpt_data, verbose=False, use_extension=True, **params)` →
  `(screen, screen_key, found_screen_value)`. Selects one screen/tout from a LUME-GPT object.
  Recognized `params`: `screen_z`/`screen_t` (nearest screen at that z/t), `tout_z`/`tout_t` (nearest
  tout), or explicit `screen_key`+`screen_value`. Defaults to `screen[0]`. With `use_extension=True`
  the returned screen is wrapped as a `ParticleGroupExtension`.

**Unit scaling**
- `scale_and_get_units(x, x_base_units)` → `(x_scaled, x_unit_str, x_scale)`.
- `scale_mean_and_get_units(x, x_base_units, subtract_mean=True, weights=None)` →
  `(x, x_unit_str, x_scale, mean_x, mean_x_unit_str, mean_x_scale)`. Optionally subtracts the
  (weighted) mean and returns separately-scaled units for the spread and the mean.
- `check_subtract_mean(var)` → bool; True for variables (`x,y,z,t,energy,…`) where plots normally
  subtract the mean.
- `add_row(data, **params)` → appends a row to a `dict(Name=[], Value=[], Units=[])` table.

**2D coloring (used by `gpt_plot_dist2d`)**
- `scatter_color(fig, ax, pmd, x, y, weights=None, color_var='density', bins=100, colormap=jet, is_radial_var=[False,False], zlim=None)` →
  matplotlib colorbar. Draws a scatter plot colored by density or by any particle variable. `color_var`
  may be a tuple `(var, other_ParticleGroup)` to color by a value taken from a different screen
  (matched by particle id).
- `hist2d(fig, ax, pmd, x, y, weights, color_var='density', bins=[100,100], colormap=jet, is_radial_var=[False,False], zlim=None)` →
  colorbar. Same idea as a `pcolormesh` density/mean image; empty bins render as NaN/"bad" color.

```python
from GPT_tools.tools import get_screen_data, scale_and_get_units, std_weights

# Grab the screen nearest z = 58 mm from a LUME-GPT run and report its beam size
screen, key, z = get_screen_data(G, screen_z=0.058)
sx, units, scale = scale_and_get_units(screen.x, screen.units('x').unitSymbol)
print(f"screen at {key}={z:.4f}: sigma_x = {std_weights(sx, screen.weight):.3g} {units}")
```

---

## `ParticleGroupExtension.py` — extended particle group

### class `ParticleGroupExtension(ParticleGroup)`

`__init__(self, input_particle_group=None, data=None)` — wraps an existing `ParticleGroup` (copying
its settable keys, plus `id`) or builds from a `data` dict. Sets `n_slices=50` and `slice_key='t'`
(used by the slice-emittance properties) and registers extra unit entries in
`PARTICLEGROUP_UNITS` for the derived quantities below.

**Derived properties** (all read unless a setter is noted):

| Property | Meaning |
|---|---|
| `transverse_energy` | √(px²+py²+m²) − m, the transverse kinetic energy (eV). |
| `ptrans` (get/set) | √(px²+py²). Setter rescales px,py to a target transverse momentum. |
| `rp` | √(px²+py²)/pz, the transverse divergence angle. |
| `pr_centered` (get/set) | radial momentum about the (weighted) centroid. Setter rebuilds px,py. |
| `r_centered` | radius about the weighted (x,y) centroid. |
| `energy_spread_fraction` | σ_energy / mean kinetic energy. |
| `core_emit_x`, `core_emit_y` | core emittance in x / y (via `core_emit_calc`). |
| `core_emit_4d` | 4D core emittance (via `core_emit_calc_4d`). |
| `sqrt_norm_emit_4d` | √(4D normalized emittance) over planes x,y. |
| `root_norm_emit_6d` | cube-root of the 6D normalized emittance (after drift-to-t). |
| `slice_emit_x`, `slice_emit_y`, `slice_emit_4d` | charge-weighted average slice emittance over `n_slices` slices of `slice_key`. |
| `action_x`, `action_y` | single-particle Courant–Snyder action in x / y from the 2×2 σ-matrix. |
| `crazy_action_x`, `crazy_action_y`, `action_4d` | eigen-emittance (normal-mode) actions from the full 4×4 σ-matrix, with `action_4d = √(Jᵤ·Jᵥ)`. |

**Module helpers**
- `slice_emit(p_list, key)` → charge-weighted average of `key` over a list of slice ParticleGroups
  (slices with < 5 particles are ignored).
- `convert_gpt_data(gpt_data_input)` → deep-copied LUME-GPT object whose every `particles[i]` has
  been converted to `ParticleGroupExtension`.
- `divide_particles(particle_group, nbins=100, key='t')` → `(plist, edges, density_norm)`. Splits a
  group into `nbins` equal slices along `key` (handles radial keys `r`/`r_centered`/`rp` by binning
  in r²). `density_norm` = 1 / bin width for converting sums to densities.
- `core_emit_calc(x, xp, w, show_fit=False)` → scalar core emittance. Fits the central phase-space
  density (in rotated round coordinates) and extrapolates ε at r→0; needs ≥10000 particles.
  `show_fit=True` plots the fit.
- `core_emit_calc_4d(x, xp, y, yp, w, show_fit=False)` → 4D core emittance, removing the x–y angular
  (solenoid) correlation first.

```python
from GPT_tools.ParticleGroupExtension import ParticleGroupExtension, divide_particles

# Wrap any ParticleGroup / GPT screen to unlock the derived quantities
pg = ParticleGroupExtension(input_particle_group=G.screen[-1])
print("4D emittance:", pg.sqrt_norm_emit_4d)
print("transverse energy (eV):", pg.transverse_energy.mean())

# Per-slice emittance along t (uses pg.n_slices / pg.slice_key)
pg.n_slices = 100
print("avg slice emittance x:", pg.slice_emit_x)

# Manually split a beam into 50 longitudinal slices
slices, edges, density_norm = divide_particles(pg, nbins=50, key='t')
```

---

## `postprocess.py` — distribution filtering

The central dispatcher is `postprocess_screen`; the rest are the operations it can apply. All
operations take `make_copy=False` by default and modify in place unless told to copy.

- `postprocess_screen(screen, **params)` → processed screen. Applies, in order, any of:
  `kill_zero_weight` (bool), `include_ids` (list of ids), `take_range` ((var,min,max)),
  `take_slice` ((var,index,n_slices)), `clip_to_charge` (charge), `clip_to_emit` (emittance),
  `cylindrical_copies` (int n), `remove_spinning` (bool), `remove_correlation` ((var1,var2,power)),
  `random_N` (int) or `first_N` (int). Copies the input automatically if any copy-requiring op is
  requested (overridable with `need_copy`).
- `id_of_nearest_N(screen_input, center_particle_id, N, ndim=4)` → array of `N` particle ids nearest
  to a given particle in 2D/4D/6D phase space (distance measured after whitening the σ-matrix to
  identity).
- `random_N(screen, N, random=True, make_copy=False)` → screen reduced to a random (or first) `N`
  surviving particles.
- `include_ids(screen_input, ids, make_copy=False)` → screen with only the given ids kept
  (others zero-weighted then removed).
- `remove_spinning(screen_input, make_copy=False)` → removes the x–py / y–px correlations from
  solenoid rotation.
- `kill_zero_weight(screen_input, make_copy=False)` → drops all zero-weight particles.
- `take_range(screen_input, take_range_var, range_min, range_max, make_copy=False)` → keeps particles
  with `var` in `[min,max]` (mean-subtracted for x,y,z,t).
- `take_slice(screen_input, take_slice_var, slice_index, n_slices, make_copy=False)` → returns one
  slice (the `slice_index`-th of `n_slices`).
- `remove_correlation(screen_input, var1, var2, max_power, make_copy=False)` → subtracts a degree-
  `max_power` polynomial fit of `var2` vs `var1` from `var2`.
- `clip_to_charge(PG_input, clipping_charge, verbose=True, make_copy=False)` → radially clips the
  beam (smallest radius kept) until the enclosed charge equals `clipping_charge`.
- `clip_to_emit(PG_input, clipping_emit, verbose=False, make_copy=False)` → radially clips until the
  4D √-emittance drops to `clipping_emit`.
- `add_cylindrical_copies(screen_input, n_copies, make_copy=False)` → returns a new group with every
  particle replicated `n_copies` times, rotated uniformly about z (weights divided by `n_copies`).
  Useful for visually smoothing cylindrically-symmetric distributions.

```python
from GPT_tools.postprocess import postprocess_screen, clip_to_charge

scr = G.screen[-1]

# One call, several operations: drop zero-weight, keep |t-mean| inside a window,
# radially clip to 10 pC, and replicate for a smooth cylindrical plot.
clean = postprocess_screen(
    scr,
    kill_zero_weight=True,
    take_range=('t', -2e-12, 2e-12),
    clip_to_charge=10e-12,
    cylindrical_copies=200,
)

# Or call an operation directly (make_copy=True leaves the original untouched)
core = clip_to_charge(scr, 5e-12, make_copy=True)
```

---

## `GPTExtension.py` — the LUME-GPT run/evaluate layer

The most important module. It wraps `gpt.GPT`, DistGen beam creation, auto-phasing, restarts,
multithreading, and merit-function evaluation. Common keyword arguments across the run/evaluate
functions:

> `settings` (dict of GPT input-deck variables and special keys), `gpt_input_file`,
> `distgen_input_file` **or** `input_particle_group` (one of the two), `workdir`, `use_tempdir`,
> `gpt_bin='$GPT_BIN'`, `timeout`, `auto_phase`, `verbose`, `gpt_verbose`,
> `asci2gdf_bin='$ASCI2GDF_BIN'`, `kill_msgs`, `load_all_gdf_data`.

### Core run functions

- **`run_gpt_with_settings(...)`** → a LUME-GPT object `G_all`. The primary runner. It:
  1. Optionally builds the initial beam with `get_cathode_particlegroup` (or uses the supplied
     `input_particle_group`); supports specifying `final_n_particle`+`final_charge`+`total_charge` to
     back out the needed initial `n_particle`.
  2. Optionally auto-phases the deck (`auto_phase=True`) using a 10-particle phasing beam and
     `gpt.gpt_phasing.gpt_phasing`.
  3. Runs GPT, optionally a **second restart pass** if `t_restart` or `z_restart` is in `settings`
     (or resumes from `restart_file`).
  4. Applies screen filtering driven by `merit:min`/`merit:z`/`final_charge`/`final_emit` settings
     and inserts the cathode distribution as screen/tout 0.
  Returns the combined run with `.screen`, `.tout`, `.particles`, `.stat(...)`, etc.

  ```python
  from GPT_tools.GPTExtension import run_gpt_with_settings

  settings = {
      'gun_voltage': 350.0,         # any variable in your gpt.in deck
      'n_particle': 10000,
      'total_charge:value': 100, 'total_charge:units': 'pC',
  }
  G = run_gpt_with_settings(
      settings,
      gpt_input_file='examples/gpt.in',
      distgen_input_file='examples/distgen.in.yaml',
      auto_phase=True,             # phase the deck before the real run
      verbose=True,
  )
  z   = G.stat('mean_z', 'screen')
  sx  = G.stat('sigma_x', 'screen')
  emit = G.screen[-1]['norm_emit_x']

  # Restart a second pass from a time-snapshot (e.g. to switch space charge on midway)
  G2 = run_gpt_with_settings({**settings, 't_restart': 5e-12, 'space_charge': 1},
                             gpt_input_file='examples/gpt.in',
                             distgen_input_file='examples/distgen.in.yaml')
  ```

- **`run_gpt_with_particlegroup(...)`** → `G`. Older/simpler single-pass runner that always takes an
  `input_particle_group`; applies `final_charge`/`final_emit`/`final_radius` clipping to the last
  screen. Falls back to `run_gpt` if no particle group is given.

- **`multithread_gpt_with_settings(..., n_threads=None, num_particles_per_run=None, keep_only_last_pass=True, ...)`**
  → a single merged LUME-GPT object. Splits the input beam into chunks, runs them in parallel with a
  `ProcessPoolExecutor`, then stitches all screens/touts back together by particle id (filling lost
  particles with NaN placeholders that are then removed). `num_particles_per_run` must evenly divide
  the total. If `load_all_gdf_data`, also reassembles per-particle field data (`fEx…fBz`).

  ```python
  from GPT_tools.GPTExtension import multithread_gpt_with_settings
  from GPT_tools.cathode_particlegroup import get_cathode_particlegroup

  beam = get_cathode_particlegroup({'n_particle': 40000}, 'examples/distgen.in.yaml')
  G = multithread_gpt_with_settings(
      settings, gpt_input_file='examples/gpt.in',
      input_particle_group=beam,
      n_threads=8, num_particles_per_run=5000,   # 8 chunks of 5000
  )
  ```

### Evaluate functions (Xopt-style merit wrappers)

Each catches exceptions and returns a dict (`{'run_error': True, ...}` on failure):

- `evaluate_run_gpt_with_settings(settings, ..., merit_f=None, ...)` → merit dict. Runs
  `run_gpt_with_settings`, applies `merit_f` (or `default_gpt_merit`), and adds `pre_merit_*` keys
  (computed at the earliest screen) and `merit:min_*` keys (at the `merit:z` screen). Supports a
  `duplicate::<old>` settings convention to copy outputs to new keys.
- `evaluate_run_gpt_with_particlegroup(settings, ...)` → merit dict; builds a cathode (or core-shield)
  beam, runs `run_gpt_with_particlegroup`, and adds `merit:min_*` and `peak_intensity` outputs.
- `evaluate_run_gpt_with_THz(...)` / `evaluate_run_gpt_with_analytic_THz(...)` → merit dicts wrapping
  the two THz runners below.
- `evaluate_gpt_with_stability(...)` / `evaluate_multirun_gpt_with_stability(...)` → merit dict plus
  jitter-statistics outputs (`end_sigma_E_mean`, `end_sigma_t_mean`, combined energy/timing spread,
  slice timing spread). Runs the nominal case, then `stability:n_runs` (default 100) jittered runs
  using `add_jitter_to_settings`, with a fixed RNG seed for reproducibility.

```python
from GPT_tools.GPTExtension import evaluate_run_gpt_with_settings

# Returns a flat dict of merit values (never raises — errors come back as run_error=True).
# This is the shape Xopt's evaluator expects.
out = evaluate_run_gpt_with_settings(
    settings,
    gpt_input_file='examples/gpt.in',
    distgen_input_file='examples/distgen.in.yaml',
)
print(out['end_norm_emit_x'], out['end_sigma_t'], out['run_error'])

# Wire it into Xopt via examples/xopt.in.yaml, whose evaluator.function points here.
```

### THz runners

- `run_gpt_with_THz(...)` → LUME-GPT object. Auto-phases to find the mirror arrival time(s)
  (`settings['t0']`, `t02`) then runs the full deck with the GPT-side THz element. Requires the
  mirror-location settings (`z_mirror`, `z_mirror2`).
- `run_gpt_with_analytic_THz(...)` → LUME-GPT object. Runs up to `z_mirror`, applies the THz kick as
  an **analytic lump element** (`THz_lump_element`, see `THz_functions.py`), then continues. Requires
  `{z_mirror, E0, sig_x, sig_t, phi0, center_frequency, theta_THz, theta_beam, dt}`. Supports a
  second pulse and a `best_E0` auto-amplitude mode.

### Merit & helper functions

- `default_gpt_merit(G)` → dict of scalar beam stats on the **last screen**: `end_<stat>_<var>` for
  stats `mean/sigma/min/max` over all coordinates, momenta, velocities, angles, and energies, plus
  `end_norm_emit_x/y/4d`, `end_sqrt_norm_emit_4d`, `end_n_particle(_loss)`, `end_total_charge`,
  `end_z_screen`, and `error`. On a GPT error it returns the same keys with the sentinel value `1e88`
  and `error=True`.
- `filter_screen(scr, settings)` → bool; applies `final_charge`/`final_emit` clipping in place.
- `split_particle_group(PG, N, P)` → nested list of `P` groups, each a list of N-particle slices
  (used by the multithreader).
- `keep_only_last_forward_pass(PG_input)` → PG keeping, for any repeated id, only the latest-time
  (forward-going) instance.
- `run_one_thread(settings_input, PG_list, ...)` → list of LUME-GPT objects; the per-worker body of
  the multithreader.
- `smallest_factor_geq(P, M)` → smallest divisor of `P` that is ≥ `M`.
- `add_jitter_to_settings(settings_input, verbose=False)` → settings with Gaussian jitter applied to
  every variable named by a `stability:sigma_<name>` or `stability:relative_sigma_<name>` key.
- `add_stability_settings(gpt_data_input, settings_input)` → settings prepared for stability scans:
  pins phasing variables found in the deck, converts `final_charge`/`final_emit` into a fixed
  `final_radius` aperture, and turns space charge off.
- `radius_including_charge(PG_input, clipping_charge)` / `radius_including_emit(PG_input, clipping_emit)`
  → the clipping radius enclosing a given charge / giving a target emittance.
- `get_distgen_beam_for_phasing_from_particlegroup(PG, n_particle=10, verbose=False, output_PG=False)`
  → a small DistGen beam (or `ParticleGroup` if `output_PG`) whose centroid matches `PG`, used for
  auto-phasing.

---

## `gpt_plot.py` — plotting routines

Produce matplotlib figures embedded in ipywidgets `HBox`es (when shown standalone) or drawn into a
supplied `fig_ax`. They accept `**params` forwarded to `postprocess_screen`/`get_screen_data`.

- `make_dataframe_widget(df)` → an ipywidgets `Output` displaying a pandas DataFrame.
- `gpt_plot(gpt_data_input, var1, var2, units=None, fig_ax=None, format_input_data=True, show_survivors_at_z=None, show_survivors_after_z=None, show_screens=True, show_cursor=True, return_data=False, legend=True, **params)`
  → an `HBox` (or, if `return_data`, an `(N, 1+len(var2))` array). **Trend plot**: `var1` (an x-axis
  stat such as `mean_z`) vs one or more `var2` curves across all screens. Auto-chooses units, marks
  "special" screens with dots, can restrict to particles surviving at a given z, and attaches a
  `SnappingCursor`.
- `gpt_plot_dist1d(pmd, var, plot_type='charge', units=None, fig_ax=None, table_fig=None, table_on=True, subtract_mean='auto', **params)`
  → `HBox`/table widget. **1D distribution**: histogram of `var` weighted/normalized per `plot_type`
  (`charge`, `norm_emit_x`, `sigma_x`, …). Builds a side table of mean/σ/charge. `pmd` may be a
  `GPT` object (a screen is selected via `**params`), a `ParticleGroup`, or a `ParticleGroupExtension`.
- `gpt_plot_dist2d(pmd, var1, var2, plot_type='histogram', units=None, fig=None, table_fig=None, table_on=True, plot_width=500, plot_height=400, return_data=False, x_subtract_mean='auto', y_subtract_mean='auto', fig_ax=None, **params)`
  → `HBox` / `(table, colorbar)` / data array. **2D distribution**: scatter or 2D histogram of
  `var1` vs `var2`, optionally colored by a third variable (`color_var`, possibly from an alternate
  screen via a `(var, ParticleGroup)` tuple). Computes σ_x, σ_y, correlation, and (for x–px / y–py)
  the emittance, shown in a side table. Supports `axis='equal'`, `centered_at_zero`, `xlim/ylim`,
  `zlim/clim`, custom `units`, and `nbins`.
- `gpt_plot_trajectory(gpt_data_input, var1, var2, fig_ax=None, format_input_data=True, nlines=None, show_survivors_at_z=None, **params)`
  → `HBox`. Plots per-particle trajectories (`var1` vs `var2`) across screens, tracking particles by
  id; `nlines` randomly subsamples how many trajectories to draw.

```python
from GPT_tools.gpt_plot import gpt_plot, gpt_plot_dist1d, gpt_plot_dist2d

gpt_plot(G, 'mean_z', ['sigma_x', 'sigma_y'])          # beam-size trend vs z
gpt_plot(G, 'mean_z', ['norm_emit_x'], units='nm')     # emittance with forced units

gpt_plot_dist1d(G, 't', plot_type='charge', screen_z=0.058)         # current profile
gpt_plot_dist2d(G, 'x', 'px', screen_z=0.058, axis='equal')         # x–px phase space
gpt_plot_dist2d(G, 'x', 'y', color_var='kinetic_energy', screen_z=0.058)

# Pull the plotted curves out instead of drawing
xy = gpt_plot(G, 'mean_z', ['sigma_x'], return_data=True)           # (N, 2) array
```

---

## `gpt_plot_gui.py` — interactive GPT-output GUI

- `gpt_plot_gui(gpt_data_input)` → builds and `display()`s an ipywidgets GUI (no return value). A tabbed
  control panel (Trends / 1D Dist. / 2D Dist. / Postproc. / Slicing) drives `gpt_plot`,
  `gpt_plot_dist1d`, and `gpt_plot_dist2d` live as widgets change. Supports screen selection
  (all/special), survivor filtering, alternate-screen color/Y sources, and all `postprocess_screen`
  operations through checkboxes.
- `get_dist_plot_type(dist_y)` → maps a 1D-distribution menu label (`'Emittance X'`, `'Sigma E'`, …)
  to the internal variable name (`norm_emit_x`, `sigma_energy`, …).
- `get_trend_vars(trend_y)` → maps a trend menu label (`'Beam Size'`, `'Emittance (4D)'`, `'MTE'`, …)
  to the internal stat variable(s).

```python
# In a Jupyter notebook cell (needs `%matplotlib widget`):
from GPT_tools.gpt_plot_gui import gpt_plot_gui
gpt_plot_gui(G)     # builds and displays the explorer; nothing is returned
```

---

## `front_tools.py` — Xopt population (Pareto front) utilities

Tools for loading and visualizing CNSGA/Xopt optimizer output populations stored as CSV/JSON.

- `show_fronts(pop_number, obj1_key, obj2_key, obj3_key=None, pop_path='.', xopt_file=None, xscale/yscale/zscale=1.0, xlim/ylim/zlim=None, xlabel/ylabel/zlabel=None, show_constraint_violators=False, legend_color=None, dot_style='o', new_fig=True, dpi=100, best_N=None, legend='on', colorbar=True, return_fig=False, return_data=False, zorder=1, show_settings=False)`
  → a figure canvas (or data array if `return_data`). Scatter-plots objective `obj1` vs `obj2`
  (optionally colored by `obj3`) for a chosen population generation. With `show_settings=True`,
  clicking a point displays that individual's input settings in a text box.
- `find_settings(pop_number, obj1_target, pop_path='tmp', xopt_file=None, obj1_key='end_sigma_t', obj2_key='end_norm_emit_x', obj3_key=None, zlim=None, n_neighbors=10, minimize=True, xscale/yscale/zscale=1.0)`
  → prints the input settings of the individual whose `obj1` is nearest `obj1_target` and whose
  `obj2` is best among the `n_neighbors` nearest.
- `get_pop(pop_path, pop_number, xopt_file=None, show_constraint_violators=False, best_N=None)` →
  `(pop_DataFrame, pop_number, pop_filename)`. Loads the population file nearest the requested
  generation number, filters infeasible rows, and optionally down-selects to `best_N`.
- `pop_sampler(data, xopt_file, new_pop_size)` → a DataFrame down-selected to `new_pop_size` using the
  CNSGA selection operator.
- `clamp_population(pop_number, pop_path, xopt_file)` → writes a `_clamped.csv` resized to the xopt
  config's population size (most of the bound-clamping body is commented out).
- `color_all_settings(pop_number, pop_path, xopt_file=None, obj1_key, obj2_key, ...)` → a `VBox` of
  `show_fronts` plots, one per varying variable/constraint, each used as the color axis.
- `get_xopt_file(pop_path)` → path to the `xopt.in.yaml` associated with a population directory.
- `get_filename_list(pop_path)` → `(sorted_csv_filenames, index_array)`.
- `save_json_to_csv(pop_filename)` → converts an Xopt JSON population to CSV; returns the CSV path.
- `fix_xopt_pop_datafile(filename)` → overwrites error rows in a population CSV with a good row.
- `make_settings_csv(csv_filename, settings)` → writes a settings dict to an `xopt_index`-indexed CSV.
- `get_only_feasible_results(pop_df, xopt_yaml='xopt.yaml')` → DataFrame with errored and
  constraint-violating rows removed.
- `get_ind_settings_dict_from_pop_dataframe(pop_element, X)` → merged variables+constants settings
  dict for one individual.
- `reevaluate_population(xopt_file, pop_num=-1, pop_path=None)` → re-runs every individual of a
  population through `X.evaluate` in parallel and writes a `_reevaluated.csv`.
- `run_xopt_func(settings)` / `replace_pop_df_evaluation_output(...)` → helpers for the reevaluation.

> Note: `run_xopt_func` and a few helpers reference a module-global `X` (an `Xopt` instance) that the
> caller is expected to have set up for multiprocessing.

```python
from GPT_tools.front_tools import show_fronts, get_pop, find_settings

# Scatter emittance vs bunch length for generation 50, colored by energy spread
show_fronts(50, 'end_sigma_t', 'end_norm_emit_x', obj3_key='end_sigma_energy',
            pop_path='tmp/', show_settings=True)

# Load a population generation as a DataFrame (feasible rows only)
pop, gen, fname = get_pop('tmp/', pop_number=50)

# Print the settings of the lowest-emittance individual near sigma_t = 1 ps
find_settings(50, obj1_target=1e-12, pop_path='tmp/',
              obj1_key='end_sigma_t', obj2_key='end_norm_emit_x')
```

---

## `front_gui.py` — interactive Xopt-front GUI

### class `front_gui`

`__init__(self, xopt_file, pop_directory)` — builds a full ipywidgets dashboard for browsing
optimizer populations stored in `pop_directory` and configured by `xopt_file`. Features:

- Multi-file selection (wildcard + most-recent ordering), per-file color and legend.
- X/Y/color variable dropdowns (populated from the xopt vocs and CSV columns), with per-axis scale,
  units, label, and limit fields.
- Optional color fading across files, constraint-violator toggling, and "best N" CNSGA down-selection
  with save-to-CSV (`save_best_of_SIRS`).
- Rational-polynomial curve fitting overlay (`rat_poly` via `scipy.optimize.curve_fit`).
- Click-to-inspect: clicking a point shows that individual's settings, and a **Run Settings** button
  re-runs GPT for that point via `run_gpt_with_settings` (optionally archiving the result to an `.h5`).

Key methods: `make_gui`, `make_plot`, `load_files`, `pop_sampler`, `on_click`, `run_gpt`,
`save_best_of_SIRS`, plus numerous widget callbacks (`*_on_value_change`, `active_*_change`,
`show_settings`, `edit_settings_to_run`). The GPT object from the last run is stored on
`self.gpt_data`.

```python
# In a Jupyter notebook (needs `%matplotlib widget`):
from GPT_tools.front_gui import front_gui

gui = front_gui('examples/xopt.in.yaml', 'tmp/')   # dashboard is displayed on construction
# ...click a Pareto point, press "Run Settings", then read back the run:
G = gui.gpt_data
```

---

## `SnappingCursor.py`

### class `SnappingCursor`

`__init__(self, fig, ax, line_list)` — a matplotlib interaction helper that, on mouse-move, snaps a
marker (rectangle) and a text label to the nearest data point among the given lines, scaling
distance by screen aspect so "nearest" is visual. `on_mouse_move(event)` is the connected callback;
`set_visible(visible)` toggles the marker/text. After a snap, `self.pos`, `self.data_index`, and
`self.which_line` hold the selected point — used by the GUIs to identify which population individual
was picked.

---

## `cathode_particlegroup.py` — DistGen beam generation

- `get_cathode_particlegroup(settings_input, DISTGEN_INPUT_FILE, verbose=False, distgen_verbose=False, id_start=1)`
  → `ParticleGroup`. Loads a DistGen YAML, overrides its nested keys with `settings`
  (colon-delimited paths), runs DistGen, and assigns sequential ids from `id_start`. Supports a
  `cathode:sigma_xy:value`/`:units` override that rescales the radial distribution (recursively) to
  hit a target spot size.
- `get_coreshield_particlegroup(settings_input, DISTGEN_INPUT_FILE, verbose=False, distgen_verbose=False)`
  → `ParticleGroup`. Builds a two-population "core + shield" beam: the central `core_charge` fraction
  is sampled with `n_core` macroparticles and the outer ring with `n_shield`, so the core is resolved
  with more particles. Requires a `coreshield` block (`n_core`, `n_shield`, `core_charge`) in the
  settings/YAML; defaults the core to the final/half charge if unspecified.

```python
from GPT_tools.cathode_particlegroup import get_cathode_particlegroup

# Start from a DistGen deck, override nested keys, force a 0.4 mm rms spot
beam = get_cathode_particlegroup(
    {
        'n_particle': 20000,
        'total_charge:value': 50, 'total_charge:units': 'pC',
        'cathode:sigma_xy:value': 0.4, 'cathode:sigma_xy:units': 'mm',
    },
    'examples/distgen.in.yaml',
    verbose=True,
)
print(beam.n_particle, beam['sigma_x'])
```

---

## `image_charge.py` — image-charge barrier & photoemission energy models

Models photoemission through the Schottky-lowered image-charge barrier (with an optional Plummer
softening radius), for both metals (constant density of states) and semiconductors (parabolic
bands). The three top-level "make" functions overwrite the momentum distribution of a DistGen beam.

**Main generators**
- `MakeMetalParticleGroup(settings, DISTGEN_INPUT_FILE=None, verbose=True, only_survivors=False)` →
  `ParticleGroup`. Metal-cathode emission. Reads `start:MTE`, `kT`, `gun_field`, and either `QE` or
  `cathode_z_offset` (or `work_function`+`photon_energy`); computes the excess energy at the barrier
  peak and surface, then resamples momenta from the metal model. `only_survivors=True` keeps only
  electrons that clear the barrier and adds the barrier pz. Writes back SI `gun_field`,
  `cathode_z_offset`, and `QE` into `settings`.
- `MakeSemiconductorParticleGroup(settings, DISTGEN_INPUT_FILE=None, verbose=True)` → `ParticleGroup`.
  Same idea for a parabolic-band semiconductor; needs `electron_affinity`, `energy_gap`,
  `photon_energy`, `cathode_z_offset`, `gun_field`.
- `MakeEnergyOffsetParticleGroup(settings, DISTGEN_INPUT_FILE=None, verbose=True)` → `ParticleGroup`.
  Adds just enough pz to clear the barrier (single-electron weights).

**Barrier / potential**
- `ImagePotential(z, z0, r0, Egun)` → barrier potential (eV) of a constant field plus a (Plummer-
  softened) image charge.
- `PeakPotentialz(E0, z0, r0)` → z of the barrier maximum (Schottky saddle point).

**Metal model (constant DoS)**
- `getMetalEexc(settings, modify_settings=True, verbose=True)` → `(EexcAtSurface, EexcAtPeak, kT)`.
- `MakeMetalEnergyDist(pg, EexcAtSurface, kT)` → pg with metal-model momenta.
- `MTE_model(Eexcz, kT)` / `inv_MTE_model(MTE, kT, ...)` → MTE from excess energy and its Newton
  inverse. `dE_MTE_model` is the derivative.
- `QE_model(Eexcz, kT, Eexc0)` / `inv_QE_model(QE, Eexcz, kT, ...)` → QE and its Newton inverse.
  `dE_QE_model` is the derivative.
- `invEcumul(p, Eexc, kT, ...)` / `Ecumulprob(Ekin, Eexc, kT)` / `dEcumulprob(...)` → the kinetic-
  energy CDF, its Newton inverse (for sampling), and its derivative.

**Semiconductor model (parabolic bands)**
- `getSemiconductorEexc(settings, modify_settings=True, verbose=True)` →
  `(EexcAtSurface, EexcAtPeak, Ea+V0)`.
- `MakeSemiconductorEnergyDist(pg, EexcAtSurface, EaSurf)` → pg with semiconductor-model momenta.
- `EcumulprobSemi(Ekin, Eexc, Ea)` / `dEcumulprobSemi(...)` / `invEcumulSemi(p, Eexc, Ea, ...)` →
  the semiconductor kinetic-energy CDF, PDF, and a safeguarded (bracketed Newton/bisection) inverse.
  `_semi_support`/`_semi_antideriv` are internal helpers.

**Generic helpers**
- `getValueFromSettings(settings, value, desired_units, modify_settings=True, verbose=True)` →
  pulls a `value`/`units` pair from settings, converts to SI, and optionally writes the SI scalar
  back under `settings[value]`.
- `uniform_pr2_dist(n)` → `(pr, pz)` sampled uniformly in pr² (isotropic-into-half-space directions).
- `get_blank_particlegroup(n_particle, verbose=False)` → an uninitialized N-particle `ParticleGroup`
  with sequential ids (used as a placeholder; more reliable than DistGen for tiny N).
- `float_polylog(n, x)` / `np_polylog` → a NumPy-ufunc wrapper around `mpmath.polylog`.

```python
from GPT_tools.image_charge import MakeMetalParticleGroup

# Overwrite a DistGen beam's momenta with a metal photoemission energy distribution.
# settings is mutated in place to add SI 'gun_field', 'cathode_z_offset', and 'QE'.
settings = {
    'n_particle': 50000,
    'start:MTE:value': 150, 'start:MTE:units': 'meV',
    'kT:value': 25, 'kT:units': 'meV',
    'gun_field:value': 5, 'gun_field:units': 'MV/m',
    'QE': 1e-3,
    'plummer_radius:value': 0, 'plummer_radius:units': 'm',
}
pg = MakeMetalParticleGroup(settings, DISTGEN_INPUT_FILE='examples/distgen.in.yaml')
print('predicted QE:', settings['QE'], 'MTE (eV):', pg.transverse_energy.mean())
```

---

## `THz_functions.py` — THz streaking models

Analytic models for a single-cycle THz pulse reflecting off a mirror and streaking the beam, applied
as a localized "lump element."

- `THz_lump_element(scr, E0, phi0, sigx, sigt, f, tht, thb, dt)` → new screen with px,pz kicked by the
  analytic pulse. `E0` field amplitude, `phi0` CEP phase (rad), `sigx`/`sigt` spatial/temporal pulse
  size, `f` center frequency, `tht`/`thb` THz/beam incidence angles (rad), `dt` timing offset.
- `dpz(...)` / `dpx(...)` → analytic Δpz / Δpx (eV/c) imparted by the Gaussian pulse (uses
  `scipy.special.dawsn`).
- `dpzParabola(...)` / `dpzQuad(...)` → parabolic / quartic Taylor approximations of Δpz, used for
  fast initial guesses.
- `get_analytic_scr(settings, scr, guess=None, force_dt=None)` → screen after one or two full analytic
  pulses. `get_analytic_scr_para(settings, scr, guess=None, do_quad=False)` → same using the
  parabolic/quartic approximation (and a possible third pulse).
- `make_parabolic_guess(n_mirrors, settings, scr, use_ideal_size=True)` → settings dict with E0, sig_x,
  dt (and second/third pulse) guesses derived from a paraboloid fit of the screen's energy surface.
- `best_fit_paraboloid(scr, var='kinetic_energy')` → `[offset, t0, a_r, a_t]` least-squares paraboloid
  fit of `var` vs (r, t). `make_fit_paraboloid(scr, x)` evaluates such a fit. `subtract_paraboloid(scr, x)`
  subtracts a (r², t²) paraboloid from pz.
- `make_collimated_beam(scr, sig_x)` → screen transported (via a computed transfer matrix) to beam
  size `sig_x` without changing emittance (re-imposes energy conservation on pz).
- `get_THz_pulse(t, settings)` → the scalar Gaussian-modulated cosine pulse shape.
- `get_pulse_energy(E0, phi0, sigx, sigt, f)` → pulse energy (J) for the given field amplitude.
- `get_beta(EnKV)` → relativistic β for a kinetic energy in keV.
- `guess_to_settings(x, settings)` / `settings_to_guess(s, n_mirror)` → convert between an optimizer
  parameter vector and a settings dict (1–3 mirrors).
- `get_gpt_scr(settings, guess=None)` → runs GPT for a THz settings guess and returns the screen at
  `z_screen_1` (depends on module-global `GPT_INPUT_FILE`/`DISTGEN_INPUT_FILE`/`run_gpt_with_THz`).
- `get_cam_dist()` → a synthetic 4000-particle test `ParticleGroup` (fixed seed).

```python
import numpy as np
from GPT_tools.THz_functions import THz_lump_element

scr = G.screen[-1]
# Apply a single-cycle THz kick analytically (angles in radians)
streaked = THz_lump_element(scr, E0=1e6, phi0=0.0, sigx=1e-3, sigt=0.3e-12,
                            f=0.3e12, tht=np.radians(45), thb=0.0, dt=0.0)
print('induced energy spread (eV):', streaked['sigma_energy'])
```

---

## `compton.py` — inverse Compton scattering

- `inverse_compton_scatter(PG_e, n_macro_per_electron=5, max_theta=5, laser_wavelength_in_nm=1030, laser_sigma_in_micron=30, laser_energy_in_mJ=1)`
  → a `ParticleGroup` of scattered **photons**. For each electron in `PG_e`, emits
  `n_macro_per_electron` photon macroparticles sampled (Halton quasirandom) over a cone of half-angle
  `max_theta/γ`, with energies from the head-on Compton formula and weights from the
  Klein–Nishina-like differential cross-section times the local Gaussian laser photon density.
  `max_theta` is in units of 1/γ. Returns the summed photon ParticleGroup.

```python
from GPT_tools.compton import inverse_compton_scatter

photons = inverse_compton_scatter(G.screen[-1], n_macro_per_electron=10,
                                  laser_wavelength_in_nm=1030,
                                  laser_sigma_in_micron=30, laser_energy_in_mJ=5)
print('photon energies (eV):', photons.energy.min(), photons.energy.max())
```

---

## `tip_emission.py` — spheroidal field-emitter tip

Geometry and photoemission mapping for a prolate-spheroid emitter tip of height `H` and base
diameter `B`.

- `make_tip_distribution(PG_laser, H, B, theta, z0)` → `ParticleGroup`. Maps a flat-cathode laser
  distribution onto the tip surface: ray-traces each laser particle (incidence angle `theta`, aimed at
  `(0,0,z0)`) to its first intersection with the tip, sets the surface position and arrival-time
  offset, and rotates each particle's momentum into the local surface-normal frame. Lost particles
  (missing the tip) are dropped.
- `make_tip_distribution_lumerical(PG_laser, H, B)` → like the above but for a Lumerical-style (x,y)
  input distribution at normal incidence (no ray tracing).
- `FieldEnhancement(r, H, B)` → the local field-enhancement factor across the tip (prolate-spheroidal
  solution), scalar or array.
- `tip_height(r, H, B)` → surface height z(r). `tip_normal(r, H, B)` → `(nr, nz)` outward unit normal.
- `FirstIntersection(x0, y0, z0, theta, H, B)` → x of the first ray–tip intersection (NaN if none).
- `SpheroidIntersection(...)` / `PlaneIntersection(...)` → ray intersections with the spheroid cap and
  the surrounding flat plane.
- `rotation_from_z_to_n(n)` → 3×3 rotation matrix sending ẑ to unit vector `n` (Rodrigues form).

---

## `conical_tip_emission.py` — rounded-cone field-emitter tip

Same interface as `tip_emission.py` but for a cone of base diameter `B` and height `H` capped by a
sphere of radius `R` (so functions take the extra `R` argument).

- `make_tip_distribution(PG_laser, R, H, B, theta, z0)` → `ParticleGroup` (laser → cone surface).
- `cone_height(R, H, B)` → the apex height of the full (un-truncated) cone implied by the rounding.
- `tip_height(r, R, H, B)` → surface z(r) (spherical cap for r ≤ rt, linear cone flank beyond).
- `tip_normal(r, R, H, B)` → `(nr, nz)` outward unit normal.
- `FirstIntersection(x0, y0, z0, theta, R, H, B)` → x of first ray intersection (NaN if none).
- `SphereIntersection(...)`, `ConeIntersection(...)`, `PlaneIntersection(...)` → component ray
  intersections. `rotation_from_z_to_n(n)` → as above.

```python
import numpy as np
from GPT_tools.cathode_particlegroup import get_cathode_particlegroup
from GPT_tools.conical_tip_emission import make_tip_distribution

flat = get_cathode_particlegroup({'n_particle': 5000}, 'examples/distgen.in.yaml')
# Map a flat-cathode laser spot onto a rounded-cone emitter:
# R = tip radius, H = height, B = base diameter, theta = incidence angle, z0 = aim point
tip = make_tip_distribution(flat, R=50e-9, H=2e-6, B=1e-6, theta=np.radians(15), z0=2e-6)
print('particles that hit the tip:', tip.n_particle)
```

---

## `boundary_element_solver.py` — axisymmetric BEM field solver

A self-contained axisymmetric (r–z) boundary-element electrostatic solver, plus geometry import from
Poisson/SUPERFISH-style decks, mesh refinement, an RK4 particle tracker, and analytic test cases.

**Tracking**
- `BEM_track(r0, z0, pr0, pz0, dt, steps, zmax, SI_to_fieldmap_scale, voltage_scale, r_all, z_all, sigma, segment_indices)`
  → `(r_vals, z_vals, pr_vals, pz_vals)`. RK4-tracks one electron through the BEM-computed field until
  `z>zmax` or `steps` exhausted; reflects at r=0.

**Geometry import / meshing**
- `parse_geometry(filename, boundary_voltage=np.nan)` → list of conductor "elements" (each a list of
  `[x, y, voltage, (radius, nt)]` points) parsed from a Poisson-style `&REG/&PO` deck.
- `subdivide_elements(elements, max_seg_length)` → elements re-sampled into segments no longer than
  `max_seg_length` (handles straight segments and circular arcs with `nt=4/5` orientation flags).
- `refine_once(elements, sigma, r_all, z_all, segment_indices, max_verr, show_plots=True)` → elements
  with points inserted wherever the solved potential at a segment midpoint deviates from the target by
  more than `max_verr` (adaptive refinement). `refine_element(element, indices)` does the insertion.
- `segment_midpoint(p1, p2, cen)`, `is_clockwise_arc(...)`, `circle_circle_intersection(...)`,
  `arc_to_segments(...)`, `plot_geometry(elements, fig_ax=None, axis='equal', plot_type='-')` →
  geometry helpers / plotting.

**Solver**
- `flatten_conductors(r_conductors, z_conductors, V_conductors)` → `(r_all, z_all, V_all, segment_indices)`.
- `G_ring(r, z, rp, zp)` → axisymmetric ring Green's function (complete elliptic K).
- `G_ring_with_derivatives(r, z, rp, zp)` → `(G, dG/dr, dG/dz)`.
- `linear_segment_potential_contrib(ri, zi, r0, z0, r1, z1)` → the two linear-basis potential
  contributions of a segment to a vertex.
- `build_linear_BEM_matrix(r_all, z_all, segment_indices)` → the dense influence matrix `A`.
- `make_C_matrix(A, r_all, z_all, segment_indices)` → constraint rows tying the endpoints of each
  closed surface to equal charge density.
- `solve_linear_BEM(elements)` → `(sigma, A, r_all, z_all, segment_indices)`; solves `A σ = V`
  (augmented with the closed-surface constraints if present).
- `evaluate_potential(r_eval, z_eval, ...)` → potential at a point from the solved `sigma`.
- `evaluate_field(r_eval, z_eval, ...)` → `(E_r, E_z)` at a point (negative gradient).

**Analytic test cases**
- `voltage_step_cathode(r, z, a, V0, calc_voltage=True, calc_field=True)` → `(phi, Er, Ez)` for a
  z=0 plane held at V0 for r<a, 0 for r>a.
- `conducting_disk(r, z, a, V0)` → `(phi, Er, Ez)` for an isolated charged disk of radius `a`.

```python
from GPT_tools.boundary_element_solver import (
    parse_geometry, subdivide_elements, solve_linear_BEM, evaluate_field)

elements = parse_geometry('gun_geometry.am', boundary_voltage=0.0)  # Poisson-style deck
elements = subdivide_elements(elements, max_seg_length=1e-4)        # mesh it
sigma, A, r_all, z_all, seg = solve_linear_BEM(elements)            # solve for charge density
Er, Ez = evaluate_field(0.0, 1e-3, r_all, z_all, sigma, seg)        # on-axis field at z=1 mm
```

---

## `aperture_scan.py` — simulated aperture scans & 4D phase-space reconstruction

Simulates the experimental "aperture scan" MTE/emittance measurement (steering a beam across a
pinhole with corrector magnets and imaging it on a viewscreen) and reconstructs the cathode 4D
phase space and MTE from the resulting image stack.

**Forward simulation**
- `aperture_scan(settings, GPT_INPUT_FILE, DISTGEN_INPUT_FILE, noise_settings={}, root_dir='/tmp/fake_data', sub_dir='fake', max_workers=4, n_particle=5000, laser_offsets=[[0,0]], scan_nxny=[41,41], scan_n_sigma=4, viewscreen_nxny=[201,199], viewscreen_calib=25.6, z_aperture=0.9264, z_screen=1.42817, scan_x_var='scanner_x_current', scan_y_var='scanner_y_current', typical_scan_setting=0.01, random_seed=None)`
  → writes a grid of `.mat` viewscreen images (one per magnet setting, per laser offset) plus calibration
  metadata. Calibrates magnet-to-aperture/screen response, computes transfer matrices, runs GPT at each
  scan point in parallel (`run_one_point`), and bins the screen onto a simulated camera.
- `run_one_point(mx, my, settings, seed)` → runs one GPT scan point and saves its `.mat` image.
- `ensure_empty_dir(path)` → makes/empties a directory (only auto-clears under `/tmp/`).

**Transfer-matrix fitting**
- `get_transfer_matrix(gpt_data, screen_location, force_symplectic=False, return_offset=False, return_diagnostics=False, p0=None, method='BFGS')`
  → 4×4 matrix `M` (optionally offset `b` and diagnostics). Fits `final ≈ M·initial + b` from
  cathode↔screen particle pairs; `force_symplectic` constrains M = exp(JK) with K symmetric.
- `change_matrix_units(M, units=[1,mc,1,mc], inverse=False)` → rescales a transfer matrix between
  angle and momentum units.
- `best_larmor_frame(M, angle=None)` → `(M_rotated, angle)`, removing the solenoid Larmor rotation.
- Internal symplectic-fit machinery: `_symplectic_J`, `_vec_to_symmetric4`/`_symmetric4_to_vec`,
  `_fit_affine_unconstrained`, `_symplecticity_error`, `_nearest_symplectic_polar`,
  `_initial_p0_from_unconstrained`, `_fit_affine_symplectic`, `_extract_matched_coords`.

**Reconstruction from images**
- `analyze_aperture_scan_4d(image_directory, show_roi_check=True)` → dict with the assembled 4D
  histogram `data` and axes `x, y, px, py` (plus `g, gb, laser_position`). Reads the `.mat` stack,
  re-centers each image for the corrector kick, and stacks them into a 4D array.
- `project_phase_space(input_data, var1='x', var2='y', colormap='viridis')` → plots a 2D projection of
  the 4D data and annotates σ and emittance.
- `get_emit(data, y, py)` → `(emit, sig_y, sig_py, y_avg, py_avg, ypy_avg)` from a 2D histogram.
- `get_sigma(data, x, y, px, py)` → `(centroid X0, 4×4 sigma matrix S, 4D emittance eps)`.
- `process_image_directories(image_directory)` → analyzes every sub-directory, writing per-scan
  `summary/*.mat`; returns the summary directory.
- `complete_transfer_matrix(M, S, opts=None)` / `solve_in_scaled_coordinates(M, S, Omega, opts)` →
  fill in the unmeasured columns (2 and 4) of a transfer matrix using the measured beam covariance and
  symplecticity. Supporting objective/constraint helpers: `objective_centered`,
  `symplectic_col_constraint_value`, `objective2`, `nonlinear_constraint_scalar`,
  `covariance_residuals`, `choose_pair_scale`, `choose_symplectic_scaling`.
- `get_MTE(filename, M, S_input, viewscreen_resolution=0, colormap='viridis', show_plots=True)` →
  `(MTEx, MTEy, MTEeff, sigx_cath, emit)`; back-propagates the measured screen distribution to the
  cathode using `M` to extract the mean transverse energy and cathode spot size.
- `calc_reconstructions(root_directory, enforce_symplectic=True, viewscreen_resolution=0.0, colormap='viridis', show_plots=True)`
  → `(MM, MTEeff, sigx_cath, emit)`; the top-level driver: processes all scans, fits the
  laser-offset → centroid response into a (symplectic) transfer matrix, completes it, and computes MTE.
- `get_drift(d)` → a 4×4 drift transfer matrix.
- Loading helpers: `load_mat_file`, `_mat_struct_to_dict`, `load_summary_data`,
  `load_and_analyze_data_file`, `density_histogram`, `blur_aperturescan_data` (placeholder).

```python
from GPT_tools.aperture_scan import aperture_scan, calc_reconstructions

# 1) Generate the simulated image stack (one .mat per magnet point)
aperture_scan(settings, 'examples/gpt.in', 'examples/distgen.in.yaml',
              root_dir='/tmp/fake_data', laser_offsets=[[0, 0], [50, 0]])

# 2) Reconstruct the transfer matrix and MTE from those images
MM, MTEeff, sigx_cath, emit = calc_reconstructions('/tmp/fake_data')
print('reconstructed MTE (meV):', MTEeff, ' cathode sigma_x (um):', sigx_cath)
```

---

## `emittance_vs_fraction.py` — emittance-vs-fraction curves

- `emittance_vs_fraction(pg, var, number_of_points=25, plotting=True, verbose=False, show_core_emit_plot=False, title_fraction=[], title_emittance=[])`
  → `(es, fs, ec, fc)`. For phase plane `var`∈{`x`,`y`}, computes the emittance `es` of the beam core
  enclosing each beam fraction `fs` (by shrinking the bounding Twiss ellipse via simplex optimization),
  plus the core emittance `ec` and its corresponding fraction `fc`. Optionally plots the curve.
- `get_twiss(x, y, w)` → `(emittance, alpha, beta, x0, y0)` weighted Twiss parameters of a 2D
  distribution.
- `get_emit_at_frac(f_target, twiss_parameters, x, y, w)` → the emittance of the inner `f_target`
  fraction defined by single-particle Courant–Snyder amplitudes under the given Twiss ellipse.
- `minboundellipse(x_all, y_all, tolerance=1e-3, plot_on=False)` → `(emittance, alpha, beta, center,
  gamma)` minimum bounding ellipse via the Khachiyan algorithm. (No longer used by the main routine.)

```python
from GPT_tools.emittance_vs_fraction import emittance_vs_fraction

# Emittance enclosed vs beam fraction in the x plane, with the core point marked
es, fs, ec, fc = emittance_vs_fraction(G.screen[-1], 'x', number_of_points=25, plotting=True)
print(f'core emittance {ec:.3g} m at fraction {fc:.2f}')
```

---

## Typical workflows

**Run GPT and plot the result (notebook):**
```python
from GPT_tools.GPTExtension import run_gpt_with_settings
from GPT_tools.gpt_plot_gui import gpt_plot_gui

G = run_gpt_with_settings(settings,
                          gpt_input_file='gpt.in',
                          distgen_input_file='distgen.in.yaml',
                          auto_phase=True)
gpt_plot_gui(G)          # interactive explorer
```

**Make a single plot programmatically:**
```python
from GPT_tools.gpt_plot import gpt_plot, gpt_plot_dist2d
gpt_plot(G, 'mean_z', ['sigma_x', 'sigma_y'])           # beam-size trend
gpt_plot_dist2d(G, 'x', 'px', screen_z=0.058)           # x–px phase space at z
```

**Explore an Xopt optimization front:**
```python
from GPT_tools.front_gui import front_gui
front_gui('xopt.in.yaml', 'tmp/')
```

See `examples/example.ipynb` and `examples/example_population.ipynb` for end-to-end usage.
