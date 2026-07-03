# GPT_tools

Plotting, analysis, and orchestration tools for [lume-gpt](https://github.com/ColwynGulliford/lume-gpt) — a Python wrapper around **General Particle Tracer (GPT)**, a charged-particle beam dynamics simulation code used in accelerator physics.

`GPT_tools` sits on top of `lume-gpt`, [`distgen`](https://github.com/ChristopherMayes/distgen) (initial particle distribution generation), [`pmd-beamphysics`](https://github.com/ChristopherMayes/openPMD-beamphysics) (`ParticleGroup`, the particle-data container), and [`xopt`](https://github.com/ChristopherMayes/Xopt) (multi-objective optimization). It provides:

- A run layer that goes from a settings dictionary straight to simulation output (`GPTExtension.py`).
- A `ParticleGroup` subclass with extra beam-physics quantities — emittances, actions, slice properties (`ParticleGroupExtension.py`).
- Interactive Jupyter plotting and GUIs for inspecting simulation output (`gpt_plot.py`, `gpt_plot_gui.py`).
- Standalone physics models for cathode/tip emission, image-charge effects, Compton scattering, and THz-based bunch manipulation.
- Tools for inspecting and re-running `xopt` optimization populations (`front_tools.py`, `front_gui.py`).

## Typical workflow

```
distgen.in.yaml ──┐
                   ▼
settings dict ──► run_gpt_with_settings() ──► GPT object (ParticleGroup screens/touts)
   (gpt.in)                                          │
                                                       ▼
                                    gpt_plot() / gpt_plot_gui() / gpt_plot_dist1d() / gpt_plot_dist2d()
                                                       │
                                                       ▼
                                    evaluate_run_gpt_with_settings() ──► xopt optimization
                                                       │
                                                       ▼
                                          front_gui() / show_fronts() (inspect results)
```

## Installation

GPT itself is **Linux-only** and requires its own license/binary (see the [GPT website](http://www.pulsar.nl/gpt/)) — this package only wraps it, it doesn't ship it. You'll need `lume-gpt` configured against a working GPT install, with the `$GPT_BIN` (and `$ASCI2GDF_BIN`) environment variables set, or passed explicitly to the `run_gpt_with_settings`-family functions.

```bash
git clone https://github.com/AdamCBartnik/GPT_tools.git
cd GPT_tools
pip install -r requirements.txt
pip install -e .
```

> **Note:** the code currently imports `ParticleGroup` via `from beamphysics import ...`. The published package is [`pmd-beamphysics`](https://pypi.org/project/pmd-beamphysics/) (import name `pmd_beamphysics`) — `beamphysics` may be an older/internal name from before a refactor. `requirements.txt` lists the real PyPI package; if `import beamphysics` fails for you, that's a pre-existing naming mismatch in this repo, not a problem with your install.

## Quick start

```python
from gpt import GPT
from GPT_tools.GPTExtension import run_gpt_with_settings
from GPT_tools.gpt_plot import gpt_plot, gpt_plot_dist1d, gpt_plot_dist2d
from GPT_tools.gpt_plot_gui import gpt_plot_gui

settings = {
    'gun_voltage': 140,
    'buncher_voltage': 3.47,
    'sol_2_current': 1.995,
    'total_charge:value': 5,
    'total_charge:units': 'fC',
}

gpt_data = run_gpt_with_settings(
    settings,
    gpt_input_file='gpt.in',
    distgen_input_file='distgen.in.yaml',
    auto_phase=True,
)

gpt_plot_gui(gpt_data)                                  # full interactive GUI
gpt_plot(gpt_data, 'mean_z', ['sigma_x', 'sigma_y'])     # beam size vs. z
gpt_plot_dist2d(gpt_data, 'x', 'px', screen_z=1.843)     # phase space at a screen
```

See [`examples/`](examples/) for complete, runnable notebooks.

## Package overview

**Running simulations**
| Module | Purpose |
|---|---|
| `GPTExtension.py` | Run GPT from a settings dict (`run_gpt_with_settings`), including parallel multi-runs, `xopt`-compatible merit-function wrappers (`evaluate_run_gpt_with_settings`, etc.), and THz-augmented runs. |
| `cathode_particlegroup.py` | Build a cathode-emission `ParticleGroup` from a `distgen` template + settings (including a core/shield macroparticle-weighting trick for dense cores). |

**Particle data & postprocessing**
| Module | Purpose |
|---|---|
| `ParticleGroupExtension.py` | `ParticleGroup` subclass adding emittances, actions, slice quantities, and other derived beam properties used throughout the rest of the package. |
| `postprocess.py` | Named, composable postprocessing operations on a screen (`take_slice`, `clip_to_charge`, `remove_spinning`, `remove_correlation`, ...), usable standalone or as keyword options to the plotting functions. |

**Plotting & GUIs**
| Module | Purpose |
|---|---|
| `gpt_plot.py` | Main plotting API: trend plots vs. z/t, 1D/2D histograms, and particle trajectories, all with postprocessing options. |
| `gpt_plot_gui.py` | Interactive `ipywidgets` GUI wrapping `gpt_plot.py`. |
| `tools.py` | Shared plotting/formatting utilities (screen lookup, label formatting, unit scaling). |
| `nicer_units.py` | Picks sensible SI prefixes for axis labels and printed values. |
| `SnappingCursor.py` | Generic matplotlib cursor that snaps to the nearest data point and shows a tooltip. |

**Optimization (`xopt`)**
| Module | Purpose |
|---|---|
| `front_tools.py` | Pure functions for inspecting `xopt` optimization populations and Pareto fronts. |
| `front_gui.py` | Interactive GUI wrapping `front_tools.py` — load a population, plot fronts, inspect/re-run individual points. |

**Physics models**
| Module | Purpose |
|---|---|
| `compton.py` | Monte Carlo model of inverse Compton scattering of a laser pulse off an electron bunch. |
| `THz_functions.py` | Analytic model of THz-mirror beam manipulation (bunch compression/streaking). |
| `image_charge.py` | Photocathode emission models (metal/semiconductor) including image-charge effects on the emitted energy distribution. |
| `conical_tip_emission.py`, `tip_emission.py` | Geometric models of laser-driven emission off a conical or paraboloid/spheroid field-emitter tip. |
| `emittance_vs_fraction.py` | Emittance vs. enclosed-charge-fraction curves via constrained ellipse fits. |

**Advanced / research**
| Module | Purpose |
|---|---|
| `aperture_scan.py` | Parallelized aperture-scan workflow: runs GPT over a grid of aperture positions and reconstructs the 4D transfer matrix and beam matrix from the results. |
| `boundary_element_solver.py` | Standalone axisymmetric boundary-element-method electrostatics solver, independent of GPT/`ParticleGroup` — useful for custom field calculations. |

These last two modules are more specialized/research-grade than the rest of the package; expect sparser documentation and fewer guardrails.

## Examples

- [`examples/example.ipynb`](examples/example.ipynb) — end-to-end walkthrough: run GPT from settings, explore output with the plotting GUI and individual plot functions, apply postprocessing, save/load an archive, and run a single `xopt`-style merit evaluation.
- [`examples/example_population.ipynb`](examples/example_population.ipynb) — load and inspect an `xopt` optimization population with `front_gui`.
