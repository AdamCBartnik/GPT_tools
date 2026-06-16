# GPT_tools

A Python toolkit for running, analyzing, and visualizing **General Particle Tracer (GPT)**
beam-dynamics simulations through [LUME-GPT](https://github.com/ChristopherMayes/lume-gpt) and the
[openPMD-beamphysics](https://github.com/ChristopherMayes/openPMD-beamphysics) `ParticleGroup`
object.

It grew out of photoinjector / electron-source modeling work and bundles together a higher-level GPT
run layer, an extended `ParticleGroup` with many derived beam-physics quantities, interactive Jupyter
GUIs, and a collection of cathode/photoemission/field-solver physics models.

## What's in it

- **Run layer** (`GPTExtension.py`) — auto-phasing, restarts, multithreaded GPT runs, merit functions,
  and evaluation hooks for [Xopt](https://github.com/ChristopherMayes/Xopt) optimization.
- **Extended particle group** (`ParticleGroupExtension.py`) — slice/core/4D/6D emittance, action
  variables, transverse energy, and more, exposed as `ParticleGroup` properties.
- **Plotting** (`gpt_plot.py`, `gpt_plot_gui.py`) — trend, 1D/2D distribution, and trajectory plots,
  plus a tabbed ipywidgets GUI for exploring GPT output interactively.
- **Optimization fronts** (`front_tools.py`, `front_gui.py`) — load, filter, plot, and re-run Xopt
  population (Pareto-front) files.
- **Physics models** — image-charge photoemission (`image_charge.py`), THz streaking
  (`THz_functions.py`), inverse Compton scattering (`compton.py`), field-emitter tips
  (`tip_emission.py`, `conical_tip_emission.py`), an axisymmetric boundary-element field solver
  (`boundary_element_solver.py`), aperture-scan MTE reconstruction (`aperture_scan.py`), and
  emittance-vs-fraction analysis (`emittance_vs_fraction.py`).

A full, module-by-module reference of every function (with inputs and outputs) is in
[`DOCUMENTATION.md`](DOCUMENTATION.md).

## Installation

Requires **Python ≥ 3.8**. Clone and install in editable mode:

```bash
git clone https://github.com/AdamCBartnik/GPT_tools.git
cd GPT_tools
pip install -e .
```

### Dependencies

`setup.py` reads `requirements.txt`, which is currently empty, so `pip install` will **not** pull in
the scientific stack automatically — install the dependencies yourself. The package imports:

- Core: `numpy`, `scipy`, `matplotlib`, `pandas`, `sympy`, `mpmath`
- Beam dynamics: `gpt` (LUME-GPT), `distgen`, `beamphysics` (openPMD-beamphysics), `pint`
- Optimization: `xopt`
- Notebook GUIs: `ipywidgets` (run inside JupyterLab/Notebook with the `ipympl`/`widget` matplotlib
  backend)
- Misc: `psutil`, `fastnumbers`

A conda/mamba environment is the easiest route for the beam-dynamics packages:

```bash
mamba install -c conda-forge numpy scipy matplotlib pandas sympy mpmath \
    ipywidgets ipympl psutil fastnumbers pint
mamba install -c conda-forge lume-gpt distgen openpmd-beamphysics xopt
```

GPT itself (the `gpt` binary and the `asci2gdf` converter) must be installed separately and reachable
via the `$GPT_BIN` and `$ASCI2GDF_BIN` environment variables (the defaults used throughout the run
functions).

## Quick start

```python
from GPT_tools.GPTExtension import run_gpt_with_settings
from GPT_tools.gpt_plot_gui import gpt_plot_gui

# settings: dict of variables in your GPT input deck, plus optional special keys
G = run_gpt_with_settings(
    settings,
    gpt_input_file='examples/gpt.in',
    distgen_input_file='examples/distgen.in.yaml',
    auto_phase=True,
)

gpt_plot_gui(G)   # interactive explorer (run in a Jupyter notebook)
```

Non-interactive plots:

```python
from GPT_tools.gpt_plot import gpt_plot, gpt_plot_dist2d

gpt_plot(G, 'mean_z', ['sigma_x', 'sigma_y'])   # beam-size vs z
gpt_plot_dist2d(G, 'x', 'px', screen_z=0.058)   # x–px phase space at a screen
```

Explore an Xopt optimization population:

```python
from GPT_tools.front_gui import front_gui
front_gui('examples/xopt.in.yaml', 'tmp/')
```

The `examples/` directory contains runnable notebooks (`example.ipynb`,
`example_population.ipynb`) and the input files they use (`gpt.in`, `distgen.in.yaml`,
`xopt.in.yaml`, and a sample field map).

## Conventions

- Positions in meters, momenta in eV/c, time in seconds, energies in eV, charges (weights) in
  Coulombs — the openPMD-beamphysics convention.
- A *screen* is a fixed-z output plane; a *tout* is a fixed-t time snapshot. The run/plot helpers
  distinguish the two.
- Run functions resolve the GPT binary from `$GPT_BIN` and the ASCII→GDF converter from
  `$ASCI2GDF_BIN` by default.

See [`DOCUMENTATION.md`](DOCUMENTATION.md) for the complete API reference.
