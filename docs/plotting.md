# Plotting API reference

Full parameter reference for the four main plotting entry points in `GPT_tools`:

- [`gpt_plot`](#gpt_plot) — trend plots vs. `z` or `t`
- [`gpt_plot_dist1d`](#gpt_plot_dist1d) — 1D histograms at a screen/tout
- [`gpt_plot_dist2d`](#gpt_plot_dist2d) — 2D histograms/scatter plots at a screen/tout
- [`gpt_plot_gui`](#gpt_plot_gui) — interactive GUI wrapping all three

All four live in `GPT_tools.gpt_plot` (the GUI is in `GPT_tools.gpt_plot_gui`).

## Shared concepts

**Input data.** `gpt_plot` and `gpt_plot_gui` take a `GPT` object (from `lume-gpt`) directly. `gpt_plot_dist1d` and `gpt_plot_dist2d` accept either a `GPT` object — in which case a single screen/tout is selected first (see [screen selection](#screen-and-tout-selection) below) — or a `ParticleGroup`/`ParticleGroupExtension` directly.

**Variable names.** Anywhere a variable name is expected (`var1`, `var2`, `var`, ...), you can use anything exposed by `ParticleGroup`/`ParticleGroupExtension`: `x`, `y`, `z`, `t`, `r`, `r_centered`, `px`, `py`, `pz`, `pr_centered`, `ptrans`, `kinetic_energy`, `energy`, `charge`, emittances (`norm_emit_x`, `sqrt_norm_emit_4d`, `slice_emit_x`, ...), actions (`action_x`, `action_4d`, ...), and the `mean_*`/`sigma_*` statistic of any of these for trend plots (e.g. `mean_z`, `sigma_x`).

**`**params`.** All four functions accept arbitrary keyword arguments beyond what's listed below. They're forwarded into the screen-selection and postprocessing steps, which is how options like `screen_z` and `cylindrical_copies` reach the plot even though they're not named arguments of `gpt_plot_dist2d` itself. Each function checks the names it receives against the set it understands and prints a warning for anything unrecognized (typically a typo — the misspelled option is ignored, not applied).

### Screen and tout selection

These keys (passed via `**params`) pick which screen or tout `gpt_plot_dist1d`/`gpt_plot_dist2d` operate on, when given a `GPT` object. They're handled by `get_screen_data` in `tools.py`.

| Parameter | Description |
|---|---|
| `screen_z=z` | Use the screen nearest `mean_z == z`. |
| `screen_t=t` | Use the screen nearest `mean_t == t`. |
| `tout_z=z` | Use the tout nearest `mean_z == z` (touts instead of screens). |
| `tout_t=t` | Use the tout nearest `mean_t == t`. |
| `screen_key=k, screen_value=v` | Generic form: use the screen nearest `mean_k == v`. |
| `verbose=True` | Print which screen/tout was actually selected. |
| *(none of the above)* | Defaults to `screen[0]`. |

If none of `screen_z`/`screen_t`/`tout_z`/`tout_t` is given, the first screen is used.

### Postprocessing options

These keys (passed via `**params`) filter or transform the selected screen before plotting. They're handled by `postprocess_screen` in `postprocess.py`, and apply to `gpt_plot_dist1d`, `gpt_plot_dist2d`, and (per-screen, across the whole trend) to `gpt_plot`/`gpt_plot_trajectory` via `show_survivors_at_z`/`include_ids`.

| Parameter | Description |
|---|---|
| `kill_zero_weight=True` | Remove particles with zero weight. |
| `include_ids=[...]` | Keep only particles with the given IDs. |
| `take_range=(var, min, max)` | Keep only particles with `var` in `[min, max]`. For `x`/`y`/`z`/`t`, the range is relative to the mean. If no particles fall inside the range, a message is printed and the filter is not applied. |
| `take_slice=(var, slice_index, n_slices)` | Divide the screen into `n_slices` equal-width bins along `var`, and keep only the `slice_index`-th bin. |
| `clip_to_charge=target_charge` | Radially clip particles (around the beam centroid) until the remaining charge equals `target_charge`. |
| `clip_to_emit=target_emit` | Radially clip particles until the remaining `sqrt_norm_emit_4d` equals `target_emit`. |
| `cylindrical_copies=n_copies` | Add `n_copies` rotated duplicates of every particle around the z-axis, splitting each particle's weight by `n_copies`. Useful for filling in a cylindrically-symmetric distribution that was sparsely sampled in angle. |
| `remove_spinning=True` | Subtract the `x`-`py`/`y`-`px` correlations caused by solenoid rotation. |
| `remove_correlation=(var1, var2, max_power)` | Fit a degree-`max_power` polynomial of `var2` vs. `var1` and subtract it from `var2`, removing that correlation. |
| `random_N=N` | Keep a random `N` particles (takes precedence over `first_N`). |
| `first_N=N` | Keep the first `N` particles. |
| `need_copy=True/False` | Force or skip the internal deep-copy of the screen before mutating it. Normally inferred automatically from which other options are set; only override this if you specifically want the input object mutated in place. |

## `gpt_plot`

```python
gpt_plot(gpt_data_input, var1, var2, units=None, fig_ax=None, format_input_data=True,
          show_survivors_at_z=None, show_survivors_after_z=None,
          show_screens=True, show_cursor=True, return_data=False, legend=True, **params)
```

Plots one or more statistics (`var2`) vs. `var1` (typically `mean_z` or `mean_t`) across all screens/touts.

| Parameter | Default | Description |
|---|---|---|
| `gpt_data_input` | — | A `GPT` object (or anything `convert_gpt_data` accepts). |
| `var1` | — | X-axis variable, e.g. `'mean_z'`. |
| `var2` | — | Y-axis variable, or a list of variables to plot together. All entries must share the same base units. |
| `units` | `None` | Override the y-axis units, e.g. `'mm'`. Must end with the same base unit symbol as `var2`. |
| `fig_ax` | `None` | Existing `(fig, ax)` tuple to draw into. If `None`, a new figure is created and returned. |
| `format_input_data` | `True` | Run `convert_gpt_data` on the input. Set `False` if it's already a converted/extended object. |
| `show_survivors_at_z` | `None` | Only show particles that have nonzero weight (after postprocessing) at this `z`, across *every* screen/tout in the plot. |
| `show_survivors_after_z` | `None` | Like `show_survivors_at_z`, but the filter is only applied to screens/touts at or after this `z`; earlier screens still show all particles. |
| `show_screens` | `True` | Overlay point markers at "special" screens — screens spaced irregularly compared to the rest (e.g. apertures, named diagnostics), as detected by `special_screens()`. |
| `show_cursor` | `True` | Enable a hover tooltip that snaps to the nearest data point. |
| `return_data` | `False` | If `True`, return the plotted `(x, y1, y2, ...)` arrays instead of a widget. |
| `legend` | `True` | Show a legend with auto-generated labels. Pass a string to use a custom label instead (only meaningful with a single `var2`). |
| `color` *(in `**params`)* | auto (`Set1` colormap) | A single color, or a list of colors (one per `var2` entry). |
| `xlim` / `ylim` *(in `**params`)* | auto | `[min, max]` axis limits. |
| `log_scale` *(in `**params`)* | `False` | Use a log-scaled y-axis. |
| `include_ids` *(in `**params`)* | — | Restrict the plot to only these particle IDs, across every screen/tout (same effect as the postprocessing option, applied per-screen). |
| `slice_key` *(in `**params`)* | `'t'` | Variable to slice along when plotting slice statistics (`slice_emit_x`, `slice_emit_4d`, ...). |
| `n_slices` *(in `**params`)* | `50` | Number of slices used for slice statistics. |

Also accepts all [postprocessing options](#postprocessing-options) (applied to the survivor-filtering step) and figure-sizing kwargs (`plot_width`, `plot_height`, `dpi`) forwarded to `make_default_plot` when `fig_ax` is not given.

## `gpt_plot_dist1d`

```python
gpt_plot_dist1d(pmd, var, plot_type='charge', units=None, fig_ax=None, table_fig=None,
                 table_on=True, subtract_mean='auto', **params)
```

Plots a 1D histogram of `var` at a single screen/tout.

| Parameter | Default | Description |
|---|---|---|
| `pmd` | — | A `GPT` object (a screen/tout is selected per [screen selection](#screen-and-tout-selection)) or a `ParticleGroup`/`ParticleGroupExtension` directly. |
| `var` | — | Variable to histogram on the x-axis, e.g. `'x'`, `'t'`, `'r'`. |
| `plot_type` | `'charge'` | What to compute per bin. `'charge'` plots a charge-density histogram. Any other attribute name (e.g. `'norm_emit_x'`, `'sigma_x'`, `'mean_kinetic_energy'`) plots the per-bin value of that attribute instead of a density. Bins with fewer than 3 particles are left at zero for types that need a population to be meaningful (`norm_emit`, `sigma`). |
| `units` | `None` | Override the x-axis units. |
| `fig_ax` | `None` | Existing `(fig, ax)` tuple to draw into. |
| `table_fig` | `None` | Existing summary-statistics table widget to reuse (internal/GUI use). |
| `table_on` | `True` | Show a summary statistics table (total charge, mean value, mean of `var`, σ of `var`) alongside the plot. |
| `subtract_mean` | `'auto'` | Whether to subtract the mean from `var` before plotting. `'auto'` subtracts for `x`, `y`, `z`, `t`, `kinetic_energy`, `energy` and not otherwise. Pass `True`/`False` to force it, or a float to subtract that specific reference value instead of the mean. |
| `nbins` *(in `**params`)* | `50` | Number of histogram bins. |
| `xlim` / `ylim` *(in `**params`)* | auto | `[min, max]` axis limits. |
| `color` *(in `**params`)* | auto | Line color. |

Also accepts all [screen/tout selection](#screen-and-tout-selection) and [postprocessing](#postprocessing-options) options.

## `gpt_plot_dist2d`

```python
gpt_plot_dist2d(pmd, var1, var2, plot_type='histogram', units=None, table_fig=None,
                 table_on=True, plot_width=600, plot_height=400, return_data=False,
                 x_subtract_mean='auto', y_subtract_mean='auto', fig_ax=None, **params)
```

Plots `var2` vs. `var1` at a single screen/tout, as a 2D histogram or scatter plot.

| Parameter | Default | Description |
|---|---|---|
| `pmd` | — | A `GPT` object or `ParticleGroup`/`ParticleGroupExtension`, as in `gpt_plot_dist1d`. |
| `var1` | — | X-axis variable. |
| `var2` | — | Y-axis variable. Can also be a tuple `(var_name, other_pmd)` to plot a variable from a *different* screen/`ParticleGroup`, matched to `pmd`'s particles by ID. |
| `plot_type` | `'histogram'` | `'histogram'` bins the data into a 2D grid (color = density, or the mean of `color_var` per cell). `'scatter'` plots individual particles colored by `color_var`. |
| `units` | `None` | `(x_units, y_units)` tuple to override axis units. |
| `table_fig` | `None` | Existing summary table widget to reuse (internal/GUI use). |
| `table_on` | `True` | Show a summary statistics table (charge, means, σs, correlation, and emittance if `var1`/`var2` is an `x`/`px` or `y`/`py` pair). |
| `plot_width`, `plot_height` | `600`, `400` | Figure size in pixels, if creating a new figure. |
| `return_data` | `False` | If `True`, return `(x, y, weight)` arrays instead of a widget. |
| `x_subtract_mean`, `y_subtract_mean` | `'auto'` | Same semantics as `gpt_plot_dist1d`'s `subtract_mean`, applied independently per axis. |
| `fig_ax` | `None` | Existing `(fig, ax)` tuple to draw into. |
| `nbins` *(in `**params`)* | `50` | Bin count, or `[n_x, n_y]` for separate x/y bin counts. |
| `colormap` *(in `**params`)* | `'jet'` | Colormap name or a `matplotlib.colors.Colormap` instance. |
| `zlim` / `clim` *(in `**params`)* | auto | `[min, max]` color-scale limits (`zlim` and `clim` are aliases). |
| `color_var` *(in `**params`)* | `'density'` | Variable to color by. Can also be a tuple `(var_name, other_pmd)` to color by a variable from a different screen/`ParticleGroup`, matched by particle ID. |
| `axis='equal'` *(in `**params`)* | — | Force equal x/y scaling (same units, same scale) and equal aspect ratio. |
| `centered_at_zero` *(in `**params`)* | — | If `True`, force the axis limits to be symmetric around zero. |
| `xlim` / `ylim` *(in `**params`)* | auto | `[min, max]` axis limits. |

Also accepts all [screen/tout selection](#screen-and-tout-selection) and [postprocessing](#postprocessing-options) options.

## `gpt_plot_gui`

```python
gpt_plot_gui(gpt_data_input)
```

Builds an interactive `ipywidgets` GUI wrapping the three functions above. Takes a single argument — a `GPT` object (or anything `convert_gpt_data` accepts) — and exposes everything else through widgets:

- **Screen selector** (always visible): choose between "Special" screens (irregularly-spaced, e.g. apertures/diagnostics) or "All" screens, then pick a specific one.
- **Trends tab**: x-axis (`z`/`t`), y-axis (preset combos — Beam Size, Bunch Length, Emittance (x,y), Emittance (4D), Slice emit. (x,y), Slice emit. (4D), Charge, Kinetic Energy, Energy Spread, Trajectory, MTE), slice-trend variable/count when a slice-based y is chosen (`slice_key`/`n_slices`), survivor filtering (`show_survivors_at_z`), log scale, cursor toggle, x/y axis limits.
- **1D Dist. tab**: x variable, y "plot type" preset (Charge Density, Emittance X/Y/4D, Sigma X/Y/E), histogram bin count, x/y axis limits.
- **2D Dist. tab**: plot method (Scatter/Histogram), color variable with an optional alternate screen as its source, x/y variables with an optional alternate screen for y, equal-axis toggle, x/y bin counts, x/y axis limits, color-scale limits.
- **Postproc. tab**: remove zero weight, cylindrical copies (with count), remove spinning, remove correlation (with polynomial order and variable pair).
- **Slicing tab**: take a slice or range of the data (with variable/index/range controls), radial clip to charge.

The axis/color limit boxes take `min, max` (e.g. `-2, 2`); leaving one blank (or unparseable) means automatic.

**Equivalent command box.** Below the controls, the GUI shows the `gpt_plot`/`gpt_plot_dist1d`/`gpt_plot_dist2d` call that reproduces the current plot, assuming your data variable is named `gpt_data` — copy it into a notebook cell to go from interactive exploration to a reproducible plot. This is literally the call the GUI makes: anything you can do in the GUI, you can also do by calling those functions directly with the parameters documented above.

`gpt_plot_gui` returns the widget; in a notebook, make it the last expression of the cell (or call `display()` on it) to show it.
