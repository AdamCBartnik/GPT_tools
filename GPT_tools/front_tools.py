import os, yaml
import numpy as np
import pandas as pd
from glob import glob
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from xopt import Xopt, VOCS
from xopt.generators.ga.cnsga import cnsga_toolbox, pop_from_data


def load_vocs(xopt_file):
    # VOCS read directly from the yaml, so a (possibly incompatible) generator.population_file is never loaded
    with open(xopt_file, 'r') as fid:
        xopt_input = yaml.safe_load(fid)
    return VOCS.model_validate(xopt_input['vocs']), xopt_input


def pop_sampler(data, vocs, new_pop_size):
    # Best new_pop_size individuals by CNSGA selection (objectives and constraints), as front_gui's "best N"
    toolbox = cnsga_toolbox(vocs)
    pop = pop_from_data(data, vocs)
    pop = toolbox.select(pop, new_pop_size)
    return data.iloc[[int(p.index) for p in pop]]  # p.index is positional; xopt_index labels may repeat


def random_variable_values(variable, n, rng):
    if hasattr(variable, 'values') and variable.values is not None:   # DiscreteVariable
        return rng.choice(sorted(variable.values), n)
    lo, hi = variable.domain
    return rng.uniform(lo, hi, n)


def clamp_variable_values(variable, x):
    x = np.asarray(x, dtype=float)
    if hasattr(variable, 'values') and variable.values is not None:   # snap to the nearest allowed value
        allowed = np.array(sorted(variable.values), dtype=float)
        return allowed[np.argmin(np.abs(x[:, None] - allowed[None, :]), axis=1)]
    lo, hi = variable.domain
    return np.clip(x, lo, hi)


def clamp_population(xopt_file, pop_filename=None, output_filename=None, seed=None, verbose=True):
    """
    Make an existing population file consistent with an (edited) xopt file, so that it can be used as
    generator.population_file in a new optimization. pop_filename defaults to generator.population_file in the
    xopt file (a relative path is tried as given, then relative to the xopt file's folder).
      - variables are clamped to their new ranges (discrete variables snap to the nearest allowed value),
        and variables missing from the file get random values within their ranges
      - every constant is set to its value in the xopt file, including former variables that are now constants
      - if the xopt file asks for a smaller population_size, the best individuals are kept (CNSGA selection,
        as front_gui's "best N"); if larger, random individuals within the variable ranges are added
        (their outputs are left empty)
    Saves to output_filename, default <pop_filename without .csv>_clamped.csv, and returns that filename.
    """
    vocs, xopt_input = load_vocs(xopt_file)
    new_pop_size = int(xopt_input['generator']['population_size'])
    rng = np.random.default_rng(seed)

    if pop_filename is None:
        pop_filename = xopt_input['generator'].get('population_file')
        if pop_filename is None:
            raise ValueError(f'No pop_filename given and no generator: population_file in {xopt_file}')
        if not os.path.isfile(pop_filename) and not os.path.isabs(pop_filename):
            beside_xopt_file = os.path.join(os.path.dirname(os.path.abspath(xopt_file)), pop_filename)
            if os.path.isfile(beside_xopt_file):
                pop_filename = beside_xopt_file
    if not os.path.isfile(pop_filename):
        raise FileNotFoundError(f'Population file not found: {pop_filename}')

    pop = pd.read_csv(pop_filename, index_col='xopt_index')
    n_start = len(pop)
    report = []

    for name, variable in vocs.variables.items():
        if name not in pop.columns:
            pop[name] = random_variable_values(variable, len(pop), rng)
            report.append(f'{name}: not in file, filled with random values')
            continue
        old = pop[name].to_numpy(dtype=float)
        clamped = clamp_variable_values(variable, old)
        n_changed = np.count_nonzero(clamped != old)
        if n_changed > 0:
            pop[name] = clamped
            report.append(f'{name}: clamped {n_changed} of {len(pop)} value(s) into the new range')

    for name, constant in vocs.constants.items():
        value = constant.value
        if name not in pop.columns or not (pop[name] == value).all():
            pop[name] = value
            report.append(f'{name}: set to constant {value!r}')

    if new_pop_size < len(pop):
        missing = [k for k in vocs.objective_names + vocs.constraint_names if k not in pop.columns]
        if len(missing) > 0:
            report.append(f'warning: {", ".join(missing)} not in file, so ignored when choosing the best individuals')
        pop = pop_sampler(pop, vocs, new_pop_size)
        report.append(f'kept the best {new_pop_size} of {n_start} individuals')
    elif new_pop_size > len(pop):
        n_new = new_pop_size - len(pop)
        first_index = (int(pop.index.max()) + 1) if len(pop) > 0 else 0
        new_index = pd.Index(np.arange(first_index, first_index + n_new), name=pop.index.name)
        new_rows = pd.DataFrame(np.nan, index=new_index, columns=pop.columns)
        for name, variable in vocs.variables.items():
            new_rows[name] = random_variable_values(variable, n_new, rng)
        for name, constant in vocs.constants.items():
            new_rows[name] = constant.value
        pop = pd.concat([pop, new_rows])
        report.append(f'added {n_new} random individuals ({n_start} -> {new_pop_size})')

    if output_filename is None:
        output_filename = os.path.splitext(pop_filename)[0] + '_clamped.csv'
    pop.to_csv(output_filename, index_label='xopt_index')

    if verbose:
        print('\n'.join(report) if len(report) > 0 else 'Population already consistent with the xopt file.')
        print(f'Saved: {output_filename}')

    return output_filename


def get_ind_settings_dict_from_pop_dataframe(pop_element, X):
    # Xopt.evaluate() checks constants but does not pass them to the evaluate function, so include them here
    ind_dict = pop_element.to_dict()
    missing = [k for k in X.vocs.variable_names if k not in ind_dict]
    if len(missing) > 0:
        raise ValueError(f'Population is missing variable(s) {missing}; run clamp_population first.')
    settings = {k: ind_dict[k] for k in X.vocs.variable_names}
    settings.update({k: getattr(c, 'value', c) for k, c in X.vocs.constants.items()})
    return settings


def run_xopt_func(settings):
    return _reevaluate_X.evaluate(settings)  # module global set by reevaluate_population, inherited by forked workers


def replace_pop_df_evaluation_output(pop_sample, evaluation_list, settings_list):
    # Assign column by column (DataFrame.replace would swap matching values in every column). Outputs can differ
    # between evaluations (e.g. a failed run returns fewer keys), so use every key seen and NaN where one is missing.
    pop_sample = pop_sample.copy()
    output_keys = list(dict.fromkeys(k for p in evaluation_list for k in p.keys()))
    input_keys = list(dict.fromkeys(k for p in settings_list for k in p.keys()))
    for k in output_keys:
        pop_sample[k] = [p.get(k, np.nan) for p in evaluation_list]
    for k in input_keys:
        pop_sample[k] = [p.get(k, np.nan) for p in settings_list]
    return pop_sample


def reevaluate_population(xopt_file, pop_num=-1, pop_path=None, max_workers=None):
    """
    Re-run every individual of a population with the settings in xopt_file (e.g. after changing constants or
    the simulation), keeping the best population_size individuals first if the file is larger.

    The population is generator.population_file if set, otherwise pop_path/*_population_*.csv sorted by name
    (i.e. by time), index pop_num (default: newest); pop_path defaults to tmp/ next to xopt_file.
    Evaluations run in parallel on max_workers processes (default: all CPUs). Linux only (uses fork).
    Saves <population file without .csv>_reevaluated.csv and returns that filename.
    """
    ### Open Xopt ###
    global _reevaluate_X
    X = Xopt.from_file(xopt_file)
    _reevaluate_X = X
    with open(xopt_file, 'r') as fid:
        xopt_input = yaml.safe_load(fid)

    ### Find the population file ###
    if xopt_input['generator'].get('population_file') is not None:
        pop_filename = xopt_input['generator']['population_file']
    else:
        if pop_path is None:
            pop_path = os.path.join(os.path.dirname(xopt_file), 'tmp')
        filename_list = sorted(glob(os.path.join(pop_path, "*_population_*.csv")))
        if len(filename_list) == 0:
            raise FileNotFoundError(f'No *_population_*.csv files in {pop_path}')
        pop_filename = filename_list[pop_num]
    print(f'Reevaluating {pop_filename}')

    new_pop_size = int(xopt_input['generator']['population_size'])
    pop_df = pd.read_csv(pop_filename, index_col="xopt_index")
    pop_sample = pop_sampler(pop_df, X.vocs, new_pop_size) if new_pop_size < len(pop_df) else pop_df

    ### Get settings from population subset and xopt ###
    all_ind_settings = [get_ind_settings_dict_from_pop_dataframe(p, X) for i, p in pop_sample.iterrows()]

    ### Reevaluate in parallel ###
    # fork (rather than the platform default) so workers inherit _reevaluate_X
    if max_workers is None:
        max_workers = os.cpu_count()
    max_workers = int(max(1, min(max_workers, len(all_ind_settings))))
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context('fork')) as executor:
        ps = list(executor.map(run_xopt_func, all_ind_settings))

    ### Create output population ###
    pop_new = replace_pop_df_evaluation_output(pop_sample, ps, all_ind_settings)
    n_errors = int(np.sum([bool(p.get('xopt_error', False)) for p in ps]))
    if (n_errors > 0):
        print(f'Warning: {n_errors} of {len(ps)} evaluations raised an error (see xopt_error_str)')
    reevaluated_pop_filename = os.path.splitext(pop_filename)[0] + '_reevaluated.csv'
    pop_new.to_csv(reevaluated_pop_filename, index_label="xopt_index")
    print(f'Saved: {reevaluated_pop_filename}')

    return reevaluated_pop_filename
