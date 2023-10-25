import os, copy, yaml, json, re
import numpy as np
from glob import glob
from matplotlib import pyplot as plt
from matplotlib import cm as cm
from numbers import Number
import pandas as pd
from xopt.generators.ga.cnsga import cnsga_toolbox, pop_from_data
from xopt import Xopt
import multiprocessing as mp     
from concurrent.futures import ProcessPoolExecutor

def show_fronts(pop_number, pop_path, obj1_key, obj2_key, obj3_key=None, xopt_file=None, xscale=1.0, yscale=1.0, zscale=1.0, show_constraint_violators=False, 
                new_fig=True, xlim=None, ylim=None, zlim=None, dpi=100, best_N=None, legend='on', colorbar=True, 
                xlabel=None, ylabel=None, zlabel=None, return_fig=False, return_data = False, legend_color=None, dot_style='o', zorder=1):
    
    if (new_fig and return_data == False):
        fig = plt.figure(dpi=dpi)
        
    if (xopt_file is None):
        xopt_file = get_xopt_file(pop_path)
                
    obj1_scale = xscale
    obj2_scale = yscale
    if (obj3_key is not None):
        obj3_scale = zscale
    
    pop, pop_number, pop_filename = get_pop(pop_path, pop_number, xopt_file, show_constraint_violators=show_constraint_violators, best_N=best_N)
        
    obj1 = np.array(pop[obj1_key]) * obj1_scale 
    obj2 = np.array(pop[obj2_key]) * obj2_scale
        
    if (obj3_key is not None):
        #obj3_key = fix_key(obj3_key, pop, pop_filename)   
        obj3 = np.array(pop[obj3_key]) * obj3_scale

    if (xlim is not None):
        obj1[obj1<xlim[0]] = np.nan
        obj1[obj1>xlim[1]] = np.nan

    if (ylim is not None):
        obj1[obj2<ylim[0]] = np.nan
        obj1[obj2>ylim[1]] = np.nan

    if (zlim is not None):
        obj1[obj3<zlim[0]] = np.nan
        obj1[obj3>zlim[1]] = np.nan

    not_nan = np.logical_not(np.isnan(obj1))

    p_list = []
    leg_list = []
    
    if (return_data == True):
        output_data = np.array([obj1[not_nan],  obj2[not_nan]]).T
    else:
        if (obj3_key is not None):
            if (zlim is not None):
                sc = plt.scatter(obj1[not_nan], obj2[not_nan], 10, c=obj3[not_nan], cmap='jet', vmin=zlim[0], vmax=zlim[1], marker=dot_style, zorder=zorder)
                if (legend_color is None):
                    cmap = cm.get_cmap('jet')
                    #legend_color = cmap(np.median(obj3[not_nan]))
            else:
                sc = plt.scatter(obj1[not_nan], obj2[not_nan], 10, c=obj3[not_nan], cmap='jet', marker=dot_style, zorder=zorder)
                if (legend_color is None):
                    cmap = cm.get_cmap('jet')
                    #legend_color = cmap(np.median(obj3[not_nan]))
            if (colorbar):
                cbar = plt.colorbar(sc)
                if (zlabel is None):
                    cbar.set_label(f'{obj3_key}')
                else:
                    cbar.set_label(zlabel)
        else:
            line_handle, = plt.plot(obj1[not_nan], obj2[not_nan], dot_style, zorder=zorder)
            p_list.append(line_handle)
            leg_list.append(f'Pop {pop_number}')

    if (return_data == False):
        if (xlim is not None):
            plt.xlim(xlim)
        if (ylim is not None):
            plt.ylim(ylim)

        if (xlabel is None):
            plt.xlabel(f'{obj1_key}')
        else:
            plt.xlabel(xlabel)
        if (ylabel is None):
            plt.ylabel(f'{obj2_key}')
        else:
            plt.ylabel(ylabel)
        if (legend_color is not None and legend is not None):
            line_handle, = plt.plot([], [], dot_style, color=legend_color)
            line_handle.set_label(legend)
            plt.legend()

        if (legend != 'off'):
            if (legend == 'on'):
                if (len(p_list) == len(leg_list)):
                    for ii,p in enumerate(p_list):
                        if (p is not None):
                            p.set_label(leg_list[ii])
            else:
                if type(legend) not in [list, tuple]:
                    legend = [legend]
                if (len(p_list) == len(legend)):
                    for ii,p in enumerate(p_list):
                        p.set_label(legend[ii])
            if (any([p is not None for p in p_list])):
                plt.legend()
                
    if (return_fig):
        return fig
    
    if (return_data):
        return output_data

def fix_xopt_pop_datafile(filename):
    pdf = pd.read_csv(os.path.join(template_dir, '../LANL/ued/tmp', fname), index_col = "xopt_index")
    good_row = pdf[pdf["xopt_error"] == False].iloc[0]
    bad_row_numbers = np.array(pdf[pdf["xopt_error"] == True].index)
    for ii in bad_row_numbers:
        pdf[pdf.index == ii] = good_row
    pdf.to_csv(filename)
    
def make_settings_csv(csv_filename, settings):
    n_settings = np.max([np.size(settings[ii]) for ii in settings.keys()])
    data = pd.DataFrame(data=settings, index=np.arange(1,n_settings+1))
    data.index.name = 'xopt_index'
    data.to_csv(csv_filename, index_label="xopt_index")
    
def find_settings(pop_number, obj1_target, pop_path=os.path.join('tmp'), xopt_file=None, obj1_key = 'end_sigma_t', 
                  obj2_key = 'end_norm_emit_x', obj3_key = None, zlim = None, n_neighbors = 10, 
                  minimize=True, xscale=1.0, yscale=1.0, zscale=1.0):

    if (xopt_file is None):
        xopt_file = get_xopt_file(pop_path)
    
    obj1_scale = xscale
    obj2_scale = yscale
    if (obj3_key is not None):
        obj3_scale = zscale
    
    pop, pop_number, pop_filename = get_pop(pop_path, pop_number, xopt_file, show_constraint_violators=False)
                
    index = np.array(pop.index)
    obj1 = np.array(pop[obj1_key]) * obj1_scale 
    obj2 = np.array(pop[obj2_key]) * obj2_scale
    
    if (obj3_key is not None):
        obj3 = np.array(pop[obj3_key]) * obj3_scale
        if (zlim is not None):
            possible_dudes = np.logical_and(obj3>=zlim[0], obj3<=zlim[1])
            index = index[possible_dudes]
            obj1 = obj1[possible_dudes]
            obj2 = obj2[possible_dudes]
            obj3 = obj3[possible_dudes]
            
    possible_dudes = np.argsort(np.abs(obj1 - obj1_target))
    
    

    possible_dudes = possible_dudes[0:n_neighbors]
    obj2 = obj2[possible_dudes]
    
    if (len(obj2)==0):
        print('No values found.')
        return Null
    else:
        if (minimize):
            best_dude = possible_dudes[np.argmin(obj2)]
        else:
            best_dude = possible_dudes[np.argmax(obj2)]
            
        settings = pop.to_dict('index')[index[best_dude]]
            
        print(f'{obj1_key} = {settings[obj1_key]*obj1_scale}')
        print(f'{obj2_key} = {settings[obj2_key]*obj2_scale}')
        if (obj3_key is not None):
            print(f'{obj3_key} = {settings[obj3_key]*obj3_scale}')
            
        if(isinstance(xopt_file, str)):
            with open(xopt_file, 'r') as fid:
                xopt_yaml =  yaml.safe_load(fid)
            wanted_keys = {**xopt_yaml['vocs']['variables'], **xopt_yaml['vocs']['constants']}.keys()
            settings = dict((k, settings[k]) for k in wanted_keys if k in settings)
            
        print(settings)
        return 

    
    
def get_pop(pop_path, pop_number, xopt_file=None, show_constraint_violators=False, best_N=None):
    (filename_list, pop_n_list) = get_filename_list(pop_path)
        
    if (xopt_file is None):
        xopt_file = get_xopt_file(pop_path)
        
    if (len(filename_list) < 1):
        print('Could not find any files.')
        return
    
    if (isinstance(pop_number, list)):
        print('ERROR: You are probably using an old version of this code, and are using a list for the pop number')
    
    if (pop_number >= 0):
        (pop_index, pop_number) = min(enumerate(pop_n_list), key=lambda xx: abs(xx[1]-pop_number))
    else:
        pop_index = pop_number
    
    pop_number = pop_n_list[pop_index]        
    pop_filename = filename_list[pop_index]
    if ('.json' in pop_filename):
        pop_filename = save_json_to_csv(pop_filename)
            
    pop = pd.read_csv(pop_filename, index_col="xopt_index")
    initial_pop_size = len(pop)
    
    if (not show_constraint_violators):
        pop = get_only_feasible_results(pop, xopt_yaml=xopt_file)
    
    n_removed = initial_pop_size-len(pop)
    if (n_removed>0):
        print(f'Removed {n_removed} from pop. {pop_number}')
    
    if (best_N is not None):
        pop = pop_sampler(pop, xopt_file, best_N)

    #pop = add_constraints(pop)
    return (pop, pop_number, pop_filename)
    

def pop_sampler(data, xopt_file, new_pop_size):
    xopt = Xopt(xopt_file)
    vocs = xopt.vocs
    vocs.constraints = {}

    toolbox = cnsga_toolbox(vocs)

    pop = pop_from_data(data, vocs)

    pop = toolbox.select(pop, new_pop_size)

    return data.loc[[int(p.index) for p in pop]]

    
def clamp_population(pop_number, pop_path, xopt_file):
    change_applied = False

    xopt_input = yaml.safe_load(open(xopt_file))
    vocs = xopt_input['vocs']
    varkeys = sorted([*vocs['variables']])
    
    new_pop_size = xopt_input['generator']['population_size']
    
    pop, pop_number, pop_filename = get_pop(pop_path, pop_number, xopt_file, show_constraint_violators=True)
    
    if (new_pop_size < len(pop)):
        change_applied = True
        pop = pop_sampler(pop, xopt_file, new_pop_size)
    
    data_length = len(pop)
    
    #for vk in varkeys:
    #    if vk not in data:
    #        data_range = vocs['variables'][vk]
    #        print(f'Inserting random values for undefined variable: {vk} from [{data_range[0]}, {data_range[1]}]')
    #        data[vk] = (data_range[0] + np.random.rand(data_length)*(data_range[1] - data_range[0])).tolist()
    #        change_applied = True
    #old_pop_size = len(data[varkeys[0]])
        
    #vecs = np.array([data[k] for k in varkeys]).T 
    
    ## Clamp bounds within range in xopt file
    #for i, v in enumerate(varkeys):
    #    low, up = vocs['variables'][v]
    #    if (vecs[:,i].min() < low):
    #        change_applied = True
    #        print(f'Data below lower bound of {v}, clamping to {low}.')
    #    if (vecs[:,i].max() > up):
    #        change_applied = True
    #        print(f'Data below upper bound of {v}, clamping to {up}.')
    #    for j,vec in enumerate(vecs[:,i]):
    #        vecs[j,i] = sorted((low, vec, up))[1]
    #    assert vecs[:,i].min() >= low, 'Data below lower bound' 
    #    assert vecs[:,i].max() <= up,  'Data above upper bound'
        
    #vecs = vecs.T
    #vecs = vecs[:,0:new_pop_size]
    
    #if (new_pop_size > old_pop_size):
    #    change_applied = True
    #    print(f'Padding population to new size of {new_pop_size} with duplicates, old size = {old_pop_size}.')
    #    vecs = np.pad(vecs, ((0,0), (0,new_pop_size-old_pop_size)), 'wrap')
    
    #for ii, k in enumerate(varkeys):
    #    data[k] = vecs[ii].tolist()
        
    #pop['variables'] = data
                            
    if (change_applied):
        clamped_pop_filename = pop_filename.replace('.csv','_clamped.csv')
        pop.to_csv(clamped_pop_filename, index_label="xopt_index")
        print('Creating new file: ' + clamped_pop_filename)
    else:
        print('No changes to population were needed, not creating new file.')
        
    return
   
    
def color_all_settings(pop_number, pop_path, xopt_file=None, obj1_key = 'end_sigma_t', obj2_key = 'end_norm_emit_x', xlim=None, ylim=None, zlim=None, xscale=1.0, yscale=1.0, show_constraint_violators=False):
        
    if (xopt_file is None):
        xopt_file = get_xopt_file(pop_path)
    
    p_list = []
    leg_list = []
    
    pop, pop_number, pop_filename = get_pop(pop_path, pop_number, xopt_file, show_constraint_violators=show_constraint_violators)
        
    if(isinstance(xopt_file, str)):
        with open(xopt_file, 'r') as fid:
            xopt_yaml =  yaml.safe_load(fid)
            
    vars = list(xopt_yaml['vocs']['variables'].keys())
    cons = list(xopt_yaml['vocs']['constraints'].keys())
    all_items = vars + cons
        
    if (obj1_key in all_items):
        all_items.remove(obj1_key)
        
    if (obj2_key in all_items):
        all_items.remove(obj2_key)
        
    for key in all_items: 
        if (np.all([isinstance(xx, Number) for xx in pop[key]])):
            if (np.std(pop[key]) > 0):
                show_fronts(pop_number, pop_path, obj1_key, obj2_key, obj3_key=key, xlim=xlim, ylim=ylim, xscale=xscale, yscale=yscale, show_constraint_violators=show_constraint_violators)
    

def get_xopt_file(pop_path):
    pop_path_split = os.path.split(os.path.normpath(pop_path))
    if (pop_path_split[1] == 'tmp' or pop_path_split[1] == 'test'):
        xopt_file = os.path.join(pop_path_split[0], 'xopt.in.yaml')
    else:
        xopt_file = os.path.join(pop_path, 'xopt.in.yaml')
    return xopt_file
    
def get_filename_list(pop_path):
    filename_list = sorted(glob(os.path.join(pop_path, "*.csv")))
    pop_n_list = np.arange(0, len(filename_list))
        
    return (filename_list, pop_n_list)
    
    
def save_json_to_csv(pop_filename):
    pop_old = json.load(open(pop_filename))
    a = copy.copy(pop_old['outputs'])
    for kk in pop_old['variables'].keys():
        for ii, ss in enumerate(pop_old['variables'][kk]):
            a[ii][kk] = ss
            a[ii]['xopt_index'] = ii
            a[ii]['xopt_error'] = pop_old['error'][ii]
    a = pd.json_normalize(a)
    csv_filename = pop_filename.replace('.json', '.csv')
    a.to_csv(csv_filename, index=False)
    return csv_filename
    
def get_only_feasible_results(pop_df, xopt_yaml='xopt.yaml'):
    
    if(isinstance(xopt_yaml, str)):
        with open(xopt_yaml, 'r') as fid:
            xopt_yaml =  yaml.safe_load(fid)
    
    # Remove individuals that threw an error
    pop_df = pop_df[pop_df['xopt_error']!=True] 
        
    for c, v in xopt_yaml['vocs']['constraints'].items():
        
        bin_opr, bound = v[0], v[1]
        bound = float(bound)
        
        if(bin_opr=='LESS_THAN'):
            pop_df = pop_df[pop_df[c]<bound]
        elif(bin_opr=='GREATER_THAN'):
            pop_df = pop_df[pop_df[c]>bound]
            
    return pop_df



def get_ind_settings_dict_from_pop_dataframe(pop_element, X):
    
    variables = list(X.generator.vocs.variables.keys())
    constants = list(X.generator.vocs.constants.keys())
    xopt_constants = X.generator.vocs.constants
    
    #input_setting_names = variables + constants
    
    #ind_df = pop_df.loc[[xopt_ind]]
    ind_dict = pop_element.to_dict()
    settings_dict_vars = {c:ind_dict[c] for c in list(ind_dict.keys()) if c in variables}
    settings_dict_consts = {c:xopt_constants[c] for c in list(ind_dict.keys()) if c in constants}
    return {**settings_dict_vars, **settings_dict_consts}

def run_xopt_func(settings):
    return X.evaluate(settings)

def replace_pop_df_evaluation_output(pop_sample, evaluation_list, settings_list):
    output_keys = list(evaluation_list[0].keys())
    input_keys = list(settings_list[0].keys())
    for k in output_keys:
        pop_sample = pop_sample.replace(list(pop_sample[k]), [p[k] for p in evaluation_list])
    for k in input_keys:
        pop_sample = pop_sample.replace(list(pop_sample[k]), [p[k] for p in settings_list])
    return pop_sample

def reevaluate_population(xopt_file, pop_num = -1, pop_path = None):
    
    ### Open Xopt ###
    X = Xopt(xopt_file)
    xopt_input = yaml.safe_load(open(xopt_file))
    
    ### Check for population file in Xopt ###
    # If there is no population file, use file provided by user
    # Default is latest in 'tmp/' directory in xopt file path
    if 'population_file' in xopt_input['generator']:
        pop_filename = xopt_input['generator']['population_file']
    else:
        if pop_path is None:
            pop_path = os.path.join(os.path.dirname(xopt_file),'tmp/')
            print(pop_path)
        filename_list = sorted(glob(os.path.join(pop_path, "*_population_*.csv")))
        pop_filename = filename_list[pop_num]
    
    new_pop_size = xopt_input['generator']['population_size']                  # desired output population size
    pop_df = pd.read_csv(pop_filename, index_col="xopt_index")                 # load starting population
    pop_sample = pop_sampler(pop_df, xopt_file, new_pop_size)                  # subset population with desired output size
    
    ### Parallelization ###
    try:
        mp.set_start_method("fork")
    except Exception as ex:
        print(ex)

    executor = ProcessPoolExecutor()
    executor.max_workers = pop_sample.shape[0]
    
    ### Get settings from population subset and xopt ###
    all_ind_settings = [get_ind_settings_dict_from_pop_dataframe(p, X) for i, p in pop_sample.iterrows()]

    ### Reevaluate subset population ###
    with executor as p:
        ps = list(p.map(run_xopt_func, all_ind_settings))

    ### Create output population ###
    pop_new = replace_pop_df_evaluation_output(pop_sample, ps, all_ind_settings)
    clamped_pop_filename = pop_filename.replace('.csv','_reevaluated.csv')
    pop_new.to_csv(clamped_pop_filename, index_label="xopt_index")