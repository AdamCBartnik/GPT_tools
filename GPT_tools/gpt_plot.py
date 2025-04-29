import numpy as np
import matplotlib as mpl
from gpt import GPT
from .tools import *
from .nicer_units import *
from .postprocess import postprocess_screen
from pmd_beamphysics.units import c_light, e_charge
from .ParticleGroupExtension import ParticleGroupExtension, convert_gpt_data, divide_particles
from ipywidgets import HBox
import ipywidgets as widgets
from GPT_tools.SnappingCursor import SnappingCursor
import pandas as pd
import random

def make_dataframe_widget(df):
    out = widgets.Output()
    with out:
        display(df, clear=True)
    return out

def gpt_plot(gpt_data_input, var1, var2, units=None, fig_ax=None, format_input_data=True, show_survivors_at_z=None, show_survivors_after_z=None, 
             show_screens=True, show_cursor=True, return_data=False, legend=True, **params):
    if (format_input_data):
        gpt_data = convert_gpt_data(gpt_data_input)
    else:
        gpt_data = copy.deepcopy(gpt_data_input)
    
    if (show_survivors_after_z is not None):
        show_survivors_at_z = show_survivors_after_z
        show_survivors_after_z = True
    
    if (show_survivors_at_z is not None):
        survivor_params = {}
        survivor_params['screen_z'] = show_survivors_at_z
        pmd, _, _ = get_screen_data(gpt_data, **survivor_params)
        pmd = postprocess_screen(pmd, **params)
        survivor_ids = pmd.id[pmd.weight > 0]

        survivor_params = {}
        survivor_params['include_ids'] = survivor_ids
        survivor_params['need_copy'] = False
        for i, s in enumerate(gpt_data.particles):
            if ((s['mean_z'] >= show_survivors_at_z) or (show_survivors_after_z is None)):
                gpt_data.particles[i] = postprocess_screen(s, **survivor_params)
    else:
        if ('include_ids' in params):
            survivor_params = {}
            survivor_params['include_ids'] = params['include_ids']
            survivor_params['need_copy'] = False
            for i, s in enumerate(gpt_data.particles):
                gpt_data.particles[i] = postprocess_screen(s, **survivor_params)
    
    if (fig_ax==None):
        show_plot = True
        if (legend):
            fig_ax = make_default_plot(plot_width=700, plot_height=400, **params)
        else:
            fig_ax = make_default_plot(plot_width=600, plot_height=400, **params)
    else:
        show_plot = False
        if (legend):
            fig_ax[0].set_size_inches(700/fig_ax[0].dpi, 400/fig_ax[0].dpi)
        else:
            fig_ax[0].set_size_inches(600/fig_ax[0].dpi, 400/fig_ax[0].dpi)
    
    # Use MatPlotLib default line colors
    mpl_cmap = mpl.pyplot.get_cmap('Set1') # 'Set1', 'tab10'
    cmap = ["#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b, _ in 255*mpl_cmap(range(mpl_cmap.N))]
    
    # Find good units for x data
    special_i = special_screens(gpt_data.stat('mean_z', 'screen'))
    (all_x, x_units, x_scale) = scale_and_get_units(gpt_data.stat(var1, 'screen'), gpt_data.units(var1).unitSymbol)
    
    x = all_x
    screen_x = [xx for i, xx in enumerate(all_x) if i in special_i]
        
    if (not isinstance(var2, list)):
        var2 = [var2]
        
    # Combine all y data into single array to find good units
    all_y = np.array([])
    all_y_base_units = gpt_data.units(var2[0]).unitSymbol
    for var in var2:
        if (gpt_data.units(var).unitSymbol != all_y_base_units):
            raise ValueError('Plotting data with different units not allowed.')
        #all_y = np.concatenate((all_y, gpt_data.stat(var)))  # touts and screens for unit choices
        all_y = np.concatenate((all_y, gpt_data.stat(var, 'screen')))  # screens for unit choices

    # In the case of emittance, use 2*median(y) as a the default scale, to avoid solenoid growth dominating the choice of scale
    use_median_y_strs = ['norm', 'slice']
    if (any(any(substr in varstr for substr in use_median_y_strs) for varstr in var2)):
        (_, y_units, y_scale) = scale_and_get_units(2.0*np.median(all_y), all_y_base_units)
        all_y = all_y / y_scale
    else:
        (all_y, y_units, y_scale) = scale_and_get_units(all_y, all_y_base_units)
                
    # overwrite with user units
    if (units is not None):
        base_units = gpt_data.units(var2[0]).unitSymbol
        if (units.endswith(base_units)):
            all_y = all_y * y_scale
            y_units = units
            y_scale = SHORT_PREFIX_FACTOR[units[:-len(base_units)]]
            all_y = all_y / y_scale
        else:
            print('Incorrect units specified')
            units = None

    # Make sure the order of the x values is monotonic
    xi = np.argsort(x)
    x = x[xi]
            
    # Finally, actually plot the data
    if (return_data):
        output_data = np.array([x])
    
    line_list = []
    
    auto_legend = type(legend) == bool
    if (not auto_legend):
        legend_name = legend
        legend = True
    
    for i, var in enumerate(var2):
        y = gpt_data.stat(var, 'screen') / y_scale
        y = y[xi]
        if (auto_legend):
            legend_name = f'{format_label(var)}'
        
        if ('color' in params):
            if (type(params['color']) == list):
                plot_color = params['color'][i]
            else:
                plot_color = params['color']
        else:
            plot_color = cmap[i % len(cmap)]
        
        line_list = line_list + fig_ax[1].plot(x, y, '-', color=plot_color, label=legend_name)

        if (return_data):
            output_data = np.vstack((output_data, y))

        if (show_screens):
            screen_y = [yy for i, yy in enumerate(y) if i in special_i]
            if (auto_legend):
                screen_legend_name = f'Screen: {format_label(var)}'
            else:
                screen_legend_name = 'Screen: ' + legend_name
            fig_ax[1].plot(screen_x, screen_y, 'o', color=plot_color, label=screen_legend_name)
                
    # Axes labels
    ylabel_str = get_y_label(var2)
    fig_ax[1].set_xlabel(f"{format_label(var1, use_base=True)} ({x_units})")
    fig_ax[1].set_ylabel(f"{ylabel_str} ({y_units})")
                     
    # Cursor that snaps to data
    if (show_cursor):
        snap_cursor = SnappingCursor(fig_ax[0], fig_ax[1], line_list)
        fig_ax[0].canvas.mpl_connect('motion_notify_event', snap_cursor.on_mouse_move)   # button_press_event, motion_notify_event
        fig_ax[0].snap_cursor = snap_cursor # HACK: keep reference to cursor in figure so that garbage collection doesn't kill it
        
    # Turn off legend if not desired
    if (legend):
        fig_ax[1].legend(bbox_to_anchor=(1.02, 1.0), loc='upper left')
    
    fig_ax[1].set_xlim([np.min(all_x), np.max(all_x)])
        
    if ('xlim' in params):
        fig_ax[1].set_xlim(params['xlim'])

    if ('log_scale' in params):
        log_scale = params['log_scale']
    else:
        log_scale = False
            
    if (log_scale):
        fig_ax[1].set_yscale('log')
        
        if ('ylim' in params):
            fig_ax[1].set_ylim(params['ylim'])
        
    else:
        # Cases where the y-axis should be forced to start at 0
        zero_y_strs = ['sigma_', 'charge', 'energy', 'slice', 'norm']
        if (any(any(substr in varstr for substr in zero_y_strs) for varstr in var2)):
            fig_ax[1].set_ylim([0, 1.1*np.max(all_y)])      

        # Cases where the y-axis range should use the median, instead of the max (e.g. emittance plots)
        use_median_y_strs = ['norm_emit_x','norm_emit_y']
        if (any(any(substr in varstr for substr in use_median_y_strs) for varstr in var2)):
            fig_ax[1].set_ylim([0, 2.0*np.median(all_y)])

        if ('ylim' in params):
            fig_ax[1].set_ylim(params['ylim'])
            
    if (show_plot):
        return HBox([fig_ax[0].canvas], layout=widgets.Layout(width='800px'))
        
    if (return_data):
        return np.transpose(output_data)

    


def gpt_plot_dist1d(pmd, var, plot_type='charge', units=None, fig_ax=None, table_fig=None, table_on=True, **params):
    screen_key = None
    screen_value = None
    if (isinstance(pmd, GPT)):
        pmd, screen_key, screen_value = get_screen_data(pmd, **params)
    if (not isinstance(pmd, ParticleGroupExtension)):
        pmd = ParticleGroupExtension(input_particle_group=pmd)
    pmd = postprocess_screen(pmd, **params)
                
    plot_type = plot_type.lower()
    density_types = {'charge'}
    is_density = False
    if (plot_type in density_types):
        is_density = True
        
    positive_types = {'charge', 'norm_emit', 'sigma', 'slice'}
    is_positive = False
    if any([d in plot_type for d in positive_types]):
        is_positive = True
        
    min_particles = 1
    needs_many_particles_types = {'norm_emit', 'sigma'}
    if any([d in plot_type for d in positive_types]):
        min_particles = 3
        
    if (fig_ax==None):
        show_plot = True
        fig_ax = make_default_plot(plot_width=600, plot_height=400, **params)
    else:
        show_plot = False
        fig_ax[0].set_size_inches(600/fig_ax[0].dpi, 400/fig_ax[0].dpi)
            
    # Use MatPlotLib default line colors
    mpl_cmap = mpl.pyplot.get_cmap('Set1') # 'Set1', 'tab10'
    cmap = ["#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b, _ in 255*mpl_cmap(range(mpl_cmap.N))]
    
    if('nbins' in params):
        nbins = params['nbins']
    else:
        nbins = 50
                
    charge_base_units = pmd.units('charge').unitSymbol
    q_total, charge_scale, charge_prefix = nicer_array(pmd.charge)
    q = pmd.weight / charge_scale
    q_units = check_mu(charge_prefix)+charge_base_units
    
    if ('subtract_mean' in params):
        subtract_mean = params['subtract_mean']
    else:
        subtract_mean=check_subtract_mean(var)
    (x, x_units, x_scale, mean_x, mean_x_units, mean_x_scale) = scale_mean_and_get_units(getattr(pmd, var), pmd.units(var).unitSymbol,
                                                                                         subtract_mean=subtract_mean, weights=q)
    
    if (units is not None):       
        # Replace X units
        base_units = pmd.units(var).unitSymbol
        if (units.endswith(base_units)):
            x = x * x_scale
            x_units = units
            x_scale = SHORT_PREFIX_FACTOR[units[:-len(base_units)]]
            x = x / x_scale
        else:
            print('Incorrect units specified')
    
    p_list, edges, density_norm = divide_particles(pmd, nbins=nbins, key=var)
    
    is_radial_var = False
    if (var == 'r' or var == 'r_centered' or var == 'rp'):
        is_radial_var = True
    
    if (is_radial_var):
        density_norm = density_norm*(x_scale*x_scale)
    else:
        density_norm = density_norm*x_scale
    
    if (subtract_mean==True):
        edges = edges - mean_x*mean_x_scale
    edges = edges/x_scale
    
    plot_type_base_units = pmd.units(plot_type).unitSymbol
    _, plot_type_scale, plot_type_prefix = nicer_array(pmd[plot_type])
    plot_type_units = check_mu(plot_type_prefix)+plot_type_base_units
    norm = 1.0/plot_type_scale
    if (is_density):
        norm = norm*density_norm
    
    weights = np.array([0.0 for p in p_list])
    hist = np.array([0.0 for p in p_list])
    for p_i, p in enumerate(p_list):
        if (p.n_particle >= min_particles):
            hist[p_i] = p[plot_type]*norm
            weights[p_i] = p['charge']
    weights = weights/np.sum(weights)
    avg_hist = np.sum(hist*weights)
                
    edges, hist = duplicate_points_for_hist_plot(edges, hist)
    
    if (not is_radial_var):
        edges, hist = pad_data_with_zeros(edges, hist)

    if ('color' in params):
        plot_color = params['color']
    else:
        plot_color = cmap[0]

    line_list = fig_ax[1].plot(edges, hist, '-', color=plot_color, label=f'{format_label(var)}')
            
    fig_ax[1].set_xlabel(f"{format_label(var)} ({x_units})")
    
    plot_type_label = get_y_label([plot_type])
    if (is_positive):
        fig_ax[1].set_ylim([0, 1.1*np.max(hist)])  
    if (is_density):
        if (var == 'r'):
            y_axis_label=f"{plot_type_label} density ({plot_type_units}/{x_units}^2)"
        else:
            y_axis_label=f"{plot_type_label} density ({plot_type_units}/{x_units})"
    else:
        y_axis_label=f"{plot_type_label} ({plot_type_units})"
    
    fig_ax[1].set_ylabel(y_axis_label)
    
    if ('xlim' in params):
        fig_ax[1].set_xlim(params['xlim'])
    if ('ylim' in params):
        fig_ax[1].set_ylim(params['ylim'])
    
    stdx = std_weights(x,q)
        
    if(table_on):
        x_units = format_label(x_units, latex=False)
        mean_x_units = format_label(mean_x_units, latex=False)
        plot_type_units = format_label(plot_type_units, latex=False)
        q_units = format_label(q_units, latex=False)
        var_label = format_label(var, add_underscore=False, latex=False)
        plot_type_label = format_label(plot_type, add_underscore=False, latex=False)
        data = dict(Name=[], Value=[], Units=[])
        if (screen_key is not None):
            data = add_row(data, Name=f'Screen {screen_key}', Value=f'{screen_value:G}', Units='')
        data = add_row(data, Name=f'Total charge', Value=f'{q_total:G}', Units=f'{q_units}')
        if (not is_density):
            data = add_row(data, Name=f'Mean {plot_type_label}', Value=f'{avg_hist:G}', Units=f'{plot_type_units}')
        data = add_row(data, Name=f'Mean {var_label}', Value=f'{mean_x:G}', Units=f'{mean_x_units}')
        data = add_row(data, Name=f'σ_{var_label}', Value=f'{stdx:G}', Units=f'{x_units}')
    
        table_fig = make_dataframe_widget(pd.DataFrame(data=data))
    
    if (show_plot):
        if (table_on):
            return HBox([fig_ax[0].canvas, table_fig], layout=widgets.Layout(width='1000px'))
        else:
            return HBox([fig_ax[0].canvas], layout=widgets.Layout(width='700px'))

    if (table_on):
        return table_fig
    
    
    
def gpt_plot_dist2d(pmd, var1, var2, plot_type='histogram', units=None, fig=None, table_fig=None, table_on=True, plot_width=500, plot_height=400, 
                    return_data=False, x_subtract_mean='auto', y_subtract_mean='auto', fig_ax=None, **params):

    if (fig_ax==None):
        show_plot = True
        fig_ax = make_default_plot(plot_width=600, plot_height=400, **params)
    else:
        show_plot = False
        fig_ax[0].set_size_inches(600/fig_ax[0].dpi, 400/fig_ax[0].dpi)
    
    screen_key = None
    screen_value = None
    
    if (isinstance(pmd, GPT)):
        pmd, screen_key, screen_value = get_screen_data(pmd, **params)
    
    if (not isinstance(pmd, ParticleGroupExtension)):
        pmd = ParticleGroupExtension(input_particle_group=pmd)
    pmd = postprocess_screen(pmd, **params)
            
    if (isinstance(var2, tuple)):
        use_separate_data = True
        pmd2 = var2[1]
        var2 = var2[0]
        if (not isinstance(pmd2, ParticleGroupExtension)):
            pmd2 = ParticleGroupExtension(input_particle_group=pmd2)
    else:
        use_separate_data = False
        pmd2 = pmd
            
    is_radial_var = [False, False]
    if (var1 == 'r' or var1 == 'r_centered' or var1 == 'rp'):
        is_radial_var[0] = True
    if (var2 == 'r' or var2 == 'r_centered' or var2 == 'rp'):
        is_radial_var[1] = True
        
    if('nbins' in params):
        nbins = params['nbins']
    else:
        nbins = 50
        
    if (not isinstance(nbins, list)):
        nbins = [nbins, nbins]
        
    if ('colormap' in params):
        if type(params['colormap']) == str:
            colormap = mpl.cm.get_cmap(params['colormap'])
        else:
            colormap = params['colormap']
    else:
        colormap = mpl.cm.get_cmap('jet') 

    zlim = None
    if ('zlim' in params):
        zlim = params['zlim']
        
    if ('clim' in params):
        zlim = params['clim']
        
    charge_base_units = pmd.units('charge').unitSymbol
    q_total, charge_scale, charge_prefix = nicer_array(pmd.charge)
    q = pmd.weight / charge_scale
    q_units = check_mu(charge_prefix)+charge_base_units
    
    if (not isinstance(x_subtract_mean, bool)):
        x_subtract_mean = check_subtract_mean(var1)
    
    (x, x_units, x_scale, avgx, avgx_units, avgx_scale) = scale_mean_and_get_units(getattr(pmd, var1), pmd.units(var1).unitSymbol, subtract_mean=x_subtract_mean, weights=q)
    
    y = getattr(pmd2, var2)
    q_y = pmd2.weight / charge_scale
    if (use_separate_data):
        # Reorder to order from pmd
        y_id = pmd2.id
        y_dict = {id : i for i,id in enumerate(y_id)}
        y = np.array([y[y_dict[id]] if id in y_dict else 0.0 for id in pmd.id])  # The value on failure here doesn't matter since it will have weight = 0
        q_y = np.array([q_y[y_dict[id]] if id in y_dict else 0.0 for id in pmd.id])
    
    if (not isinstance(y_subtract_mean, bool)):
        y_subtract_mean = check_subtract_mean(var2)
    (y, y_units, y_scale, avgy, avgy_units, avgy_scale) = scale_mean_and_get_units(y, pmd2.units(var2).unitSymbol, subtract_mean=y_subtract_mean, weights=q_y)
        
    # overwrite with user units
    if (units is not None):
        user_x_units = units[0]
        user_y_units = units[1]
        
        # Replace X units
        base_units = pmd.units(var1).unitSymbol
        if (user_x_units.endswith(base_units)):
            x = x * x_scale
            x_units = user_x_units
            x_scale = SHORT_PREFIX_FACTOR[user_x_units[:-len(base_units)]]
            x = x / x_scale
        else:
            print('Incorrect units specified')
        
        # Replace Y units
        base_units = pmd.units(var2).unitSymbol
        if (user_y_units.endswith(base_units)):
            y = y * y_scale
            y_units = user_y_units
            y_scale = SHORT_PREFIX_FACTOR[user_y_units[:-len(base_units)]]
            y = y / y_scale
        else:
            print('Incorrect units specified')

        
    if('axis' in params and params['axis']=='equal'):
        if (x_scale > y_scale):
            rescale_ratio = y_scale/x_scale
            y = rescale_ratio*y
            y_units = x_units
            y_scale = x_scale
        else:
            rescale_ratio = x_scale/y_scale
            x = rescale_ratio*x
            x_units = y_units
            x_scale = y_scale
    
    colorbar_instance = None
    
    color_var = 'density'
    if ('color_var' in params):
        color_var = params['color_var']
    if(plot_type=="scatter"):
        colorbar_instance = scatter_color(fig_ax[0], fig_ax[1], pmd, x, y, weights=q, color_var=color_var, bins=nbins, colormap=colormap, is_radial_var=is_radial_var,zlim=zlim)
    if(plot_type=="histogram"):
        colorbar_instance = hist2d(fig_ax[0], fig_ax[1], pmd, x, y, weights=q, color_var=color_var, bins=nbins, colormap=colormap, is_radial_var=is_radial_var,zlim=zlim)

    fig_ax[1].set_xlabel(f"{format_label(var1)} ({x_units})")
    fig_ax[1].set_ylabel(f"{format_label(var2)} ({y_units})")
             
    if ('centered_at_zero' in params and params['centered_at_zero']):
        x_range_max = 1.05*np.max(np.abs(x))
        y_range_max = 1.05*np.max(np.abs(y))
        if('axis' in params and params['axis']=='equal'):
            x_range_max = np.max([x_range_max, y_range_max])
            y_range_max = x_range_max
        fig_ax[1].set_xlim([-x_range_max, x_range_max])
        fig_ax[1].set_ylim([-y_range_max, y_range_max])
        
    if ('xlim' in params):
        fig_ax[1].set_xlim(params['xlim'])
    if ('ylim' in params):
        fig_ax[1].set_ylim(params['ylim'])
        
    stdx = std_weights(x,q)
    stdy = std_weights(y,q)
    corxy = corr_weights(x,y,q)
    if (x_units == y_units):
        corxy_units = f'{x_units}²'
    else:
        corxy_units = f'{x_units}·{y_units}'
        
    if('axis' in params and params['axis']=='equal'):
        fig_ax[1].axis('equal')
    
    show_emit = False
    if ((var1 == 'x' and 'px' in var2 ) or (var1 == 'y' and 'py' in var2)):
        show_emit = True
        factor = c_light**2 /e_charge # kg -> eV
        particle_mass = 9.10938356e-31  # kg
        emitxy = (x_scale*y_scale/factor/particle_mass)*np.sqrt(stdx**2 * stdy**2 - corxy**2)
        (emitxy, emitxy_units, emitxy_scale) = scale_and_get_units(emitxy, pmd.units(var1).unitSymbol)
    
    if(table_on):
        x_units = format_label(x_units, latex=False)
        y_units = format_label(y_units, latex=False)
        corxy_units = format_label(corxy_units, latex=False)
        avgx_units = format_label(avgx_units, latex=False)
        avgy_units = format_label(avgy_units, latex=False)
        q_units = format_label(q_units, latex=False)
        if (show_emit):
            emitxy_units = format_label(emitxy_units, latex=False)
        var1_label = format_label(var1, add_underscore=False, latex=False)
        var2_label = format_label(var2, add_underscore=False, latex=False)
        data = dict(Name=[], Value=[], Units=[])
        if (screen_key is not None):
            data = add_row(data, Name=f'Screen {screen_key}', Value=f'{screen_value:G}', Units='')
        data = add_row(data, Name=f'Total charge', Value=f'{q_total:G}', Units=f'{q_units}')
        data = add_row(data, Name=f'Mean {var1_label}', Value=f'{avgx:G}', Units=f'{avgx_units}')
        data = add_row(data, Name=f'Mean {var2_label}', Value=f'{avgy:G}', Units=f'{avgy_units}')
        data = add_row(data, Name=f'σ_{var1_label}', Value=f'{stdx:G}', Units=f'{x_units}')
        data = add_row(data, Name=f'σ_{var2_label}', Value=f'{stdy:G}', Units=f'{y_units}')
        data = add_row(data, Name=f'Corr({var1_label}, {var2_label})', Value=f'{corxy:G}', Units=f'{corxy_units}')
        if (show_emit):
            data = add_row(data, Name=f'ε_{var1_label}', Value=f'{emitxy:G}', Units=f'{emitxy_units}')

        table_fig = make_dataframe_widget(pd.DataFrame(data=data))
        
    if (return_data):
        return np.transpose(np.vstack((x,y,q)))
        
    if (show_plot):
        if (table_on):
            return HBox([HBox([fig_ax[0].canvas], layout=widgets.Layout(width='700px')), table_fig])
        else:
            return HBox([fig_ax[0].canvas], layout=widgets.Layout(width='700px'))
        
    if (table_on):
        return (table_fig, colorbar_instance)



def gpt_plot_trajectory(gpt_data_input, var1, var2, fig_ax=None, format_input_data=True, nlines=None, show_survivors_at_z=None, **params):
    if (format_input_data):
        gpt_data = convert_gpt_data(gpt_data_input)
    else:
        gpt_data = copy.deepcopy(gpt_data_input)
    
    if (fig_ax==None):
        show_plot = True
        fig_ax = make_default_plot(plot_width=600, plot_height=400, **params)
    else:
        show_plot = False
        fig_ax[0].set_size_inches(600/fig_ax[0].dpi, 400/fig_ax[0].dpi)

    plot_ids = gpt_data_input.screen[0].id
        
    if (show_survivors_at_z is not None):
        survivor_params = copy.copy(params)
        survivor_params['screen_z'] = show_survivors_at_z
        pmd, _, _ = get_screen_data(gpt_data, **survivor_params)
        pmd = postprocess_screen(pmd, **params)
        plot_ids = pmd.id[pmd.weight > 0]

    if ('include_ids' in params):
        plot_ids = params['include_ids']
    
    if(len(plot_ids)==0):
        return None
    
    x = np.empty( (len(gpt_data.screen),len(plot_ids)) )
    y = np.empty( (len(gpt_data.screen),len(plot_ids)) )
    
    for i, s in enumerate(gpt_data.screen):
        index = np.argsort(s.id)
        sorted_s_id = s.id[index]
        sorted_index = np.searchsorted(sorted_s_id, plot_ids)

        found_index = np.take(index, sorted_index, mode="clip")
        mask = s.id[found_index] != plot_ids

        index_of_found_ids = np.ma.array(found_index, mask=mask)
        x[i,:] = getattr(s, var1)[index_of_found_ids]
        x[i,index_of_found_ids.mask] = np.nan
        y[i,:] = getattr(s, var2)[index_of_found_ids]
        y[i,index_of_found_ids.mask] = np.nan
            
    all_x = x.flatten()
    all_x = all_x[np.logical_not(np.isnan(all_x))]
    (_, x_units, x_scale, _, _, _) = scale_mean_and_get_units(all_x, gpt_data_input.units(var1).unitSymbol, subtract_mean=False)
    x = x/x_scale
    
    all_y = y.flatten()
    all_y = all_y[np.logical_not(np.isnan(all_y))]
    (_, y_units, y_scale, _, _, _) = scale_mean_and_get_units(all_y, gpt_data_input.units(var2).unitSymbol, subtract_mean=False)
    y = y/y_scale
    
    index_to_plot = np.arange(0,len(plot_ids)).tolist()
    if nlines is not None:
        if (nlines < len(plot_ids)):
            index_to_plot = random.sample(index_to_plot, nlines)
    for j in index_to_plot:
        fig_ax[1].plot(x[:,j], y[:,j], '-')
                
    fig_ax[1].set_xlabel(f"{format_label(var1, use_base=True)} ({x_units})")
    fig_ax[1].set_ylabel(f"{format_label(var2, use_base=True)} ({y_units})")
    
    if ('xlim' in params):
        fig_ax[1].set_xlim(params['xlim'])
    if ('ylim' in params):
        fig_ax[1].set_ylim(params['xlim'])
            
    if (show_plot):
        return HBox([fig_ax[0].canvas], layout=widgets.Layout(width='800px'))
    
