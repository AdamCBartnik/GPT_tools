import time, os, copy, numbers
import numpy as np
import re
from gpt import GPT
from gpt.gpt_phasing import gpt_phasing
from .ParticleGroupExtension import ParticleGroupExtension
from pmd_beamphysics import ParticleGroup
from distgen import Generator
from distgen.writers import write_gpt
from .tools import get_screen_data
from .postprocess import kill_zero_weight, clip_to_charge, take_range, clip_to_emit
from .cathode_particlegroup import get_coreshield_particlegroup, get_cathode_particlegroup
from pint import UnitRegistry
import matplotlib.pyplot as plt


def default_gpt_merit(G):
    
    
    """
    default merit function to operate on an evaluated LUME-GPT object G.  
    
    Returns dict of scalar values containing all stat quantities a particle group can compute 
    """
    # Check for error
    if G.error:
        # Make a GPT() that does not have an error
        PG = {}
        for k in ['x', 'px', 'y', 'py', 'z', 'pz', 't', 'id', 'weight']:
            PG[k] = [1,2,3]
        PG['species'] = 'electron'
        PG['status'] = [0,0,0]
        PG = ParticleGroup(data=PG)

        GG = GPT(initial_particles = PG)
        GG.output['particles'] = [PG]
        GG.output['n_screen'] = 1
        GG.output['n_tout'] = 0

        error_output = default_gpt_merit(GG)
        
        # Replace all values with Ivan's favorite error number
        for k in output.keys():
            if (isinstance(error_output[k], numbers.Number)  and not isinstance(error_output[k], bool)):
                error_output[k] = 1.0e88
        
        # Make sure error is true
        error_output['error'] = True
        return error_output
    
    else:
        m= {'error':False}

    if(G.initial_particles):
        start_n_particle = G.initial_particles['n_particle']

    elif(G.get_dist_file()):

        iparticles=read_particle_gdf_file(os.path.join(G.path, G.get_dist_file()))
        start_n_particle = len(iparticles['x'])

    else:
        raise ValueError('evaluate.default_gpt_merit: could not find initial particles.')


    try:

        # Load final screen for calc
        if(len(G.screen)>0):

            screen = G.screen[-1]   # Get data on last screen

            cartesian_coordinates = ['x', 'y', 'z']
            cylindrical_coordinates = ['r', 'theta']
            all_coordinates = cartesian_coordinates + cylindrical_coordinates

            all_momentum = [f'p{var}' for var in all_coordinates]
            cartesian_velocity = [f'beta_{var}' for var in cartesian_coordinates]
            angles = ['xp', 'yp']
            energy = ['energy', 'kinetic_energy', 'p', 'gamma']

            all_variables = all_coordinates + all_momentum + cartesian_velocity + angles + energy + ['t']

            keys =  ['n_particle', 'norm_emit_x', 'norm_emit_y', 'norm_emit_4d', 'higher_order_energy_spread']

            stats = ['mean', 'sigma', 'min', 'max']
            for var in all_variables:
                for stat in stats:
                    keys.append(f'{stat}_{var}')

            for key in keys:
                m[f'end_{key}']=screen[key]

            # Extras
            m['end_z_screen']=screen['mean_z']
            m['end_n_particle_loss'] = start_n_particle - m['end_n_particle']
            m['end_total_charge'] = screen['charge']

            # Basic Custom paramters:
            m['end_sqrt_norm_emit_4d'] = np.sqrt(m['end_norm_emit_4d'])
            m['end_max[sigma_x, sigma_y]'] = max([m['end_sigma_x'], m['end_sigma_y']])
            m['end_max[norm_emit_x, norm_emit_y]'] = max([m['end_norm_emit_x'], m['end_norm_emit_y']])
            
    except Exception as ex:

        m['error']=True
    
    # Remove annoying strings
    if 'why_error' in m:
        m.pop('why_error')
        
    return m




def multirun_gpt_with_particlegroup(settings=None,
                             gpt_input_file=None,
                             input_particle_group=None,
                             workdir=None, 
                             use_tempdir=True,
                             gpt_bin='$GPT_BIN',
                             timeout=2500,
                             auto_phase=False,
                             verbose=False,
                             gpt_verbose=False,
                             asci2gdf_bin='$ASCI2GDF_BIN',
                             kill_msgs=[]
                             ):
    """
    Run gpt with particles from ParticleGroup. 
    
        settings: dict with keys that are in gpt input file.    
        
    """

    unit_registry = UnitRegistry()
    
    # Call simpler evaluation if there is no input_particle_group:
    if (input_particle_group is None):
        raise ValueError('Must supply input_particle_group')
    
    if(verbose):
        print('Run GPT with ParticleGroup:') 

    if ('clipping_charge' in settings):
        raise ValueError('clipping_charge is deprecated, please specify value and units instead.')
    if ('final_charge' in settings):
        raise ValueError('final_charge is deprecated, please specify value and units instead.')    
    if (('t_restart' not in settings) and ('z_restart' not in settings)):
        raise ValueError('t_restart or z_restart must be supplied')
    if (('t_restart' in settings) and ('z_restart' in settings)):
        raise ValueError('Please use either t_restart or z_restart, not both')
                
    if ('restart_file' not in settings):
        # Make gpt and generator objects
        G = GPT(gpt_bin=gpt_bin, input_file=gpt_input_file, initial_particles=input_particle_group, workdir=workdir, use_tempdir=use_tempdir, parse_layout=False, kill_msgs=kill_msgs)
        G.timeout=timeout
        G.verbose = verbose


        # Set inputs
        if settings:
            for k, v in settings.items():
                G.set_variable(k,v)
        else:
            raise ValueError('Must supply settings')

        G.set_variable('multi_run',0)
        if(auto_phase): 

            if(verbose):
                print('\nAuto Phasing >------\n')
            t1 = time.time()

            # Create the distribution used for phasing
            if(verbose):
                print('****> Creating initial distribution for phasing...')

            phasing_beam = get_distgen_beam_for_phasing_from_particlegroup(input_particle_group, n_particle=10, verbose=verbose)
            phasing_particle_file = os.path.join(G.path, 'gpt_particles.phasing.gdf')
            write_gpt(phasing_beam, phasing_particle_file, verbose=verbose, asci2gdf_bin=asci2gdf_bin)

            if(verbose):
                print('<**** Created initial distribution for phasing.\n')    

            G.write_input_file()   # Write the unphased input file

            phased_file_name, phased_settings = gpt_phasing(G.input_file, path_to_gpt_bin=G.gpt_bin[:-3], path_to_phasing_dist=phasing_particle_file, verbose=verbose)
            G.set_variables(phased_settings)
            t2 = time.time()

            if(verbose):
                print(f'Time Ellapsed: {t2-t1} sec.')
                print('------< Auto Phasing\n')

        G.set_variable('multi_run',1)
        G.set_variable('last_run',2)
        G.set_variable('t_start', 0.0)
        
        if ('t_restart' in settings):
            G.set_variable('t_restart', settings['t_restart'])   
        if ('z_restart' in settings):
            G.set_variable('z_restart', settings['z_restart'])

        # If here, either phasing successful, or no phasing requested
        G.run(gpt_verbose=gpt_verbose)
    else:
        G = GPT()
        G.load_archive(settings['restart_file'])
        if settings:
            for k, v in settings.items():
                G.input['variables'][k]=v    # G.set_variable(k,v) does not add items to the dictionary, so just do this instead
                        
    if ('t_restart' in settings):
        # Remove touts and screens that are after restart point
        t_restart = settings['t_restart']
        t_restart_with_fudge = t_restart + 1.0e-18 # slightly larger that t_restart to avoid floating point comparison problem
        G.output['n_tout'] = np.count_nonzero(G.stat('mean_t', 'tout') <= t_restart_with_fudge)
        G.output['n_screen'] = np.count_nonzero(G.stat('mean_t', 'screen') <= t_restart_with_fudge)
        for p in reversed(G.particles):
            if (p['mean_t'] > t_restart_with_fudge):
                G.particles.remove(p)

        G_all = G  # rename it, and then overwrite G

        if (verbose):
            print(f'Looking for tout at t = {t_restart}')
        restart_particles = get_screen_data(G, tout_t = t_restart, use_extension=False, verbose=verbose)[0]
    else:
        # Remove screens after z_restart
        z_restart = settings['z_restart']
        z_restart_with_fudge = z_restart + 1.0e-9 # slightly larger that z_restart to avoid floating point comparison problem
        G.output['n_tout'] = np.count_nonzero(G.stat('mean_z', 'tout') <= z_restart_with_fudge)
        G.output['n_screen'] = np.count_nonzero(G.stat('mean_z', 'screen') <= z_restart_with_fudge)
        for p in reversed(G.particles):
            if (p['mean_z'] > z_restart_with_fudge):
                G.particles.remove(p)
                
        G_all = G  # rename it, and then overwrite G

        if (verbose):
            print(f'Looking for screen at z = {z_restart}')
        restart_particles = get_screen_data(G, screen_z = z_restart, use_extension=False, verbose=verbose)[0]
        
        t_restart = restart_particles['mean_t']
        restart_particles.drift_to_t(t_restart) # Change to an effective tout
        
    if (verbose):
        print(f'Found {len(restart_particles.x)} particles')
        
    if ('clipping_charge:value' in settings and 'clipping_charge:units' in settings):
        clipping_charge = settings['clipping_charge:value'] * unit_registry.parse_expression(settings['clipping_charge:units'])
        clipping_charge = clipping_charge.to('coulomb').magnitude
        restart_particles = clip_to_charge(restart_particles, clipping_charge, make_copy=False)
                
    if ('clipping_emit:value' in settings and 'clipping_emit:units' in settings):
        clipping_emit = settings['clipping_emit:value'] * unit_registry.parse_expression(settings['clipping_emit:units'])
        clipping_emit = clipping_emit.to('meter').magnitude
        restart_particles = clip_to_emit(restart_particles, clipping_emit, make_copy=False)
                
    if ('clipping_radius:value' in settings and 'clipping_radius:units' in settings):
        clipping_radius = settings['clipping_radius:value'] * unit_registry.parse_expression(settings['clipping_radius:units'])
        clipping_radius = clipping_radius.to('m').magnitude
        restart_particles = take_range(restart_particles, 'r', 0, clipping_radius, make_copy=False)
            
    G = GPT(gpt_bin=gpt_bin, input_file=gpt_input_file, initial_particles=restart_particles, workdir=workdir, use_tempdir=use_tempdir, parse_layout=False, kill_msgs=kill_msgs)
    G.timeout = timeout
    G.verbose = verbose

    for k, v in G_all.input["variables"].items():
        G.set_variable(k,v)
    
    G.set_variable('multi_run',2)
    G.set_variable('last_run',2)
    G.set_variable('t_start', t_restart)
    if (verbose):
        print('Starting second run of GPT.')
    G.run(gpt_verbose=gpt_verbose)
        
    G_all.output['particles'][G_all.output['n_tout']:G_all.output['n_tout']] = G.tout
    G_all.output['particles'] = G_all.output['particles'] + G.screen
    G_all.output['n_tout'] = G_all.output['n_tout']+G.output['n_tout']
    G_all.output['n_screen'] = G_all.output['n_screen']+G.output['n_screen']
    
    if ('final_charge:value' in settings and 'final_charge:units' in settings and len(G_all.screen)>0):
        final_charge = settings['final_charge:value'] * unit_registry.parse_expression(settings['final_charge:units'])
        final_charge = final_charge.to('coulomb').magnitude
        clip_to_charge(G_all.screen[-1], final_charge, make_copy=False)
        
    if ('final_emit:value' in settings and 'final_emit:units' in settings and len(G_all.screen)>0):
        final_emit = settings['final_emit:value'] * unit_registry.parse_expression(settings['final_emit:units'])
        final_emit = final_emit.to('meters').magnitude
        clip_to_emit(G_all.screen[-1], final_emit, make_copy=False)
        
    if (input_particle_group['sigma_t'] == 0.0):
        # Initial distribution is a tout
        if (G_all.output['n_tout'] > 0):
            # Don't include the cathode if there are no other screens. Screws up optimizations of "final" screen when there is an error
            G_all.output['particles'].insert(0, input_particle_group)
            G_all.output['n_tout'] = G_all.output['n_tout']+1
    else:
        # Initial distribution is a screen
        if (G_all.output['n_screen'] > 0):
            # Don't include the cathode if there are no other screens. Screws up optimizations of "final" screen when there is an error
            G_all.output['particles'].insert(G_all.output['n_tout'], input_particle_group)
            G_all.output['n_screen'] = G_all.output['n_screen']+1
        
    return G_all

    

def evaluate_multirun_gpt_with_particlegroup(settings,
                                             archive_path=None,
                                             merit_f=None, 
                                             gpt_input_file=None,
                                             distgen_input_file=None,
                                             workdir=None, 
                                             use_tempdir=True,
                                             gpt_bin='$GPT_BIN',
                                             timeout=2500,
                                             auto_phase=False,
                                             verbose=False,
                                             gpt_verbose=False,
                                             asci2gdf_bin='$ASCI2GDF_BIN'):    
    """
    Will raise an exception if there is an error. 
    """
    if ('final_charge' in settings and 'coreshield:core_charge_fraction' not in settings):
        settings['coreshield:core_charge_fraction'] = 0.5
        
    if ('coreshield' not in settings):
        input_particle_group = get_cathode_particlegroup(settings, distgen_input_file, verbose=verbose)
    else:
        input_particle_group = get_coreshield_particlegroup(settings, distgen_input_file, verbose=verbose)
    
    G = multirun_gpt_with_particlegroup(settings=settings,
                             gpt_input_file=gpt_input_file,
                             input_particle_group=input_particle_group,
                             workdir=workdir, 
                             use_tempdir=use_tempdir,
                             gpt_bin=gpt_bin,
                             timeout=timeout,
                             auto_phase=auto_phase,
                             verbose=verbose,
                             gpt_verbose=gpt_verbose,
                             asci2gdf_bin=asci2gdf_bin)
                        
    if merit_f:
        output = merit_f(G)
    else:
        output = default_gpt_merit(G)
    
    if ('merit:z' in settings.keys()):
        z_list = settings['merit:z']
        if (not isinstance(z_list, list)):
            z_list = [z_list]
        r_clip_list = None
        if ('merit:r_clip' in settings.keys()):
            r_clip_list = settings['merit:r_clip']
            if (not isinstance(r_clip_list, list)):
                r_clip_list = [r_clip_list]
        for ii, z in enumerate(z_list):
            g = copy.deepcopy(G)
            scr = get_screen_data(g, screen_z=z)[0]
            if (r_clip_list is not None):
                r_clip = r_clip_list[ii]
                take_range(scr, 'r', 0, r_clip)
            g.particles.clear()
            g.particles.insert(0,scr)
            g.output['n_tout'] = 0
            g.output['n_screen'] = 1
            if merit_f:
                g_output = merit_f(g)
            else:
                g_output = default_gpt_merit(g)
            for j in g_output.keys():
                if ('end_' in j):
                    output[j.replace('end_', f'merit:{z}_')] = g_output[j]
                
    if ('merit:peak_intensity_fraction' in settings.keys()):
        peak_intensity_fraction = settings['merit:peak_intensity_fraction']
        g = copy.deepcopy(G)
        scr = g.screen[-1]
        peak_radius = int(np.floor(scr.r.size * peak_intensity_fraction))
        r_sort = np.sort(scr.r)
        scr.weight[scr.r > r_sort[peak_radius]] = 0.0
        output['peak_intensity'] = 490206980 * scr.charge / (np.pi * r_sort[peak_radius]**2)
            
    if output['error']:
        raise ValueError('error occured!')
              
    return output




def run_gpt_with_particlegroup(settings=None,
                             gpt_input_file=None,
                             input_particle_group=None,
                             workdir=None, 
                             use_tempdir=True,
                             gpt_bin='$GPT_BIN',
                             timeout=2500,
                             auto_phase=False,
                             verbose=False,
                             gpt_verbose=False,
                             asci2gdf_bin='$ASCI2GDF_BIN',
                             kill_msgs=[]
                             ):
    """
    Run gpt with particles from ParticleGroup. 
    
        settings: dict with keys that are in gpt input file.    
        
    """

    # Call simpler evaluation if there is no input_particle_group:
    if (input_particle_group is None):
        return run_gpt(settings=settings, 
                       gpt_input_file=gpt_input_file, 
                       workdir=workdir,
                       use_tempdir=use_tempdir,
                       gpt_bin=gpt_bin, 
                       timeout=timeout, 
                       verbose=verbose,
                       kill_msgs=kill_msgs)
    
    if(verbose):
        print('Run GPT with ParticleGroup:') 

    unit_registry = UnitRegistry()
        
    if ('ignore_gpt_warnings' not in settings):
        settings['ignore_gpt_warnings'] = 0
        
    # Make gpt and generator objects
    if (settings['ignore_gpt_warnings'] == 1):
        # Allow things like particles with gamma > 1, etc etc, that normally LUME would kill immediately
        G = GPT(gpt_bin=gpt_bin, input_file=gpt_input_file, initial_particles=input_particle_group, workdir=workdir, use_tempdir=use_tempdir, parse_layout=False, kill_msgs=[])
    else:
        G = GPT(gpt_bin=gpt_bin, input_file=gpt_input_file, initial_particles=input_particle_group, workdir=workdir, use_tempdir=use_tempdir, parse_layout=False)
    G.timeout=timeout
    G.verbose = verbose

    # Set inputs
    if settings:
        for k, v in settings.items():
            G.set_variable(k,v)
            
    if ('final_charge' in settings):
        raise ValueError('final_charge is deprecated, please specify value and units instead.')
            
    # Run
    if(auto_phase): 

        if(verbose):
            print('\nAuto Phasing >------\n')
        t1 = time.time()

        # Create the distribution used for phasing
        if(verbose):
            print('****> Creating initial distribution for phasing...')

        phasing_beam = get_distgen_beam_for_phasing_from_particlegroup(input_particle_group, n_particle=10, verbose=verbose)
        phasing_particle_file = os.path.join(G.path, 'gpt_particles.phasing.gdf')
        write_gpt(phasing_beam, phasing_particle_file, verbose=verbose, asci2gdf_bin=asci2gdf_bin)
    
        if(verbose):
            print('<**** Created initial distribution for phasing.\n')    

        G.write_input_file()   # Write the unphased input file

        phased_file_name, phased_settings = gpt_phasing(G.input_file, path_to_gpt_bin=G.gpt_bin[:-3], path_to_phasing_dist=phasing_particle_file, verbose=verbose)
        G.set_variables(phased_settings)
        t2 = time.time()

        if(verbose):
            print(f'Time Ellapsed: {t2-t1} sec.')
            print('------< Auto Phasing\n')
            
    # If here, either phasing successful, or no phasing requested
    G.run(gpt_verbose=gpt_verbose)
    
    if ('final_charge:value' in settings and 'final_charge:units' in settings and len(G.screen)>0):
        final_charge = settings['final_charge:value'] * unit_registry.parse_expression(settings['final_charge:units'])
        final_charge = final_charge.to('coulomb').magnitude
        clip_to_charge(G.screen[-1], final_charge, make_copy=False)
        
    if ('final_emit:value' in settings and 'final_emit:units' in settings and len(G_all.screen)>0):
        final_emit = settings['final_emit:value'] * unit_registry.parse_expression(settings['final_emit:units'])
        final_emit = final_emit.to('meters').magnitude
        clip_to_emit(G_all.screen[-1], final_emit, make_copy=False)

    if ('final_radius:value' in settings and 'final_radius:units' in settings and len(G.screen)>0):
        final_radius = settings['final_radius:value'] * unit_registry.parse_expression(settings['final_radius:units'])
        final_radius = final_radius.to('meter').magnitude
        take_range(G.screen[-1], 'r', 0, final_radius)
    
    if (input_particle_group['sigma_t'] == 0.0):
        # Initial distribution is a tout
        if (G.output['n_tout'] > 0):
            G.output['particles'].insert(0, input_particle_group)
            G.output['n_tout'] = G.output['n_tout']+1
    else:
        # Initial distribution is a screen
        if (G.output['n_screen'] > 0):
            G.output['particles'].insert(G.output['n_tout'], input_particle_group)
            G.output['n_screen'] = G.output['n_screen']+1
    
    
    return G





def evaluate_run_gpt_with_particlegroup(settings,
                                         archive_path=None,
                                         merit_f=None, 
                                         gpt_input_file=None,
                                         distgen_input_file=None,
                                         workdir=None, 
                                         use_tempdir=True,
                                         gpt_bin='$GPT_BIN',
                                         timeout=2500,
                                         auto_phase=False,
                                         verbose=False,
                                         gpt_verbose=False,
                                         asci2gdf_bin='$ASCI2GDF_BIN',
                                         debug=False):    
    """
    Will raise an exception if there is an error. 
    """
    
    unit_registry = UnitRegistry()
    
    if (gpt_input_file is None):
        raise ValueError('You must specify the GPT input file')
        
    if (distgen_input_file is None):
        raise ValueError('You must specify the distgen input file')
    
    if ('final_charge' in settings and 'coreshield:core_charge_fraction' not in settings):
        settings['coreshield:core_charge_fraction'] = 0.5
        
    if ('final_n_particle' in settings and 'final_charge:value' in settings and 'final_charge:units' in settings and 'total_charge:value' in settings and 'total_charge:units' in settings):
        final_charge = settings['final_charge:value'] * unit_registry.parse_expression(settings['final_charge:units'])
        final_charge = final_charge.to('coulomb').magnitude
        total_charge = settings['total_charge:value'] * unit_registry.parse_expression(settings['total_charge:units'])
        total_charge = total_charge.to('coulomb').magnitude
        n_particle = int(np.ceil(settings['final_n_particle'] * total_charge / final_charge))
        settings['n_particle'] = int(np.max([n_particle, int(settings['final_n_particle'])]))
        if(verbose):
            print(f'<**** Setting n_particle = {n_particle}.\n')    
        
    if ('coreshield' not in settings):
        input_particle_group = get_cathode_particlegroup(settings, distgen_input_file, verbose=verbose)
    else:
        input_particle_group = get_coreshield_particlegroup(settings, distgen_input_file, verbose=verbose)

    G = run_gpt_with_particlegroup(settings=settings,
                         gpt_input_file=gpt_input_file,
                         input_particle_group=input_particle_group,
                         workdir=workdir, 
                         use_tempdir=use_tempdir,
                         gpt_bin=gpt_bin,
                         timeout=timeout,
                         auto_phase=auto_phase,
                         verbose=verbose,
                         gpt_verbose=gpt_verbose,
                         asci2gdf_bin=asci2gdf_bin)
        
    if merit_f:
        output = merit_f(G)
    else:
        output = default_gpt_merit(G)
    
    if ('merit:min' in settings.keys()):
        which_setting = settings['merit:min']
        z_list = np.array([scr['mean_z'] for scr in G.screen])
        merit_list = np.array([ParticleGroupExtension(scr)[which_setting] for scr in G.screen])
        
        merit_list = merit_list[z_list > 0.0]
        z_list = z_list[z_list > 0.0]
        
        settings['merit:z'] = z_list[np.argmin(merit_list)]
                

    if ('merit:z' in settings.keys()):
        z = settings['merit:z']
        g = copy.deepcopy(G)
        scr = get_screen_data(g, screen_z=z)[0]
        
        if ('final_charge:value' in settings and 'final_charge:units' in settings):
            final_charge = settings['final_charge:value'] * unit_registry.parse_expression(settings['final_charge:units'])
            final_charge = final_charge.to('coulomb').magnitude
            clip_to_charge(scr, final_charge, make_copy=False)
        
        g.particles.clear()
        g.particles.insert(0,scr)
        g.output['n_tout'] = 0
        g.output['n_screen'] = 1
        if merit_f:
            g_output = merit_f(g)
        else:
            g_output = default_gpt_merit(g)
        for j in g_output.keys():
            if ('end_' in j):
                output[j.replace('end_', f'merit:min_')] = g_output[j]
    
    
    if ('merit:peak_intensity_fraction' in settings.keys()):
        peak_intensity_fraction = settings['merit:peak_intensity_fraction']
        g = copy.deepcopy(G)
        scr = g.screen[-1]
        peak_radius = int(np.floor(scr.r.size * peak_intensity_fraction))
        r_sort = np.sort(scr.r)
        scr.weight[scr.r > r_sort[peak_radius]] = 0.0
        output['peak_intensity'] = 490206980 * scr.charge / (np.pi * r_sort[peak_radius]**2)
        
    if output['error']:
        raise ValueError('error occured!')
            
    return output


def evaluate_gpt_with_stability(settings,
                             archive_path=None,
                             merit_f=None, 
                             gpt_input_file=None,
                             distgen_input_file=None,
                             workdir=None, 
                             use_tempdir=True,
                             gpt_bin='$GPT_BIN',
                             timeout=2500,
                             auto_phase=False,
                             verbose=False,
                             gpt_verbose=False,
                             asci2gdf_bin='$ASCI2GDF_BIN',
                             plot_on=False):    
    """
    Will raise an exception if there is an error. 
    """
    unit_registry = UnitRegistry()
    
    random_state = np.random.get_state()
    np.random.seed(seed=6858)  # temporary seed to make the stability calculations reproducible
    
    if (gpt_input_file is None):
        raise ValueError('You must specify the GPT input file')
        
    if (distgen_input_file is None):
        raise ValueError('You must specify the distgen input file')
    
    input_particle_group = get_cathode_particlegroup(settings, distgen_input_file, verbose=verbose)
    
    G = run_gpt_with_particlegroup(settings=settings,
                         gpt_input_file=gpt_input_file,
                         input_particle_group=input_particle_group,
                         workdir=workdir, 
                         use_tempdir=use_tempdir,
                         gpt_bin=gpt_bin,
                         timeout=timeout,
                         auto_phase=auto_phase,
                         verbose=verbose,
                         gpt_verbose=gpt_verbose,
                         asci2gdf_bin=asci2gdf_bin)
            
    if merit_f:
        output = merit_f(G)
    else:
        output = default_gpt_merit(G)
        
    stability_settings = add_stability_settings(G, settings)
    
    if 'stability:n_runs' in settings:
        n_runs = settings['stability:n_runs']
    else:
        n_runs = 100
        
    arrival_t = np.empty(n_runs)
    arrival_t[:] = np.nan
    final_E = copy.copy(arrival_t)
        
    for ii in range(n_runs):
        auto_phase = False
        reduced_timeout = 20
        s = add_jitter_to_settings(stability_settings, verbose=verbose)
        stability_beam = get_distgen_beam_for_phasing_from_particlegroup(input_particle_group, n_particle=10, verbose=verbose, output_PG=True)
        
        G = run_gpt_with_particlegroup(s, gpt_input_file, input_particle_group=stability_beam, workdir=workdir, use_tempdir=use_tempdir, gpt_bin=gpt_bin, 
                                            timeout=reduced_timeout, auto_phase=auto_phase, verbose=verbose, gpt_verbose=gpt_verbose, asci2gdf_bin=asci2gdf_bin)
        arrival_t[ii] = G.stat('mean_t', 'screen')[-1]
        final_E[ii] = G.stat('mean_energy', 'screen')[-1]
        if ('stability:sigma_global_phase' in s):
            # global phase is used to mimic the laser, not an actual global shift, so we need to shift the time back by the global shift amount
            arrival_t[ii] = arrival_t[ii] + (s['global_phase'] - stability_settings['global_phase']) * 2.13675214e-12
        
    arrival_t = arrival_t - np.mean(arrival_t)
    final_E = final_E - np.mean(final_E)
        
    if (plot_on):
        plt.plot(final_E*1e-3, arrival_t*1e15, 'ro')
        plt.ylabel('Arrival time error (fs)')
        plt.xlabel('Energy error (kV)')
        plt.show()
        
    output['end_sigma_E_mean'] = np.std(final_E)
    output['end_avg_Et_mean'] = np.mean(final_E*arrival_t)
    output['end_sigma_E_combined'] = np.sqrt(output['end_sigma_E_mean']**2 + output['end_sigma_energy']**2)
    output['end_sigma_E_combined_fraction'] = output['end_sigma_E_combined']/output['end_mean_energy']
    
    output['end_sigma_t_mean'] = np.std(arrival_t)
    output['end_sigma_t_mean_slice'] = np.sqrt(output['end_sigma_t_mean']**2 - output['end_avg_Et_mean']**2 / output['end_sigma_E_mean']**2)
    output['end_sigma_t_combined'] = np.sqrt(output['end_sigma_t_mean']**2 + output['end_sigma_t']**2)
    output['end_sigma_t_combined_slice'] = np.sqrt(output['end_sigma_t_mean_slice']**2 + output['end_sigma_t']**2)
            
    np.random.set_state(random_state)   # return the RNG to what it was doing before this function seeded it
    
    if output['error']:
        raise ValueError('error occured!')
            
    return output


def evaluate_multirun_gpt_with_stability(settings,
                             archive_path=None,
                             merit_f=None, 
                             gpt_input_file=None,
                             distgen_input_file=None,
                             workdir=None, 
                             use_tempdir=True,
                             gpt_bin='$GPT_BIN',
                             timeout=2500,
                             auto_phase=False,
                             verbose=False,
                             gpt_verbose=False,
                             asci2gdf_bin='$ASCI2GDF_BIN',
                             plot_on=False):    
    """
    Will raise an exception if there is an error. 
    """
    unit_registry = UnitRegistry()
    
    random_state = np.random.get_state()
    np.random.seed(seed=6858)  # temporary seed to make the stability calculations reproducible
    
    if (gpt_input_file is None):
        raise ValueError('You must specify the GPT input file')
        
    if (distgen_input_file is None):
        raise ValueError('You must specify the distgen input file')
    
    input_particle_group = get_cathode_particlegroup(settings, distgen_input_file, verbose=verbose)
    
    G = multirun_gpt_with_particlegroup(settings=settings,
                         gpt_input_file=gpt_input_file,
                         input_particle_group=input_particle_group,
                         workdir=workdir, 
                         use_tempdir=use_tempdir,
                         gpt_bin=gpt_bin,
                         timeout=timeout,
                         auto_phase=auto_phase,
                         verbose=verbose,
                         gpt_verbose=gpt_verbose,
                         asci2gdf_bin=asci2gdf_bin)
            
    if merit_f:
        output = merit_f(G)
    else:
        output = default_gpt_merit(G)
        
    stability_settings = add_stability_settings(G, settings)
    
    if 'stability:n_runs' in settings:
        n_runs = settings['stability:n_runs']
    else:
        n_runs = 100
        
    arrival_t = np.empty(n_runs)
    arrival_t[:] = np.nan
    final_E = copy.copy(arrival_t)
        
    for ii in range(n_runs):
        auto_phase = False
        reduced_timeout = 20
        s = add_jitter_to_settings(stability_settings, verbose=verbose)
        stability_beam = get_distgen_beam_for_phasing_from_particlegroup(input_particle_group, n_particle=10, verbose=verbose, output_PG=True)
        
        G = run_gpt_with_particlegroup(s, gpt_input_file, input_particle_group=stability_beam, workdir=workdir, use_tempdir=use_tempdir, gpt_bin=gpt_bin, 
                                            timeout=reduced_timeout, auto_phase=auto_phase, verbose=verbose, gpt_verbose=gpt_verbose, asci2gdf_bin=asci2gdf_bin)
        arrival_t[ii] = G.stat('mean_t', 'screen')[-1]
        final_E[ii] = G.stat('mean_energy', 'screen')[-1]
        if ('stability:sigma_global_phase' in s):
            # global phase is used to mimic the laser, not an actual global shift, so we need to shift the time back by the global shift amount
            arrival_t[ii] = arrival_t[ii] + (s['global_phase'] - stability_settings['global_phase']) * 2.13675214e-12
        
    arrival_t = arrival_t - np.mean(arrival_t)
    final_E = final_E - np.mean(final_E)
        
    if (plot_on):
        plt.plot(final_E*1e-3, arrival_t*1e15, 'ro')
        plt.ylabel('Arrival time error (fs)')
        plt.xlabel('Energy error (kV)')
        plt.show()
        
    output['end_sigma_E_mean'] = np.std(final_E)
    output['end_avg_Et_mean'] = np.mean(final_E*arrival_t)
    output['end_sigma_E_combined'] = np.sqrt(output['end_sigma_E_mean']**2 + output['end_sigma_energy']**2)
    output['end_sigma_E_combined_fraction'] = output['end_sigma_E_combined']/output['end_mean_energy']
    
    output['end_sigma_t_mean'] = np.std(arrival_t)
    output['end_sigma_t_mean_slice'] = np.sqrt(output['end_sigma_t_mean']**2 - output['end_avg_Et_mean']**2 / output['end_sigma_E_mean']**2)
    output['end_sigma_t_combined'] = np.sqrt(output['end_sigma_t_mean']**2 + output['end_sigma_t']**2)
    output['end_sigma_t_combined_slice'] = np.sqrt(output['end_sigma_t_mean_slice']**2 + output['end_sigma_t']**2)
            
    np.random.set_state(random_state)   # return the RNG to what it was doing before this function seeded it
    
    if output['error']:
        raise ValueError('error occured!')
                    
    return output


def add_jitter_to_settings(settings_input, verbose=False):    
    settings = copy.deepcopy(settings_input)
    
    for k in settings.keys():
        if ('stability:sigma_' in k):
            sigma = settings[k]
            setting_name = k.replace('stability:sigma_', '')
            original_value = settings[setting_name]
            new_value = original_value + sigma * np.random.randn()
            settings[setting_name] = new_value
            if verbose:
                print(f'Changing {setting_name} from {original_value} -> {new_value}')
        if ('stability:relative_sigma_' in k):
            rel_sigma = settings[k]
            setting_name = k.replace('stability:relative_sigma_', '')
            original_value = settings[setting_name]
            new_value = original_value*(1.0 + rel_sigma * np.random.randn())
            settings[setting_name] = new_value
            if verbose:
                print(f'Changing {setting_name} from {original_value} -> {new_value}')
    
    return settings
    

def add_stability_settings(gpt_data_input, settings_input):
    settings = copy.deepcopy(settings_input)
    gpt_data = copy.deepcopy(gpt_data_input)
        
    unit_registry = UnitRegistry()
    
    # Add phasing settings
    lines = gpt_data.input['lines']
    vars = [];
    for line in lines:
        if('phasing_on_crest' in line):
            var_name = (line.split('=')[1]).split(';')[0].strip()
            vars.append(var_name)
    for var in vars:
        for line in lines:
            if (re.match(rf'\s*{var}\s*=.*',line) is not None):
                var_name = (line.split('=')[0]).strip()
                val = float((line.split('=')[1]).split(';')[0].strip())
                # print(f'settings[\'{var_name}\'] = {val}')
                settings[var_name] = val
                
    # If 'final_charge' is used, replace it with a fixed aperture size
    if ('final_charge:value' in settings and 'final_charge:units' in settings and len(gpt_data.screen)>0):
        final_charge = settings['final_charge:value'] * unit_registry.parse_expression(settings['final_charge:units'])
        final_charge = final_charge.to('coulomb').magnitude
        
        final_screen = gpt_data.screen[-1]
        r_clip = radius_including_charge(final_screen, final_charge)
        
        del(settings['final_charge:value'])
        del(settings['final_charge:units'])
        settings['final_radius:value'] = r_clip
        settings['final_radius:units'] = 'm'
    
    # If 'final_emit' is used, replace it with a fixed aperture size
    if ('final_emit:value' in settings and 'final_emit:units' in settings and len(gpt_data.screen)>0):
        final_emit = settings['final_emit:value'] * unit_registry.parse_expression(settings['final_emit:units'])
        final_emit = final_emit.to('meter').magnitude
        
        final_screen = gpt_data.screen[-1]
        r_clip = radius_including_emit(final_screen, final_emit)
        
        del(settings['final_emit:value'])
        del(settings['final_emit:units'])
        settings['final_radius:value'] = r_clip
        settings['final_radius:units'] = 'm'
    
    # Turn off space charge
    settings['space_charge'] = 0
    
    return settings


def radius_including_charge(PG_input, clipping_charge):
    PG = copy.deepcopy(PG_input)
    
    min_final_particles = 3
    
    r_i = np.argsort(PG.r)
    r = PG.r[r_i]
    w = PG.weight[r_i]
    w_sum = np.cumsum(w)
    if (clipping_charge >= w_sum[-1]):
        n_clip = -1
    else:
        n_clip = np.argmax(w_sum > clipping_charge)
    if (n_clip < (min_final_particles-1) and n_clip > -1):
        n_clip = min_final_particles-1
    r_cut = r[n_clip]
    return r_cut


def radius_including_emit(PG_input, clipping_emit):
    PG = ParticleGroupExtension(copy.deepcopy(PG_input))
    
    min_final_particles = 3
    
    r_i = np.argsort(PG.r)
    r = PG.r[r_i]
    emit_i = np.zeros(len(r_i))
    emit_i[len(r_i)-1] = PG.sqrt_norm_emit_4d

    for ii in np.arange(len(r_i)-1,min_final_particles,-1):
        PG.weight[r_i[ii]] = 0
        emit_i[ii-1] = PG.sqrt_norm_emit_4d
    
    if (clipping_emit >= emit_i[-1]):
        n_clip = -1
    else:
        n_clip = np.argmax(emit_i > clipping_emit)
    if (n_clip < (min_final_particles-1) and n_clip > -1):
        n_clip = min_final_particles-1
    
    r_cut = r[n_clip]
    return r_cut


def get_distgen_beam_for_phasing_from_particlegroup(PG, n_particle=10, verbose=False, output_PG = False):

    variables = ['x', 'y', 'z', 'px', 'py', 'pz', 't']

    transforms = { f'avg_{var}':{'type': f'set_avg {var}', f'avg_{var}': { 'value': PG['mean_'+var], 'units':  PG.units(var).unitSymbol  } } for var in variables }

    phasing_distgen_input = {'n_particle':n_particle, 'random_type':'hammersley', 'transforms':transforms,
                             'total_charge':{'value':1.0, 'units':'pC'},
                             'start': {'type':'time', 'tstart':{'value': 0.0, 'units': 's'}},}
    
    gen = Generator(phasing_distgen_input, verbose=verbose) 
    
    if (not output_PG):
        pbeam = gen.beam()
        return pbeam
    else:
        gen.run()
        PG = gen.particles
        return PG

