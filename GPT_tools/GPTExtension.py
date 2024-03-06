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


def evaluate_run_gpt_with_settings(settings,
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
    
    G = run_gpt_with_settings(settings=settings,
                         gpt_input_file=gpt_input_file,
                         distgen_input_file=distgen_input_file,
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
        
    # Add beginning screen
    g = copy.deepcopy(G)
    z_list = g.stat('mean_z', 'screen')
    min_z = np.min(z_list[z_list>0.0])

    scr = get_screen_data(g, screen_z=min_z)[0]

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
            output[j.replace('end_', 'pre_merit_')] = g_output[j]
        
    # Add merit_z screen
    ran_settings = copy.copy(G.input['variables'])

    if ('merit:z' in ran_settings.keys()):
        z = ran_settings['merit:z']
        g = copy.deepcopy(G)
        scr = get_screen_data(g, screen_z=z)[0]
        
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
            
    return output


def run_gpt_with_settings(settings=None,
                             gpt_input_file=None,
                             distgen_input_file=None,
                             input_particle_group=None,  # use either distgen file or particle group, not both
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

    unit_registry = UnitRegistry()
    
    if settings is None:
        raise ValueError('Must supply settings')
    
    if (gpt_input_file is None):
        raise ValueError('You must specify the GPT input file')
        
    if (distgen_input_file is None and input_particle_group is None):
        raise ValueError('You must specify the distgen input file or provide a particle group')
            
    if (input_particle_group is not None and distgen_input_file is not None):
        raise ValueError('Use either distgen input file or input particle group, not both')
            
    if (input_particle_group is None):
        # Modify settings for input particlegroup as needed
        if ('final_n_particle' in settings and 'final_charge:value' in settings and 'final_charge:units' in settings and 'total_charge:value' in settings and 'total_charge:units' in settings):
            # user specifies final n_particles, rather than initial
            final_charge = settings['final_charge:value'] * unit_registry.parse_expression(settings['final_charge:units'])
            final_charge = final_charge.to('coulomb').magnitude
            total_charge = settings['total_charge:value'] * unit_registry.parse_expression(settings['total_charge:units'])
            total_charge = total_charge.to('coulomb').magnitude
            n_particle = int(np.ceil(settings['final_n_particle'] * total_charge / final_charge))
            settings['n_particle'] = int(np.max([n_particle, int(settings['final_n_particle'])]))
            if(verbose):
                print(f'<**** Setting n_particle = {n_particle}.\n')    

        # Make initial distribution
        input_particle_group = get_cathode_particlegroup(settings, distgen_input_file, verbose=verbose)
    
    # Check restart parameters self-consistency
    if (('t_restart' in settings) and ('z_restart' in settings)):
        raise ValueError('Please use either t_restart or z_restart, not both')
    
    if ('restart_file' not in settings):
        # Starting from scratch, either single or multiple passes
        
        # Make gpt and generator objects
        G = GPT(gpt_bin=gpt_bin, input_file=gpt_input_file, initial_particles=input_particle_group, workdir=workdir, use_tempdir=use_tempdir, parse_layout=False, kill_msgs=kill_msgs)
        G.timeout=timeout
        G.verbose = verbose

        # Put settings into GPT object
        for k, v in settings.items():
            G.input['variables'][k]=v

        if(auto_phase): 
            # Zeroth pass = auto-phasing
            G.input['variables']['multi_pass']=0
            
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
            
            # Put phased settings into GPT object
            G.set_variables(phased_settings) # Note: G.set_variable(k,v) does not add items to the dictionary, not sure about this form of the function
            t2 = time.time()

            if(verbose):
                print(f'Time Ellapsed: {t2-t1} sec.')
                print('------< Auto Phasing\n')

        # First (or only) pass
        G.input['variables']['multi_pass']=1
        G.input['variables']['last_pass']=2
        G.input['variables']['t_start']=0.0
        
        G.run(gpt_verbose=gpt_verbose)
    else:
        # Restarting from file
        G = GPT()
        G.load_archive(settings['restart_file'])
        for k, v in settings.items():
            G.input['variables'][k]=v
    
    if (('t_restart' in settings) or ('z_restart' in settings)):
        # Run second pass
        
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
            restart_particles.drift_to_t(t_restart) # Change to an effective tout... even though this almost always a bad idea

        # Do second GPT call
        G = GPT(gpt_bin=gpt_bin, input_file=gpt_input_file, initial_particles=restart_particles, workdir=workdir, use_tempdir=use_tempdir, parse_layout=False, kill_msgs=kill_msgs)
        G.timeout = timeout
        G.verbose = verbose

        # Put settings in new GPT object
        for k, v in G_all.input["variables"].items():
            G.input['variables'][k]=v

        G.input['variables']['multi_pass']=2
        G.input['variables']['last_pass']=2
        G.input['variables']['t_start']=t_restart

        if (verbose):
            print('Starting second run of GPT.')
        G.run(gpt_verbose=gpt_verbose)

        G_all.output['particles'][G_all.output['n_tout']:G_all.output['n_tout']] = G.tout
        G_all.output['particles'] = G_all.output['particles'] + G.screen
        G_all.output['n_tout'] = G_all.output['n_tout']+G.output['n_tout']
        G_all.output['n_screen'] = G_all.output['n_screen']+G.output['n_screen']
    else:
        # Just a single run is needed
        G_all = G

    used_filter = False
        
    # Filter screens (clip, remove, etc.) depending on settings
    if ('merit:min' in settings.keys()):
        if (verbose):
            print('Finding screen for merit:min...')
        # user wants to find a screen with a minimum value of a parameter        
        G_merit_min = copy.deepcopy(G_all)
        which_setting = settings['merit:min']
        z_list = np.array([scr['mean_z'] for scr in G_merit_min.screen])
        index_list = np.arange(0, len(z_list))
        for scr in G_merit_min.screen:
            # if requested, filter at this screen
            filter_screen(scr, settings)
        merit_list = np.array([ParticleGroupExtension(scr)[which_setting] for scr in G_merit_min.screen])
        
        merit_list = merit_list[z_list > 0.0]
        index_list = index_list[z_list > 0.0]
        z_list = z_list[z_list > 0.0]
        
        if ('merit:z_min' in settings.keys()):
            merit_list = merit_list[z_list >= settings['merit:z_min']]
            index_list = index_list[z_list >= settings['merit:z_min']]
            z_list = z_list[z_list >= settings['merit:z_min']]
            
        if ('merit:z_max' in settings.keys()):
            merit_list = merit_list[z_list <= settings['merit:z_max']]
            index_list = index_list[z_list <= settings['merit:z_max']]
            z_list = z_list[z_list <= settings['merit:z_max']]
        
        z_ii = np.argmin(merit_list)
        ii = index_list[z_ii]
        settings['merit:z'] = z_list[z_ii]  # set a merit:z so that screens will be removed if filtering was used
        G_all.input['variables']['merit:z'] = settings['merit:z']  # also update the settings in the GPT object so that the evaluate function can use it
        if (verbose):
            print(f'Found z = {z_list[z_ii]}')
        
        # if requested, filter at this screen
        used_filter = filter_screen(G_all.screen[ii], settings)
        
        if (verbose):
            print('Finished.')
    
    if ('merit:z' in settings.keys()):
        z = settings['merit:z']
                
        if (used_filter):
            # All screens after z are now incorrect, and should be removed
            n_tout = len(G_all.stat('mean_z', 'tout') <= z + 1.0e-9) # slightly larger that z to avoid floating point comparison problem
            n_screen = len(G_all.stat('mean_z', 'screen') <= z + 1.0e-9) # slightly larger that z to avoid floating point comparison problem

            G_all.output['particles'] = list(filter(lambda dummy: dummy['mean_z'] <= z + 1.0e-9, G_all.output['particles']))

            G_all.output['n_tout'] = n_tout
            G_all.output['n_screen'] = n_screen
    
    if (len(G_all.screen)>0 and used_filter == False):
        # Filter last screen as needed if nothing has been done yet
        used_filter = filter_screen(G_all.screen[-1], settings)
        
    # insert initial particle distribution into list of screens or touts
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

    
def filter_screen(scr, settings):
    unit_registry = UnitRegistry()
    
    filtered = False
    
    clip_particles = True
    if ('clip_final_particles' in settings):
        clip_particles = settings['clip_final_particles'] == 1
        
    if (clip_particles):
        if ('final_charge:value' in settings and 'final_charge:units' in settings):
            final_charge = settings['final_charge:value'] * unit_registry.parse_expression(settings['final_charge:units'])
            final_charge = final_charge.to('coulomb').magnitude
            clip_to_charge(scr, final_charge, verbose=False, make_copy=False)
            filtered = True

        if ('final_emit:value' in settings and 'final_emit:units' in settings):
            final_emit = settings['final_emit:value'] * unit_registry.parse_expression(settings['final_emit:units'])
            final_emit = final_emit.to('meters').magnitude
            clip_to_emit(scr, final_emit, verbose=False, make_copy=False)
            filtered = True
        
    return filtered



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
        for k in error_output.keys():
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
        
    #if output['error']:
    #    raise ValueError('error occured!')
            
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

