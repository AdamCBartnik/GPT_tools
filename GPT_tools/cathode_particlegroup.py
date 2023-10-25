import yaml, copy
import numpy as np
from distgen import Generator
from distgen.tools import update_nested_dict
from pint import UnitRegistry

def get_cathode_particlegroup(settings_input, DISTGEN_INPUT_FILE, verbose=False, distgen_verbose=False, id_start=1):
    unit_registry = UnitRegistry()
    settings = copy.copy(settings_input)
    
    distgen_input = yaml.safe_load(open(DISTGEN_INPUT_FILE))
    for k, v in settings.items():
        distgen_input = update_nested_dict(distgen_input, {k:v}, verbose=verbose, create_new=False)
    gen = Generator(distgen_input,verbose=distgen_verbose)
    gen.run()
    PG = gen.particles
    
    if ('cathode:sigma_xy' in settings):
        raise ValueError('cathode:sigma_xy is deprecated, please specify value and units instead.')
    if ('cathode:sigma_xy:value' in settings and 'cathode:sigma_xy:units' in settings):
        sigma_xy_value = settings.pop('cathode:sigma_xy:value') # remove from dictionary to avoid recursion problem
        sigma_xy_units = settings.pop('cathode:sigma_xy:units') # remove from dictionary to avoid recursion problem
        sigma_xy = sigma_xy_value * unit_registry.parse_expression(sigma_xy_units)
        sigma_xy = sigma_xy.to('m').magnitude # convert to meters
        
        sx_orig = 0.5*(PG['sigma_x'] + PG['sigma_y'])
        sig_ratio = sigma_xy/sx_orig
        settings_1 = copy.copy(settings)
        
        var_list = ['r_dist:sigma_xy:value', 'r_dist:truncation_radius_right:value', 'r_dist:truncation_radius_left:value']
        for var in var_list:
            if (var in settings):
                settings_1[var] = settings[var] * sig_ratio
        PG = get_cathode_particlegroup(settings_1, DISTGEN_INPUT_FILE, verbose=verbose, distgen_verbose=distgen_verbose, id_start=id_start)
        if (verbose):
            print(f'Rescaling sigma_xy from {sx_orig} -> {sigma_xy}. Achieved: {PG["sigma_x"]}')
        return PG
    
    PG.assign_id()
    PG.id = np.arange(id_start,id_start+gen['n_particle'])
    
    return PG

    

def get_coreshield_particlegroup(settings_input, DISTGEN_INPUT_FILE, verbose=False, distgen_verbose=False):
    unit_registry = UnitRegistry()
    settings = copy.copy(settings_input)
    
    all_settings = yaml.safe_load(open(DISTGEN_INPUT_FILE))
    for k, v in settings.items():
        all_settings = update_nested_dict(all_settings, {k:v}, verbose=False, create_new=True)
    
    if ('coreshield' not in all_settings):
        raise ValueError('No coreshield settings specificed.')
    
    coreshield_settings = all_settings['coreshield']
    
    if ('n_core' in coreshield_settings):
        n_core = coreshield_settings['n_core']
    else:
        raise ValueError('Please specify n_core.')
    
    if ('n_shield' in coreshield_settings):
        n_shield = coreshield_settings['n_shield']
    else:
        raise ValueError('Please specify n_shield.')
        
    if ('core_charge_fraction' in coreshield_settings):
        raise ValueError('core_charge_fraction is deprecated, please use core_charge instead.')
    if ('core_charge' not in coreshield_settings):
        if ('final_charge' in all_settings):
            if (verbose):
                print('Defaulting to core charge = final charge.')
            core_charge = all_settings['final_charge']['value'] * unit_registry.parse_expression(all_settings['final_charge']['units'])
            core_charge_fraction = core_charge.to(all_settings['total_charge']['units']).magnitude / all_settings['total_charge']['value']
        else:
            if (verbose):
                print('Defaulting to half of the charge in the core.')
            core_charge_fraction = 0.5
    else:
        core_charge = coreshield_settings['core_charge']['value'] * unit_registry.parse_expression(coreshield_settings['core_charge']['units'])
        core_charge_fraction = core_charge.to(all_settings['total_charge']['units']).magnitude / all_settings['total_charge']['value']
    if (core_charge_fraction < 0.0 or core_charge_fraction > 1.0):
        core_charge_fraction = 0.5
        if (verbose):
            print('Invalid core charge fraction, defaulting to 0.5')
    
    sigma_xy = None
    if ('cathode:sigma_xy' in settings):
        raise ValueError('cathode:sigma_xy is deprecated, please specify value and units instead.')
    if ('cathode:sigma_xy:value' in settings and 'cathode:sigma_xy:units' in settings):
        sigma_xy_value = settings.pop('cathode:sigma_xy:value') # Remove from dictionary so that calls to get_cathode_particlegroup do not see it
        sigma_xy_units = settings.pop('cathode:sigma_xy:units') # Remove from dictionary so that calls to get_cathode_particlegroup do not see it
        sigma_xy = sigma_xy_value * unit_registry.parse_expression(sigma_xy_units)
        sigma_xy = sigma_xy.to('m').magnitude # convert to meters
    
    PG = get_cathode_particlegroup(settings, DISTGEN_INPUT_FILE, verbose=False)
        
    sx = PG['sigma_x']
    sig_ratio = 1.0
    if (sigma_xy is not None):
        sig_ratio = sigma_xy/sx
    r_i = np.argsort(PG.r)
    r = PG.r[r_i]
    w = PG.weight[r_i]
    w_sum = np.cumsum(w)
    n_core_orig = np.argmax(w_sum > core_charge_fraction * w_sum[-1])
    n_shield_orig = len(w)-n_core_orig
    r_cut = r[n_core_orig]
    
    if (n_core is None):
        n_core = n_core_orig
        
    if (n_shield is None):
        n_shield = n_shield_orig
        
    settings_1 = copy.copy(settings)
    settings_1['r_dist:sigma_xy:value'] = sig_ratio*settings['r_dist:sigma_xy:value']
    settings_1['r_dist:truncation_radius_right:value'] = sig_ratio*settings['r_dist:truncation_radius_right:value']
    settings_1['r_dist:truncation_radius_left:value'] = sig_ratio*r_cut
    settings_1['r_dist:truncation_radius_left:units'] = 'm'
    settings_1['n_particle'] = n_shield
    settings_1['total_charge:value'] = (1.0-core_charge_fraction) * settings['total_charge:value']
    
    settings_2 = copy.copy(settings)
    settings_2['r_dist:sigma_xy:value'] = sig_ratio*settings['r_dist:sigma_xy:value']
    settings_2['r_dist:truncation_radius_right:value'] = sig_ratio*r_cut
    settings_2['r_dist:truncation_radius_right:units'] = 'm'
    settings_2['r_dist:truncation_radius_left:value'] = sig_ratio*settings['r_dist:truncation_radius_left:value']
    settings_2['n_particle'] = n_core
    settings_2['total_charge:value'] = core_charge_fraction * settings['total_charge:value']
    
    PG_shield = get_cathode_particlegroup(settings_1, DISTGEN_INPUT_FILE, verbose=False, id_start=n_core+1)
    PG_core = get_cathode_particlegroup(settings_2, DISTGEN_INPUT_FILE, verbose=False, id_start=1)
    
    PG = PG_core+PG_shield
        
    if (verbose):
        if (sigma_xy is not None):
            print(f'Rescaling sigma_xy from {sx} -> {sigma_xy}. Achieved: {PG["sigma_x"]}')
        
    return PG



