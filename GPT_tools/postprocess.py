import numpy as np
import copy
from .ParticleGroupExtension import ParticleGroupExtension, divide_particles
import numpy.polynomial.polynomial as poly

def postprocess_screen(screen, **params):
    need_copy_params = ['take_slice', 'take_range', 'cylindrical_copies', 'remove_correlation', 'kill_zero_weight',
                        'remove_spinning', 'include_ids', 'random_N', 'first_N', 'clip_to_charge', 'clip_to_emit']
    need_copy = any([p in params for p in need_copy_params])
    
    if ('need_copy' in params):
        need_copy = params['need_copy']
    
    if (need_copy):
        screen = copy.deepcopy(screen)
    
    if ('kill_zero_weight' in params):
        if (params['kill_zero_weight']):
            screen = kill_zero_weight(screen, make_copy=False)
    
    if ('include_ids' in params):
        ids = params['include_ids']
        if (len(ids) > 0):
            screen = include_ids(screen, ids, make_copy=False)
    
    if ('take_range' in params):
        (take_range_var, range_min, range_max) = params['take_range']
        if (range_min < range_max):
            screen = take_range(screen, take_range_var, range_min, range_max, make_copy=False)
    
    if ('take_slice' in params):
        (take_slice_var, slice_index, n_slices) = params['take_slice']
        if (n_slices > 1):
            screen = take_slice(screen, take_slice_var, slice_index, n_slices, make_copy=False)
            
    if ('clip_to_charge' in params):
        target_charge = params['clip_to_charge']
        if (target_charge > 0):
            screen = clip_to_charge(screen, target_charge, verbose=False, make_copy=False)
    
    if ('clip_to_emit' in params):
        target_emit = params['clip_to_emit']
        if (target_emit > 0):
            screen = clip_to_emit(screen, target_emit, verbose=False, make_copy=False)
            
    if ('cylindrical_copies' in params):
        cylindrical_copies_n = params['cylindrical_copies']
        if (cylindrical_copies_n > 0):
            screen = add_cylindrical_copies(screen, params['cylindrical_copies'], make_copy=False)
            
    if ('remove_spinning' in params):
        if (params['remove_spinning']):
            screen = remove_spinning(screen, make_copy=False)
               
    if ('remove_correlation' in params):
        (remove_correlation_var1, remove_correlation_var2, remove_correlation_n) = params['remove_correlation']
        if (remove_correlation_n >= 0):
            screen = remove_correlation(screen, remove_correlation_var1, remove_correlation_var2, remove_correlation_n, make_copy=False)
    
    if ('random_N' in params):
        N = params['random_N']
        if (N > 0):
            screen = random_N(screen, N, random=True, make_copy=False)
    else:
        if ('first_N' in params):
            N = params['first_N']
            if (N > 0):
                screen = random_N(screen, N, random=False, make_copy=False)
        
    return screen


# Returns IDs of the N nearest particles to center_particle_id in the ndim dimensional phase space
# "Nearest" is determined by changing coordinates to ones with sigma_matrix = identity_matrix
def id_of_nearest_N(screen_input, center_particle_id, N, ndim=4):
    screen = copy.deepcopy(screen_input)
    
    if (ndim == 6):
        screen.drift_to_t()
    
    x = screen.x
    px = screen.px
    w = screen.weight
    pid = screen.id
    
    if (center_particle_id not in pid):
        print('Cannot find center particle')
        return np.array([])
    
    if (ndim == 2):
        x = x - np.sum(x*w)/np.sum(w)
        px = px - np.sum(px*w)/np.sum(w)
        u0 = np.vstack((x, px))
    if (ndim == 4):
        y = screen.y
        py = screen.py
        x = x - np.sum(x*w)/np.sum(w)
        px = px - np.sum(px*w)/np.sum(w)
        y = y - np.sum(y*w)/np.sum(w)
        py = py - np.sum(py*w)/np.sum(w)
        u0 = np.vstack((x, px, y, py))
    if (ndim == 6):
        y = screen.y
        py = screen.py
        z = screen.z
        pz = screen.pz
        
        x = x - np.sum(x*w)/np.sum(w)
        px = px - np.sum(px*w)/np.sum(w)
        y = y - np.sum(y*w)/np.sum(w)
        py = py - np.sum(py*w)/np.sum(w)
        z = z - np.sum(z*w)/np.sum(w)
        pz = pz - np.sum(pz*w)/np.sum(w)
        u0 = np.vstack((x, px, y, py, z, pz))
    
    sigma_matrix = np.cov(u0, aweights=w)
            
    # Change into round phase space coordinates
    (E, V) = np.linalg.eigh(sigma_matrix)
    u1 = np.diag(1.0/np.sqrt(E)) @ np.linalg.solve(V, u0)
        
    u1_cen = u1[:, pid == center_particle_id]
    d = np.sum((u1 - u1_cen)**2, 0)
    sorted_index = np.argsort(d)
        
    return pid[sorted_index[0:N]]
    

    
# Returns a screen with either only the first N or a random N particles remaining
def random_N(screen, N, random=True, make_copy=False, seed=None):
    alive_ids = screen.id[screen.weight > 0]
    if (random):
        alive_ids = np.random.default_rng(seed).permutation(alive_ids)  # pass seed for a reproducible selection
    if (N < len(alive_ids)):
        alive_ids = alive_ids[0:N]
    return include_ids(screen, alive_ids, make_copy)


# Returns a screen with only the particles with id = ids remaining
def include_ids(screen_input, ids, make_copy=False):
    if (make_copy==True):
        screen = copy.deepcopy(screen_input)
    else:
        screen = screen_input
    screen.weight[np.logical_not(np.isin(screen.id, ids))] = 0.0
    
    return kill_zero_weight(screen, make_copy=False)


# Removes the rotation (angular momentum about the beam centroid) from particles spinning in a solenoid.
# Only the antisymmetric part of the x-py, y-px correlation is removed, so e.g. skew-quad coupling is kept.
def remove_spinning(screen_input, make_copy=False):
    if (make_copy==True):
        screen = copy.deepcopy(screen_input)
    else:
        screen = screen_input
    w = screen.weight
    sumw = np.sum(w)

    x = screen.x - np.sum(screen.x*w)/sumw
    px = screen.px - np.sum(screen.px*w)/sumw
    y = screen.y - np.sum(screen.y*w)/sumw
    py = screen.py - np.sum(screen.py*w)/sumw

    u2 = 0.5*(np.sum(x*x*w) + np.sum(y*y*w))/sumw
    L = 0.5*(np.sum(x*py*w) - np.sum(y*px*w))/sumw    # angular momentum per particle / 2, about the centroid

    # Rigid-rotation kick about the centroid: px -> px + (L/u2)*y, py -> py - (L/u2)*x makes <x py - y px> = 0
    # without moving the centroid (same form as in core_emit_calc_4d)
    C = L/u2
    screen.px = screen.px + C*y
    screen.py = screen.py - C*x

    return screen


# Removes particles that have zero weight from the distribution
def kill_zero_weight(screen_input, make_copy=False):
    w = screen_input.weight
    
    if (make_copy==False):
        for k in screen_input._settable_array_keys:
            screen_input.data[k] = screen_input[k][w>0]
        new_screen = screen_input
    else:
        data = {}
        for k in screen_input._settable_array_keys:
            data[k] = screen_input[k][w>0]

        for k in screen_input._settable_scalar_keys:
            data[k] = screen_input[k]

        new_screen = ParticleGroupExtension(data=data)

    return new_screen
    

# Removes particles that are outside of a given range of a variable
def take_range(screen_input, take_range_var, range_min, range_max, make_copy=False):
    if (make_copy==True):
        screen = copy.deepcopy(screen_input)
    else:
        screen = screen_input
    x = getattr(screen, take_range_var) 
    
    if (take_range_var in ['x','y','z','t']):
        # Subtract mean
        x = x - np.sum(x*screen.weight)/np.sum(screen.weight)
    
    out_of_range = np.logical_or(x < range_min, x > range_max)

    if (np.count_nonzero(out_of_range) < len(out_of_range)):
        screen.weight[out_of_range] = 0.0
    else:
        print(f'take_range: no particles with {take_range_var} in [{range_min:G}, {range_max:G}], range not applied')

    return kill_zero_weight(screen, make_copy=False)

    
# Takes n_slices slices over the full range of the variable take_slice_var, and then returns a screen with the particles in the slice_index'th slice
def take_slice(screen_input, take_slice_var, slice_index, n_slices, make_copy=False):
    if (make_copy==True):
        screen = copy.deepcopy(screen_input)
    else:
        screen = screen_input
    p_list, edges, density_norm = divide_particles(screen, nbins=n_slices, key=take_slice_var)
    if (slice_index>=0 and slice_index<len(p_list)):
        return p_list[slice_index]
    else:
        return screen


# Removes a polynomial correlation in the var1-var2 phase space. Subtracts from var2 to remove correlation.
def remove_correlation(screen_input, var1, var2, max_power, make_copy=False):
    if (make_copy==True):
        screen = copy.deepcopy(screen_input)
    else:
        screen = screen_input
    
    x = getattr(screen,var1)
    y = getattr(screen,var2)
    w = screen.weight
    w_sum = np.sum(w)
    x_mean = np.sum(x*w)/w_sum
    y_mean = np.sum(y*w)/w_sum
    
    c = poly.polyfit(x-x_mean, y-y_mean, max_power, w=w)
    y_fit = poly.polyval(x-x_mean, c)+y_mean
    
    setattr(screen, var2, y-y_fit)
    
    return screen


def clip_to_charge(PG_input, clipping_charge, verbose=True, make_copy=False):
    if (make_copy==True):
        PG = copy.deepcopy(PG_input)
    else:
        PG = PG_input

    min_final_particles = 3
    
    w = PG.weight / np.sum(PG.weight)
    r_centered = np.sqrt((PG.x - np.sum(PG.x * w))**2 + (PG.y - np.sum(PG.y * w))**2)
    r_i = np.argsort(r_centered)
    r = r_centered[r_i]
    w = PG.weight[r_i]
    w_sum = np.cumsum(w)
    if (clipping_charge >= w_sum[-1]):
        n_clip = -1
    else:
        n_clip = np.argmax(w_sum > clipping_charge)
    if (n_clip < (min_final_particles-1) and n_clip > -1):
        n_clip = min_final_particles-1
    r_cut = r[n_clip]
    PG.weight[r_centered>r_cut] = 0
    if (verbose):
        print(f'Clipping at r = {r_cut}')
    PG = kill_zero_weight(PG, make_copy=False)
    
    return PG



def clip_to_emit(PG_input, clipping_emit, verbose=False, make_copy=False):
    if (make_copy==True):
        PG = copy.deepcopy(PG_input)
    else:
        PG = PG_input

    min_final_particles = 3

    # Radius about the beam centroid (as in clip_to_charge)
    w_all = PG.weight
    r_centered = np.sqrt((PG.x - np.sum(PG.x*w_all)/np.sum(w_all))**2 + (PG.y - np.sum(PG.y*w_all)/np.sum(w_all))**2)
    r_i = np.argsort(r_centered)
    r = r_centered[r_i]

    # emit_i[j] = sqrt_norm_emit_4d of the j+1 innermost particles, from running weighted sums
    # (same weighted covariance as ParticleGroup.cov, i.e. np.cov with aweights)
    u = np.array([PG.x, PG.px, PG.y, PG.py])[:, r_i]
    u = u - np.mean(u, axis=1, keepdims=True)                   # reduces roundoff in the running sums
    w = w_all[r_i]
    v1 = np.cumsum(w)
    v2 = np.cumsum(w*w)
    S1 = np.cumsum(u*w, axis=1).T                               # (N, 4)
    S2 = np.cumsum(np.einsum('in,jn->nij', u, u)*w[:, None, None], axis=0)   # (N, 4, 4)

    with np.errstate(divide='ignore', invalid='ignore'):
        C = S2 - S1[:, :, None]*S1[:, None, :]/v1[:, None, None]
        C = C / (v1 - v2/v1)[:, None, None]
        emit_i = np.power(np.linalg.det(C), 0.25) / PG.mass     # NaN where det < 0 (too few particles), as before
    emit_i[:min_final_particles] = 0

    if (clipping_emit >= emit_i[-1]):
        n_clip = -1
    else:
        n_clip = np.argmax(emit_i > clipping_emit)
    if (n_clip < (min_final_particles-1) and n_clip > -1):
        n_clip = min_final_particles-1
    r_cut = r[n_clip]
    PG.weight[r_centered>r_cut] = 0
    if (verbose):
        print(f'Clipping at r = {r_cut}')
    PG = kill_zero_weight(PG, make_copy=False)

    return PG


# Duplicates all particles n_copies times, uniformly rotated around the z-axis. Useful for making pretty plots when the screen is cylindrically symmetric
def add_cylindrical_copies(screen_input, n_copies, make_copy=False):
    screen = screen_input  # a new group is always returned; make_copy is kept for a consistent signature
    npart = len(screen.x)

    data = {k: np.tile(screen[k], n_copies) for k in screen._settable_array_keys}
    for k in screen._settable_scalar_keys:
        data[k] = screen[k]

    # Copy c of every particle is rotated by 2*pi*c/n_copies (copy 0 is the original)
    theta = np.repeat(2*np.pi*np.arange(n_copies)/n_copies, npart)
    costh = np.cos(theta)
    sinth = np.sin(theta)

    x, y, px, py = data['x'], data['y'], data['px'], data['py']
    data['x'] = x*costh - y*sinth
    data['y'] = x*sinth + y*costh
    data['px'] = px*costh - py*sinth
    data['py'] = px*sinth + py*costh

    data['weight'] = data['weight']/n_copies

    # Unique ids: copy c of particle id gets id + c*(max_id+1), so copy 0 keeps the original ids
    ids = screen.id
    data['id'] = np.tile(ids, n_copies) + np.repeat(np.arange(n_copies), npart)*(np.max(ids)+1)

    return ParticleGroupExtension(data=data)
