
from pmd_beamphysics import ParticleGroup
import numpy as np
import scipy

import copy
   


def make_parabolic_guess(n_mirrors, settings, scr, use_ideal_size=True):
    
    new_settings = copy.copy(settings)
    
    omega0 = 2*np.pi*settings['center_frequency']
    sigt = settings['sig_t']
    tht = np.radians(settings['theta_THz'])
    thb = np.radians(settings['theta_beam'])
    g = scr['mean_energy'] / 510998.95
    gb = np.sqrt((g-1)*(g+1))
    beta = gb/g
    phi0 = np.radians(settings['phi0'])

    x = best_fit_paraboloid(scr, var='kinetic_energy')
    dt = x[1]
    ar = x[2] / beta
    at = x[3] / beta

    A = dpzParabola(1, 0, 0, 0, omega0, sigt, tht, thb, phi0, beta, 1)

    if (n_mirrors == 1):
        w1 = np.sqrt(2.0 * at / (ar * omega0**2))
        dp1 = ar * w1**2
        w2 = w1
        dp2 = 0.0
    else:
        if (use_ideal_size):
            # w1 gets the ideal single pulse size, w2 is arbitrarily made a few times larger
            w1 = 2.0 * np.sqrt(2.0 * at / (ar * omega0**2))
            w2 = 1.5*w1
        else:
            w1 = 2*settings['sig_x']
            w2 = 2*settings['sig_x2']
        dp1 = w1**2 * w2**2 / (w2**2 - w1**2) * (ar - 2.0*at/(w2**2 * omega0**2))
        dp2 = w1**2 * w2**2 / (w2**2 - w1**2) * (2.0*at/(w1**2 * omega0**2) - ar)

    new_settings['E0'] = dp1 / A
    new_settings['sig_x'] = w1 / 2
    new_settings['dt'] = dt

    new_settings['E02'] = dp2 / A
    new_settings['sig_x2'] = w2 / 2
    new_settings['dt2'] = dt
    
    if (n_mirrors == 3):
        new_settings['E03'] = 0.0
        new_settings['sig_x3'] = settings['sig_x3'] 
        new_settings['dt3'] = dt
    
    return new_settings

def get_THz_pulse(t, settings):
    return np.exp(-t*t/(4*settings['sig_t']**2))*np.cos(2*np.pi*settings['center_frequency']*t + settings['phi0']*np.pi/180.0)

# Applies a transfer matrix to a screen such that a desired final beam size is achieved, without changing the emittance
def make_collimated_beam(scr, sig_x):
    scr2 = copy.deepcopy(scr)
    n = np.sum(scr.weight)
    xx = np.sum(scr.x * scr.x * scr.weight)/n
    xpx = np.sum(scr.x * scr.px * scr.weight)/n
    pxpx = np.sum(scr.px * scr.px * scr.weight)/n

    S = np.array([[xx, xpx], [xpx, pxpx]])
    eps = np.sqrt(np.linalg.det(S))
    
    D, V = np.linalg.eigh(S)  # warning, don't use np.linalg.eig without first scaling x and px. Instead, use np.linalg.eigh
        
    if (np.isclose(np.linalg.det(V), -1, atol=1.0e-3)):
        idx = [1,0]
        D = D[idx]
        V = V[:,idx]

    Ds = np.sqrt(D)
    
    Dp, Vp = np.linalg.eigh(np.array([[sig_x**2,0], [0,(eps/sig_x)**2]]))
    Dps = np.sqrt(Dp)
    
    #T = np.diag(Dps/Ds)@(V.T)  # full form is Vp @ np.diag(Dps/Ds) @ V.T, but in this case Vp is the identity
        
    T = Vp @ (Dps * Ds) @ V.T
        
    print(T)
        
    x_px = T@np.array([scr.x, scr.px])
    y_py = T@np.array([scr.y, scr.py])
    
    scr2.x = x_px[0,:]
    scr2.px = x_px[1,:]
    scr2.y = y_py[0,:]
    scr2.py = y_py[1,:]
    
    scr2.pz = np.sqrt((scr.energy - 510998.95)*(scr.energy + 510998.95) - scr2.px**2 - scr2.py**2) # require energy conservation
        
    return scr2    

# Analytic effect of mirror delta_pz in units of eV/c if SI units are passed in
def dpz(E0, x0, y0, t0, omega0, sigt, tht, thb, phi0, beta, w0):
    c = 299792458
    x0p = x0 * np.cos(tht) / np.cos(thb)
    A = np.cos(thb)*(beta*np.sin(thb) - np.sin(tht))/((beta*np.cos(thb - tht)-1.0)*(beta*np.cos(thb+tht)+1.0))
    B = np.exp(-(x0p**2 + y0**2)/w0**2)
    R = 0.5*(t0/sigt - x0/np.cos(thb)*(beta*np.sin(tht) - np.sin(thb))/(beta*c*sigt) )
    
    C = -4*sigt*(0.5*np.sqrt(np.pi)*np.exp(-omega0**2*sigt**2)*np.cos(phi0) + np.imag(np.exp(-R**2 + 1j*(phi0 + 2.0*R*omega0*sigt))*scipy.special.dawsn(-omega0*sigt-1j*R)))
    
    conv = c  # (1 volt / m) * (electron charge) * (1 s) in eV/c  =  c
        
    return E0*conv * A * B * C

def dpzParabola(E0, x0, y0, t0, omega0, sigt, tht, thb, phi0, beta, w0):
    x0p = x0 * np.cos(tht) / np.cos(thb)
    r2 = (x0p**2 + y0**2)
    A = 2.0*np.cos(thb)*(beta*np.sin(thb) - np.sin(tht))/((beta*np.cos(thb - tht)-1.0)*(beta*np.cos(thb+tht)+1.0))
    B = (1.0 - r2/w0**2 - 0.5*omega0**2*t0**2)
    c = 299792458
    conv = c  # (1 volt / m) * (electron charge) * (1 s) in eV/c  =  c
    
    return E0*conv * A * B / omega0

def dpzQuad(E0, x0, y0, t0, omega0, sigt, tht, thb, phi0, beta, w0):
    x0p = x0 * np.cos(tht) / np.cos(thb)
    r2 = (x0p**2 + y0**2)
    A = 2.0*np.cos(thb)*(beta*np.sin(thb) - np.sin(tht))/((beta*np.cos(thb - tht)-1.0)*(beta*np.cos(thb+tht)+1.0))
    B = (1.0 - r2/w0**2 - 0.5*omega0**2*t0**2 + 0.5*r2*t0**2*omega0**2/w0**2 + 0.5*r2**2/w0**4 - omega0**4*t0**4/24.0)
    c = 299792458
    conv = c  # (1 volt / m) * (electron charge) * (1 s) in eV/c  =  c
    
    return E0*conv * A * B / omega0


def get_beta(EnKV):
    g = 1 + EnKV/511
    return np.sqrt((g+1)*(g-1))/g

def get_pulse_energy(settings, which_screen):
    if (which_screen == 1):
        phi0 = np.radians(settings['phi0'])
        w0 = settings['sig_x']*2
        E0 = settings['E0']
    if (which_screen == 2):
        phi0 = np.radians(settings['phi02'])
        w0 = settings['sig_x2']*2
        E0 = settings['E02']
    
    omega0 = 2*np.pi*settings['center_frequency']
    sigt = settings['sig_t']
    
    # Note phip in radians
    c = 299792458
    mu0 = 1.25663706e-6
    return E0**2 * np.pi**1.5 * w0**2 * sigt * (1 + np.exp(-2.0*sigt**2*omega0**2)*np.cos(2.0*phi0)) / (2.0 * np.sqrt(2.0) * c * mu0)


def get_analytic_scr_para(settings, scr, guess=None, do_quad=False):
    scr_new = copy.deepcopy(scr)
    
    s = guess_to_settings(guess, settings)    

    omega0 = 2*np.pi*s['center_frequency']
    tht = np.radians(s['theta_THz'])
    thb = np.radians(s['theta_beam'])
    g = scr['mean_energy'] / 510998.95
    gb = np.sqrt((g-1)*(g+1))
    beta = gb/g
    sigt = s['sig_t']
    
    w0 = s['sig_x']*2
    phi0 = np.radians(s['phi0'])
    E0 = s['E0']
    
    w02 = s['sig_x2']*2
    phi02 = np.radians(s['phi02'])
    E02 = s['E02']

    x = scr_new.x
    y = scr_new.y
    t = scr_new.t - scr_new['mean_t'] - s['dt']
    
    if (do_quad):
        scr_new.pz = scr_new.pz + dpzQuad(E0, x, y, t, omega0, sigt, tht, thb, phi0, beta, w0)
    else:
        scr_new.pz = scr_new.pz + dpzParabola(E0, x, y, t, omega0, sigt, tht, thb, phi0, beta, w0)
    
    x2 = scr_new.x
    y2 = scr_new.y
    t2 = scr_new.t - scr_new['mean_t'] - s['dt2']
    
    if (np.abs(E02) > 0.0):
        if (do_quad):
            scr_new.pz = scr_new.pz + dpzQuad(E02, x2, y2, t2, omega0, sigt, tht, thb, phi02, beta, w02)
        else:
            scr_new.pz = scr_new.pz + dpzParabola(E02, x2, y2, t2, omega0, sigt, tht, thb, phi02, beta, w02)

    if ('sig_x3' in s.keys()):
        w03 = s['sig_x3']*2
        phi03 = np.radians(s['phi03'])
        E03 = s['E03']
        
        x3 = scr_new.x
        y3 = scr_new.y
        t3 = scr_new.t - scr_new['mean_t'] - s['dt3']
        
        if (do_quad):
            scr_new.pz = scr_new.pz + dpzQuad(E03, x3, y3, t3, omega0, sigt, tht, thb, phi03, beta, w03)
        else:
            scr_new.pz = scr_new.pz + dpzParabola(E03, x3, y3, t3, omega0, sigt, tht, thb, phi03, beta, w03)
        
    return scr_new

def get_analytic_scr(settings, scr, guess=None, force_dt=None):
    scr_new = copy.deepcopy(scr)
    
    s = guess_to_settings(guess, settings)

    if (force_dt is not None):
        s['dt'] = force_dt
        s['dt2'] = force_dt
        s['dt3'] = force_dt
    
    omega0 = 2*np.pi*s['center_frequency']
    tht = np.radians(s['theta_THz'])
    thb = np.radians(s['theta_beam'])
    g = scr['mean_energy'] / 510998.95
    gb = np.sqrt((g-1)*(g+1))
    beta = gb/g
    sigt = s['sig_t']
    
    w0 = s['sig_x']*2
    phi0 = np.radians(s['phi0'])
    E0 = s['E0']
    
    w02 = s['sig_x2']*2
    phi02 = np.radians(s['phi02'])
    E02 = s['E02']
    
    x = scr_new.x
    y = scr_new.y
    t = scr_new.t - scr_new['mean_t'] - s['dt']
    
    scr_new.pz = scr_new.pz + dpz(E0, x, y, t, omega0, sigt, tht, thb, phi0, beta, w0)
    
    x2 = scr_new.x
    y2 = scr_new.y
    t2 = scr_new.t - scr_new['mean_t'] - s['dt2']
    
    scr_new.pz = scr_new.pz + dpz(E02, x2, y2, t2, omega0, sigt, tht, thb, phi02, beta, w02)

    if ('sig_x3' in s.keys()):
        w03 = s['sig_x3']*2
        phi03 = np.radians(s['phi03'])
        E03 = s['E03']
        
        x3 = scr_new.x
        y3 = scr_new.y
        t3 = scr_new.t - scr_new['mean_t'] - s['dt3']
        
        scr_new.pz = scr_new.pz + dpz(E03, x3, y3, t3, omega0, sigt, tht, thb, phi03, beta, w03)

    
    return scr_new


def get_gpt_scr(settings, guess=None):
    s = guess_to_settings(guess, settings)   
    
    gpt_data_temp = run_gpt_with_THz(s,
                                 gpt_input_file=GPT_INPUT_FILE,
                                 distgen_input_file=DISTGEN_INPUT_FILE,
                                 verbose=False,
                                 gpt_verbose=False,
                                 auto_phase=False,
                                 timeout=1000)

    sigE = get_screen_data(gpt_data_temp, screen_z=s["z_screen_1"])[0]['sigma_kinetic_energy']
    
    print(f'{1e3*sigE:.2f} , {guess}')
    return get_screen_data(gpt_data_temp, screen_z=s["z_screen_1"])[0]

def subtract_paraboloid(scr, x):
    scr_new = copy.deepcopy(scr)
    
    r = scr_new.r
    t = scr_new.t - scr_new['mean_t']
    
    t0 = x[0] * 1e-12
    invR2 = x[1] / (1e-3)**2
    invT2 = x[2] / (1e-12)**2
    
    scr_new.pz = scr_new.pz + (invR2 * r**2 + invT2 * (t-t0)**2)
    return scr_new


def guess_to_settings(x, settings):
    s = copy.copy(settings)
    
    if x is not None:
        s['E0'] = x[0] * 10000
        s['dt'] = x[1] / 1e12
        s['sig_x'] = x[2] * 1.0e-3

        if (len(x) > 3):
            s['E02'] = x[3] * 10000
            s['dt2'] = x[4] / 1e12
            s['sig_x2'] = x[5] * 1.0e-3
        else:
            s['E02'] = 0.0
            
        if (len(x) > 6):
            s['E03'] = x[6] * 10000
            s['dt3'] = x[7] / 1e12
            s['sig_x3'] = x[8] * 1.0e-3
        else:
            s['E03'] = 0.0
    return s

def settings_to_guess(s, n_mirror):
    x0 = s['E0'] / 10000
    x1 = s['dt'] * 1e12
    x2 = s['sig_x'] / 1.0e-3
    
    if n_mirror == 1:
        return [x0, x1, x2]
    
    x3 = s['E02'] / 10000
    x4 = s['dt2'] * 1e12
    x5 = s['sig_x2'] / 1.0e-3
    
    if n_mirror == 2:
        return [x0, x1, x2, x3, x4, x5]
    
    x6 = s['E03'] / 10000
    x7 = s['dt3'] * 1e12
    x8 = s['sig_x3'] / 1.0e-3
    
    return [x0, x1, x2, x3, x4, x5, x6, x7, x8]


def best_fit_paraboloid(scr, var='kinetic_energy'):   
    r = scr.r
    t = scr.t - np.mean(scr.t)
    K = scr[var] - np.mean(scr[var])
    
    M = np.matrix([[len(r) , np.sum(r*r),     np.sum(t),     np.sum(t*t)],
               [np.sum(r*r), np.sum(r**4),    np.sum(t*r*r), np.sum(t*t*r*r)],
               [np.sum(t),   np.sum(r*r*t),   np.sum(t*t),   np.sum(t*t*t)],
               [np.sum(t*t), np.sum(r*r*t*t), np.sum(t*t*t), np.sum(t**4)]])
    
    b = np.matrix([[np.sum(K)], [np.sum(K*r*r)], [np.sum(K*t)], [np.sum(K*t*t)]])
    x = np.linalg.solve(M, b)
    x = np.array(x).flatten()
    
    return [np.mean(scr[var]) + x[0] - 0.25*x[2]*x[2]/x[3], -0.5*x[2]/x[3], x[1], x[3]]

def make_fit_paraboloid(scr, x):
    r = scr.r
    t = scr.t - np.mean(scr.t)
    
    return x[0] + x[2]*(r)**2 + x[3]*(t-x[1])**2



def get_cam_dist():
    np.random.seed(1)

    n = 4000
    K = 1
    me = 5.11e5
    maxgamma = K/me
    sigmax = 5e-8
    tmax = 3e-14

    x_list = []
    y_list = []
    t_list = []
    z_list = np.zeros(n)
    Bx_list = []
    By_list = []
    Bz_list = []
    status = np.ones(n)
    weight = np.ones(n) / n
    
    for gamma, theta, cosphi, x, y, t in zip(np.random.uniform(low = 0, high = maxgamma, size=n),
                                     np.random.uniform(low = 0, high = 2*math.pi, size = n),
                                     np.random.uniform(low = 0, high = 1, size = n),
                                     np.random.normal(scale=sigmax, size=n),
                                     np.random.normal(scale=sigmax, size=n),
                                     np.random.uniform(low = 0, high = tmax, size = n)):
        Bx = math.sqrt(2*gamma)*math.sqrt(1-(cosphi**2))*math.cos(theta)
        By = math.sqrt(2*gamma)*math.sqrt(1-(cosphi**2))*math.sin(theta)
        Bz = math.sqrt(2*gamma)*cosphi
        x_list = x_list + [x]
        y_list = y_list + [y]
        t_list = t_list + [t]
        Bx_list = Bx_list + [Bx]
        By_list = By_list + [By]
        Bz_list = Bz_list + [Bz]
    
    Bx_list = np.array(Bx_list)
    By_list = np.array(By_list)
    Bz_list = np.array(Bz_list)
    
    data = {}
    data['species'] = 'electron'
    data['x'] = x_list
    data['y'] = y_list
    data['z'] = z_list
    data['t'] = t_list
    b_list = np.sqrt(Bx_list**2 + By_list**2 + Bz_list**2)
    g_list = 1.0/np.sqrt((1 - b_list)*(1 + b_list))
    data['px'] = g_list*Bx_list * 510998.95
    data['py'] = g_list*By_list * 510998.95
    data['pz'] = g_list*Bz_list * 510998.95
    data['status'] = status
    data['weight'] = weight
    PG = ParticleGroup(data=data)
    
    return PG
