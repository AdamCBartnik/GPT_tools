import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from scipy.interpolate import PchipInterpolator
from .ParticleGroupExtension import core_emit_calc
from .nicer_units import *
from .tools import scale_and_get_units

def emittance_vs_fraction(pg, var, number_of_points=25, plotting=True, verbose=False, show_core_emit_plot=False, title_fraction=[], title_emittance=[]):
    # pg:   Input ParticleGroup
    # var:  'x' or 'y'
    
    var1 = var
    var2 = 'p' + var
    
    # Check input and perform initializations:
    x = getattr(pg, var1)
    y = getattr(pg, var2)/pg.mass
    w = pg.weight
    
    (full_emittance, alpha, beta, center_x, center_y) = get_twiss(x, y, w)
    
    fs = np.linspace(0,1,number_of_points)
    es = np.zeros(number_of_points)
    es[-1] = full_emittance

    twiss_parameters = np.array([alpha, beta, center_x, center_y])
    twiss_scales = np.abs(np.array([alpha, beta, np.max([1.0e-6, np.abs(center_x)]), np.max([1.0e-6, np.abs(center_y)])]))  # scale of each fit parameter, helps simplex dimensions all be similar
    normed_twiss_parameters = twiss_parameters/twiss_scales
    
    # Computation of emittance vs. fractions
    
    # Run through bounding ellipse areas (largest to smallest) and compute the
    # enclosed fraction and emittance of inclosed beam.  The Twiss parameters
    # computed for the minimum bounding ellipse for the entire distribution is
    # used as an initial guess:

    if verbose:
       print('')
       print('   computing emittance vs. fraction curve...') 
        
    indices = np.arange(len(es)-2,0,-1)  # every fraction except f=0 (emittance 0) and f=1 (full rms emittance)
    for ind, ii in enumerate(indices):
        # use previous ellipse as a guess point to compute next one:
        twiss_parameter_guess = normed_twiss_parameters
        
        normed_twiss_parameters = fmin(lambda xx: get_emit_at_frac(fs[ii],xx*twiss_scales,x,y,w), twiss_parameter_guess, args=(), maxiter=None, disp=verbose)  # xtol=0.01, ftol=1, 
        es[ii] = get_emit_at_frac(fs[ii],normed_twiss_parameters*twiss_scales,x,y,w)
            
    if verbose:
        print('   ...done.')

    # Compute core fraction and emittance:

    if verbose:
        print('')
        print('   computing core emittance and fraction: ')
        
    ec = core_emit_calc(x, y, w, show_fit=show_core_emit_plot)
                    
    if verbose:
        print('done.')
            
    fc = np.interp(ec,es,fs)    
        
    # Plot results

    if plotting:
        if verbose:
            print('   plotting data: ')

        plot_points=100
          
        base_units = 'm'
        (es_plot, emit_units, emit_scale) = scale_and_get_units(es, base_units)
        ec_plot = ec/emit_scale
            
        fc1s = np.ones(plot_points)*fc
        ec1s = np.linspace(0.0,1.0,plot_points)*ec_plot

        ec2s = np.ones(plot_points)*ec_plot
        fc2s = np.linspace(0.0,1.0,plot_points)
        
        plt.figure(dpi=100)

        plt.plot(fc1s, ec1s, 'r--')
        plt.plot(fc2s, ec2s, 'r--')
        plt.plot(fs, ec_plot*fs, 'r')
        plt.plot(fs, es_plot, 'b.')
        
        pchip = PchipInterpolator(fs, es_plot)
        plt.plot(fc2s, pchip(fc2s), 'b-')
                
        plt.xlim([0,1])
        plt.ylim(bottom=0)
        
        plt.xlabel('Fraction')
        plt.ylabel(f'Emittance ({emit_units})')

        title_str = rf'$\epsilon_{{core}}$ = {ec_plot:.3g} {emit_units}, $f_{{core}}$ = {fc:.3f}'
        if (title_fraction):
            title_str = title_str + rf', $\epsilon_{{{title_fraction}}}$ = {pchip(title_fraction):.3g} {emit_units}'   # np.interp(title_fraction, fs, es)
        plt.title(title_str)
        
        if verbose:
            print('done.')

    return (es, fs, ec, fc)

def get_twiss(x, y, w):
    w_sum = np.sum(w)

    x0=np.sum(x*w)/w_sum
    y0=np.sum(y*w)/w_sum
    dx=x-x0
    dy=y-y0

    x2 = np.sum(dx**2*w)/w_sum
    y2 = np.sum(dy**2*w)/w_sum
    xy = np.sum(dx*dy*w)/w_sum

    e=np.sqrt(x2*y2-xy**2)
    a = -xy/e
    b =  x2/e

    return (e,a,b,x0,y0)

             
def get_emit_at_frac(f_target, twiss_parameters, x, y, w):
    alpha = twiss_parameters[0]
    beta = twiss_parameters[1]
    x0 = twiss_parameters[2]
    y0 = twiss_parameters[3]
    
    # subtract out centroids:
    dx=x-x0
    dy=y-y0

    # compute and compare single particle emittances to emittance from Twiss parameters
    gamma=(1.0+alpha**2)/beta
    e_particles = 0.5*(gamma*dx**2 + beta*dy**2 + 2.0*alpha*dx*dy)
    
    # Mean single-particle emittance of the innermost fraction f_target (by weight). Minimizing this over
    # the Twiss parameters gives the smallest rms emittance of any subset holding that fraction.
    if np.all(w == w[0]):
        idx_target = int(np.floor(f_target * len(e_particles)))
        if (idx_target == 0):
            return 0.0
        return np.mean(np.partition(e_particles, idx_target-1)[:idx_target])  # partial sort is enough
    
    order = np.argsort(e_particles)
    cum_w = np.cumsum(w[order])
    idx_target = int(np.searchsorted(cum_w, f_target * cum_w[-1]))
    if (idx_target == 0):
        return 0.0
    inner = order[:idx_target]
    return np.sum(e_particles[inner]*w[inner])/np.sum(w[inner])
