import numpy as np
import copy
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from scipy.interpolate import PchipInterpolator
from .ParticleGroupExtension import core_emit_calc
from .nicer_units import *
from .tools import scale_and_get_units

def emittance_vs_fraction(pg, var, number_of_points=25, plotting=True, verbose=False, show_core_emit_plot=False, title_fraction=[], title_emittance=[]):
    # pg:   Input ParticleGroup
    # var:  'x' or 'y'
    
    pg = copy.deepcopy(pg)
        
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
                                
    aa = np.empty(len(fs))
    bb = np.empty(len(fs))
    cx = np.empty(len(fs))
    cp = np.empty(len(fs))
    aa[:] = np.nan
    bb[:] = np.nan
    cx[:] = np.nan
    cp[:] = np.nan
    
    # Computation of emittance vs. fractions
    
    # Run through bounding ellipse areas (largest to smallest) and compute the
    # enclosed fraction and emittance of inclosed beam.  The Twiss parameters
    # computed for the minimum bounding ellipse for the entire distribution is
    # used as an initial guess:

    if verbose:
       print('')
       print('   computing emittance vs. fraction curve...') 
        
    indices = np.arange(len(es)-2,1,-1)
    for ind, ii in enumerate(indices):
        # use previous ellipse as a guess point to compute next one:
        twiss_parameter_guess = normed_twiss_parameters
        
        normed_twiss_parameters = fmin(lambda xx: get_emit_at_frac(fs[ii],xx*twiss_scales,x,y,w), twiss_parameter_guess, args=(), maxiter=None, disp=verbose)  # xtol=0.01, ftol=1, 
        es[ii] = get_emit_at_frac(fs[ii],normed_twiss_parameters*twiss_scales,x,y,w)
        aa[ii] = normed_twiss_parameters[0]*twiss_scales[0]
        bb[ii] = normed_twiss_parameters[1]*twiss_scales[1]
        cx[ii] = normed_twiss_parameters[2]*twiss_scales[2]
        cp[ii] = normed_twiss_parameters[3]*twiss_scales[3]
            
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
    ac = np.interp(fc,fs,aa)
    bc = np.interp(fc,fs,bb)
    gc = (1.0+ac**2)/bc
        
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

        title_str = f'$\epsilon_{{core}}$ = {ec_plot:.3g} {emit_units}, $f_{{core}}$ = {fc:.3f}'
        if (title_fraction):
            title_str = title_str + f', $\epsilon_{{{title_fraction}}}$ = {pchip(title_fraction):.3g} {emit_units}'   # np.interp(title_fraction, fs, es)
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
    e_particles = np.sort(e_particles)
    
    idx_target = int(np.floor(f_target * len(e_particles)))
    frac_emit = np.sum(e_particles[0:idx_target])/(idx_target+1.0)
    
    return frac_emit
    
            

# This function is no longer used, alas
def minboundellipse( x_all, y_all, tolerance=1.0e-3, plot_on=False):

    # x_all and y_all are rows of points

    # reduce set of points to just the convex hull of the input
    ch = ConvexHull(np.array([x_all,y_all]).transpose())
    
    x = x_all[ch.vertices]
    y = y_all[ch.vertices]

    d = 2
    N = len(x)
    P = np.array([x, y])
    Q = np.array([x, y, np.ones(N)])

    # Initialize
    count = 1
    err = 1
    u = (1.0/N) * np.array([np.ones(N)]).transpose()

    # Khachiyan Algorithm
    while (err > tolerance):
        X = Q @ np.diag(u.reshape(len(u))) @ Q.transpose()        
        M = np.diag(Q.transpose() @ np.linalg.solve(X, Q))

        j = np.argmax(M)
        maximum = M[j]
        step_size = (maximum-d-1.0)/((d+1.0)*(maximum-1.0))

        new_u = (1.0 - step_size)*u
        new_u[j] = new_u[j] + step_size

        err = np.linalg.norm(new_u - u)

        count = count + 1
        u = new_u

    U = np.diag(u.reshape(len(u)))

    # Compute the twiss parameters    
    A = (1.0/d) * np.linalg.inv(P @ U @ P.transpose() - (P @ u) @ (P @ u).transpose() )

    (U, D, V) = np.linalg.svd(A)
    
    a = 1/np.sqrt(D[0]) # major axis
    b = 1/np.sqrt(D[1]) # minor axis

    # make sure V gives pure rotation
    if (np.linalg.det(V) < 0):
        V = V @ np.array([[-1, 0], [0, 1]])

    emittance = a*b

    gamma = A[0,0]*emittance;
    beta = A[1,1]*emittance;
    alpha = A[1,0]*emittance;

    # And the center
    c = P @ u
    center = np.reshape(c, len(c))

    if (plot_on):

        plt.figure(dpi=100)

        theta = np.linspace(0,2*np.pi,100)

        state = np.array([a*np.cos(theta), b*np.sin(theta)])

        X = V @ state
        X[0,:] = X[0,:] + c[0]
        X[1,:] = X[1,:] + c[1]

        plt.plot(X[0,:], X[1,:], 'r-')
        plt.plot(c[0], c[1], 'r*')
        plt.plot(x_all, y_all, 'b.')
                
    
    return (emittance, alpha, beta, center, gamma)
    
             