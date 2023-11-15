from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.units import unit, PARTICLEGROUP_UNITS
from pmd_beamphysics.statistics import norm_emit_calc
from matplotlib import pyplot as plt
import numpy.polynomial.polynomial as poly
import numpy as np
import copy

class ParticleGroupExtension(ParticleGroup):
    
    def __init__(self, input_particle_group=None, data=None):
        self.n_slices = 50
        self.slice_key = 't'
        
        if (input_particle_group):
            data={}
            for key in input_particle_group._settable_keys:
                data[key] = copy.copy(input_particle_group[key])  # is deepcopy needed?
        
            if ('id' not in input_particle_group._settable_keys and hasattr(input_particle_group, 'id')):
                data['id'] = copy.copy(input_particle_group['id'])
        
        super().__init__(data=data)
        
        new_units = {}
        new_units['r_centered'] = 'm'
        new_units['sqrt_norm_emit_4d'] = 'm'
        new_units['root_norm_emit_6d'] = 'm'
        new_units['slice_emit_x'] = 'm'
        new_units['slice_emit_y'] = 'm'
        new_units['core_emit_x'] = 'm'
        new_units['core_emit_y'] = 'm'
        new_units['core_emit_4d'] = 'm'
        new_units['slice_emit_4d'] = 'm'
        new_units['action_x'] = 'm'
        new_units['action_y'] = 'm'
        new_units['rp'] = 'rad'
        new_units['action_4d'] = 'm'
        new_units['crazy_action_x'] = 'm'
        new_units['crazy_action_y'] = 'm'
        new_units['pr_centered'] = 'eV/c'
        new_units['ptrans'] = 'eV/c'
        
        for k in new_units:
            if (k not in PARTICLEGROUP_UNITS.keys()):
                PARTICLEGROUP_UNITS[k] = unit(new_units[k])
         
    @property
    def ptrans(self):
        return np.sqrt(self.px*self.px + self.py*self.py)
  
    @ptrans.setter
    def ptrans(self, pt_input):
        pt_ratio = pt_input/np.sqrt(self.px * self.px + self.py * self.py)
        self.px = self.px * pt_ratio
        self.py = self.py * pt_ratio
        
    @property
    def rp(self):
        return np.sqrt(self.px*self.px + self.py*self.py) / self.pz
    
    @property
    def pr_centered(self):
        x_mean = np.sum(self.x * self.weight)/np.sum(self.weight)
        y_mean = np.sum(self.y * self.weight)/np.sum(self.weight)
        r = np.sqrt((self.x-x_mean)*(self.x-x_mean) + (self.y-y_mean)*(self.y-y_mean))
        return ((self.x - x_mean)*self.px + (self.y - y_mean)*self.py)/r
    
    @property
    def r_centered(self):
        x_mean = np.sum(self.x * self.weight)/np.sum(self.weight)
        y_mean = np.sum(self.y * self.weight)/np.sum(self.weight)
        return np.sqrt((self.x-x_mean)*(self.x-x_mean) + (self.y-y_mean)*(self.y-y_mean))
    
    @pr_centered.setter
    def signed_pr(self, pr_input):
        x_mean = np.sum(self.x * self.weight)/np.sum(self.weight)
        y_mean = np.sum(self.y * self.weight)/np.sum(self.weight)
        x = self.x-x_mean
        y = self.y-y_mean
        
        pdotr = self.px * x + self.py * y
        r2 = x*x + y*y
        r = np.sqrt(r2)
        px_new = self.px - pdotr * x / r2 + pr_input * x/r
        py_new = self.py - pdotr * y / r2 + pr_input * y/r
        self.px = px_new
        self.py = py_new
    
    @property
    def energy_spread_fraction(self):
        return self['sigma_energy']/self['mean_kinetic_energy']
    
    @property
    def core_emit_x(self):
        return core_emit_calc(self.x, self.xp, self.weight)
    
    @property
    def core_emit_y(self):
        return core_emit_calc(self.y, self.yp, self.weight)
    
    @property
    def core_emit_4d(self):
        return core_emit_calc_4d(self.x, self.xp, self.y, self.yp, self.weight)
        
    @property
    def sqrt_norm_emit_4d(self):
        return np.sqrt(norm_emit_calc(self, planes=['x', 'y']))

    @property
    def root_norm_emit_6d(self):
        pg = copy.deepcopy(self)
        pg.drift_to_t()
        return np.power(norm_emit_calc(pg, planes=['x', 'y', 'z']), 1.0/3.0)
    
    @property
    def slice_emit_x(self):
        (p_list, _, _) = divide_particles(self, nbins = self.n_slices, key=self.slice_key)
        return slice_emit(p_list, 'norm_emit_x')

    @property
    def slice_emit_y(self):
        (p_list, _, _) = divide_particles(self, nbins = self.n_slices, key=self.slice_key)
        return slice_emit(p_list, 'norm_emit_y')

    @property
    def slice_emit_4d(self):
        (p_list, _, _) = divide_particles(self, nbins = self.n_slices, key=self.slice_key)
        return slice_emit(p_list, 'sqrt_norm_emit_4d')

    @property
    def action_x(self):
        sig = self.cov('x', 'xp')
        emit = np.sqrt(np.linalg.det(sig))/(self.gamma*self.beta)
        beta = sig[0,0] / emit
        alpha = -sig[0,1] / emit
        gamma = sig[1,1] / emit
        return 0.5*(gamma*self.x*self.x + 2.0*alpha*self.x*self.xp + beta*self.xp*self.xp)
    
    @property
    def action_y(self):
        sig = self.cov('y', 'yp')
        emit = np.sqrt(np.linalg.det(sig))/(self.gamma*self.beta)
        beta = sig[0,0] / emit
        alpha = -sig[0,1] / emit
        gamma = sig[1,1] / emit
        return 0.5*(gamma*self.y*self.y + 2.0*alpha*self.y*self.yp + beta*self.yp*self.yp)
        
    @property
    def crazy_action_x(self):
        sig = self.cov('x', 'xp', 'y', 'yp')
        S = np.array([[0, 1, 0, 0],[-1,0,0,0],[0,0,0,1],[0,0,-1,0]])
        Q = np.array([[1, 1j, 0, 0], [1, -1j, 0, 0], [0, 0, 1, 1j], [0, 0, 1, -1j]]) / np.sqrt(2.0)
        (w, E) = np.linalg.eig(sig.dot(S))
        E = E / np.power(np.abs(np.linalg.det(E)), 0.25)
        E = E[:,[1,0,3,2]] # reorder to make N symplectic
        N = E.dot(Q)
        N = N.real 
        x = np.array([self.x,self.xp, self.y, self.yp])
        x_new = np.linalg.solve(N, x)
        return 0.5*(x_new[0,:]**2 + x_new[1,:]**2)*(self.gamma*self.beta)
    
    @property
    def crazy_action_y(self):
        sig = self.cov('x', 'xp', 'y', 'yp')
        S = np.array([[0, 1, 0, 0],[-1,0,0,0],[0,0,0,1],[0,0,-1,0]])
        Q = np.array([[1, 1j, 0, 0], [1, -1j, 0, 0], [0, 0, 1, 1j], [0, 0, 1, -1j]]) / np.sqrt(2.0)
        (w, E) = np.linalg.eig(sig.dot(S))
        E = E / np.power(np.abs(np.linalg.det(E)), 0.25)
        E = E[:,[1,0,3,2]] # reorder to make N symplectic
        N = E.dot(Q)
        N = N.real 
        x = np.array([self.x,self.xp, self.y, self.yp])
        x_new = np.linalg.solve(N, x)
        return 0.5*(x_new[2,:]**2 + x_new[3,:]**2)*(self.gamma*self.beta)
    
    @property
    def action_4d(self):
        sig = self.cov('x', 'xp', 'y', 'yp')
        S = np.array([[0, 1, 0, 0],[-1,0,0,0],[0,0,0,1],[0,0,-1,0]])
        Q = np.array([[1, 1j, 0, 0], [1, -1j, 0, 0], [0, 0, 1, 1j], [0, 0, 1, -1j]]) / np.sqrt(2.0)
        (w, E) = np.linalg.eig(sig.dot(S))
        E = E / np.power(np.abs(np.linalg.det(E)), 0.25)
        E = E[:,[1,0,3,2]] # reorder to make N symplectic
        N = E.dot(Q)
        N = N.real 
        x = np.array([self.x,self.xp, self.y, self.yp])
        x_new = np.linalg.solve(N, x)
        Ju = 0.5*(x_new[0,:]**2 + x_new[1,:]**2)*(self.gamma*self.beta)
        Jv = 0.5*(x_new[2,:]**2 + x_new[3,:]**2)*(self.gamma*self.beta)
        return np.sqrt(Ju*Jv)

#-----------------------------------------
# helper functions for ParticleGroupExtension class

def slice_emit(p_list, key):
    min_particles = 5
    weights = np.array([0.0 for p in p_list])
    emit = np.array([0.0 for p in p_list])
    for p_i, p in enumerate(p_list):
        if (p.n_particle >= min_particles):
            emit[p_i] = p[key]
            weights[p_i] = p['charge']
    weights = weights/np.sum(weights)
    avg_emit = np.sum(emit*weights)
    
    return avg_emit



def convert_gpt_data(gpt_data_input):
    gpt_data = copy.deepcopy(gpt_data_input)
    for i, pmd in enumerate(gpt_data_input.particles):
        gpt_data.particles[i] = ParticleGroupExtension(input_particle_group=pmd)  # This copies the data again
    #for tout in gpt_data.tout:
    #    tout.drift_to_z() # Turn all the touts into quasi-screens
    return gpt_data



def divide_particles(particle_group, nbins = 100, key='t'):
    """
    Splits a particle group into even slices of 'key'. Returns a list of particle groups. 
    """
    x = getattr(particle_group, key) 
    
    is_radial_var = False
    if (key == 'r' or key == 'r_centered' or key == 'rp'):
        is_radial_var = True
    
    if (is_radial_var):
        x = x*x
        xmin = 0  # force r=0 as min, could use min(x) here, optionally
        xmax = max(x)
        dx = (xmax-xmin)/(nbins-1)
        edges = np.linspace(xmin, xmax + 0.01*dx, nbins+1) # extends slightly further than max(r2)
        dx = edges[1]-edges[0]
    else:
        dx = (max(x)-min(x))/(nbins-1)
        edges = np.linspace(min(x) - 0.01*dx, max(x) + 0.01*dx, nbins+1) # extends slightly further than range(r2)
        dx = edges[1]-edges[0]
    
    which_bins = np.digitize(x, edges)-1
    
    if (is_radial_var):
        x = np.sqrt(x)
        edges = np.sqrt(edges)
            
    # Split particles
    plist = []
    for bin_i in range(nbins):
        chunk = which_bins==bin_i
        # Prepare data
        data = {}
        #keys = ['x', 'px', 'y', 'py', 'z', 'pz', 't', 'status', 'weight'] 
        for k in particle_group._settable_array_keys:
            data[k] = getattr(particle_group, k)[chunk]
        # These should be scalars
        data['species'] = particle_group.species
        
        # New object
        p = ParticleGroupExtension(data=data)
        plist.append(p)
    
    # normalization for sums of particle properties, = 1 / histogram bin width
    if (is_radial_var):
        density_norm = 1.0/(np.pi*(edges[1]**2 - edges[0]**2))
    else:
        density_norm = 1.0/(edges[1] - edges[0])
    
    return plist, edges, density_norm



def core_emit_calc_4d(x, xp, y, yp, w, show_fit=False):

    x = copy.copy(x)
    xp = copy.copy(xp)
    y = copy.copy(y)
    yp = copy.copy(yp)
    
    sumw = np.sum(w)
    
    x = x - np.sum(x*w)/sumw
    xp = xp - np.sum(xp*w)/sumw
    y = y - np.sum(y*w)/sumw
    yp = yp - np.sum(yp*w)/sumw
    
    x2 = np.sum(x*x*w)/sumw
    y2 = np.sum(y*y*w)/sumw

    u2 = (x2+y2)/2.0

    xpy = np.sum(x*yp*w)/sumw
    ypx = np.sum(y*xp*w)/sumw

    L = (xpy-ypx)/2.0

    C = -L/u2

    xp = xp - C*y
    yp = yp + C*x

    ec4x = core_emit_calc(x, xp, w, show_fit=show_fit)
    ec4y = core_emit_calc(y, yp, w, show_fit=show_fit)

    return 0.5*(ec4x+ec4y)


def core_emit_calc(x, xp, w, show_fit=False):

    x = copy.copy(x)
    xp = copy.copy(xp)
    
    emit_change_factor = 3 # fit data in range where emittance changes by less than this factor
    
    min_particle_count = 10000   # minimum number of particles required to compute a core emittance
    average_count_per_bin = int(min([len(x)/50, 1000]))

    if (len(x) < min_particle_count):
        raise ValueError('Too few particles to calculate core emittance.')

    x = x - np.sum(x*w)/np.sum(w)
    xp = xp - np.sum(xp*w)/np.sum(w)
    
    u0 = np.vstack((x, xp))
    sigma_matrix = np.cov(u0, aweights=w)
            
    if (np.sqrt(np.linalg.det(sigma_matrix)) < 1e-11):
        print('Possible zero emittance found, assuming core emittance is zero.')
        return 0

    # Change into better (round phase space) coordinates
    (_, V) = np.linalg.eig(sigma_matrix)
    u1 = np.linalg.solve(V, u0)

    # Now get the sigma matrix in the new coordinates
    sigma_matrix = np.cov(u1, aweights=w)
    
    r = np.sqrt(np.array([1.0/np.diag(sigma_matrix)]).dot(u1**2))[0]    
    dr = np.sort(r)[average_count_per_bin-1] # first dr includes exactly average_count_per_bin particles
    
    rbin = np.arange(0, np.max(r), dr)
    
    rhor = np.histogram(r, bins=rbin)[0]
        
    rbin = rbin[0:-1] + 0.5*(rbin[1] - rbin[0])
    rhonorm = np.trapz(rhor, rbin)
    
    rho = rhor / (rbin * rhonorm * 2 * np.pi * np.sqrt(np.prod(np.diag(sigma_matrix))));
            
    emit_in_range = rho > np.max(rho) / emit_change_factor
    max_fit_r = np.max(rbin[emit_in_range])
    plot_range = rbin < max_fit_r
    
    core_eps = 1.0/(4.0 * np.pi * rho[plot_range])
    rbin_fit = rbin[plot_range]   
        
    best_fit = poly.polyfit(rbin_fit, core_eps, 2);
    
    ec = best_fit[0]
    
    if (show_fit):
        plt.figure()
        p_list = []
        leg_list = []
        
        line_handle, = plt.plot(rbin[plot_range], core_eps, 'o')
        p_list.append(line_handle)
        leg_list.append('Data')
        
        r_plot = np.linspace(0, np.max(rbin_fit), 300)
        line_handle, = plt.plot(r_plot, poly.polyval(r_plot, best_fit), '-')
        p_list.append(line_handle)
        leg_list.append('Fit')
                                
        plt.xlim([0, np.max(rbin_fit)])
        plt.ylim([0, 1.1*np.max(core_eps)])
        plt.xlabel('Normalized radius^2');
        plt.ylabel('Emittance');
        plt.legend(p_list, leg_list)
    
        
    
    return ec