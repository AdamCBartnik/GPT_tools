from pmd_beamphysics import ParticleGroup
import numpy as np
import copy
import scipy

def inverse_compton_scatter(PG_e, n_macro_per_electron=5, max_theta=5, laser_wavelength_in_nm=1030, laser_sigma_in_micron=30, laser_energy_in_mJ=1):
    # max_theta is in units of 1/gamma for each particle
    
    data = {}
    data['x'] = np.zeros(n_macro_per_electron)
    data['y'] = np.zeros(n_macro_per_electron)
    data['z'] = np.zeros(n_macro_per_electron)
    data['t'] = np.zeros(n_macro_per_electron)
    data['px'] = np.zeros(n_macro_per_electron)
    data['py'] = np.zeros(n_macro_per_electron)
    data['pz'] = np.zeros(n_macro_per_electron)
    data['weight'] = np.ones(n_macro_per_electron)
    data['species'] = 'electron'
    data['status'] = np.ones(n_macro_per_electron)
    
    mean_t = PG_e['mean_t']
    
    E_laser = 1239.84198/laser_wavelength_in_nm
    N_laser = laser_energy_in_mJ * 6.24150907e15 / E_laser
    r02 = (2.8179403262e-15)**2
    c = 299792458
    w0 = 2 * laser_sigma_in_micron * 1e-6
    zR = np.pi * w0*w0/(laser_wavelength_in_nm * 1.0e-9)
    
    costh0 = 1.0 # head on collision, assumed in other parts of the code
        
    sampler = scipy.stats.qmc.Halton(d=2, scramble=True)
    
    PG_template = ParticleGroup(data=data)
    
    PG_list = []
    
    for p in PG_e:
        PG = copy.deepcopy(PG_template)
        
        x = p.x[0]
        y = p.y[0]
        t = p.t[0] - mean_t
        px = p.px[0]
        py = p.py[0]
        pz = p.pz[0]

        n = np.array([px,py,pz]) / np.sqrt(px*px + py*py + pz*pz)

        # make rotation matrix from lab coords to particle coords
        # So, R @ n = [[0],[0],[1]]
        nm = np.sqrt(n[0]*n[0] + n[2]*n[2])
        R = np.array([[n[2]/nm, 0, -n[0]/nm],[ -n[0]*n[1]/nm, -(n[1]*n[1]-1.0)/nm, -n[1]*n[2]/nm],[n[0], n[1], n[2]]])
        n = np.array([n]).T

        g = p.gamma[0]
        b = p.beta[0]
        Ne = np.abs(p.weight[0]) / 1.60217663e-19
        
        sample = sampler.random(n=n_macro_per_electron)
        phi = 2 * np.pi * sample[:,0]
        th = max_theta/g * np.sqrt(sample[:,1])
    
        th2g2 = th*th*g*g
        Egamma = E_laser * 2*g*g*(1 + b*costh0)/(1 + th2g2)
        dsdw = 4*r02*g*g*(1 - (4*th2g2*np.sin(phi)**2)/(1 + th2g2)**2)/(1 + th2g2)**2 # units of area / angle     
        
        r2 = x*x + y*y
        ct = c*t
        xth = x*th
        zR2 = zR*zR
        w02 = w0*w0
        laser_factor = 4.0*N_laser*zR2*(1+b)*np.exp(-8.0*zR2*(r2 + ct*xth)/(w02*(4.0*zR2 + (ct-xth)**2)))*(4.0*zR2 + ct*(ct+2.0*xth)) / (np.pi*w02*(ct*ct + 4.0*zR2)**2)
        
        px = th*np.cos(phi)  # small angle.... units of Egamma/c
        py = th*np.sin(phi)
        pz = np.sqrt(1 - px*px - py*py)
        
        pxpypz = [px, py, pz] 
        pxpypz = (R.T)@pxpypz

        deltaomega = np.pi*(max_theta/g)**2/n_macro_per_electron # angle occupied by each particle
        
        PG.px = Egamma * pxpypz[0, :] # eV/c
        PG.py = Egamma * pxpypz[1, :]
        PG.pz = Egamma * pxpypz[2, :]
        PG.weight = Ne * dsdw * deltaomega  * laser_factor # 
        
        if Ne > 0.0:
            PG_list.append(PG)

    PG_sum = copy.deepcopy(PG_list[0])
        
    for ii in np.arange(1,len(PG_list)):
        PG_sum = PG_sum + PG_list[ii]
        
    return PG_sum
            
        
    
    
    