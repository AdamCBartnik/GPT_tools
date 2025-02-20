import sys, os, copy, time
import numpy as np
from scipy.special import spence, factorial
from mpmath import polylog
from distgen import Generator
from GPT_tools.GPTExtension import get_cathode_particlegroup
from pint import UnitRegistry

def float_polylog(n, x):
    return float(polylog(n,x))

np_polylog = np.frompyfunc(float_polylog, 2, 1)


# -----------------------------------------------------------------------------
# This is the main function
# -----------------------------------------------------------------------------
def MakeMetalParticleGroup(settings, DISTGEN_INPUT_FILE=None, verbose=True):
    # Makes a normal particlegroup using settings and distgen_input_file, and then overwrites the momentum distribution with
    # the distribution from a flat density of states metal. Useful only for modeling individual electrons 
    #    The following values in settings are needed:
    #
    #    settings['start:MTE:value'] : desired MTE (in the limit where the particles are far apart)
    #    settings['kT:value'] : Temperature
    #    settings['gun_field:value'] : Field at the cathode surface
    #    Two choices to specify QE: (here, QE means the fraction of emitted electrons that have enough energy to get over the image charge barrier)
    #          settings['QE'] : Directly set a QE (in the limit where the particles are far apart)
    #          settings['cathode_z_offset'] : Directly set the cathode offset value in the image charge model
    #
    #    Note: two values of settings are modified (or added) in this code:
    #    settings['cathode_z_offset'] : This value is overwritten or created in SI units, intended to be used in GPT
    #    settings['gun_field'] : This value is overwritten or created in SI units, intended to be used in GPT

    (EexcAtSurface, EexcAtPeak, kT) = getEexc(settings, modify_settings=True, verbose=verbose)
    return MakeEnergyDist(get_cathode_particlegroup(settings, DISTGEN_INPUT_FILE=DISTGEN_INPUT_FILE), EexcAtSurface, kT)




def getEexc(settings, modify_settings=True, verbose=True):
    unit_registry = UnitRegistry()
    E1 = 1.43996455e-9  #  e^2/(4*pi*epsilon_0) in eV-meters
    
    if ('start:MTE:value' not in settings or 'start:MTE:units' not in settings):
        if (verbose):
            print('Need start:MTE:value and units in settings')
        return None   
    MTE_value = settings['start:MTE:value']
    MTE_units = settings['start:MTE:units']
    MTE = MTE_value * unit_registry.parse_expression(MTE_units)
    MTE = MTE.to('eV').magnitude # convert to eV

    if ('kT:value' not in settings or 'kT:units' not in settings):
        if (verbose):
            print('Need kT:value and kT:units in settings')
        return None
    kT_value = settings['kT:value']
    kT_units = settings['kT:units']
    kT = kT_value * unit_registry.parse_expression(kT_units)
    kT = kT.to('eV').magnitude # convert to eV  
    
    if ('gun_field:value' not in settings or 'gun_field:units' not in settings):
        if (verbose):
            print('Need gun_field:value and gun_field:units in settings')
        return None 
    gun_field_value = settings['gun_field:value']
    gun_field_units = settings['gun_field:units']
    gun_field = gun_field_value * unit_registry.parse_expression(gun_field_units)
    gun_field = gun_field.to('V/m').magnitude # convert to V/m    
    
    if (MTE <= 1.096144454*kT):
        if (verbose):
            print(f'MTE must be larger than 9*zeta(3)/pi^2*kT = {1.096144454*kT}')
        return None    
    
    EexcAtPeak = inv_MTE_model(MTE, kT)
    
    if ('QE' in settings and 'cathode_z_offset:value' in settings and 'cathode_z_offset:units' in settings):
        if (verbose):
            print('Error, specify only QE or cathode_z_offset')
        return None
    
    if ('QE' in settings):
        QE = settings['QE']
        EexcAtSurface = inv_QE_model(QE, EexcAtPeak, kT)
        alp = np.sqrt(E1*gun_field)
        z0 = (EexcAtSurface - EexcAtPeak + alp - np.sqrt((EexcAtSurface - EexcAtPeak)*(EexcAtSurface - EexcAtPeak + 2.0*alp)))/(2.0*gun_field)
    else:
        if ('cathode_z_offset:value' not in settings or 'cathode_z_offset:units' not in settings):
            if (verbose):
                print('Need either QE or both cathode_z_offset:value and cathode_z_offset:units in settings')
            return None 
        cathode_z_offset_value = settings['cathode_z_offset:value']
        cathode_z_offset_units = settings['cathode_z_offset:units']
        z0 = cathode_z_offset_value * unit_registry.parse_expression(cathode_z_offset_units)
        z0 = z0.to('m').magnitude # convert to m

    if (gun_field < 0):
        if (verbose):
            print('Warning: changing sign of gun field')
        gun_field = np.abs(gun_field)

    # Update settings 
    if (modify_settings):
        settings['cathode_z_offset'] = z0  # now in SI units for GPT
        settings['gun_field'] = gun_field  # now in SI units for GPT
        if (verbose):
            print(f'Adding settings["cathode_z_offset"] = {settings["cathode_z_offset"]} for use in GPT')
            print(f'Adding settings["gun_field"] = {settings["gun_field"]} for use in GPT')
        
    zpeak = 0.5*np.sqrt(E1/gun_field) - z0
    barrierV = ImagePotential(zpeak, z0, 0.0, gun_field) - ImagePotential(0, z0, 0.0, gun_field)
    EexcAtSurface = EexcAtPeak + barrierV
    QEatPeak = QE_model(EexcAtPeak, kT, EexcAtSurface)
    if (verbose):
        print(f'Initial N = {settings["n_particle"]:.0f} particles = {1.60217663e-1*settings["n_particle"]:.3g} aC')
        print(f'Predicted final N = {settings["n_particle"]*QEatPeak:.1f} particles = {1.60217663e-1*settings["n_particle"]*QEatPeak:.3g} aC = {100*QEatPeak:.3g}% QE')
        print(f'Predicted final MTE = {1e3*MTE_model(EexcAtPeak, kT):.3g} meV')
        print(f'Peak potential barrier at z = {1e9*zpeak:.3g} nm')
        
    return (EexcAtSurface, EexcAtPeak, kT)

def MakeEnergyDist(pg, EexcAtSurface, kT):   
    # Make energy distribution for the constant DoS model
    #    EexcAtSurface: Excess energy at cathode surface, eV
    #    kT: eV
    pnorm = 1010.93912  # sqrt(2* (electron mass) * (1 eV)) in eV/c
    Ekin = invEcumul(np.random.rand(len(pg)), EexcAtSurface, kT)
    (pr, pz) = uniform_pr2_dist(len(pg))
    pz = np.abs(pz)    
    pr = pr * pnorm * np.sqrt(Ekin)
    pz = pz * pnorm * np.sqrt(Ekin)
    phi = 2 * np.pi * np.random.rand(len(pg))
    pg.pz = pz
    pg.px = pr * np.cos(phi)
    pg.py = pr * np.sin(phi)
    
    pg.weight = 1.60217663e-19
    
    return pg

def ImagePotential(z, z0, r0, Egun):
    # Potential from a constant field gun and an image charge
    #    r0 : Plummer radius, m
    #    z0 : effective cathode offset, m
    #    Egun : Gun field in V/m
    #    Output : Energy in eV
    
    z = 1e9 * z
    r0 = 1e9 * r0
    z0 = 1e9 * z0
    Egun = 1e-6 * Egun
    return -1.43996455 / (2.0 * np.sqrt(r0**2 + (2 * (z + z0))**2)) - 1.0e-3*Egun * z

def inv_MTE_model(MTE, kT, MTEtol = 1.0e-9):
    # Uses Newton's method to invert MTE(peak excess energy)
    
    MTE = np.array(MTE)
    guess = np.array(3.0*MTE)
    
    fguess = np.array(MTE_model(guess, kT))
    needs_work = np.array(np.abs(fguess - MTE) > MTEtol)
                
    while (np.any(needs_work)):
        guess[needs_work] = guess[needs_work] - (fguess[needs_work] - MTE[needs_work])/dE_MTE_model(guess[needs_work], kT)
        fguess[needs_work] = MTE_model(guess[needs_work], kT)
        needs_work = np.array(np.abs(fguess - MTE) > MTEtol)

    return guess

def inv_QE_model(QE, Eexcz, kT, QEtol = 1.0e-9):
    # Uses Newton's method to invert QE(surface excess energy)
    
    QE = np.array(QE)
    guess = np.array(2*Eexcz)
    Eexcz = np.array(Eexcz)
    
    fguess = np.array(QE_model(Eexcz, kT, guess))
    needs_work = np.array(np.abs(fguess - QE) > QEtol)

    while (np.any(needs_work)):
        guess[needs_work] = guess[needs_work] - (fguess[needs_work] - QE[needs_work])/dE_QE_model(Eexcz[needs_work], kT, guess[needs_work])
        fguess[needs_work] = np.array(QE_model(Eexcz, kT, guess))
        needs_work = np.array(np.abs(fguess - QE) > QEtol)

    return guess

def MTE_model(Eexcz, kT):
    # Expected MTE given an excess energy and kT, for the constant DoS model
    #    Eexcz : Excess energy (at position z) : eV
    #    kT : eV
    return (kT*np_polylog(3, -np.exp(Eexcz/kT)))/spence(np.exp(Eexcz/kT)+1.0)

def QE_model(Eexcz, kT, Eexc0):
    # Expected QE given an excess energy and kT, for the constant DoS model
    #    Eexcz : Excess energy (at position z) : eV
    #    kT : eV
    #    Eexc0 : Excess energy (at z=0) : eV
    return spence(np.exp(Eexcz/kT)+1.0)/spence(np.exp(Eexc0/kT)+1.0)

def dE_MTE_model(Eexcz, kT):
    # Derivative of MTE w.r.t. Eexcz
    #    Eexcz : Excess energy (at position z) : eV
    #    kT : eV
    return 1.0 + (np.log(1.0 + np.exp(Eexcz/kT))*np_polylog(3, -np.exp(Eexcz/kT)))/spence(np.exp(Eexcz/kT)+1.0)**2

def dE_QE_model(Eexcz, kT, Eexc0):
    # Derivative of QE w.r.t. Eexc0
    #    Eexcz : Excess energy (at position z) : eV
    #    kT : eV
    #    Eexc0 : Excess energy (at z=0) : eV
    return np.log(1.0 + np.exp(Eexc0/kT))*spence(np.exp(Eexcz/kT)+1.0)/spence(np.exp(Eexc0/kT)+1.0)**2/kT

def invEcumul(p, Eexc, kT, ptol=1.0e-7):
    # Uses Newton's method to invert the cumulative probability distribution of kinetic energy
    
    p1 = np.exp(Eexc/kT)
    p3 = spence(p1 + 1.0)
    guess = np.sqrt(p/(-0.5*p1/((1.0+p1)*kT*kT*p3)))

    fguess = Ecumulprob(guess, Eexc, kT)
    needs_work = np.abs(fguess - p) > ptol
    
    while (np.any(needs_work)):
        guess[needs_work] = guess[needs_work] - (fguess[needs_work] - p[needs_work])/dEcumulprob(guess[needs_work], Eexc, kT)
        fguess[needs_work] = Ecumulprob(guess[needs_work], Eexc, kT)
        needs_work = np.abs(fguess - p) > ptol

    return guess

def Ecumulprob(Ekin, Eexc, kT):
    # Cumulative probaiblity distribution of kinetic energy
    
    Ek = Ekin/kT
    Ee = Eexc/kT
    eEeEk = np.exp(Ee-Ek)
    p1 = Ek*np.log(1.0+eEeEk)
    p2 = spence(eEeEk + 1.0)
    p3 = spence(np.exp(Ee) + 1.0)
    return 1 + p1/p3 - p2/p3

def dEcumulprob(Ekin, Eexc, kT):
    # Derivative of the cumulative probaiblity distribution of kinetic energy w.r.t. energy
    
    Ek = Ekin/kT
    Ee = Eexc/kT

    p1 = np.exp(Ee-Ek)
    p3 = spence(np.exp(Ee) + 1.0)

    return -Ek*p1/((1.0+p1)*p3*kT)

def uniform_pr2_dist(n):
    # Generates a uniform distribution of pr^2
    
    u = np.random.rand(n)
    pr = np.sqrt(u)
    pz = np.sqrt(1.0-u)
    return (pr,pz)

def get_blank_particlegroup(n_particle, verbose=False):
    # Returns an uninitialized particlegroup of size N. Distgen seems to suck at this for small N
    
    variables = ['x', 'y', 'z', 'px', 'py', 'pz', 't']
    phasing_distgen_input = {'n_particle':n_particle, 'random_type':'hammersley', 'total_charge':{'value':1.0, 'units':'pC'}, 'species':'electron', 'start': {'type':'time', 'tstart':{'value': 0.0, 'units': 's'}},}
    gen = Generator(phasing_distgen_input, verbose=verbose) 
    gen.run()
    PG = gen.particles
    return PG
