import copy
import numpy as np
from scipy.special import spence
from mpmath import polylog
from GPT_tools.GPTExtension import get_cathode_particlegroup
from pint import UnitRegistry

def float_polylog(n, x):
    return float(polylog(n,x))

np_polylog = np.frompyfunc(float_polylog, 2, 1)


# -----------------------------------------------------------------------------
# This is one of the main functions
# -----------------------------------------------------------------------------
def MakeMetalParticleGroup(settings, DISTGEN_INPUT_FILE=None, verbose=True, only_survivors=False, rng = np.random.default_rng()):
    # Makes a normal particlegroup using settings and DISTGEN_INPUT_FILE, and then overwrites the momentum distribution with
    # the distribution from a flat density of states metal. Useful only for modeling individual electrons
    #    only_survivors: If true, then it only makes particles that will definitely escape the barrier
    #
    #    The following values in settings are needed:
    #
    #    settings['photon_energy:value'] : Energy of exciting photon
    #    settings['work_function:value'] : Metal work function   (only the difference between the photon energy and this actually matter)
    #    settings['kT:value'] : Temperature of cathode
    #    settings['gun_field:value'] : Field at the cathode surface
    #    settings['cathode_z_offset:value'] : Cathode 'fudge-factor' that keeps the potential finite at z=0
    #
    #    Note: two values of settings are modified (or added) in this code:
    #    settings['cathode_z_offset'] : This value is overwritten or created in SI units, intended to be used in GPT
    #    settings['gun_field'] : This value is overwritten or created in SI units, intended to be used in GPT

    (EexcAtSurface, EexcAtPeak, kT) = getMetalEexc(settings, modify_settings=True, verbose=verbose)
    
    if (only_survivors):
        PG = MakeMetalEnergyDist(get_cathode_particlegroup(settings, DISTGEN_INPUT_FILE=DISTGEN_INPUT_FILE), EexcAtPeak, kT, rng=rng)
        barrierV = EexcAtSurface - EexcAtPeak
        pz_min = 1010.93912*np.sqrt(barrierV) # goes from eV to eV/c for an electron
        PG.pz = np.sqrt(PG.pz**2 + pz_min**2) # add energy to get over barrier
        
    else:
        PG = MakeMetalEnergyDist(get_cathode_particlegroup(settings, DISTGEN_INPUT_FILE=DISTGEN_INPUT_FILE), EexcAtSurface, kT, rng=rng)
    
    return PG


# -----------------------------------------------------------------------------
# This is one of the main functions
# -----------------------------------------------------------------------------
def MakeSemiconductorParticleGroup(settings, DISTGEN_INPUT_FILE=None, verbose=True, only_survivors=False, rng = np.random.default_rng()):
    # Makes a normal particlegroup using settings and DISTGEN_INPUT_FILE, and then overwrites the momentum distribution with
    # the distribution from a parabolic density of states with energy gap. Useful only for modeling individual electrons
    #    only_survivors: If true, then it only makes particles that will definitely escape the barrier
    #
    #    The following values in settings are needed:
    #
    #    settings['photon_energy:value'] : Energy of exciting photon
    #    settings['energy_gap:value'] : Semiconductor energy gap  (only the difference between the photon energy and this actually matter)
    #    settings['electron_affinity:value'] : Semiconductor electron affinity. Negative = easier to escape
    #    settings['gun_field:value'] : Field at the cathode surface
    #    settings['cathode_z_offset:value'] : Cathode 'fudge-factor' that keeps the potential finite at z=0
    #
    #    Optional:
    #    settings['effective_mass'] : Positive m*/m_e, defaults to 1.0.
    #                                 Transverse momentum is always conserved at the interface.
    #
    #    Note: two values of settings are modified (or added) in this code:
    #    settings['cathode_z_offset'] : This value is overwritten or created in SI units, intended to be used in GPT
    #    settings['gun_field'] : This value is overwritten or created in SI units, intended to be used in GPT

    (EexcAtSurface, EexcAtPeak, EaSurf) = getSemiconductorEexc(settings, modify_settings=True, verbose=verbose)

    effective_mass = _semi_mass(settings.get('effective_mass', 1.0))

    if (only_survivors):
        PG = MakeSemiconductorEnergyDist(
            get_cathode_particlegroup(settings, DISTGEN_INPUT_FILE=DISTGEN_INPUT_FILE),
            EexcAtPeak,
            EaSurf + (EexcAtSurface - EexcAtPeak),
            effective_mass=effective_mass,
            rng=rng,
        )
        barrierV = EexcAtSurface - EexcAtPeak
        pz_min = 1010.93912*np.sqrt(barrierV) # goes from eV to eV/c for an electron
        PG.pz = np.sqrt(PG.pz**2 + pz_min**2) # add energy to get over barrier
    else:
        PG = MakeSemiconductorEnergyDist(
            get_cathode_particlegroup(settings, DISTGEN_INPUT_FILE=DISTGEN_INPUT_FILE),
            EexcAtSurface,
            EaSurf,
            effective_mass=effective_mass,
            rng=rng,
        )
    
    return PG
    

# -----------------------------------------------------------------------------
# This is one of the main functions
# -----------------------------------------------------------------------------
def MakeEnergyOffsetParticleGroup(settings, DISTGEN_INPUT_FILE=None, verbose=True):
    # Makes a normal particlegroup using settings and DISTGEN_INPUT_FILE, and then adds enough extra pz to overcome the 
    # image charge barrier. Useful only for modeling individual electrons
    #    The following values in settings are needed:
    #    settings['cathode_z_offset:value'] and settings['cathode_z_offset:units']   : offset in modified image charge model
    #    settings['gun_field:value'] and settings['gun_field:units']:   Field at the cathode surface
    #
    #    Note: two values of settings are modified (or added) in this code:
    #    settings['cathode_z_offset'] : This value is overwritten or created in SI units, intended to be used in GPT
    #    settings['gun_field'] : This value is overwritten or created in SI units, intended to be used in GPT

    gun_field = getValueFromSettings(settings, 'gun_field', 'V/m', modify_settings=True, verbose=verbose)
    z0 = getValueFromSettings(settings, 'cathode_z_offset', 'm', modify_settings=True, verbose=verbose)
    plummer_radius = getValueFromSettings(settings, 'plummer_radius', 'm', modify_settings=True, verbose=verbose)
                    
    zpeak = PeakPotentialz(gun_field, z0, plummer_radius)
    barrierV = ImagePotential(zpeak, z0, plummer_radius, gun_field) - ImagePotential(0, z0, plummer_radius, gun_field)

    delta_pz = 1010.93912*np.sqrt(barrierV) # goes from eV to eV/c for an electron
    PG = get_cathode_particlegroup(settings, DISTGEN_INPUT_FILE=DISTGEN_INPUT_FILE)
    PG.pz = np.sqrt(PG.pz**2 + delta_pz**2)  # add energy to get over barrier
    PG.weight = 1.60217663e-19   # force single electrons
    
    return PG


def getValueFromSettings(settings, value, desired_units, modify_settings=True, verbose=True):
    unit_registry = UnitRegistry()
    val = None
    
    if (value+':value' in settings and value+':units' in settings):
        val = settings[value+':value']
        units = settings[value+':units']
        val = val * unit_registry.parse_expression(units)
        val = val.to(desired_units).magnitude 
        if (modify_settings):
            settings[value] = val  
            if (verbose):
                print(f'Adding settings["{value}"] = {settings[value]} for use in GPT')
    else:
        if (value in settings):
            if (verbose):
                print(f'Assuming settings["{value}"] is in units = {desired_units}')
            val = settings[value]
        else:
            if (verbose):
                print(f'Need either (a) {value}:value and {value}:units or (b) {value} in settings')
    return val
    
def getMetalEexc(settings, modify_settings=True, verbose=True):
    # Gets the excess energy at both the peak of the image charge potential and the cathode surface
    # The final MTE of the bunch is just a function of the excess energy at the peak, while
    # the QE is a function of both
    #
    # If modify_settings=True, then settings['cathode_z_offset'] and settings['gun_field'] are added to 
    # the dictionary (to be used outside this function) in SI units. These are derived from value/unit pairs in settings
    #
    # Returns : (EexcAtSurface, EexcAtPeak, kT)
    
    E1 = 1.43996455e-9  #  e^2/(4*pi*epsilon_0) in eV-meters

    kT = getValueFromSettings(settings, 'kT', 'eV', modify_settings=modify_settings, verbose=verbose)
    gun_field = getValueFromSettings(settings, 'gun_field', 'V/m', modify_settings=modify_settings, verbose=verbose)
    plummer_radius = getValueFromSettings(settings, 'plummer_radius', 'm', modify_settings=modify_settings, verbose=verbose)

    if (gun_field < 0):
        if (verbose):
            print('Warning: changing sign of gun field')
        gun_field = np.abs(gun_field)
    
    phi = getValueFromSettings(settings, 'work_function', 'eV', modify_settings=False, verbose=False) # ignore user modify_settings, don't add 'start:MTE' to settings
    hv = getValueFromSettings(settings, 'photon_energy', 'eV', modify_settings=False, verbose=False) # ignore user modify_settings, don't add 'start:MTE' to settings

    if (phi is not None and hv is not None):
        # user is picking the work function, photon energy, and cathode_z_offset
        z0 = getValueFromSettings(settings, 'cathode_z_offset', 'm', modify_settings=modify_settings, verbose=verbose)
        if (z0 is None):
            print('Need to specify cathode_z_offset')
            return None
        EexcAtSurface = hv - phi - ImagePotential(0, z0, plummer_radius, gun_field)
        zpeak = PeakPotentialz(gun_field, z0, plummer_radius)
        EexcAtPeak = hv - phi - ImagePotential(zpeak, z0, plummer_radius, gun_field)
        
    else:
        # user is picking the desired MTE and either the QE or the cathode_z_offset
        MTE = getValueFromSettings(settings, 'start:MTE', 'eV', modify_settings=False, verbose=verbose) # ignore user modify_settings, don't add 'start:MTE' to settings
        
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
            if (modify_settings):
                settings['cathode_z_offset'] = z0
                if (verbose):
                    print(f'Adding settings["cathode_z_offset"] = {settings["cathode_z_offset"]} for use in GPT')
        else:
            z0 = getValueFromSettings(settings, 'cathode_z_offset', 'm', modify_settings=modify_settings, verbose=verbose)

        zpeak = PeakPotentialz(gun_field, z0, plummer_radius)
        barrierV = ImagePotential(zpeak, z0, plummer_radius, gun_field) - ImagePotential(0, z0, plummer_radius, gun_field)
        EexcAtSurface = EexcAtPeak + barrierV
        
    QEatPeak = QE_model(EexcAtPeak, kT, EexcAtSurface)
    MTEatPeak = MTE_model(EexcAtPeak, kT)
    
     # Update settings 
    if (modify_settings and 'QE' not in settings):
        settings['QE'] = QEatPeak
        if (verbose):
            print(f'Adding settings["QE"] = {settings["QE"]} for use in GPT')

    if (modify_settings and 'MTE' not in settings):
        settings['MTE'] = MTEatPeak
        if (verbose):
            print(f'Adding settings["MTE"] = {settings["MTE"]} for use in GPT')
    
    if (verbose):
        print(f'Initial N = {settings["n_particle"]:.0f} particles = {1.60217663e-1*settings["n_particle"]:.3g} aC')
        print(f'Predicted final N = {settings["n_particle"]*QEatPeak:.1f} particles = {1.60217663e-1*settings["n_particle"]*QEatPeak:.3g} aC = {100*QEatPeak:.3g}% QE')
        print(f'Predicted final MTE = {1e3*MTEatPeak:.3g} meV')
        print(f'Peak potential barrier at z = {1e9*zpeak:.5g} nm')
        print(f'Eexc at surface = {EexcAtSurface}, Eexc at peak = {EexcAtPeak}, kT = {kT}')
    
    return (EexcAtSurface, EexcAtPeak, kT)


def getSemiconductorEexc(settings, modify_settings=True, verbose=True):
    # Gets the excess energy at both the peak of the image charge potential and the cathode surface
    # The final MTE of the bunch is just a function of the excess energy at the peak, while
    # the QE is a function of both
    #
    # If modify_settings=True, then settings['cathode_z_offset'] and settings['gun_field'] are added to 
    # the dictionary (to be used outside this function) in SI units. These are derived from value/unit pairs in settings
    #
    # Returns : (EexcAtSurface, EexcAtPeak, Ea + V(0))
    
    gun_field = getValueFromSettings(settings, 'gun_field', 'V/m', modify_settings=modify_settings, verbose=verbose)
    z0 = getValueFromSettings(settings, 'cathode_z_offset', 'm', modify_settings=modify_settings, verbose=verbose)
    plummer_radius = getValueFromSettings(settings, 'plummer_radius', 'm', modify_settings=modify_settings, verbose=verbose)

    if (gun_field < 0):
        if (verbose):
            print('Warning: using opposite sign of gun field')
        gun_field = np.abs(gun_field)
    
    Ea = getValueFromSettings(settings, 'electron_affinity', 'eV', modify_settings=False, verbose=False)
    Eg = getValueFromSettings(settings, 'energy_gap', 'eV', modify_settings=False, verbose=False) 
    hv = getValueFromSettings(settings, 'photon_energy', 'eV', modify_settings=False, verbose=False) 

    if (z0 is None or Ea is None or Eg is None or hv is None):
        print('Need to specify cathode_z_offset, electron_affinity, energy_gap, and photon_energy')
        return None

    EexcAtSurface = hv - Eg - Ea - ImagePotential(0, z0, plummer_radius, 0.0) # at z=0, doesn't depend on gun field
    zpeak = PeakPotentialz(gun_field, z0, plummer_radius)
    EexcAtPeak = hv - Eg - Ea - ImagePotential(zpeak, z0, plummer_radius, gun_field) 
    
    if (verbose):
        print(f'Peak potential barrier at z = {1e9*zpeak:.3g} nm')
        print(f'Eexc at surface = {EexcAtSurface}, Eexc at peak = {EexcAtPeak}')
    
    return (EexcAtSurface, EexcAtPeak, Ea + ImagePotential(0, z0, plummer_radius, gun_field))


    
def PeakPotentialz(E0, z0, r0):
    E1 = 1.43996455e-9  #  e^2/(4*pi*epsilon_0) in eV-meters
    
    a = np.power((-9.0*E0**4*E1**2*r0**2 + np.emath.sqrt(-12.0*E0**6*E1**6 + 81.0*E0**8*E1**4*r0**4)) / (2.0/3.0), 1.0/3.0)

    zpeak = 0.5*np.sqrt(-r0**2 + 2.0*np.real(E1**2/a) ) - z0
    if (zpeak < 0.0):
        zpeak = 0.0
    
    return zpeak
    
    # return 0.5*np.sqrt(E1/E0) - z0  # this is for r0 = 0, in case my crazy formula above doesn't work in some fringe case


def MakeSemiconductorEnergyDist(pg, EexcAtSurface, EaSurf, rng=np.random.default_rng(), *, effective_mass=1.0):
    # Make energy distribution for the parabolic density of states (with energy gap) model
    #    EexcAtSurface: Excess energy at cathode surface, eV
    #    EaSurf: Electron affinity plus the image potential at the surface, eV
    #    effective_mass: Positive m*/m_e, defaults to 1.0.
    #    Population flux: min(K+EaSurf, K/r)*sqrt(EexcAtSurface-K).
    #    Transverse momentum is conserved across the surface.
    effective_mass = _semi_mass(effective_mass)

    pnorm = 1010.93912  # sqrt(2* (electron mass) * (1 eV)) in eV/c
    Ekin = invEcumulSemi(
        rng.random(len(pg)).ravel(),
        EexcAtSurface,
        EaSurf,
        effective_mass=effective_mass,
    )

    # The internal population flux is uniform in sin^2(theta_i). Refraction
    # maps the admitted directions to a uniform outgoing sin^2 cone.
    q = np.maximum(0.0, EaSurf + Ekin)
    ratio = np.divide(
        effective_mass * q,
        Ekin,
        out=np.full_like(Ekin, np.inf, dtype=float),
        where=Ekin > 0.0,
    )
    sin2_theta = rng.random(len(pg)).ravel() * np.clip(ratio, 0.0, 1.0)
    pr = np.sqrt(sin2_theta)
    pz = np.sqrt(1.0 - sin2_theta)

    pr = pr * pnorm * np.sqrt(Ekin)
    pz = pz * pnorm * np.sqrt(Ekin)
    phi = 2 * np.pi * rng.random(len(pg)).ravel()
    pg.pz = pz
    pg.px = pr * np.cos(phi)
    pg.py = pr * np.sin(phi)
    
    pg.weight = 1.60217663e-19
    
    return pg


def MakeMetalEnergyDist(pg, EexcAtSurface, kT, rng=np.random.default_rng()):   
    # Make energy distribution for the constant DoS model
    #    EexcAtSurface: Excess energy at cathode surface, eV
    #    kT: eV

    pnorm = 1010.93912  # sqrt(2* (electron mass) * (1 eV)) in eV/c
    Ekin = invEcumul(rng.random(len(pg)).ravel(), EexcAtSurface, kT)
    (pr, pz) = uniform_pr2_dist(len(pg), rng=rng)
    pz = np.abs(pz)    
    pr = pr * pnorm * np.sqrt(Ekin)
    pz = pz * pnorm * np.sqrt(Ekin)
    phi = 2 * np.pi * rng.random(len(pg)).ravel()
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
        fguess[needs_work] = QE_model(Eexcz[needs_work], kT, guess[needs_work])
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

def MTE_model_semi(h, v, ea, effective_mass=1.0):
    """Population-flux MTE in eV; h=hv-Eg, v=local potential, ea=affinity.

    Energy inputs broadcast. The positive mass ratio m*/m_e defaults to 1.
    An empty emitting population has undefined MTE (NaN).
    """
    r = _semi_mass(effective_mass)
    hh, vv, ee = np.broadcast_arrays(np.asarray(h, dtype=float),
                                     np.asarray(v, dtype=float),
                                     np.asarray(ea, dtype=float))
    out = np.empty_like(vv)
    for idx in np.ndindex(vv.shape):
        chi = ee[idx] + vv[idx]
        Eexc = hh[idx] - chi
        denom = _semi_total_weight(Eexc, chi, r)
        numer = _semi_total_mte_weight(Eexc, chi, r)
        out[idx] = numer / denom if denom > 0.0 else np.nan
    return float(out) if out.ndim == 0 else out


def QE_model_semi(h, v, v0, ea, effective_mass=1.0):
    """Surface-to-local-potential transmission, not absolute photon QE.

    h=hv-Eg, v0=surface potential, ea=affinity; energy inputs broadcast.
    The positive mass ratio m*/m_e defaults to 1. An empty surface population
    gives undefined transmission (NaN); a closed final barrier gives zero.
    """
    r = _semi_mass(effective_mass)
    hh, vv, vv0, ee = np.broadcast_arrays(
        np.asarray(h, dtype=float), np.asarray(v, dtype=float),
        np.asarray(v0, dtype=float), np.asarray(ea, dtype=float))
    out = np.empty_like(vv)
    for idx in np.ndindex(vv.shape):
        chi, chi0 = ee[idx] + vv[idx], ee[idx] + vv0[idx]
        numer = _semi_total_weight(hh[idx] - chi, chi, r)
        denom = _semi_total_weight(hh[idx] - chi0, chi0, r)
        out[idx] = numer / denom if denom > 0.0 else np.nan
    return float(out) if out.ndim == 0 else out


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
    # Cumulative probability distribution of kinetic energy
    
    Ek = Ekin/kT
    Ee = Eexc/kT
    eEeEk = np.exp(Ee-Ek)
    p1 = Ek*np.log(1.0+eEeEk)
    p2 = spence(eEeEk + 1.0)
    p3 = spence(np.exp(Ee) + 1.0)
    return 1 + p1/p3 - p2/p3

def dEcumulprob(Ekin, Eexc, kT):
    # Derivative of the cumulative probability distribution of kinetic energy w.r.t. energy
    
    Ek = Ekin/kT
    Ee = Eexc/kT

    p1 = np.exp(Ee-Ek)
    p3 = spence(np.exp(Ee) + 1.0)

    return -Ek*p1/((1.0+p1)*p3*kT)

def uniform_pr2_dist(n, rng=None):
    # Generates a uniform distribution of pr^2.
    # If rng is supplied, use it so callers can reproduce the complete distribution.
    
    if rng is None:
        u = np.random.rand(n)
    else:
        u = rng.random(n)
    pr = np.sqrt(u)
    pz = np.sqrt(1.0-u)
    return (pr,pz)

def _semi_support(Eexc, Ea):
    """
    Return the lower and upper kinetic-energy bounds for the semiconductor model.
    """
    if Ea >= 0:
        Elo = 0.0
    else:
        Elo = -Ea

    Ehi = Eexc

    if Ehi <= Elo:
        raise ValueError(
            f"Invalid support: need Eexc > Elo, but got Eexc={Eexc}, Elo={Elo}."
        )

    return Elo, Ehi


def _semi_mass(effective_mass):
    """Validate a scalar conduction-band/free-electron mass ratio."""
    try:
        r = float(effective_mass)
    except (TypeError, ValueError):
        raise ValueError("effective_mass must be a finite positive m*/m_e.") from None
    if not np.isfinite(r) or r <= 0:
        raise ValueError("effective_mass must be a finite positive m*/m_e.")
    return r


_SEMI_QUAD_NODES, _SEMI_QUAD_WEIGHTS = np.polynomial.legendre.leggauss(4)


def _semi_integral(Ekin, Eexc, Ea, effective_mass, moment=False):
    """Definite population-flux integral, optionally weighted by mean K_perp.

    W(K) = min(K+Ea, K/r)*sqrt(Eexc-K). With t=sqrt(Eexc-K),
    the integrands are polynomials of degree four (normalization) or six
    (MTE numerator) on each side of K=r*Ea/(1-r). Four-point Gaussian
    quadrature is exact on each branch up to roundoff. Positive definite
    intervals avoid cancellation from subtracting large antiderivatives.
    Ekin may have any shape; Eexc, Ea, and effective_mass are scalar.
    """
    r = _semi_mass(effective_mass)
    Elo, Ehi = _semi_support(Eexc, Ea)
    x = np.clip(np.asarray(Ekin, dtype=float), Elo, Ehi)
    crossing = r * Ea / (1.0 - r) if r != 1.0 else Ehi
    bounds = [Elo]
    if Elo < crossing < Ehi:
        bounds.append(crossing)
    bounds.append(Ehi)
    total = np.zeros_like(x)
    for lower, end in zip(bounds[:-1], bounds[1:]):
        upper = np.clip(x, lower, end)
        t_hi = np.sqrt(Ehi - lower)
        t_lo = np.sqrt(np.maximum(0.0, Ehi - upper))
        # Rationalize t_hi-t_lo to retain precision on narrow intervals.
        width = (upper - lower) / (t_hi + t_lo)
        for node, weight in zip(_SEMI_QUAD_NODES, _SEMI_QUAD_WEIGHTS):
            fraction = 0.5 * (node + 1.0)
            t = t_hi - fraction * width
            increment = fraction * width * (t_hi + t)
            K = lower + increment
            q = (Ea + lower) + increment
            admitted = np.minimum(q, K / r)
            integrand = t**2 * admitted
            if moment:
                integrand *= 0.5 * r * admitted
            total += weight * width * integrand
    return total


def _semi_cumul_raw(Ekin, Eexc, Ea, effective_mass=1.0):
    """Integral of min(K+Ea, K/r)*sqrt(Eexc-K)."""
    return _semi_integral(Ekin, Eexc, Ea, effective_mass)


def _semi_mte_cumul_raw(Ekin, Eexc, Ea, effective_mass=1.0):
    """Population-flux integral weighted by min(K, r*(K+Ea))/2."""
    return _semi_integral(Ekin, Eexc, Ea, effective_mass, moment=True)


def _semi_total_weight(Eexc, Ea, effective_mass=1.0):
    """Total unnormalized population flux; zero for empty support."""
    r = _semi_mass(effective_mass)
    if Eexc <= max(0.0, -Ea):
        return 0.0
    return float(_semi_cumul_raw(Eexc, Eexc, Ea, r))


def _semi_total_mte_weight(Eexc, Ea, effective_mass=1.0):
    """Total transverse-energy numerator; zero for empty support."""
    r = _semi_mass(effective_mass)
    if Eexc <= max(0.0, -Ea):
        return 0.0
    return float(_semi_mte_cumul_raw(Eexc, Eexc, Ea, r))


def EcumulprobSemi(Ekin, Eexc, Ea, effective_mass=1.0):
    """CDF of W(K) proportional to min(K+Ea, K/r)*sqrt(Eexc-K).

    r=m*/m_e defaults to 1. Transverse momentum is always conserved.
    """
    r = _semi_mass(effective_mass)
    norm = _semi_cumul_raw(Eexc, Eexc, Ea, r)
    F = np.clip(_semi_cumul_raw(Ekin, Eexc, Ea, r) / norm, 0.0, 1.0)
    return float(F) if np.isscalar(Ekin) else F


def dEcumulprobSemi(Ekin, Eexc, Ea, effective_mass=1.0):
    """Normalized population-flux PDF; zero outside its physical support."""
    r = _semi_mass(effective_mass)
    scalar_input = np.isscalar(Ekin)
    Ekin = np.asarray(Ekin, dtype=float)
    Elo, Ehi = _semi_support(Eexc, Ea)
    weight = np.maximum(0.0, np.minimum(Ea + Ekin, Ekin / r))
    root = np.sqrt(np.maximum(0.0, Eexc - Ekin))
    norm = _semi_cumul_raw(Ehi, Eexc, Ea, r)
    pdf = np.where((Ekin >= Elo) & (Ekin <= Ehi), weight * root / norm, 0.0)
    return float(pdf) if scalar_input else pdf


def invEcumulSemi(p, Eexc, Ea, ptol=1.0e-7, max_iter=200, *, effective_mass=1.0):
    """
    Invert the semiconductor cumulative probability distribution using a
    safeguarded Newton iteration.

    Parameters
    ----------
    p : float or array_like
        Cumulative probabilities in [0, 1].
    Eexc : float
        Excess energy at the position where the distribution is being sampled.
    Ea : float
        Local affinity-like energy parameter, chi = Ea + V(z).
    effective_mass : float
        Positive m*/m_e, defaults to 1. Transverse momentum is conserved.
        The energy distribution uses the revised population-flux weight.
    ptol : float
        Absolute tolerance in cumulative probability.
    max_iter : int
        Maximum number of safeguarded Newton iterations.
    """
    scalar_input = np.isscalar(p)
    p = np.asarray(p, dtype=float)

    if np.any(~np.isfinite(p) | (p < 0.0) | (p > 1.0)):
        raise ValueError("p must be in the interval [0, 1].")

    effective_mass = _semi_mass(effective_mass)

    Elo, Ehi = _semi_support(Eexc, Ea)

    # Exact endpoint handling.
    Eout = np.empty_like(p, dtype=float)
    at_low = p == 0.0
    at_high = p == 1.0
    active = ~(at_low | at_high)

    Eout[at_low] = Elo
    Eout[at_high] = Ehi

    if np.any(active):
        pp = p[active]

        lo = np.full_like(pp, Elo, dtype=float)
        hi = np.full_like(pp, Ehi, dtype=float)

        # Initial guess: linear in CDF. Not perfect, but bracketed Newton fixes it.
        x = Elo + pp * (Ehi - Elo)

        for _ in range(max_iter):
            F = EcumulprobSemi(x, Eexc, Ea, effective_mass=effective_mass)
            err = F - pp

            done = np.abs(err) <= ptol
            if np.all(done):
                break

            # Update the brackets.
            too_low = err < 0.0
            lo = np.where(too_low, x, lo)
            hi = np.where(too_low, hi, x)

            pdf = dEcumulprobSemi(x, Eexc, Ea, effective_mass=effective_mass)

            # Newton proposal.
            with np.errstate(divide="ignore", invalid="ignore"):
                x_newton = x - err / pdf

            # Fall back to bisection if Newton is unsafe.
            x_bisect = 0.5 * (lo + hi)

            bad_newton = (
                ~np.isfinite(x_newton)
                | (x_newton <= lo)
                | (x_newton >= hi)
                | (pdf <= 0.0)
            )

            x_new = np.where(bad_newton, x_bisect, x_newton)

            # Keep already-converged values fixed.
            x = np.where(done, x, x_new)

        else:
            F = EcumulprobSemi(x, Eexc, Ea, effective_mass=effective_mass)
            if np.any(np.abs(F - pp) > ptol):
                raise RuntimeError("invEcumulSemi failed to converge for some entries.")

        Eout[active] = x

    if scalar_input:
        return float(Eout)

    return Eout
