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
    #    settings['effective_mass'] : m*/m_e. If present, conserve transverse momentum at the semiconductor/vacuum interface.
    #                                 If absent, retain the original disordered-crystal model exactly.
    #
    #    Note: two values of settings are modified (or added) in this code:
    #    settings['cathode_z_offset'] : This value is overwritten or created in SI units, intended to be used in GPT
    #    settings['gun_field'] : This value is overwritten or created in SI units, intended to be used in GPT

    (EexcAtSurface, EexcAtPeak, EaSurf) = getSemiconductorEexc(settings, modify_settings=True, verbose=verbose)

    effective_mass = settings.get('effective_mass', None)
    if effective_mass is not None and effective_mass <= 0:
        raise ValueError('settings["effective_mass"] must be positive and should be given as m*/m_e.')

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


def MakeSemiconductorEnergyDist(pg, EexcAtSurface, EaSurf, rng=np.random.default_rng(), *, effective_mass=None):
    # Make energy distribution for the parabolic density of states (with energy gap) model
    #    EexcAtSurface: Excess energy at cathode surface, eV
    #    EaSurf: Electron affinity plus the image potential at the surface, eV
    #    effective_mass: m*/m_e. If None, use the original disordered-crystal model.
    #                    If specified, conserve transverse momentum across the surface.
    
    pnorm = 1010.93912  # sqrt(2* (electron mass) * (1 eV)) in eV/c
    Ekin = invEcumulSemi(
        rng.random(len(pg)).ravel(),
        EexcAtSurface,
        EaSurf,
        effective_mass=effective_mass,
    )

    if effective_mass is None:
        # Original model: the emitted flux has sin^2(theta) uniformly distributed
        # over the full outgoing hemisphere.
        (pr, pz) = uniform_pr2_dist(len(pg), rng=rng)
    else:
        if effective_mass <= 0:
            raise ValueError('effective_mass must be positive and should be given as m*/m_e.')

        # q = K + chi is the conduction-band kinetic energy inside the material.
        # Conservation of transverse momentum gives
        #
        #   sin^2(theta_out) <= min(1, (m*/m_e) q / K).
        #
        # The transmitted flux is uniform in sin^2(theta_out) over this range.
        q = EaSurf + Ekin
        ratio = np.divide(
            effective_mass * q,
            Ekin,
            out=np.full_like(Ekin, np.inf, dtype=float),
            where=Ekin > 0.0,
        )
        sin2_max = np.clip(ratio, 0.0, 1.0)
        sin2_theta = rng.random(len(pg)).ravel() * sin2_max
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

def MTE_model_semi(h, v, ea, effective_mass=None):
    # h = hv-Eg, 
    # v = ImagePotential(z, z0, plummer_radius, gun_field)
    # ea = Ea
    # effective_mass = m*/m_e. None retains the original disordered-crystal model.
    if effective_mass is not None:
        if effective_mass <= 0:
            raise ValueError('effective_mass must be positive and should be given as m*/m_e.')

        scalar_input = np.isscalar(v)
        vv = np.asarray(v, dtype=float)
        out = np.empty_like(vv, dtype=float)

        for idx in np.ndindex(vv.shape):
            chi = ea + vv[idx]
            Eexc = h - chi
            denom = _semi_total_weight(Eexc, chi, effective_mass)
            numer = _semi_total_mte_weight(Eexc, chi, effective_mass)
            out[idx] = numer/denom if denom > 0.0 else np.nan

        if scalar_input:
            return float(out)
        return out

    out = np.empty_like(v, dtype=float)
    m = ea + v > 0

    x = v[m]
    s = np.sqrt(-1 + h/(ea + x))*np.arcsin(np.sqrt(-(ea - h + x)/h))
    k = (ea - h + x)*(4*ea**2 - 4*ea*h + 3*h**2 + 8*ea*x - 4*h*x + 4*x**2) - 3*h**2*(-2*ea + h - 2*x)*s
    n = (ea - h + x)*(2*ea - h + 2*x)*(8*ea**2 + 15*h**2 - 8*ea*(h - 2*x) - 8*h*x + 8*x**2) + 3*h**2*(16*ea**2 + 5*h**2 - 16*ea*(h - 2*x) - 16*h*x + 16*x**2)*s
    out[m] = -n/(16*k)

    x = v[~m]
    out[~m] = (5*h**2 + 16*(ea + x)*(ea - h + x))/(16*(-2*ea + h - 2*x))
    return out


def QE_model_semi(h, v, v0, ea, effective_mass=None):
    # h = hv-Eg, 
    # v = ImagePotential(z, z0, plummer_radius, gun_field)
    # v0 = ImagePotential(0, z0, plummer_radius, gun_field)
    # ea = Ea
    # effective_mass = m*/m_e. None retains the original disordered-crystal model.
    if effective_mass is not None:
        if effective_mass <= 0:
            raise ValueError('effective_mass must be positive and should be given as m*/m_e.')

        scalar_input = np.isscalar(v) and np.isscalar(v0)
        vv, vv0 = np.broadcast_arrays(np.asarray(v, dtype=float), np.asarray(v0, dtype=float))
        out = np.empty_like(vv, dtype=float)

        for idx in np.ndindex(vv.shape):
            chi = ea + vv[idx]
            chi0 = ea + vv0[idx]
            Eexc = h - chi
            Eexc0 = h - chi0
            numer = _semi_total_weight(Eexc, chi, effective_mass)
            denom = _semi_total_weight(Eexc0, chi0, effective_mass)
            out[idx] = numer/denom if denom > 0.0 else np.nan

        if scalar_input:
            return float(out)
        return out

    out = np.empty_like(v, dtype=float)

    m = ea + v0 > 0
    x, x0 = v[m], v0[m]
    S = lambda y: np.sqrt(-1 + h/(ea + y))*np.arcsin(np.sqrt(-(ea - h + y)/h))
    K = lambda y: (ea - h + y)*(4*ea**2 - 4*ea*h + 3*h**2 + 8*ea*y - 4*h*y + 4*y**2) - 3*h**2*(-2*ea + h - 2*y)*S(y)
    out[m] = np.sqrt(-(ea + x)/(ea - h + x))*K(x)/(np.sqrt(-(ea + x0)/(ea - h + x0))*K(x0))

    m1 = (~m) & (ea + v > 0)
    x, x0 = v[m1], v0[m1]
    s = np.sqrt(-1 + h/(ea + x))*np.arcsin(np.sqrt(-(ea - h + x)/h))
    k = (ea - h + x)*(4*ea**2 - 4*ea*h + 3*h**2 + 8*ea*x - 4*h*x + 4*x**2) - 3*h**2*(-2*ea + h - 2*x)*s
    out[m1] = -(2*np.sqrt(-(ea + x)/(ea - h + x))*k)/(3*h**2*np.pi*(-2*ea + h - 2*x0))

    m2 = ~(m | m1)
    out[m2] = (-2*ea + h - 2*v[m2])/(-2*ea + h - 2*v0[m2])
    return out
    
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


def _semi_antideriv(Ekin, Eexc, Ea):
    """
    Antiderivative of

        Ekin * sqrt((Ea + Ekin) * (Eexc - Ekin))

    on the physical interval. This is the energy weight in the original
    disordered-crystal semiconductor model.
    """
    Ekin = np.asarray(Ekin, dtype=float)

    m = 0.5 * (Eexc - Ea)
    R = 0.5 * (Eexc + Ea)

    if R <= 0:
        raise ValueError("Need Eexc + Ea > 0 for a non-empty physical interval.")

    t = (Ekin - m) / R
    t = np.clip(t, -1.0, 1.0)

    s2 = np.maximum(0.0, 1.0 - t**2)
    s = np.sqrt(s2)

    return R**2 * (
        0.5 * m * (t * s + np.arcsin(t))
        - (R / 3.0) * s**3
    )


def _semi_antideriv_q(Ekin, Eexc, Ea):
    """
    Antiderivative of

        (Ea + Ekin) * sqrt((Ea + Ekin) * (Eexc - Ekin)).
    """
    Ekin = np.asarray(Ekin, dtype=float)

    m = 0.5 * (Eexc - Ea)
    R = 0.5 * (Eexc + Ea)

    if R <= 0:
        raise ValueError("Need Eexc + Ea > 0 for a non-empty physical interval.")

    t = (Ekin - m) / R
    t = np.clip(t, -1.0, 1.0)

    s2 = np.maximum(0.0, 1.0 - t**2)
    s = np.sqrt(s2)

    return R**3 * (
        0.5 * (t * s + np.arcsin(t))
        - s**3 / 3.0
    )


def _semi_antideriv_E2(Ekin, Eexc, Ea):
    """
    Antiderivative of

        Ekin**2 * sqrt((Ea + Ekin) * (Eexc - Ekin)).
    """
    Ekin = np.asarray(Ekin, dtype=float)

    m = 0.5 * (Eexc - Ea)
    R = 0.5 * (Eexc + Ea)

    if R <= 0:
        raise ValueError("Need Eexc + Ea > 0 for a non-empty physical interval.")

    t = (Ekin - m) / R
    t = np.clip(t, -1.0, 1.0)

    s2 = np.maximum(0.0, 1.0 - t**2)
    s = np.sqrt(s2)

    A0 = 0.5 * (t * s + np.arcsin(t))
    A1 = -s**3 / 3.0
    A2 = (np.arcsin(t) - t * s * (1.0 - 2.0*t**2)) / 8.0

    return R**2 * (
        m**2 * A0
        + 2.0*m*R * A1
        + R**2 * A2
    )


def _semi_antideriv_q2(Ekin, Eexc, Ea):
    """
    Antiderivative of

        (Ea + Ekin)**2 * sqrt((Ea + Ekin) * (Eexc - Ekin)).
    """
    Ekin = np.asarray(Ekin, dtype=float)

    m = 0.5 * (Eexc - Ea)
    R = 0.5 * (Eexc + Ea)

    if R <= 0:
        raise ValueError("Need Eexc + Ea > 0 for a non-empty physical interval.")

    t = (Ekin - m) / R
    t = np.clip(t, -1.0, 1.0)

    s2 = np.maximum(0.0, 1.0 - t**2)
    s = np.sqrt(s2)

    A0 = 0.5 * (t * s + np.arcsin(t))
    A1 = -s**3 / 3.0
    A2 = (np.arcsin(t) - t * s * (1.0 - 2.0*t**2)) / 8.0

    return R**4 * (A0 + 2.0*A1 + A2)


def _semi_piecewise_integral(Ekin, Eexc, Ea, effective_mass,
                             primitive_q, primitive_E,
                             q_scale=1.0, E_scale=1.0):
    """
    Integrate a piecewise semiconductor weight whose active branch is selected
    by

        min(Ea + E, E/effective_mass).

    primitive_q and primitive_E are antiderivatives for the corresponding q
    and E branches before q_scale and E_scale are applied.
    """
    r = float(effective_mass)
    if r <= 0:
        raise ValueError("effective_mass must be positive and should be given as m*/m_e.")

    Elo, Ehi = _semi_support(Eexc, Ea)
    x = np.clip(np.asarray(Ekin, dtype=float), Elo, Ehi)

    def Pq(y):
        return q_scale * primitive_q(y, Eexc, Ea)

    def PE(y):
        return E_scale * primitive_E(y, Eexc, Ea)

    def q_branch_is_smaller(y):
        return (Ea + y) <= y / r

    # For r=1 there is no finite crossing unless Ea=0.
    if np.isclose(r, 1.0, rtol=0.0, atol=1.0e-14):
        use_q = Ea <= 0.0
        P = Pq if use_q else PE
        return P(x) - P(Elo)

    Eswitch = r * Ea / (1.0 - r)

    # No branch crossing inside the physical support.
    if not (Elo < Eswitch < Ehi):
        mid = 0.5 * (Elo + Ehi)
        P = Pq if q_branch_is_smaller(mid) else PE
        return P(x) - P(Elo)

    # There is one crossing. Determine which branch is active on the left.
    left_mid = 0.5 * (Elo + Eswitch)
    left_is_q = q_branch_is_smaller(left_mid)

    if left_is_q:
        Pleft, Pright = Pq, PE
    else:
        Pleft, Pright = PE, Pq

    I_switch = Pleft(Eswitch) - Pleft(Elo)

    return np.where(
        x <= Eswitch,
        Pleft(x) - Pleft(Elo),
        I_switch + Pright(x) - Pright(Eswitch),
    )


_SEMI_QUAD_NODES, _SEMI_QUAD_WEIGHTS = np.polynomial.legendre.leggauss(16)


def _semi_threshold_cumul(Ekin, Eexc, Ea, effective_mass, power=1):
    """Stable definite integral for positive affinity near emission threshold.

    K = Eexc * (1-t**2) removes the upper-endpoint square root. Integrate
    each side of the escape-cone crossing separately using Gaussian quadrature.
    In this regime q varies by less than 5%, so the transformed integrands
    are smooth. All terms are nonnegative; no large primitives are subtracted.
    power=2 includes the r/2 factor for the transverse-energy numerator.
    """
    r = float(effective_mass)
    if r <= 0:
        raise ValueError("effective_mass must be positive and should be given as m*/m_e.")
    x = np.clip(np.asarray(Ekin, dtype=float), 0.0, Eexc)
    crossing = r * Ea / (1.0 - r) if r != 1.0 else Eexc
    split = crossing if 0.0 < crossing < Eexc else Eexc
    total = np.zeros_like(x)
    for lower, upper in ((0.0, np.minimum(x, split)),
                         (np.minimum(x, split), x)):
        t_hi = np.sqrt(1.0 - lower / Eexc)
        t_lo = np.sqrt(np.maximum(0.0, 1.0 - upper / Eexc))
        # Rationalized difference preserves narrow integration intervals.
        width = np.divide((upper - lower) / Eexc, t_hi + t_lo,
                          out=np.zeros_like(x), where=(t_hi + t_lo) > 0.0)
        for node, weight in zip(_SEMI_QUAD_NODES, _SEMI_QUAD_WEIGHTS):
            fraction = 0.5 * (node + 1.0)
            t = t_hi - fraction * width
            K = lower + Eexc * fraction * width * (t_hi + t)
            q = Ea + K
            total += weight * width * t**2 * np.minimum(q, K/r)**power * np.sqrt(q)
    total *= Eexc**1.5
    return total if power == 1 else total * (0.5*r)


def _semi_cumul_raw(Ekin, Eexc, Ea, effective_mass):
    """
    Unnormalized cumulative emitted-electron energy weight when transverse
    momentum is conserved at a mass-discontinuous semiconductor/vacuum surface.

    The differential weight is

        min(q, K/r) * sqrt(q * (Eexc - K)),

    where q = Ea + K and r = m*/m_e.
    """
    r = float(effective_mass)
    if 0.0 < Eexc < 0.05 * Ea:
        return _semi_threshold_cumul(Ekin, Eexc, Ea, r)
    return _semi_piecewise_integral(
        Ekin,
        Eexc,
        Ea,
        r,
        primitive_q=_semi_antideriv_q,
        primitive_E=_semi_antideriv,
        q_scale=1.0,
        E_scale=1.0/r,
    )


def _semi_mte_cumul_raw(Ekin, Eexc, Ea, effective_mass):
    """
    Unnormalized cumulative transverse-energy weight for the momentum-conserving
    semiconductor model.

    At fixed K, the mean outgoing transverse energy is

        0.5 * min(K, r*q),

    so the transverse-energy numerator has the piecewise integrand

        (r/2) * min(q, K/r)**2 * sqrt(q * (Eexc - K)).
    """
    r = float(effective_mass)
    if 0.0 < Eexc < 0.05 * Ea:
        return _semi_threshold_cumul(Ekin, Eexc, Ea, r, power=2)
    return _semi_piecewise_integral(
        Ekin,
        Eexc,
        Ea,
        r,
        primitive_q=_semi_antideriv_q2,
        primitive_E=_semi_antideriv_E2,
        q_scale=0.5*r,
        E_scale=0.5/r,
    )


def _semi_total_weight(Eexc, Ea, effective_mass):
    """Total unnormalized momentum-conserving semiconductor emission weight."""
    Elo = max(0.0, -Ea)
    if Eexc <= Elo:
        return 0.0
    return float(_semi_cumul_raw(Eexc, Eexc, Ea, effective_mass))


def _semi_total_mte_weight(Eexc, Ea, effective_mass):
    """Total unnormalized transverse-energy numerator."""
    Elo = max(0.0, -Ea)
    if Eexc <= Elo:
        return 0.0
    return float(_semi_mte_cumul_raw(Eexc, Eexc, Ea, effective_mass))


def EcumulprobSemi(Ekin, Eexc, Ea, effective_mass=None):
    """
    Cumulative probability distribution for the semiconductor kinetic-energy
    distribution.

    If effective_mass is None, this is the original disordered-crystal model:

        P(K) proportional to K*sqrt((Ea+K)*(Eexc-K)).

    If effective_mass = m*/m_e is supplied, transverse momentum is conserved
    across the semiconductor/vacuum interface and the energy weight becomes

        P(K) proportional to min(Ea+K, K/effective_mass)
                            * sqrt((Ea+K)*(Eexc-K)).
    """
    scalar_input = np.isscalar(Ekin)
    Ekin = np.asarray(Ekin, dtype=float)

    Elo, Ehi = _semi_support(Eexc, Ea)
    Eclip = np.clip(Ekin, Elo, Ehi)

    if effective_mass is None:
        A_lo = _semi_antideriv(Elo, Eexc, Ea)
        A_hi = _semi_antideriv(Ehi, Eexc, Ea)
        norm = A_hi - A_lo
        F = (_semi_antideriv(Eclip, Eexc, Ea) - A_lo) / norm
    else:
        if effective_mass <= 0:
            raise ValueError("effective_mass must be positive and should be given as m*/m_e.")
        norm = _semi_cumul_raw(Ehi, Eexc, Ea, effective_mass)
        F = _semi_cumul_raw(Eclip, Eexc, Ea, effective_mass) / norm

    F = np.clip(F, 0.0, 1.0)

    if scalar_input:
        return float(F)

    return F


def dEcumulprobSemi(Ekin, Eexc, Ea, effective_mass=None):
    """
    Derivative of the semiconductor CDF with respect to Ekin.
    This is the normalized semiconductor kinetic-energy PDF.
    """
    scalar_input = np.isscalar(Ekin)
    Ekin = np.asarray(Ekin, dtype=float)

    Elo, Ehi = _semi_support(Eexc, Ea)
    q = Ea + Ekin
    inside = q * (Eexc - Ekin)
    root = np.sqrt(np.maximum(0.0, inside))

    if effective_mass is None:
        A_lo = _semi_antideriv(Elo, Eexc, Ea)
        A_hi = _semi_antideriv(Ehi, Eexc, Ea)
        norm = A_hi - A_lo
        pdf = Ekin * root / norm
    else:
        r = float(effective_mass)
        if r <= 0:
            raise ValueError("effective_mass must be positive and should be given as m*/m_e.")
        weight = np.minimum(q, Ekin/r)
        weight = np.maximum(0.0, weight)
        norm = _semi_cumul_raw(Ehi, Eexc, Ea, r)
        pdf = weight * root / norm

    in_range = (Ekin >= Elo) & (Ekin <= Ehi)
    pdf = np.where(in_range, pdf, 0.0)

    if scalar_input:
        return float(pdf)

    return pdf


def invEcumulSemi(p, Eexc, Ea, ptol=1.0e-7, max_iter=200, *, effective_mass=None):
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
    effective_mass : float or None
        If None, use the original disordered-crystal model. Otherwise this is
        m*/m_e and transverse momentum is conserved across the surface.
    ptol : float
        Absolute tolerance in cumulative probability.
    max_iter : int
        Maximum number of safeguarded Newton iterations.
    """
    scalar_input = np.isscalar(p)
    p = np.asarray(p, dtype=float)

    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p must be in the interval [0, 1].")

    if effective_mass is not None and effective_mass <= 0:
        raise ValueError("effective_mass must be positive and should be given as m*/m_e.")

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
