import numpy as np
import copy

def make_tip_distribution_lumerical(PG_laser, H, B):
    # PG_laser : distribution lumerical output, from distgen, in x-y plane
    # H : height of tip (m)
    # B : diameter of tip base (m)

    # For now, this assumes theta = 90 degrees incidence and a (x,y) dist from lumerical
        
    PG_tip = copy.deepcopy(PG_laser)

    PG_tip.z = tip_height(np.sqrt(PG_tip.x**2 + PG_tip.y**2), H, B)
        
    # reference particle to define when t=0 happens
    ref_x = 0.0
    ref_z = H
    
    dt = (PG_tip.x - ref_x) / 299792458
    PG_tip.t = PG_tip.t + dt 
    
    r = PG_tip.r
    phi = PG_tip.theta
    px = PG_tip.px
    py = PG_tip.py
    pz = PG_tip.pz
    
    for ii in np.arange(0, len(px)):
        nr, nz = tip_normal(r[ii], H, B)
        nx = nr * np.cos(phi[ii])
        ny = nr * np.sin(phi[ii])
        Rmat = rotation_from_z_to_n([nx, ny, nz])
        pnew = Rmat @ np.array([px[ii], py[ii], pz[ii]])
        PG_tip.px[ii] = pnew[0]
        PG_tip.py[ii] = pnew[1]
        PG_tip.pz[ii] = pnew[2]
    
    return PG_tip

def make_tip_distribution(PG_laser, H, B, theta, z0):
    # PG_laser : distribution from laser hitting flat surface (i.e. from normal distgen)
    # H : height of tip (m)
    # B : diameter of tip base (m)
    # theta : laser incident angle (rad)
    # z0 : laser is pointed at the point (0, 0, z0)
        
    PG_tip = copy.deepcopy(PG_laser)

    # reference particle to define when t=0 happens
    laser_ref_x = 0.0
    ref_x = FirstIntersection(laser_ref_x, 0, z0, theta, H, B)
    if (ref_x is None):
        raise ValueError("Error: center of laser does not hit tip")
        # If this causes problems, might have to define the reference particle some other way
    ref_z = tip_height(ref_x, H, B)
    
    for ii in np.arange(0, len(PG_laser)):
        PG_tip.x[ii] = FirstIntersection(PG_laser.x[ii], PG_laser.y[ii], z0, theta, H, B)

    # temporarily put non-nan values into lost particles, to calculate dt
    lost_particles = np.isnan(PG_tip.x)
    PG_tip.x[lost_particles] = PG_laser.x[lost_particles]
    PG_tip.y[lost_particles] = PG_laser.y[lost_particles]
    PG_tip.z = tip_height(PG_tip.r, H, B)

    dt = np.sqrt((PG_tip.x - ref_x)**2 + (PG_tip.z - ref_z)**2 - (PG_laser.x - laser_ref_x)**2) / 299792458
    PG_tip.t = PG_tip.t + dt 

    # Now remove lost particles
    PG_tip = PG_tip[~lost_particles]
    
    r = PG_tip.r
    phi = PG_tip.theta
    px = PG_tip.px
    py = PG_tip.py
    pz = PG_tip.pz
    
    for ii in np.arange(0, len(px)):
        nr, nz = tip_normal(r[ii], H, B)
        nx = nr * np.cos(phi[ii])
        ny = nr * np.sin(phi[ii])
        Rmat = rotation_from_z_to_n([nx, ny, nz])
        pnew = Rmat @ np.array([px[ii], py[ii], pz[ii]])
        PG_tip.px[ii] = pnew[0]
        PG_tip.py[ii] = pnew[1]
        PG_tip.pz[ii] = pnew[2]
    
    return PG_tip
    

def FieldEnhancement(r, H, B):
    isscalar = False
    if (np.isscalar(r)):
        isscalar = True
        r = [r]
    
    r = np.array(r)
    a = 0.5*np.sqrt((2.0*H-B)*(2.0*H+B))
    xi0 = H/a
    rin = r < 0.5*B
    eta = np.zeros(len(r))
    eta[rin] = (H/B)*np.sqrt((B-2.0*r[rin])*(B+2.0*r[rin]))/(a*xi0)

    xi = np.ones(len(r))*xi0
    xi[~rin] = np.sqrt(1 + (r[~rin]/a)*(r[~rin]/a))

    F = np.zeros(len(r))
    F[rin] = eta[rin] / (np.sqrt(eta[rin]*eta[rin] - (1.0 + eta[rin]*eta[rin])*xi0*xi0 + xi0*xi0*xi0*xi0)*(xi0*np.arctanh(1.0/xi0) - 1.0))
    F[~rin] = (xi[~rin] - xi0 + xi[~rin] * xi0 * np.arctanh((xi[~rin] - xi0)/(1.0 - xi[~rin]*xi0))) / (xi[~rin] - xi[~rin] * xi0 * np.arctanh(1.0/xi0))

    if (isscalar):
        F = F[0]
    
    return F

def FirstIntersection(x0, y0, z0, theta, H, B):
    theta = np.abs(theta)
    if theta > 0.5*np.pi:
        return np.nan
    
    pi_valid = False
    si_valid = False    
    
    pi_x = PlaneIntersection(x0, y0, z0, theta)
    if (pi_x is not None):
        pi_valid = pi_x**2 + y0**2 > (0.5*B)**2
    
    si_x, si_z = SpheroidIntersection(x0, y0, z0, theta, B, H)
    if (si_x is not None):
        si_valid = si_z > 0
    
    valid_intersections = []
    if (pi_valid):
        valid_intersections.append(pi_x)
    if (si_valid):
        valid_intersections.append(si_x)

    if (len(valid_intersections) > 0):
        return np.max(valid_intersections)
    else:
        return np.nan

def SpheroidIntersection(x0, y0, z0, theta, B, H):
    cos = np.cos(theta)
    sin = np.sin(theta)

    if (theta == 0.0):
        z = tip_height(np.sqrt(x0*x0+y0*y0), H, B)
        return x0, z
        
    cot = cos/sin
    tan = sin/cos

    rad = B**2 * (-4 * x0**2 + (B**2 - 4 * y0**2) * cos**2) + 8 * B**2 * x0 * z0 * sin + 4 * (-4 * H**2 * y0**2 + B**2 * (H - z0) * (H + z0)) * sin**2
    if (rad < 0):
        return None, None
    num = (B**2 * cos * (x0 - z0 * sin) + H * sin * np.sqrt(rad))
    den = B**2 * cos**2 + 4 * H**2 * sin**2
    x = num / den
    z = -x0/sin + x*(cos/sin) + z0
    return x, z

def PlaneIntersection(x0, y0, z0, theta):
    if theta >= 0.5*np.pi:
        return None
    return x0/np.cos(theta) - z0*np.tan(theta)

def tip_height(r, H, B):
    scalar_input = np.isscalar(r)
    r_arr = np.array(r, copy=False, ndmin=1, dtype=float)  # now r_arr.shape is (n,) or (1,) for a scalar

    h_arr = np.zeros_like(r_arr)

    # Region 1: r < 0.5*B
    mask = (r_arr < 0.5 * B)
    if mask.any():
        h_arr[mask] = (H/B)*np.sqrt(B**2-4.0*r_arr[mask]**2)

    # For r >= 0.5*B, h_arr stays zero

    # If the original input was a scalar, return a scalar
    if scalar_input:
        return h_arr[0]
    else:
        return h_arr

def tip_normal(r, H, B):
    scalar_input = np.isscalar(r)
    r_arr = np.array(r, copy=True, ndmin=1, dtype=float)  # now r_arr.shape is (n,) or (1,) for a scalar
    
    s_arr = np.zeros_like(r_arr)

    # Region 1: r < 0.5*B
    mask = (r_arr < 0.5 * B)
    if mask.any():
        s_arr[mask] = -4.0*r_arr[mask]*(H/B)/np.sqrt(B**2-4.0*r_arr[mask]**2)
    
    # For r >=rt, s_arr stays zero

    nr_arr = np.abs(-s_arr/np.sqrt(1 + s_arr**2))
    nz_arr = 1.0/np.sqrt(1 + s_arr**2)
    
    # If the original input was a scalar, return a scalar
    if scalar_input:
        return nr_arr[0], nz_arr[0]
    else:
        return nr_arr, nz_arr

def rotation_from_z_to_n(n):
    """
    Return a 3×3 rotation matrix R that sends e_z = (0,0,1) to n = (nx, ny, nz).
    Accepts any nonzero vector n (it normalizes internally).
    """
    n = np.array(n)
    norm_n = np.linalg.norm(n)
    if norm_n < 1e-12:
        raise ValueError("Input vector n must be nonzero.")
    # Normalize n to unit length
    nx, ny, nz = (n / norm_n)

    # Compute s = sqrt(nx^2 + ny^2)
    s = np.hypot(nx, ny)

    # Handle special cases where n ≈ ±e_z
    if s < 1e-12:
        if nz > 0:
            # n ≈ +e_z → identity
            return np.eye(3)
        else:
            # n ≈ –e_z → 180° rotation about any axis in the xy-plane. 
            # For example, flip y and z:
            return np.diag([1.0, -1.0, -1.0])

    # Otherwise build R from the closed-form Rodrigues construction:
    one_minus_nz = 1.0 - nz
    s2 = s * s  # = nx^2 + ny^2

    R = np.array([
        [nz + (ny*ny / s2) * one_minus_nz,   - (nx*ny / s2) * one_minus_nz,   nx],
        [- (nx*ny / s2) * one_minus_nz,       nz + (nx*nx / s2) * one_minus_nz, ny],
        [- nx,                               - ny,                             nz    ]
    ])

    return R