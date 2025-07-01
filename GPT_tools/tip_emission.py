import numpy as np
import copy

def make_tip_distribution(PG_laser, R, H, B, theta, z0):
    # PG_laser : distribution from laser hitting flat surface (i.e. from normal distgen)
    # R : radius of curvature of tip (m)
    # H : height of tip (m)
    # B : diameter of tip base (m)
    # theta : laser incident angle (rad)
    # z0 : laser is pointed at the point (0, 0, z0)
        
    PG_tip = copy.deepcopy(PG_laser)

    # reference particle to define when t=0 happens
    laser_ref_x = 0.0
    ref_x = FirstIntersection(laser_ref_x, 0, z0, theta, R, H, B)
    if (ref_x is None):
        raise ValueError("Error: center of laser does not hit tip")
        # If this causes problems, might have to define the reference particle some other way
    ref_z = tip_height(ref_x, R, H, B)
    
    for ii in np.arange(0, len(PG_laser)):
        PG_tip.x[ii] = FirstIntersection(PG_laser.x[ii], PG_laser.y[ii], z0, theta, R, H, B)

    # temporarily put non-nan values into lost particles, to calculate dt
    lost_particles = np.isnan(PG_tip.x)
    PG_tip.x[lost_particles] = PG_laser.x[lost_particles]
    PG_tip.y[lost_particles] = PG_laser.y[lost_particles]
    PG_tip.z = tip_height(PG_tip.r, R, H, B)

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
        nr, nz = tip_normal(r[ii], R, H, B)
        nx = nr * np.cos(phi[ii])
        ny = nr * np.sin(phi[ii])
        Rmat = rotation_from_z_to_n([nx, ny, nz])
        pnew = Rmat @ np.array([px[ii], py[ii], pz[ii]])
        PG_tip.px[ii] = pnew[0]
        PG_tip.py[ii] = pnew[1]
        PG_tip.pz[ii] = pnew[2]
    
    return PG_tip
    

def cone_height(R, H, B):
    rt = (2.0 * R / (B**2 + 4.0 * (H - R)**2) *(np.sqrt(B**2 + 4.0 * H * (H - 2.0 * R)) * (H - R) + B * R))
    ht = (H - R) + np.sqrt(R**2 - rt**2)
    return B * ht / (B - 2.0*rt)

def PlaneIntersection(x0, y0, z0, theta):
    theta = np.abs(theta)
    if theta >= 0.5*np.pi:
        return None
    return x0/np.cos(theta) - z0*np.tan(theta)

def SphereIntersection(x0, y0, z0, theta, R, H):
    if (theta == 0.0):
        return x0, 0.0
    cot = np.cos(theta) / np.sin(theta)
    csc = 1.0 / np.sin(theta)
    term1 = cot * (H - R - z0 + x0 * csc)
    term2 = -y0**2- (H - z0) * (H - 2*R - z0)+ (R - y0) * (R + y0) * cot**2+ x0 * csc * (2 * (-H + R + z0) - x0 * csc)
    if (term2 < 0):
        return None, None
    x = (term1 + np.sqrt(term2)) * np.sin(theta)**2
    z = -x0*csc + x*cot + z0
    return x, z

def ConeIntersection(x0, y0, z0, theta, R, H, B):
    if (theta == 0.0):
        return x0, 0.0
    cot = 1.0/np.tan(theta)
    csc = 1.0/np.sin(theta)
    CH = cone_height(R, H, B)
    term = z0 - x0 * csc
    term2 =  CH**2 * (B**2 - 4 * y0**2) - 2 * B**2 * CH * term + B**2 * (y0**2 * cot**2 + term**2)
    if (term2 < 0.0):
        return None, None
    num = (B**2 * cot * (-CH + term)+ 2 * CH * np.sqrt(term2))
    den = 4 * CH**2 - B**2 * cot**2
    x = num / den
    z = -x0*csc + x*cot + z0
    return x, z

def FirstIntersection(x0, y0, z0, theta, R, H, B):
    rt = (2.0 * R / (B**2 + 4.0 * (H - R)**2) *(np.sqrt(B**2 + 4.0 * H * (H - 2.0 * R)) * (H - R) + B * R))
    ht = (H - R) + np.sqrt(R**2 - rt**2)

    pi_valid = False
    si_valid = False
    ci_valid = False
    
    
    pi_x = PlaneIntersection(x0, y0, z0, theta)
    if (pi_x is not None):
        pi_valid = pi_x**2 + y0**2 > (0.5*B)**2
    
    si_x, si_z = SphereIntersection(x0, y0, z0, theta, R, H)
    if (si_x is not None):
        si_valid = (si_x**2 + y0**2 <= rt**2) & (si_z > ht)
    
    ci_x, ci_z = ConeIntersection(x0, y0, z0, theta, R, H, B)
    if (ci_x is not None):
        ci_valid = (ci_x**2 + y0**2 > rt**2) & (ci_x**2 + y0**2 < (0.5*B)**2) & (ci_z < ht) & (ci_z > 0)
    
    valid_intersections = []
    if (pi_valid):
        valid_intersections.append(pi_x)
    if (si_valid):
        valid_intersections.append(si_x)
    if (ci_valid):
        valid_intersections.append(ci_x)

    if (len(valid_intersections) > 0):
        return np.max(valid_intersections)
    else:
        return np.nan
    

def tip_height(r, R, H, B):
    scalar_input = np.isscalar(r)
    r_arr = np.array(r, copy=False, ndmin=1, dtype=float)  # now r_arr.shape is (n,) or (1,) for a scalar

    rt = (2.0 * R / (B**2 + 4.0 * (H - R)**2) * (np.sqrt(B**2 + 4.0 * H * (H - 2.0 * R)) * (H - R) + B * R))
    ht = (H - R) + np.sqrt(R**2 - rt**2)

    h_arr = np.zeros_like(r_arr)

    # Region 1: r <= rt
    mask1 = (r_arr <= rt)
    if mask1.any():
        h_arr[mask1] = (H - R) + np.sqrt(R**2 - r_arr[mask1]**2)

    # Region 2: rt < r < 0.5*B
    mask2 = (r_arr > rt) & (r_arr < 0.5 * B)
    if mask2.any():
        h_arr[mask2] = ht - ((r_arr[mask2] - rt) / (0.5 * B - rt)) * ht

    # For r >= 0.5*B, h_arr stays zero

    # If the original input was a scalar, return a scalar
    if scalar_input:
        return h_arr[0]
    else:
        return h_arr

def tip_normal(r, R, H, B):
    scalar_input = np.isscalar(r)
    r_arr = np.array(r, copy=False, ndmin=1, dtype=float)  # now r_arr.shape is (n,) or (1,) for a scalar

    rt = (2.0 * R / (B**2 + 4.0 * (H - R)**2) *(np.sqrt(B**2 + 4.0 * H * (H - 2.0 * R)) * (H - R) + B * R))
    ht = (H - R) + np.sqrt(R**2 - rt**2)
    
    s_arr = np.zeros_like(r_arr)

    # Region 1: r <= rt
    mask1 = (r_arr <= rt)
    if mask1.any():
        s_arr[mask1] = -r_arr[mask1]/np.sqrt(R**2 - r_arr[mask1]**2)

    # Region 2: rt < r < 0.5*B
    mask2 = (r_arr > rt) & (r_arr < 0.5 * B)
    if mask2.any():
        s_arr[mask2] = ht / (0.5*B - rt)
    
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