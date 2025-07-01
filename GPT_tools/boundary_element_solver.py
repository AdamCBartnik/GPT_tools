import numpy as np
import matplotlib.pyplot as plt
import re, copy
import time

from scipy.integrate import quad
from scipy.special import ellipk, ellipe
from mpmath import ellippi

#from numba import njit, jit

def BEM_track(r0, z0, pr0, pz0, dt, steps, zmax, SI_to_fieldmap_scale, voltage_scale, r_all, z_all, sigma, segment_indices):
    # Constants
    eoverm = -1.75882001e11  # F/m

    # Convert momentum [eV/c] to velocity [m/s]
    vr = pr0 * 586.679206
    vz = pz0 * 586.679206

    r_vals = np.zeros(steps)
    z_vals = np.zeros(steps)
    pr_vals = np.zeros(steps)
    pz_vals = np.zeros(steps)

    r, z = r0, z0

    start = time.perf_counter()
    field_time_all = 0
    
    for i in range(steps):
        if z > zmax:
            return r_vals[:i], z_vals[:i], pr_vals[:i], pz_vals[:i]

        r_vals[i] = r
        z_vals[i] = z
        pr_vals[i] = vr / 586.679206
        pz_vals[i] = vz / 586.679206

        # k1
        Er1, Ez1 = evaluate_field(r*SI_to_fieldmap_scale, z*SI_to_fieldmap_scale, r_all, z_all, sigma, segment_indices)
        ar1 = eoverm * Er1 * voltage_scale * SI_to_fieldmap_scale
        az1 = eoverm * Ez1 * voltage_scale * SI_to_fieldmap_scale

        # k2
        r2 = r + 0.5 * dt * vr
        z2 = z + 0.5 * dt * vz
        vr2 = vr + 0.5 * dt * ar1
        vz2 = vz + 0.5 * dt * az1
        Er2, Ez2 = evaluate_field(r2*SI_to_fieldmap_scale, z2*SI_to_fieldmap_scale, r_all, z_all, sigma, segment_indices)
        ar2 = eoverm * Er2 * voltage_scale * SI_to_fieldmap_scale
        az2 = eoverm * Ez2 * voltage_scale * SI_to_fieldmap_scale

        # k3
        r3 = r + 0.5 * dt * vr2
        z3 = z + 0.5 * dt * vz2
        vr3 = vr + 0.5 * dt * ar2
        vz3 = vz + 0.5 * dt * az2
        Er3, Ez3 = evaluate_field(r3*SI_to_fieldmap_scale, z3*SI_to_fieldmap_scale, r_all, z_all, sigma, segment_indices)
        ar3 = eoverm * Er3 * voltage_scale * SI_to_fieldmap_scale
        az3 = eoverm * Ez3 * voltage_scale * SI_to_fieldmap_scale

        # k4
        r4 = r + dt * vr3
        z4 = z + dt * vz3
        vr4 = vr + dt * ar3
        vz4 = vz + dt * az3
        Er4, Ez4 = evaluate_field(r4*SI_to_fieldmap_scale, z4*SI_to_fieldmap_scale, r_all, z_all, sigma, segment_indices)
        ar4 = eoverm * Er4 * voltage_scale * SI_to_fieldmap_scale
        az4 = eoverm * Ez4 * voltage_scale * SI_to_fieldmap_scale

        # RK4 update
        r += dt * (vr + 2*vr2 + 2*vr3 + vr4) / 6
        z += dt * (vz + 2*vz2 + 2*vz3 + vz4) / 6
        vr += dt * (ar1 + 2*ar2 + 2*ar3 + ar4) / 6
        vz += dt * (az1 + 2*az2 + 2*az3 + az4) / 6

        if (r < 0):
            r = -r
            vr = -vr
                    
    time_all = time.perf_counter() - start
    print(f'{time_all} in total')
            
    return r_vals, z_vals, pr_vals, pz_vals



# --------------------------------------------------------------------------------------------------
#   IMPORTING FILES FROM POISSON SECTION
# --------------------------------------------------------------------------------------------------

def parse_geometry(filename, boundary_voltage = np.nan):
    lines = []
    with open(filename, 'r') as f:
        in_geometry = False
        for line in f:
            stripped = line.strip().upper()
            if not stripped or stripped.startswith("!"):
                continue
            if stripped.startswith("&"):
                if ('KPROB' not in stripped):
                    lines.append(stripped.split("!")[0].strip())

    parts = []
    elements = []

    voltage = np.nan
        
    for line in lines:
        if line == '&':
            if (np.isnan(boundary_voltage)):
                print('found boundary, ignoring')
            else:
                print(f'found boundary, voltage = {boundary_voltage}')
            if (parts):
                if not np.isnan(voltage):
                    elements.append(parts)
                parts = []
            voltage = boundary_voltage
            continue
            
        if  line.startswith('&REG'):
            
            if (parts):
                if not np.isnan(voltage):
                    elements.append(parts)
                parts = []
            voltage = float(re.search(r'VOLTAGE\s*=\s*([-+]?[0-9]*\.?[0-9]+)', line).group(1))
            print(f'Found material, voltage = {voltage}')
            continue

        if line.startswith("&PO"):
            line = line.replace('&po', '')
            x = float(re.search(r'X\s*=\s*([-+]?[0-9]*\.?[0-9]+)', line).group(1))
            y = float(re.search(r'Y\s*=\s*([-+]?[0-9]*\.?[0-9]+)', line).group(1))
            
            if 'RADIUS' in line:
                r = float(re.search(r'RADIUS\s*=\s*([-+]?[0-9]*\.?[0-9]+)', line).group(1))
                nt = float(re.search(r'NT\s*=\s*([-+]?[0-9]*\.?[0-9]+)', line).group(1))
                p = [x, y, voltage, r, nt]
            else:
                p = [x, y, voltage]
            
            parts.append(p)
            
    if (parts):
        if (parts):
            if not np.isnan(voltage):
                elements.append(parts)
            parts = []
            
    return elements

def is_clockwise_arc(center, p1, p2):
    v1x, v1y = p1[0] - center[0], p1[1] - center[1]
    v2x, v2y = p2[0] - center[0], p2[1] - center[1]
    cross = v1x * v2y - v1y * v2x
    return cross < 0  # True if clockwise


def segment_midpoint(p1, p2, cen):
    x1 = p1[0]
    y1 = p1[1]
    xc = p1[3]
    yc = p1[4]
    x2 = p2[0]
    y2 = p2[1]
    if (np.isnan(xc)):
        # straight line
        return [0.5*(x1+x2), 0.5*(y1+y2)]

    else:
        # circle with center at [cx, cy]
        xm = 0.5*(x1+x2)
        ym = 0.5*(y1+y2)
        R = np.sqrt((x1-xc)**2 + (y1-yc)**2)
        r = np.sqrt((xm-xc)**2 + (ym-yc)**2)
        x3 = xc + (R/r) * (xm - xc)
        y3 = yc + (R/r) * (ym - yc)
        return [x3, y3]
    
    
    
# 
def refine_once(elements, sigma, r_all, z_all, segment_indices, max_verr, show_plots=True):
    new_elements = []

    for (e, maxv) in zip(elements, max_verr):
        ind_to_fix = []
        midpoints_to_fix = []
        verr_list = []

        for p_ii in np.arange(0, len(e)-1):
            p1 = e[p_ii]
            p2 = e[p_ii+1]
            v1 = p1[2]
            v2 = p2[2]
            rc = p1[3]
            zc = p1[4]
            pm = segment_midpoint(p1, p2, [rc, zc])

            vm = evaluate_potential(pm[0], pm[1], r_all, z_all, sigma, segment_indices)
            verr = vm - 0.5*(v1 + v2)
            verr_list.append(verr)
            if (np.abs(verr) > maxv):
                #print(f'Point {p_ii}: dV({pm[0]}, {pm[1]}) = {verr}')
                ind_to_fix.append(p_ii)
                midpoints_to_fix.append(pm)
                                
        e2 = refine_element(e, ind_to_fix)
        new_elements.append(e2)

        if (show_plots):
            v_avg = np.mean([p[2] for p in e2])
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
            ax2.plot(verr_list, 'bo')
            ax2.set_xlabel('Point index')
            ax2.set_ylabel('Voltage error')
            ax2.set_title(f'<V_target> = {v_avg}')
            plot_geometry([e2], fig_ax = (fig, ax1))
            for p in midpoints_to_fix:
                ax1.plot(p[0], p[1], 'ko')
            
            plt.tight_layout()

    return new_elements


#Adds a new point in between chosen points
def refine_element(element, starting_indices_to_refine):
    new_element = []
    
    if (len(starting_indices_to_refine)==0):
        return copy.deepcopy(element)
        
    for ii in np.arange(len(element)-1):
        new_element.append(element[ii])
        
        if (ii in starting_indices_to_refine):
            p1 = element[ii]
            p2 = element[ii+1]
            v1 = p1[2]
            v2 = p2[2]
            xc = p1[3]
            yc = p1[4]
            rm = segment_midpoint(p1, p2, [xc, yc])
            new_element.append([rm[0], rm[1], 0.5*(v1+v2), xc, yc])
                
    new_element.append(element[-1])
    return new_element
                

def subdivide_elements(elements, max_seg_length):
    if np.isscalar(max_seg_length):
        max_seg_length = np.full(len(elements), max_seg_length)
    
    new_elements = []
    
    for jj, e in enumerate(elements):
        points = []
        
        max_l = max_seg_length[jj]
        
        for ii in np.arange(len(e)-1):
            p1 = e[ii]
            p2 = e[ii+1]
            x1 = p1[0]
            y1 = p1[1]
            x2 = p2[0]
            y2 = p2[1]
            
            v1 = p1[2]
            v2 = p2[2]
        
            is_arc = len(p2) > 3
            if (is_arc):
                is_arc = p2[4] > 1
        
            if (is_arc):
                # connect to the point with an arc
                r = p2[3]
                nt = p2[4]
                                
                cen = circle_circle_intersection((x1, y1), (x2, y2), r)
                
                isCW = is_clockwise_arc(cen[0], (x1, y1), (x2, y2))
                                
                if (nt == 5):  # clockwise == bad math way
                    if (isCW):
                        cen = cen[0]
                    else:
                        cen = cen[1]
                else:
                    if (nt == 4):
                        if (isCW): # CCW == good math way
                            cen = cen[1]
                        else:
                            cen = cen[0]
                    
                xc = cen[0]
                yc = cen[1]

                theta1 = np.arctan2(y1 - yc, x1 - xc)
                theta2 = np.arctan2(y2 - yc, x2 - xc)
                    
                if (theta2 < theta1 and nt == 4):
                    theta2 += 2 * np.pi
                    
                if (theta1 < theta2 and nt == 5):
                    theta2 -= 2 * np.pi

                arc_length = abs(r * (theta2 - theta1))
                n_segments = max(2, int(np.ceil(arc_length / max_l)))
                thetas = np.linspace(theta1, theta2, n_segments + 1)
                thetas = thetas[:-1]
                
                x = xc + r * np.cos(thetas)
                y = yc + r * np.sin(thetas)
            else:
                xc = np.nan
                yc = np.nan
                length = np.hypot(x2 - x1, y2 - y1)
                n_segments = max(2, int(np.ceil(length / max_l)))
                x = np.linspace(x1, x2, n_segments + 1)
                y = np.linspace(y1, y2, n_segments + 1)
                x = x[:-1]
                y = y[:-1]
                
            v = np.linspace(v1, v2, n_segments + 1)
            for jj in np.arange(len(x)):
                points.append([x[jj], y[jj], v[jj], xc, yc])
            
        points.append(e[-1])
        new_elements.append(points)

    return new_elements

def circle_circle_intersection(p1, p2, r):
    x1, y1 = p1
    x2, y2 = p2

    dx = x2 - x1
    dy = y2 - y1
    d = np.hypot(dx, dy)

    if d > 2 * r:
        raise ValueError("No intersection: circles too far apart")
    if d == 0:
        raise ValueError("Infinite solutions: centers are the same")
    
    # Midpoint between p1 and p2
    x3 = (x1 + x2) / 2
    y3 = (y1 + y2) / 2

    # Distance from midpoint to intersection points
    h = np.sqrt(r**2 - (d/2)**2)

    # Unit perpendicular vector
    rx = -dy / d
    ry = dx / d

    # Two intersection points
    xi1 = x3 + h * rx
    yi1 = y3 + h * ry
    xi2 = x3 - h * rx
    yi2 = y3 - h * ry

    return (xi1, yi1), (xi2, yi2)

def arc_to_segments(x1, y1, x2, y2, xc, yc, max_seg_length):
    r = np.hypot(x1 - xc, y1 - yc)
    theta1 = np.arctan2(y1 - yc, x1 - xc)
    theta2 = np.arctan2(y2 - yc, x2 - xc)
    if theta2 < theta1:
        theta2 += 2 * np.pi

    arc_length = abs(r * (theta2 - theta1))
    n_segments = max(2, int(np.ceil(arc_length / max_seg_length)))
    thetas = np.linspace(theta1, theta2, n_segments + 1)

    x_points = xc + r * np.cos(thetas)
    y_points = yc + r * np.sin(thetas)

    return np.column_stack((x_points[:-1], y_points[:-1], x_points[1:], y_points[1:]))


def plot_geometry(elements, fig_ax=None, axis='equal', plot_type='-'):
    if (fig_ax is None):
        fig, ax = plt.subplots()
    else:
        fig = fig_ax[0]
        ax = fig_ax[1]

    for e in elements:
        x = [p[0] for p in e]
        y = [p[1] for p in e]
                
        ax.plot(x, y, plot_type)
        
        if (axis):
            ax.set_aspect(axis)

# --------------------------------------------------------------------------------------------------
#   BOUNDARY ELEMENT SOLVER SECTION     
# --------------------------------------------------------------------------------------------------
        

def G_ring(r, z, rp, zp):
    """Green's function for a ring in cylindrical symmetry (unitless, no 1/4πε₀)."""
    m_numerator = 4 * r * rp
    m_denominator = (r + rp)**2 + (z - zp)**2
    if m_denominator == 0:
        return 0
    m = m_numerator / m_denominator
    m = min(m, 1.0)
    return (1 / np.pi) * ellipk(m) / np.sqrt(m_denominator)

def linear_segment_potential_contrib(ri, zi, r0, z0, r1, z1):
    """Contributions from segment (r0,z0)-(r1,z1) to vertex at (ri, zi), including rp Jacobian."""
    def integrand(s, target_r, target_z):
        rp = r0 + s * (r1 - r0)
        zp = z0 + s * (z1 - z0)
        G = G_ring(target_r, target_z, rp, zp)
        return (1 - s) * rp * G, s * rp * G  # includes Jacobian rp

    length = np.sqrt((r1 - r0)**2 + (z1 - z0)**2)

    def integrand0(s): return integrand(s, ri, zi)[0]
    def integrand1(s): return integrand(s, ri, zi)[1]

    phi_ij, _ = quad(integrand0, 0, 1, limit=200, points=[0,0.5,1])
    phi_ij1, _ = quad(integrand1, 0, 1, limit=200, points=[0,0.5,1])
    
    return phi_ij * length, phi_ij1 * length

def flatten_conductors(r_conductors, z_conductors, V_conductors):
    """Flatten multiple conductor lists and track segment ownership."""
    r_all, z_all, V_all = [], [], []
    segment_indices = []  # list of (start_index, end_index) for each conductor
    idx = 0
    for r_list, z_list, V_list in zip(r_conductors, z_conductors, V_conductors):
        n = len(r_list)
        r_all.extend(r_list)
        z_all.extend(z_list)
        V_all.extend(V_list)
        segment_indices.append((idx, idx + n))
        idx += n
    return np.array(r_all), np.array(z_all), np.array(V_all), segment_indices

def build_linear_BEM_matrix(r_all, z_all, segment_indices):
    """Assemble BEM matrix A using linear interpolation basis."""
    N = len(r_all)
    A = np.zeros((N, N))
    for i in range(N):
        ri, zi = r_all[i], z_all[i]
        for start, end in segment_indices:
            for j in range(start, end - 1):
                rj0, zj0 = r_all[j], z_all[j]
                rj1, zj1 = r_all[j + 1], z_all[j + 1]
                phi_j, phi_j1 = linear_segment_potential_contrib(ri, zi, rj0, zj0, rj1, zj1)
                A[i, j]     += phi_j
                A[i, j + 1] += phi_j1
    return A


def make_C_matrix(A, r_all, z_all, segment_indices):
    C = []

    for s in segment_indices:
        i0 = s[0]
        i1 = s[1]-1

        dr = np.sqrt((r_all[i0] - r_all[i1])**2 + (z_all[i0] - z_all[i1])**2)

        if dr < 1.0e-12:
            C_row = [0.0] * len(A)
            C_row[i0] = 1.0
            C_row[i1] = -1.0
            C.append(C_row)

    return np.matrix(C)


def solve_linear_BEM(elements):

    r = [[eii[0] for eii in e] for e in elements]
    z = [[eii[1] for eii in e] for e in elements]
    V = [[eii[2] for eii in e] for e in elements]

    r_all, z_all, V_all, segment_indices = flatten_conductors(r, z, V)
    A = build_linear_BEM_matrix(r_all, z_all, segment_indices)   
    C = make_C_matrix(A, r_all, z_all, segment_indices)
    n_closed = C.shape[0]
    
    if (n_closed == 0):
        # No closed surface elements
        sigma = np.linalg.solve(A, V_all)
    else:
        print(f'Found {n_closed} closed surfaces... handling extra constraints.')
        # Handle extra contraint with closed surfaces that it begins and end with same sigma
        bigAL = np.concatenate((A, C), axis=0)
        bigAR = np.concatenate((C.T, np.zeros((n_closed, n_closed))), axis=0)
        
        #print(f'{bigAL.shape} ,  {bigAR.shape}')
        
        bigA = np.concatenate((bigAL, bigAR), axis=1)
        
        bigV = np.concatenate((V_all, np.zeros(n_closed)), axis=0)
        
        sigma = np.linalg.solve(bigA, bigV)
        sigma = sigma[:-n_closed]
        
    return sigma, A, r_all, z_all, segment_indices
        
def evaluate_potential(r_eval, z_eval, r_all, z_all, sigma, segment_indices):
    """Evaluate potential at arbitrary (r_eval, z_eval) from BEM result."""
    phi = 0.0
    for start, end in segment_indices:
        for j in range(start, end - 1):
            r0, z0 = r_all[j], z_all[j]
            r1, z1 = r_all[j + 1], z_all[j + 1]
            sigma0, sigma1 = sigma[j], sigma[j + 1]

            def integrand(s):
                rp = r0 + s * (r1 - r0)
                zp = z0 + s * (z1 - z0)
                G = G_ring(r_eval, z_eval, rp, zp)
                sigma_s = (1 - s) * sigma0 + s * sigma1
                return sigma_s * rp * G  # includes Jacobian

            length = np.sqrt((r1 - r0)**2 + (z1 - z0)**2)
            phi_seg, _ = quad(integrand, 0, 1, limit=200, points=(0,0.5,1))
            if np.isnan(phi_seg):
                raise ValueError(f'NaN found at segment: ({r0}, {z0}) and ({r1}, {z1})')
                
            phi += phi_seg * length
    return phi    

def G_ring_with_derivatives(r, z, rp, zp):
    """Return G, dG/dr, and dG/dz for the Green's function of a ring."""
    #start_t = time.perf_counter()
    
    delta_rp = r + rp
    dz = z - zp
    denom = delta_rp**2 + dz**2
    sqrt_denom = np.sqrt(denom)

    # Avoid division by zero at singularity
    if denom==0:
        return 0.0, 0.0, 0.0
    
    m = 4 * r * rp / denom
    m = min(m, 1.0)
    
    if (m == 0):
        G = 0.5/sqrt_denom
        dG_dr = -0.5*delta_rp/sqrt_denom**3
        dG_dz = -0.5*dz/sqrt_denom**3
        return G, dG_dr, dG_dz
    
    #start_e = time.perf_counter()
    K = ellipk(m)
    E = ellipe(m)
    #time_ellip = time.perf_counter() - start_e
    
    # Green's function itself
    G = (1 / np.pi) * K / sqrt_denom

    # dK/dm
    dK_dm = (E - (1 - m) * K) / (2 * m * (1 - m))

    # Derivatives of m
    dm_dr = (4 * rp * (rp**2 - r**2 + dz**2)) / denom**2
    dm_dz = -8 * r * rp * dz / denom**2

    # Derivatives of sqrt_denom
    d_denom_dr = 2 * delta_rp
    d_denom_dz = 2 * dz
    d_sqrt_denom_dr = d_denom_dr / (2 * sqrt_denom)
    d_sqrt_denom_dz = d_denom_dz / (2 * sqrt_denom)

    # Derivatives of G
    dG_dr = (1 / np.pi) * (dK_dm * dm_dr / sqrt_denom - K * d_sqrt_denom_dr / denom)
    dG_dz = (1 / np.pi) * (dK_dm * dm_dz / sqrt_denom - K * d_sqrt_denom_dz / denom)

    #time_all = time.perf_counter() - start_t
    
    #print(f'{time_ellip*1e6} in ellip')
    #print(f'{time_all*1e6} in function')
    
    return G, dG_dr, dG_dz

def evaluate_field(r_eval, z_eval, r_all, z_all, sigma, segment_indices):
    """Evaluate (E_r, E_z) at (r_eval, z_eval) from BEM charge segments."""
    
    #start_t = time.perf_counter()
    #time_quad = 0
    
    Er_total, Ez_total = 0.0, 0.0
   
    for start, end in segment_indices:
        for j in range(start, end - 1):
            r0, z0 = r_all[j], z_all[j]
            r1, z1 = r_all[j + 1], z_all[j + 1]
            sigma0, sigma1 = sigma[j], sigma[j + 1]

            def integrand(s):
                rp = r0 + s * (r1 - r0)
                zp = z0 + s * (z1 - z0)
                sigma_s = (1 - s) * sigma0 + s * sigma1
                _, dG_dr, dG_dz = G_ring_with_derivatives(r_eval, z_eval, rp, zp)
                return sigma_s * rp * dG_dr, sigma_s * rp * dG_dz

            length = np.sqrt((r1 - r0)**2 + (z1 - z0)**2)

            def integrand_Er(s): return integrand(s)[0]
            def integrand_Ez(s): return integrand(s)[1]

            #start_i = time.perf_counter()
            Er_seg, _ = quad(integrand_Er, 0, 1, limit=100, points=[0,1])
            Ez_seg, _ = quad(integrand_Ez, 0, 1, limit=100, points=[0,1])
            #Er_seg = 0
            #Ez_seg = 0
            
            #time_quad = time_quad + time.perf_counter() - start_i

            Er_total += Er_seg * length
            Ez_total += Ez_seg * length

    if (r_eval == 0.0):
        Er_total = 0.0
            
    #time_all = time.perf_counter() - start_t
    #print(f'{time_all} in total')
    #print(f'{time_quad} in quad')
            
    return -Er_total, -Ez_total  # Electric field is negative gradient of potential


# --------------------------------------------------------------------------------------------------
#   ANALYTIC TEST CASES
# --------------------------------------------------------------------------------------------------
        
        
def voltage_step_cathode(r, z, a, V0, calc_voltage=True, calc_field=True):
    # On the z=0 plane, a conductor with V(r < a) = V0    and    V(r > a) = 0
    # Returns voltage, Er, and Ez
    # To skip (time-consuming) calculation of either the field or voltage, change flags
        
    phi = np.nan
    Er = np.nan
    Ez = np.nan
        
    k_sq = (4 * r) / (a * ((1 + r/a)**2 + (z/a)**2))
    ele = ellipe(k_sq)
    elk = ellipk(k_sq)
    
    if (calc_voltage):
        if (r == a):
            if (z==0):
                phi = 0.5*V0
            else:
                phi =  0.5*V0 - V0/np.pi*ellipk(-4*(a/z)**2)
            
        else:
            b0 = a * np.sqrt((1 + r/a)**2 + (z/a)**2)
            n = (4 * r) / (a * (1 + r/a)**2)

            phi = V0 * np.heaviside(1 - r/a, 0.5) - (V0 / np.pi * z / b0) * (ellipk(k_sq) + (1 - r/a) / (1 + r/a) * float(ellippi(n, k_sq)))       
        
    if (calc_field):
        E0 = V0 / (a * np.pi * np.sqrt((1 + r/a)**2 + (z/a)**2))

        b1 = 1 - (r**2)/(a**2) - (z**2)/(a**2)
        b2 = (1 - r/a)**2 + (z/a)**2
        b3 = 1 + (r**2)/(a**2) + (z**2)/(a**2)

        if (r == 0.0):
            Er = 0.0
        else:
            Er = E0 * (z/r)*((b3 / b2) * ele - elk)
        Ez = E0 * ((b1 / b2) * ele + elk)
        
    return (phi, Er, Ez)
    

def conducting_disk(r, z, a, V0):
    # On the z=0 plane, a conducting disk of voltage V0 with radius a, nothing outside of it
    # Returns voltage, Er, and Ez
    
    phi = np.nan
    Er = np.nan
    Ez = np.nan
    
    rpa = np.sqrt((r + a)**2 + z**2)
    rma = np.sqrt((r - a)**2 + z**2)
    
    phi = (2.0/np.pi) * V0 * np.arcsin((2*a)/(rpa + rma))
    
    if (z == 0):
        if (r < a):
            Ez = 2.0*V0/(np.pi * np.sqrt(a**2 - r**2))
            Er = 0.0
        else:
            Ez = 0.0
            Er = 2.0*a*V0/(np.pi * r * np.sqrt(r**2 - a**2))
    else:
        Ez = 2.0*np.sqrt(2.0) * a * z * V0 / (np.pi* rpa * rma * np.sqrt(-a**2 + r**2 + z**2 + np.sqrt((a**2 - r**2)**2 + 2*(a**2 + r**2) * z**2 + z**4)))
        Er = -((rma - rpa) * np.sqrt(1 - (4 * a**2) / (rma + rpa)**2) * V0) / (np.pi * rpa * rma)
    
    return (phi, Er, Ez)