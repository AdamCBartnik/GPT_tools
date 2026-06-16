from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
import matplotlib.pyplot as plt
import warnings

import copy, os, shutil
import numpy as np
from scipy.io import savemat, loadmat
from scipy.stats import binned_statistic_2d
from scipy.ndimage import shift
from scipy.signal import fftconvolve
from scipy.optimize import minimize
from scipy.linalg import expm, logm, sqrtm
from GPT_tools.tools import get_screen_data
from GPT_tools.GPTExtension import run_gpt_with_settings


def aperture_scan(settings, GPT_INPUT_FILE, DISTGEN_INPUT_FILE,
                    noise_settings = {},
                    root_dir = '/tmp/fake_data',
                    sub_dir = 'fake',
                    max_workers = 4,
                    n_particle = 5000,            # overwrite what is in settings
                    laser_offsets = [[0, 0]],     # micron
                    scan_nxny = [41,41],          # number of scan points
                    scan_n_sigma = 4,             # scan +/- n_sigma of beam width at aperture
                    viewscreen_nxny = [201, 199], # pixels in viewscreen
                    viewscreen_calib = 25.6,      # microns / pixel
                    z_aperture = 0.9264,
                    z_screen = 1.42817,
                    scan_x_var = 'scanner_x_current',
                    scan_y_var = 'scanner_y_current',
                    typical_scan_setting = 0.01,   # something to give a small beam kick, for calibration
                    random_seed = None
                 ):
    
    required_parameters_in_settings = {'n_particle', 'aperture_on', 'n_screens', 'gun_voltage', 'transforms:tx:avg_x:value', 'transforms:ty:avg_y:value'}
    
    for p in required_parameters_in_settings:
        if p not in settings.keys():
            raise ValueError(f'ERROR: {p} not in settings.')

    for p in noise_settings.keys():
        if p not in settings.keys():
            raise ValueError(f'ERROR: {p} not in settings, but it is in noise_settings.')
    
    ensure_empty_dir(root_dir)
    
    s = copy.copy(settings)
    s['n_screens'] = 0
    s['aperture_on'] = 0
    s['n_particle'] = n_particle
    s['transforms:tx:avg_x:value'] = 0.0
    s['transforms:ty:avg_y:value'] = 0.0
    
    s[scan_x_var] = 0.0
    s[scan_y_var] = 0.0
    s['n_particle'] = 2000  # For initial rough runs
    g = run_gpt_with_settings(s, gpt_input_file=GPT_INPUT_FILE, distgen_input_file=DISTGEN_INPUT_FILE, verbose=False, gpt_verbose=False, auto_phase=False, timeout=1000)
    aper = get_screen_data(g, screen_z=z_aperture)[0]
    scr = get_screen_data(g, screen_z=z_screen)[0]
    total_charge = scr.charge
    
    Taper = get_transfer_matrix(g, z_aperture) # transfer matrix, [x, px, y, py]
    Tscr = get_transfer_matrix(g, z_screen) # transfer matrix, [x, px, y, py]
    
    scr_x0 = scr['mean_x'] # center of viewscreen image, in case beamline has kick after aperture
    scr_y0 = scr['mean_y']
    aper_x0 = aper['mean_x']
    aper_y0 = aper['mean_y']
    aper_sigx = aper['sigma_x']
    aper_sigy = aper['sigma_y']
    
    # calibrate magnet on aperture and screen
    s[scan_x_var] = typical_scan_setting
    s[scan_y_var] = typical_scan_setting
    g = run_gpt_with_settings(s, gpt_input_file=GPT_INPUT_FILE, distgen_input_file=DISTGEN_INPUT_FILE, verbose=False, gpt_verbose=False, auto_phase=False, timeout=1000)
    aper = get_screen_data(g, screen_z=z_aperture)[0]
    scr = get_screen_data(g, screen_z=z_screen)[0]
    aper_x_cal = (aper['mean_x']-aper_x0) / typical_scan_setting  #  meters / amp
    aper_y_cal = (aper['mean_y']-aper_y0) / typical_scan_setting
    scr_x_cal = (scr['mean_x']-scr_x0) / typical_scan_setting #  meters / amp
    scr_y_cal = (scr['mean_y']-scr_y0) / typical_scan_setting
    
    s['n_particle'] = n_particle
    s['aperture_on'] = 1
    s['transforms:tx:avg_x:units'] = 'um'
    s['transforms:ty:avg_y:units'] = 'um'
    
    for laser_offset in laser_offsets:
        print(f'Beginning laser offset: {laser_offset}')
        s['transforms:tx:avg_x:value'] = laser_offset[0]
        s['transforms:ty:avg_y:value'] = laser_offset[1]
        
        # Magnet kick to recenter beam    
        aper_offset = Taper @ np.array([laser_offset[0] * 1e-6, 0.0, laser_offset[1] * 1e-6, 0.0])
        scan_x_offset = -aper_offset[0] / aper_x_cal
        scan_y_offset = -aper_offset[2] / aper_y_cal
    
        # Magnet scan settings
        scan_x_list = scan_x_offset + np.linspace(-scan_n_sigma * aper_sigx / aper_x_cal, scan_n_sigma * aper_sigx / aper_x_cal, scan_nxny[0])
        scan_y_list = scan_y_offset + np.linspace(-scan_n_sigma * aper_sigy / aper_y_cal, scan_n_sigma * aper_sigy / aper_y_cal, scan_nxny[1])
    
        # Expected viewscreen centroid with laser offset and magnet kick
        scr_offset = Tscr @ np.array([laser_offset[0] * 1e-6, 0.0, laser_offset[1] * 1e-6, 0.0])
        view_x_offset = scr_offset[0] + scan_x_offset * scr_x_cal
        view_y_offset = scr_offset[2] + scan_y_offset * scr_y_cal
    
        view_x_offset_pixel = np.round(view_x_offset / (1e-6*viewscreen_calib))
        view_y_offset_pixel = np.round(view_y_offset / (1e-6*viewscreen_calib))
    
        xedges = 1e-6*viewscreen_calib * (view_x_offset_pixel + np.arange(0, viewscreen_nxny[0]+1) - 0.5 - np.floor(0.5*viewscreen_nxny[0]))
        yedges = 1e-6*viewscreen_calib * (view_y_offset_pixel + np.arange(0, viewscreen_nxny[1]+1) - 0.5 - np.floor(0.5*viewscreen_nxny[1]))
        xcenters = 0.5*(xedges[0:-1] + xedges[1:])
        ycenters = 0.5*(yedges[0:-1] + yedges[1:])
    
        calib_data = {}
        calib_data['viewscreen_scale'] = [float(viewscreen_calib), float(viewscreen_calib)]
        calib_data['camera_roi_offsets'] = [float(view_x_offset_pixel), float(view_y_offset_pixel)]
        calib_data['magnet_center_values'] = [scan_x_offset, scan_y_offset]
        calib_data['aperture_scale'] = [aper_x_cal*1e6, aper_y_cal*1e6]        # micron / amp
        calib_data['screen_scale'] = [scr_x_cal*1e6, scr_y_cal*1e6]            # micron / amp
        calib_data['aperture_screen_separation'] = (z_screen - z_aperture)*1e6 # micron
        calib_data['gun_voltage'] = float(s['gun_voltage'])   # kV
        calib_data['magnet_x_values'] = scan_x_list
        calib_data['magnet_y_values'] = scan_y_list
    
        sub_dir_laser = sub_dir + f'_{laser_offset[0]}_{laser_offset[1]}_SNInf'
        ensure_empty_dir(Path(root_dir) / sub_dir_laser)
    
        with open(Path(root_dir) / sub_dir_laser / 'laser_info.txt', 'w') as f:
            f.write(f'{laser_offset[0]} {laser_offset[1]}\n')
    
        process_vars = {
            's': s,
            'noise_settings': noise_settings,
            'scan_x_var': scan_x_var,
            'scan_y_var': scan_y_var,
            'GPT_INPUT_FILE': GPT_INPUT_FILE,
            'DISTGEN_INPUT_FILE': DISTGEN_INPUT_FILE,
            'z_screen': z_screen,
            'total_charge': total_charge,
            'xedges': xedges,
            'yedges': yedges,
            'viewscreen_nxny': viewscreen_nxny,
            'calib_data': calib_data,
            'scan_x_offset': scan_x_offset,
            'scan_y_offset': scan_y_offset,
            'aper_x_cal': aper_x_cal,
            'aper_y_cal': aper_y_cal,
            'root_dir': root_dir,
            'sub_dir_laser': sub_dir_laser,
        }
        
        points = list(product(scan_x_list, scan_y_list))
        n_total = len(scan_x_list) * len(scan_y_list)

        master_ss = np.random.SeedSequence(random_seed)
        child_seeds = master_ss.spawn(len(points))
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_one_point, mx, my, process_vars, child_seed) for (mx, my), child_seed in zip(points, child_seeds)]
    
            for i, future in enumerate(as_completed(futures), start=1):
                future.result()
    
                if (i % len(scan_x_list) == 0) or (i == n_total):
                    print(f"[{i}/{n_total} | {100 * i / n_total:5.1f}%] Finished")
    
def change_matrix_units(M, units = [1, 510998.95e6, 1, 510998.95e6], inverse=False):
    units = np.array(units)
    if (not inverse):
        Dunits = np.diag(units)
    else:
        Dunits = np.diag(1.0/units)
    DunitsInv = np.diag(1.0/np.diag(Dunits))
    return Dunits@M@DunitsInv



def best_larmor_frame(M, angle=None):
    def larmor_rotation(theta):
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([
            [ c, 0,  s, 0],
            [ 0, c,  0, s],
            [-s, 0,  c, 0],
            [ 0,-s,  0, c],
        ], dtype=float)

    if (angle is None):
        A = M[0:2, 0:2]
        B = M[0:2, 2:4]
        C = M[2:4, 0:2]
        D = M[2:4, 2:4]
    
        a = np.linalg.norm(B, 'fro')**2 + np.linalg.norm(C, 'fro')**2
        b = np.linalg.norm(A, 'fro')**2 + np.linalg.norm(D, 'fro')**2
        d = np.sum(A * C) - np.sum(B * D)
    
        angle = 0.5 * (np.arctan2(d, 0.5 * (a - b)) + np.pi)
    Mp = larmor_rotation(-angle) @ M
    return Mp, angle

def _symplectic_J():
    return np.array([
        [0.0,  1.0, 0.0,  0.0],
        [-1.0, 0.0, 0.0,  0.0],
        [0.0,  0.0, 0.0,  1.0],
        [0.0,  0.0, -1.0, 0.0]
    ])


def _vec_to_symmetric4(p):
    """
    Map 10 parameters to a symmetric 4x4 matrix.
    Order:
      [k11, k22, k33, k44, k12, k13, k14, k23, k24, k34]
    """
    K = np.array([
        [p[0], p[4], p[5], p[6]],
        [p[4], p[1], p[7], p[8]],
        [p[5], p[7], p[2], p[9]],
        [p[6], p[8], p[9], p[3]],
    ], dtype=float)
    return K


def _symmetric4_to_vec(K):
    return np.array([
        K[0, 0],
        K[1, 1],
        K[2, 2],
        K[3, 3],
        K[0, 1],
        K[0, 2],
        K[0, 3],
        K[1, 2],
        K[1, 3],
        K[2, 3],
    ], dtype=float)


def _fit_affine_unconstrained(X, Y):
    """
    Fit Y ~= M X + b
    X, Y shape = (4, N)
    """
    N = X.shape[1]
    A = np.vstack([X, np.ones(N)])                # (5, N)
    Z = np.linalg.lstsq(A.T, Y.T, rcond=None)[0].T   # (4, 5)
    M = Z[:, :4]
    b = Z[:, 4]
    return M, b


def _symplecticity_error(M):
    J = _symplectic_J()
    return np.linalg.norm(M.T @ J @ M - J)


def _nearest_symplectic_polar(A):
    """
    Return a symplectic matrix M near A using the symplectic polar decomposition.

    Formula:
        M = A ( -J A^T J A )^{-1/2}

    For symplectic A, the factor in parentheses is I.
    """
    J = _symplectic_J()
    P = -J @ A.T @ J @ A

    # Matrix square root and inverse square root
    P_sqrt = sqrtm(P)
    P_inv_sqrt = np.linalg.inv(P_sqrt)

    M = A @ P_inv_sqrt

    # Numerical cleanup: keep only real part if tiny imaginary noise appears
    M = np.real_if_close(M, tol=1000)
    return np.asarray(M, dtype=float)


def _initial_p0_from_unconstrained(M_unc):
    """
    Build an initial 10-parameter vector p0 for the symplectic fit
    from an unconstrained 4x4 matrix M_unc.

    Steps:
      1) Project M_unc to nearby symplectic matrix Ms
      2) Compute L = log(Ms)
      3) Set K = -J L  so that Ms = exp(J K)
      4) Symmetrize K and pack into 10 params
    """
    J = _symplectic_J()

    Ms = _nearest_symplectic_polar(M_unc)

    # Guard against tiny numerical complex parts
    L = logm(Ms)
    L = np.real_if_close(L, tol=1000)
    L = np.asarray(L, dtype=float)

    K = -J @ L

    # In exact arithmetic K should be symmetric if Ms = exp(JK)
    K = 0.5 * (K + K.T)

    return _symmetric4_to_vec(K), Ms


def _fit_affine_symplectic(X, Y, p0=None, method='BFGS'):
    """
    Fit Y ~= M X + b with M symplectic by construction:
        M = expm(J K),  K = K^T
    """
    J = _symplectic_J()

    xbar = np.mean(X, axis=1)
    ybar = np.mean(Y, axis=1)

    # For fixed M, best-fit offset is b = ybar - M xbar
    Xc = X - xbar[:, None]
    Yc = Y - ybar[:, None]

    def objective(p):
        K = _vec_to_symmetric4(p)
        JK = J @ K
    
        # Optional early bailout if generator is too large
        gen_norm = np.linalg.norm(JK, ord='fro')
        if gen_norm > 50:
            return 1e300
    
        try:
            M = expm(JK)
        except Exception:
            return 1e300
    
        M = np.asarray(M)
        if np.iscomplexobj(M):
            max_imag = np.max(np.abs(np.imag(M)))
            if max_imag < 1e-9:
                M = np.real(M)
            else:
                return 1e300
    
        if not np.all(np.isfinite(M)):
            return 1e300
    
        R = Yc - M @ Xc
    
        if not np.all(np.isfinite(R)):
            return 1e300
    
        f = np.sum(R**2)
        if not np.isfinite(f):
            return 1e300
    
        return f

    if p0 is None:
        M_unc, _ = _fit_affine_unconstrained(X, Y)
        p0, M0 = _initial_p0_from_unconstrained(M_unc)
    else:
        M0 = expm(J @ _vec_to_symmetric4(p0))

    res = minimize(objective, p0, method=method)

    K = _vec_to_symmetric4(res.x)
    M = expm(J @ K)
    b = ybar - M @ xbar

    return M, b, res, M0


def _extract_matched_coords(gpt_data, screen_location):
    cathode = get_screen_data(gpt_data, screen_z=0)[0]
    final_screen = get_screen_data(gpt_data, screen_z=screen_location)[0]

    cathode_idx = {pid: i for i, pid in enumerate(cathode.id)}
    final_idx   = {pid: i for i, pid in enumerate(final_screen.id)}

    common_ids = np.array(sorted(set(cathode.id).intersection(final_screen.id)))
    if len(common_ids) < 5:
        raise ValueError("Need at least 5 common particles for affine 4D fit.")

    ic = np.array([cathode_idx[pid] for pid in common_ids])
    fc = np.array([final_idx[pid]   for pid in common_ids])

    X = np.vstack((
        cathode.x[ic],
        cathode.px[ic],
        cathode.y[ic],
        cathode.py[ic]
    ))

    Y = np.vstack((
        final_screen.x[fc],
        final_screen.px[fc],
        final_screen.y[fc],
        final_screen.py[fc]
    ))

    return X, Y, common_ids


def get_transfer_matrix(
    gpt_data,
    screen_location,
    force_symplectic=False,
    return_offset=False,
    return_diagnostics=False,
    p0=None,
    method='BFGS'
):
    """
    Fit final_coords ~= M @ initial_coords + b

    Parameters
    ----------
    gpt_data : object
    screen_location : float
    force_symplectic : bool
        If True, constrain M to be symplectic.
    return_offset : bool
        If True, also return b.
    return_diagnostics : bool
        If True, also return diagnostic info.
    p0 : array-like or None
        Optional initial 10-parameter guess for the symplectic fit.
        If None and force_symplectic=True, the unconstrained affine fit
        is used to construct an initial guess automatically.
    method : str
        scipy.optimize.minimize method for the symplectic fit.

    Returns
    -------
    M : (4,4) ndarray
    b : (4,) ndarray, optional
    info : dict, optional
    """
    X, Y, common_ids = _extract_matched_coords(gpt_data, screen_location)

    if force_symplectic:
        M, b, optres, M0 = _fit_affine_symplectic(X, Y, p0=p0, method=method)

        R = Y - (M @ X + b[:, None])

        info = {
            'common_ids': common_ids,
            'n_particles': X.shape[1],
            'optimizer_success': optres.success,
            'optimizer_message': optres.message,
            'objective_value': optres.fun,
            'optimizer_result': optres,
            'initial_symplectic_guess_matrix': M0,
            'initial_symplectic_guess_error': _symplecticity_error(M0),
            'final_symplecticity_error': _symplecticity_error(M),
            'rms_residual_per_coordinate': np.sqrt(np.mean(R * R, axis=1)),
            'residual_sum_squares': np.sum(R * R),
        }
    else:
        M, b = _fit_affine_unconstrained(X, Y)
        R = Y - (M @ X + b[:, None])

        info = {
            'common_ids': common_ids,
            'n_particles': X.shape[1],
            'residual_sum_squares': np.sum(R * R),
            'rms_residual_per_coordinate': np.sqrt(np.mean(R * R, axis=1)),
            'final_symplecticity_error': _symplecticity_error(M),
        }

    if return_diagnostics and return_offset:
        return M, b, info
    elif return_diagnostics:
        return M, info
    elif return_offset:
        return M, b
    else:
        return M




    

'''
def get_transfer_matrix(gpt_data, screen_location):
     # final_coords_pred = T @ initial_coords
    cathode = get_screen_data(gpt_data, screen_z = 0)[0]
    final_screen = get_screen_data(gpt_data, screen_z = screen_location)[0]
    idi = np.argsort(cathode.id)
    idf = np.argsort(final_screen.id)
    initial_coords = np.vstack((cathode.x[idi], cathode.px[idi], cathode.y[idi], cathode.py[idi]))
    final_coords = np.vstack((final_screen.x[idf], final_screen.px[idf], final_screen.y[idf], final_screen.py[idf]))
    return np.linalg.lstsq(initial_coords.T, final_coords.T, rcond=None)[0].T
'''

def ensure_empty_dir(path):
    path = Path(path)

    if not path.exists():
        path.mkdir(parents=True)
    elif not path.is_dir():
        raise RuntimeError(f"{path} exists but is not a directory")
    elif any(path.iterdir()):
        path_str = str(path)
        if path_str.startswith("/tmp/"):
            for item in path.iterdir():
                if item.is_dir() and not item.is_symlink():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        else:
            raise RuntimeError(f"{path} already exists and is not empty")

    return path




def run_one_point(mx, my, settings, seed):
    # Unpack only what we need
    s = settings['s']
    noise_settings = settings['noise_settings']
    scan_x_var = settings['scan_x_var']
    scan_y_var = settings['scan_y_var']

    GPT_INPUT_FILE = settings['GPT_INPUT_FILE']
    DISTGEN_INPUT_FILE = settings['DISTGEN_INPUT_FILE']

    z_screen = settings['z_screen']
    total_charge = settings['total_charge']
    xedges = settings['xedges']
    yedges = settings['yedges']
    viewscreen_nxny = settings['viewscreen_nxny']

    calib_data = settings['calib_data']
    scan_x_offset = settings['scan_x_offset']
    scan_y_offset = settings['scan_y_offset']
    aper_x_cal = settings['aper_x_cal']
    aper_y_cal = settings['aper_y_cal']

    root_dir = settings['root_dir']
    sub_dir_laser = settings['sub_dir_laser']

    # Make per-point settings
    s_ii = copy.copy(s)
    s_ii[scan_x_var] = mx
    s_ii[scan_y_var] = my
    
    # Noise
    rng = np.random.default_rng(seed)
    for ns in noise_settings.keys():
        s_ii[ns] = s_ii[ns] + rng.normal(0.0, noise_settings[ns])
    
    # Run GPT
    g_ii = run_gpt_with_settings(
        s_ii,
        gpt_input_file=GPT_INPUT_FILE,
        distgen_input_file=DISTGEN_INPUT_FILE,
        verbose=False,
        gpt_verbose=False,
        auto_phase=False,
        timeout=1000
    )

    scr_ii = get_screen_data(g_ii, screen_z=z_screen)[0]
    w_ii = scr_ii.weight / total_charge

    if np.abs(scr_ii['mean_z'] - z_screen) < 1.0e-6:
        viewscr, _, _, _ = binned_statistic_2d(
            scr_ii.x,
            scr_ii.y,
            w_ii,
            statistic='sum',
            bins=(xedges, yedges)
        )
    else:
        print('WTTTFFFFFFFFF')
        print(g_ii.stat('mean_z'))
        viewscr = np.zeros(viewscreen_nxny)

    calib_data_ii = copy.copy(calib_data)
    calib_data_ii['magnet_values'] = [mx, my]

    file_x = (mx - scan_x_offset) * aper_x_cal * 1e6
    file_y = (my - scan_y_offset) * aper_y_cal * 1e6
    filename = f'X@{file_x:.2f}@Y@{file_y:.2f}@N@0001@.mat'

    outpath = Path(root_dir) / sub_dir_laser / filename

    savemat(outpath, {
        'calib_data': calib_data_ii,
        'plot_data': viewscr.T
    })




def analyze_aperture_scan_4d(image_directory, show_roi_check=True):
    
    image_directory = Path(image_directory)

    if not image_directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {image_directory}")

    # Laser position
    laser_position = np.array([np.nan, np.nan], dtype=float)
    laser_info_file = image_directory / "laser_info.txt"
    if laser_info_file.is_file():
        laser_position = np.loadtxt(laser_info_file)
    else:
        laser_position = np.nan
        print("Could not find laser position file.")

    # File list
    file_list = sorted(image_directory.glob("*.mat"))
    if len(file_list) == 0:
        raise FileNotFoundError(f"No .mat files found in {image_directory}")

    # Load example image / calib
    example_image = load_mat_file(file_list[0])
    example_plot_data = np.array(example_image["plot_data"]).T
    calib_data = example_image["calib_data"]

    gun_voltage = float(calib_data["gun_voltage"])
    camera_roi_offsets = np.array(calib_data["camera_roi_offsets"], dtype=float)
    viewscreen_scale = np.array(calib_data["viewscreen_scale"], dtype=float)
    aperture_screen_separation = np.array(calib_data["aperture_screen_separation"], dtype=float)
    aperture_scale = np.array(calib_data["aperture_scale"], dtype=float)
    screen_scale = np.array(calib_data["screen_scale"], dtype=float)
    magnet_center_values = np.array(calib_data["magnet_center_values"], dtype=float)
    magnet_x_values = np.array(calib_data["magnet_x_values"], dtype=float)
    magnet_y_values = np.array(calib_data["magnet_y_values"], dtype=float)

    magnet_aperture_separation = (
        aperture_scale / (screen_scale - aperture_scale) * aperture_screen_separation
    )

    g = 1.0 + gun_voltage / 510.99895
    gb = np.sqrt(g**2 - 1.0)

    x = np.sort(-magnet_x_values * aperture_scale[0])
    y = np.sort(-magnet_y_values * aperture_scale[1])

    # phasespace_size = [len(x), len(y), n_px, n_py]
    phasespace_size = (len(x), len(y), *example_plot_data.shape)

    # Until the end of the code, px is an angle not a momentum
    px = (
        (np.arange(phasespace_size[2]) + camera_roi_offsets[0])
        * viewscreen_scale[0]
        / aperture_screen_separation
    )
    py = (
        (np.arange(phasespace_size[3]) + camera_roi_offsets[1])
        * viewscreen_scale[1]
        / aperture_screen_separation
    )

    px_correction = aperture_scale[0] * magnet_center_values[0] / magnet_aperture_separation[0]
    py_correction = aperture_scale[1] * magnet_center_values[1] / magnet_aperture_separation[1]

    px = px - px_correction
    py = py - py_correction

    data = np.zeros(phasespace_size, dtype=float)

    max_pixel_im = np.zeros_like(example_plot_data, dtype=float)

    for ii, fname in enumerate(file_list):
        im_data = load_mat_file(fname)
        calib_data = im_data["calib_data"]
        im = np.array(im_data["plot_data"]).T.astype(float)

        calib_magnet_values = np.array(calib_data["magnet_values"], dtype=float)

        im_x = -calib_magnet_values[0] * aperture_scale[0]
        im_y = -calib_magnet_values[1] * aperture_scale[1]

        scr_x = (calib_magnet_values[0] - magnet_center_values[0]) * (
            screen_scale[0] - aperture_scale[0]
        )
        scr_y = (calib_magnet_values[1] - magnet_center_values[1]) * (
            screen_scale[1] - aperture_scale[1]
        )

        im_x_index = np.argmin(np.abs(im_x - x))
        im_y_index = np.argmin(np.abs(im_y - y))

        # Shift pixels to fix kick from corrector
        px_correction_pixels = scr_x / viewscreen_scale[0]
        py_correction_pixels = scr_y / viewscreen_scale[1]

        # Matlab imtranslate(im, [-py_correction_pixels, -px_correction_pixels])
        # For scipy.ndimage.shift, shift is (rows, cols)
        im = shift(
            im,
            shift=(-px_correction_pixels, -py_correction_pixels),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=True,
        )

        data[im_x_index, im_y_index, :, :] = im

        big_stuff = max_pixel_im < im
        max_pixel_im[big_stuff] = im[big_stuff]

    if (show_roi_check):
        plt.figure()
        plt.clf()
        plt.imshow(max_pixel_im.T, origin="lower", aspect="equal")
        plt.title(f"ROI check")

    output_data = {
        "data": data,
        "g": g,
        "gb": gb,
        "x": x,
        "y": y,
        "px": px * gb,  # change to momentum
        "py": py * gb,
        "laser_position": laser_position,
    }

    return output_data



def project_phase_space(input_data, var1="x", var2="y", colormap="viridis"):
    show_absolute_coords = False

    spatial_plot_limits = "auto"   # micron
    momentum_plot_limits = "auto"  # mrad

    data = input_data["data"]
    gb = input_data["gb"]
    x = np.asarray(input_data["x"])
    y = np.asarray(input_data["y"])
    px = np.asarray(input_data["px"]) * 1e3  # mrad
    py = np.asarray(input_data["py"]) * 1e3  # mrad

    coords = [x, y, px, py]

    vars_ = ["x", "y", "px", "py"]
    units = [r"$\mu$m", r"$\mu$m", "mrad", "mrad"]
    emit_units = "nm"

    try:
        var1_ind = [s.lower() for s in vars_].index(var1.lower())
        var2_ind = [s.lower() for s in vars_].index(var2.lower())
    except ValueError:
        raise ValueError(f"var1 and var2 must be in {vars_}")

    not_var = [i for i in range(4) if i not in (var1_ind, var2_ind)]

    u = coords[var1_ind]
    v = coords[var2_ind]

    # Sum over the other two dimensions.
    # Do larger axis first so axis numbering stays valid.
    proj_data = data
    for ax in sorted(not_var, reverse=True):
        proj_data = np.sum(proj_data, axis=ax)
    proj_data = np.squeeze(proj_data)

    emit, sig_u, sig_v, avg_u, avg_v, avg_uv = get_emit(proj_data, u, v)

    show_emit = False
    if ("p" in vars_[var2_ind]) and ("p" not in vars_[var1_ind]):
        show_emit = True

    equal_axes = False
    if (("p" in vars_[var2_ind]) and ("p" in vars_[var1_ind])) or (
        ("p" not in vars_[var2_ind]) and ("p" not in vars_[var1_ind])
    ):
        equal_axes = True

    if not show_absolute_coords:
        avg_u_plot = avg_u
        avg_v_plot = avg_v
    else:
        avg_u_plot = 0.0
        avg_v_plot = 0.0

    fig, ax = plt.subplots()
    
    # Use extent so axes correspond to u and v, similar to Matlab imagesc(u,v,data.')
    du = abs(u[1] - u[0]) if len(u) > 1 else 1.0
    dv = abs(v[1] - v[0]) if len(v) > 1 else 1.0

    extent = [
        (u[0] - avg_u_plot) - 0.5 * du,
        (u[-1] - avg_u_plot) + 0.5 * du,
        (v[0] - avg_v_plot) - 0.5 * dv,
        (v[-1] - avg_v_plot) + 0.5 * dv,
    ]

    im = ax.imshow(
        proj_data.T,
        origin="lower",
        aspect="equal" if equal_axes else "auto",
        extent=extent,
        cmap=colormap,
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax)

    ax.set_xlabel(f"{vars_[var1_ind]} ({units[var1_ind]})")
    ax.set_ylabel(f"{vars_[var2_ind]} ({units[var2_ind]})")

    title_1 = (
        f"$\\sigma_{{{vars_[var1_ind]}}}$ = {sig_u:.3g} {units[var1_ind]}, "
        f"$\\sigma_{{{vars_[var2_ind]}}}$ = {sig_v:.3g} {units[var2_ind]}"
    )
    title_2 = f"$\\epsilon_{{{vars_[var1_ind]}}}$ = {emit:.3g} {emit_units}"
    title_3 = (
        f"<{vars_[var1_ind]}> = {avg_u:.5g} {units[var1_ind]}, "
        f"<{vars_[var2_ind]}> = {avg_v:.5g} {units[var2_ind]}, "
        f"<{vars_[var1_ind]}{vars_[var2_ind]}> = {avg_uv:.5g}"
    )

    if show_emit:
        ax.set_title(title_1 + ", " + title_2 + "\n" + title_3)
    else:
        ax.set_title(title_1 + "\n" + title_3)

    # Auto plot limits based on nonzero support
    u_data = np.abs(np.sum(proj_data, axis=1))
    nonzero_u = np.where(u_data > 0.0)[0]
    if len(nonzero_u) > 0:
        xmin_ind = nonzero_u[0]
        xmax_ind = nonzero_u[-1]
        ax.set_xlim(
            np.sort(
                [
                    u[xmin_ind] - avg_u_plot - 0.5 * du,
                    u[xmax_ind] - avg_u_plot + 0.5 * du,
                ]
            )
        )

    v_data = np.abs(np.sum(proj_data, axis=0))
    nonzero_v = np.where(v_data > 0.0)[0]
    if len(nonzero_v) > 0:
        ymin_ind = nonzero_v[0]
        ymax_ind = nonzero_v[-1]
        ax.set_ylim(
            np.sort(
                [
                    v[ymin_ind] - avg_v_plot - 0.5 * dv,
                    v[ymax_ind] - avg_v_plot + 0.5 * dv,
                ]
            )
        )

    if isinstance(spatial_plot_limits, (list, tuple, np.ndarray)) and len(spatial_plot_limits) == 2:
        ax.set_xlim(spatial_plot_limits)

    if isinstance(momentum_plot_limits, (list, tuple, np.ndarray)) and len(momentum_plot_limits) == 2:
        ax.set_ylim(momentum_plot_limits)

    # Get average coordinates in x-y
    u = coords[0]
    v = coords[1]
    proj_xy = data
    for axsum in sorted([2, 3], reverse=True):
        proj_xy = np.sum(proj_xy, axis=axsum)
    proj_xy = np.squeeze(proj_xy)
    _, _, _, avg_x, avg_y, _ = get_emit(proj_xy, u, v)

    # Get average coordinates in px-py
    u = coords[2]
    v = coords[3]
    proj_pp = data
    for axsum in sorted([0, 1], reverse=True):
        proj_pp = np.sum(proj_pp, axis=axsum)
    proj_pp = np.squeeze(proj_pp)
    _, _, _, avg_px, avg_py, _ = get_emit(proj_pp, u, v)

    avg_coords = np.array([avg_x, avg_px, avg_y, avg_py])

    plt.tight_layout()
    plt.show()

    #return avg_coords

def get_emit(data, y, py):
    y = np.asarray(y)
    py = np.asarray(py)
    data = np.asarray(data, dtype=float)

    total = np.sum(data)
    if total == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    data = data / total

    PY, Y = np.meshgrid(py, y)

    y_avg = np.sum(Y * data)
    py_avg = np.sum(PY * data)

    yy = np.sum((Y - y_avg) ** 2 * data)
    pypy = np.sum((PY - py_avg) ** 2 * data)
    ypy = np.sum((Y - y_avg) * (PY - py_avg) * data)

    sig_y = np.sqrt(yy)
    sig_py = np.sqrt(pypy)
    ypy_avg = ypy

    emit = np.sqrt(max(yy * pypy - ypy**2, 0.0))

    return emit, sig_y, sig_py, y_avg, py_avg, ypy_avg



# --------------------------------------------------------------------------------------------
# Calculating MTE / transfer matrix from many aperture scans
# --------------------------------------------------------------------------------------------


def process_image_directories(image_directory):
    temp_directory = Path(image_directory)
    if (not temp_directory.is_dir()):
        raise ValueError(f'Could not find directory: {temp_directory}')
    summary_dir = temp_directory / "summary"
    if (not summary_dir.is_dir()):
        summary_dir.mkdir(exist_ok=True)
        os.chmod(summary_dir, 0o777)
        print(f'Making directory for analysis files: {str(summary_dir)}')
    
    for folder in temp_directory.iterdir():
        if folder.is_dir() and folder.name not in (".", "..", "summary"):
            output_file = summary_dir / f"{folder.name}.mat"
    
            if output_file.exists():
                continue

            print(f'Analyzing files in {str(folder)}...')
            data = analyze_aperture_scan_4d(str(folder))

            [dd0, S, eps] = get_sigma(data['data'], data['x'], data['y'], data['px'], data['py'])
            data["sigma_matrix"] = S
            data["phase_space_centroid"] = dd0
            savemat(output_file, {"data": data})
            os.chmod(output_file, 0o666)
    
    summary_dir = temp_directory / "summary"
    return summary_dir


def get_sigma(data, x, y, px, py):
    data = np.asarray(data, dtype=float)
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    px = np.asarray(px).reshape(-1)
    py = np.asarray(py).reshape(-1)

    Q = np.sum(data)
    if Q == 0:
        X0 = np.full(4, np.nan)
        S = np.full((4, 4), np.nan)
        return X0, S, np.nan

    data = data / Q

    X = [x.copy(), y.copy(), px.copy(), py.copy()]
    X0 = np.full(4, np.nan)

    for ii in range(4):
        inds = [0, 1, 2, 3]
        inds.remove(ii)
        d = np.sum(data, axis=tuple(inds))
        d = np.asarray(d).reshape(-1)
        X0[ii] = np.sum(d * X[ii])
        X[ii] = X[ii] - X0[ii]

    S = np.full((4, 4), np.nan)

    for ii in range(4):
        for jj in range(ii, 4):
            inds = [0, 1, 2, 3]
            if ii == jj:
                inds.remove(ii)
                d = np.sum(data, axis=tuple(inds))
                d = np.asarray(d).reshape(-1)
                val = np.sum(d * X[ii] * X[ii])
            else:
                inds.remove(max(ii, jj))
                inds.remove(min(ii, jj))
                d = np.sum(data, axis=tuple(inds))
                B, A = np.meshgrid(X[jj], X[ii], indexing="xy")
                val = np.sum(d * B * A)
            S[ii, jj] = val

    for ii in range(4):
        for jj in range(ii):
            S[ii, jj] = S[jj, ii]

    # Matlab reorder:
    # S = S([1,3,2,4],[1,3,2,4]); X0 = X0([1,3,2,4]);
    perm = [0, 2, 1, 3]
    S = S[np.ix_(perm, perm)]
    X0 = X0[perm]

    detS = np.linalg.det(S)
    eps = np.sqrt(np.sqrt(max(detS, 0.0)))

    return X0, S, eps



# ----------------------------------------------------------------------
# Basic loading helpers
# ----------------------------------------------------------------------

def _mat_struct_to_dict(obj):
    if hasattr(obj, "_fieldnames"):
        return {field: _mat_struct_to_dict(getattr(obj, field)) for field in obj._fieldnames}
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            if obj.size == 1:
                return _mat_struct_to_dict(obj.item())
            return np.array([_mat_struct_to_dict(x) for x in obj.flat], dtype=object).reshape(obj.shape)
        obj2 = np.squeeze(obj)
        if obj2.shape == ():
            return obj2.item()
        return obj2
    return obj


def load_mat_file(filename):
    raw = loadmat(filename, struct_as_record=False, squeeze_me=True)
    out = {}
    for k, v in raw.items():
        if not k.startswith("__"):
            out[k] = _mat_struct_to_dict(v)
    return out


# ----------------------------------------------------------------------
# Plot helpers / simple replacements
# ----------------------------------------------------------------------

def density_histogram(x, y, w, x_edges, y_edges):
    H, xe, ye = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=w)
    x_plot = 0.5 * (xe[:-1] + xe[1:])
    y_plot = 0.5 * (ye[:-1] + ye[1:])
    return x_plot, y_plot, H.T


def blur_aperturescan_data(data, viewscreen_res):
    """
    Placeholder. Matlab code references this, but definition was not included.
    For now, return unchanged.
    """
    return data

def load_summary_data(filename):
    """
    Load one of the saved summary .mat files and return its contained data struct.
    Matlab version did:
        data = load(data_name);
        d = fieldnames(data);
        data = data.(d{1});
    """
    raw = load_mat_file(filename)
    if len(raw) == 0:
        raise ValueError(f"No data variables found in {filename}")
    first_key = list(raw.keys())[0]
    return raw[first_key], first_key

# ----------------------------------------------------------------------
# Analysis / centroid loading
# ----------------------------------------------------------------------

def load_and_analyze_data_file(filename, viewscreen_resolution):

    data_input, data_field_name =  load_summary_data(filename)
    
    laser = np.asarray(data_input["laser_position"])
    dd0 = np.asarray(data_input["phase_space_centroid"])
    S_calc = np.asarray(data_input["sigma_matrix"])
    eps = np.sqrt(np.sqrt(max(np.linalg.det(S_calc), 0.0)))
    return dd0, S_calc, eps, laser


# ----------------------------------------------------------------------
# Symplectic fit for columns 1 and 3
# ----------------------------------------------------------------------

def objective_centered(m, Xc, Rc):
    M = np.reshape(m, (4, 2), order="F")
    E = M @ Xc - Rc
    return np.sum(E**2)


def symplectic_col_constraint_value(m, J):
    M = np.reshape(m, (4, 2), order="F")
    c1 = M[:, 0]
    c3 = M[:, 1]
    return c1.T @ J @ c3


# ----------------------------------------------------------------------
# Complete transfer matrix
# ----------------------------------------------------------------------

def choose_pair_scale(vpos, vmom, eps_floor, label):
    if not (np.isfinite(vpos) and np.isfinite(vmom)):
        warnings.warn(f"Non-finite covariance entries for {label}; using unit scale.")
        return 1.0
    if vpos <= 0 or vmom <= 0:
        warnings.warn(f"Non-positive covariance diagonal entries for {label}; using unit scale.")
        return 1.0

    ratio = max(vmom, eps_floor) / max(vpos, eps_floor)
    s = ratio ** 0.25
    if not np.isfinite(s) or s <= 0:
        warnings.warn(f"Could not determine stable scale for {label}; using unit scale.")
        return 1.0
    return s


def choose_symplectic_scaling(S):
    eps_floor = 1e-30
    sx = choose_pair_scale(S[0, 0], S[1, 1], eps_floor, "x/px")
    sy = choose_pair_scale(S[2, 2], S[3, 3], eps_floor, "y/py")
    return np.diag([sx, 1.0 / sx, sy, 1.0 / sy])


def covariance_residuals(c2, c4, B):
    return np.concatenate([B @ c2, B @ c4])


def objective2(z, c20, c40, N, B, wCov):
    alpha = z[0:2]
    beta = z[2:4]
    c2 = c20 + N @ alpha
    c4 = c40 + N @ beta
    r = covariance_residuals(c2, c4, B)
    return np.sum((wCov * r) ** 2)


def nonlinear_constraint_scalar(z, c20, c40, N, Omega):
    alpha = z[0:2]
    beta = z[2:4]
    c2 = c20 + N @ alpha
    c4 = c40 + N @ beta
    return c2.T @ Omega @ c4


def solve_in_scaled_coordinates(M, S, Omega, opts=None):
    if opts is None:
        opts = {}

    c1 = M[:, 0]
    c3 = M[:, 2]

    A = np.vstack([c1.T @ Omega, c3.T @ Omega])

    if np.linalg.matrix_rank(A) < 2:
        raise ValueError("The measured columns c1 and c3 do not provide a full-rank linear symplectic system.")

    known_known = c1.T @ Omega @ c3
    if abs(known_known) > 1e-5:
        warnings.warn(f"Known columns c1 and c3 are not symplectically compatible: {known_known:.3e}")

    c20 = np.linalg.lstsq(A, np.array([1.0, 0.0]), rcond=None)[0]
    c40 = np.linalg.lstsq(A, np.array([0.0, 1.0]), rcond=None)[0]

    U, s, Vh = np.linalg.svd(A)
    rank = np.sum(s > 1e-12)
    N = Vh[rank:].T
    if N.shape[1] != 2:
        raise ValueError("Unexpected nullspace dimension; expected 2.")

    r2 = np.array([-c1[1], c1[0], -c1[3], c1[2]])
    r4 = np.array([-c3[1], c3[0], -c3[3], c3[2]])
    B = np.vstack([r2 @ S @ Omega, r4 @ S @ Omega])

    Ahat = np.vstack([A, B])
    c2_lin = np.linalg.lstsq(Ahat, np.array([1.0, 0.0, 0.0, 0.0]), rcond=None)[0]
    c4_lin = np.linalg.lstsq(Ahat, np.array([0.0, 1.0, 0.0, 0.0]), rcond=None)[0]

    alpha0 = N.T @ (c2_lin - c20)
    beta0 = N.T @ (c4_lin - c40)
    z0 = np.concatenate([alpha0, beta0])

    wCov = np.asarray(opts.get("wCov", np.ones(4)))

    cons = {
        "type": "eq",
        "fun": lambda z: nonlinear_constraint_scalar(z, c20, c40, N, Omega),
    }

    res = minimize(
        fun=lambda z: objective2(z, c20, c40, N, B, wCov),
        x0=z0,
        method="SLSQP",
        constraints=[cons],
        options={
            "maxiter": int(opts.get("maxiter", 500)),
            "ftol": opts.get("ftol", 1e-12),
            "disp": opts.get("disp", False),
        },
    )

    z_opt = res.x
    alpha = z_opt[0:2]
    beta = z_opt[2:4]

    c2 = c20 + N @ alpha
    c4 = c40 + N @ beta

    Mfull = M.copy()
    Mfull[:, 1] = c2
    Mfull[:, 3] = c4

    Rcol = Mfull.T @ Omega @ Mfull - Omega
    Rrow = Mfull @ Omega @ Mfull.T - Omega

    info = {
        "c2_linear_init": c2_lin,
        "c4_linear_init": c4_lin,
        "z0": z0,
        "z_opt": z_opt,
        "fval": res.fun,
        "exitflag": res.status,
        "output": res,
        "covResidual": covariance_residuals(c2, c4, B),
        "A": A,
        "B": B,
        "N": N,
        "c20": c20,
        "c40": c40,
        "known_known": known_known,
        "Rcol_scaled": Rcol,
        "Rrow_scaled": Rrow,
        "RcolFro_scaled": np.linalg.norm(Rcol, ord="fro"),
        "RrowFro_scaled": np.linalg.norm(Rrow, ord="fro"),
        "cond_scaled": np.linalg.cond(Mfull),
        "remainingBilinearConstraint": c2.T @ Omega @ c4,
    }
    return Mfull, info


def complete_transfer_matrix(M, S, opts=None):
    if opts is None:
        opts = {}

    M = np.asarray(M, dtype=float)
    S = np.asarray(S, dtype=float)
    S = 0.5 * (S + S.T)

    if M.shape != (4, 4):
        raise ValueError("M must be 4x4")
    if S.shape != (4, 4):
        raise ValueError("S must be 4x4")

    if np.any(np.isnan(M[:, 0])) or np.any(np.isnan(M[:, 2])):
        raise ValueError("Columns 1 and 3 of M must be known.")
    if not np.all(np.isnan(M[:, 1])) or not np.all(np.isnan(M[:, 3])):
        raise ValueError("Columns 2 and 4 of M should be NaN.")

    Omega = np.array([
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, -1, 0]
    ], dtype=float)

    if "D" in opts and opts["D"] is not None:
        D = np.asarray(opts["D"], dtype=float)
    elif opts.get("autoScale", True):
        D = choose_symplectic_scaling(S)
    else:
        D = np.eye(4)

    Dinv = np.linalg.inv(D)

    Mscaled = np.full((4, 4), np.nan)
    Mscaled[:, 0] = D @ M[:, 0] / D[0, 0]
    Mscaled[:, 2] = D @ M[:, 2] / D[2, 2]

    Sscaled = D @ S @ D.T

    Mscaled_full, scaled_info = solve_in_scaled_coordinates(Mscaled, Sscaled, Omega, opts)
    Mfull = Dinv @ Mscaled_full @ D

    Mfull[:, 0] = M[:, 0]
    Mfull[:, 2] = M[:, 2]

    Rcol = Mfull.T @ Omega @ Mfull - Omega
    Rrow = Mfull @ Omega @ Mfull.T - Omega

    info = dict(scaled_info)
    info.update({
        "D": D,
        "Dinv": Dinv,
        "Mscaled_full": Mscaled_full,
        "Rcol_original": Rcol,
        "Rrow_original": Rrow,
        "RcolFro_original": np.linalg.norm(Rcol, ord="fro"),
        "RrowFro_original": np.linalg.norm(Rrow, ord="fro"),
        "cond_original": np.linalg.cond(Mfull),
        "known_known_original": Mfull[:, 0].T @ Omega @ Mfull[:, 2],
    })
    return Mfull, info


def get_MTE(filename, M, S_input, viewscreen_resolution=0, colormap='viridis', show_plots=True):
    dx = M[:, 0]
    dy = M[:, 2]

    data, _ = load_summary_data(filename)

    x = data['x']
    y = data['y']
    px = data['px']
    py = data['py']

    if (show_plots):
        project_phase_space(data, "x", "px", colormap=colormap)

    gb = data["gb"]
    arr = data["data"]
    _, S = get_sigma(arr, x, y, px, py)[:2]
    emit = np.linalg.det(S) ** 0.25

    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    px = np.asarray(px).reshape(-1)
    py = np.asarray(py).reshape(-1)

    m = 510998950.0
    mc = 510998950.0
    mc2 = 510998950.0

    dx_units = np.array([1.0, mc, 1.0, mc])
    px_evc = px * mc
    py_evc = py * mc

    omega = np.array([
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, -1, 0]
    ], dtype=float)

    Q = np.transpose(arr, (0, 2, 1, 3))
    X, PX, Y, PY = np.meshgrid(x, px_evc, y, py_evc, indexing="ij")

    X = X.ravel()
    Y = Y.ravel()
    PX = PX.ravel()
    PY = PY.ravel()
    Q = Q.ravel()

    R = np.vstack([X, PX, Y, PY])

    px_cath = ((dx_units * dx).T @ omega @ R)
    py_cath = ((dx_units * dy).T @ omega @ R)

    R0 = (np.diag(dx_units) @ (-omega @ M.T @ omega) @ np.diag(1.0 / dx_units)) @ R
    x_cath = R0[0, :]
    y_cath = R0[2, :]

    nz = Q > 0
    Q = Q[nz]
    Q = Q / np.sum(Q)

    px_cath = px_cath[nz]
    py_cath = py_cath[nz]
    x_cath = x_cath[nz]
    y_cath = y_cath[nz]

    avg_px = np.sum(px_cath * Q)
    avg_py = np.sum(py_cath * Q)
    avg_x = np.sum(x_cath * Q)
    avg_y = np.sum(y_cath * Q)

    px_cath = px_cath - avg_px
    py_cath = py_cath - avg_py
    x_cath = x_cath - avg_x
    y_cath = y_cath - avg_y

    sigma_px = np.sqrt(np.sum(px_cath**2 * Q))
    sigma_py = np.sqrt(np.sum(py_cath**2 * Q))
    sigma_x = np.sqrt(np.sum(x_cath**2 * Q))
    sigma_y = np.sqrt(np.sum(y_cath**2 * Q))

    avg_pxpy = np.sum(px_cath * py_cath * Q)
    sigma_p_effective = np.sqrt(np.sqrt(sigma_px**2 * sigma_py**2 - avg_pxpy**2))

    max_sigma_p = max(sigma_px, sigma_py)
    n_points_in_histogram_plot = 30
    n_sigma_plot = 3
    p_edges = np.linspace(-n_sigma_plot * max_sigma_p, n_sigma_plot * max_sigma_p, n_points_in_histogram_plot)

    if (show_plots):
        plt.figure()
        plt.clf()
        x_plot, y_plot, H = density_histogram(px_cath, py_cath, Q, p_edges, p_edges)
        plt.imshow(
            H,
            origin="lower",
            aspect="equal",
            extent=[1e-3 * x_plot[0], 1e-3 * x_plot[-1], 1e-3 * y_plot[0], 1e-3 * y_plot[-1]],
            interpolation="nearest",
            cmap=colormap,
        )
        plt.colorbar()
        plt.xlabel("Cathode p_x (eV/c)")
        plt.ylabel("Cathode p_y (eV/c)")

    #print(f"sigma_px = {1e-3 * sigma_px:.3g} eV/c")
    #print(f"sigma_py = {1e-3 * sigma_py:.3g} eV/c")

    MTEx = sigma_px**2 / m
    MTEy = sigma_py**2 / m
    MTEeff = sigma_p_effective**2 / m
    sigx_cath = emit / np.sqrt(MTEeff / mc2)

    if (show_plots):
        plt.title(f"MTE_x = {MTEx:.3g} meV, MTE_y = {MTEy:.3g} meV\nMTE = {MTEeff:.3g} meV")
        plt.show()

    max_sigma_x = max(sigma_x, sigma_y)
    x_edges = np.linspace(-n_sigma_plot * max_sigma_x, n_sigma_plot * max_sigma_x, n_points_in_histogram_plot)

    if (show_plots):
        plt.figure()
        plt.clf()
        x_plot, y_plot, H = density_histogram(x_cath, y_cath, Q, x_edges, x_edges)
        plt.imshow(
            H,
            origin="lower",
            aspect="equal",
            extent=[x_plot[0], x_plot[-1], y_plot[0], y_plot[-1]],
            interpolation="nearest",
            cmap=colormap,
        )
        plt.colorbar()
        plt.xlabel("Cathode x ($\\mu$m)")
        plt.ylabel("Cathode y ($\\mu$m)")
        plt.title(f"$\\sigma_x$ = {sigma_x:.3g} um, $\\sigma_y$ = {sigma_y:.3g} um")
        plt.show()

    return MTEx, MTEy, MTEeff, sigx_cath, emit


# ----------------------------------------------------------------------
# Drift helper
# ----------------------------------------------------------------------

def get_drift(d):
    return np.array([
        [1, d, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, d],
        [0, 0, 0, 1]
    ], dtype=float)


# ----------------------------------------------------------------------
# Main reconstruction function
# ----------------------------------------------------------------------

def calc_reconstructions(root_directory, enforce_symplectic = True, viewscreen_resolution = 0.0, colormap='viridis', show_plots=True):
   
    directory = process_image_directories(root_directory)
    
    directory = Path(directory)
    file_list = sorted(directory.glob("*.mat"))
    name_list = [f.name for f in file_list]

    R = np.full((4, len(name_list)), np.nan)
    sigma_list = [None] * len(name_list)
    X = np.full((2, len(name_list)), np.nan)

    for ii, name in enumerate(name_list):
        r, s, eps, laser = load_and_analyze_data_file(directory / name,viewscreen_resolution)
        R[:, ii] = r
        sigma_list[ii] = s
        X[:, ii] = laser

    N = len(name_list)
    w = np.ones((N, 1))

    R = R - R[:, [0]]
    X = X - X[:, [0]]

    if not enforce_symplectic:
        M = (
            (R @ X.T - (1.0 / N) * (R @ w) @ (X @ w).T)
            @ np.linalg.inv(X @ X.T - (1.0 / N) * (X @ w) @ (X @ w).T)
        )
        r0 = (1.0 / N) * (R - M @ X) @ w
        r0 = r0.reshape(-1)
    else:
        J = np.array([
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, -1, 0]
        ], dtype=float)

        xbar = (1.0 / N) * (X @ w)
        rbar = (1.0 / N) * (R @ w)

        Xc = X - xbar @ w.T
        Rc = R - rbar @ w.T

        M0 = (
            (R @ X.T - (1.0 / N) * (R @ w) @ (X @ w).T)
            @ np.linalg.inv(X @ X.T - (1.0 / N) * (X @ w) @ (X @ w).T)
        )
        m0 = M0.flatten(order="F")

        cons = {
            "type": "eq",
            "fun": lambda m: symplectic_col_constraint_value(m, J),
        }

        res = minimize(
            fun=lambda m: objective_centered(m, Xc, Rc),
            x0=m0,
            method="SLSQP",
            constraints=[cons],
            options={"maxiter": 10000, "ftol": 1e-12, "disp": False},
        )

        m_fit = res.x
        M = np.reshape(m_fit, (4, 2), order="F")
        r0 = (rbar - M @ xbar).reshape(-1)

    Rp = M @ X + r0[:, None]
    dx = M[:, 0]
    dy = M[:, 1]

    MM = np.full((4, 4), np.nan)
    MM[:, 0] = M[:, 0]
    MM[:, 2] = M[:, 1]

    labels = [r"$\Delta x$ (um)", r"$\Delta px$ (rad)", r"$\Delta y$ (um)", r"$\Delta py$ (rad)"]

    if (show_plots):
        for ii in range(4):
            x_plot = np.arange(1, R.shape[1] + 1)
            plt.figure()
            plt.clf()
            plt.plot(x_plot, R[ii, :], "ro")
            plt.plot(x_plot, Rp[ii, :], "bo")
            plt.xlabel("Data set")
            plt.ylabel(labels[ii])
            plt.xlim([0.5, R.shape[1] + 0.5])
            plt.legend(["Measured", "Best fit"])
            plt.draw()
            plt.show()

    data_0_name = name_list[0]
    s0 = sigma_list[0]
    s1 = s0

    omega = np.array([
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, -1, 0]
    ], dtype=float)

    Sp = omega @ s1 @ omega.T
    dxa = dx.copy()
    dya = dy.copy()

    MM, test_info = complete_transfer_matrix(MM, s1)

    MTEx, MTEy, MTEeff, sigx_cath, emit = get_MTE(directory / data_0_name, MM, s0, viewscreen_resolution, colormap, show_plots=show_plots)

    MM = change_matrix_units(MM) # Add mc to momentum
    
    return MM, MTEeff, sigx_cath, emit