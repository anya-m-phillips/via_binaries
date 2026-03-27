import petar
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib import gridspec
from astropy.table import Table

import astropy.coordinates as coord
from astropy.coordinates import Galactocentric, ICRS, CartesianRepresentation, CartesianDifferential
from astropy.coordinates import SkyCoord

import astropy.units as u
import numpy as np


from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors

import astropy.constants as const
from streamframe import StreamFrame
from scipy.stats import binned_statistic
import scipy.stats as stats
from scipy.optimize import curve_fit
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap

from scipy.stats import binom
from tqdm import tqdm

from scipy.ndimage import gaussian_filter1d

 
##### FUNCTIONS
def define_time_cmap():
    colors=["#212738","#4D3041","#79394A","#994250","#B24B54","#C55756","#D26A54","#DC8051","#E5974D","#EDAE49"]
    colors.reverse()
    time_cmap = LinearSegmentedColormap.from_list('time_cmap', colors)
    return time_cmap

def define_simcolors():
    lm_colors = ["#2D3047","#246569","#1B998B","#8DCB87"] # increasing diffuseness
    hm_colors = ["#BB333C", "#E84855","#FF9B71","#FFCC7A"]#,"#FFE57E"] # same; densest to most diffuse. 

    simcolors = [lm_colors[0],hm_colors[0],lm_colors[1],hm_colors[1],lm_colors[2],hm_colors[2],lm_colors[3],hm_colors[3]] * 3
    return lm_colors, hm_colors, simcolors


def calc_P(a, Mtot):
    """
    period from semimajor axis, make the inputs astropy unit quantities
    """
    P = 2*np.pi * np.sqrt((a**3)/(const.G * Mtot))
    return P

def calc_K(Mtot, M2, a, e, P, i):
    a1 = (M2/Mtot) * a

    num = 2*np.pi*a1 *np.sin(i)# nominally this would include sin(i)
    denom = P * np.sqrt(1-(e**2))
    K = num/denom # velocity semiamplitude - multiply by 2 for full amplitude.
    return K 

def calc_a(P, Mtot):
    """
    asdf
    """
    a = (P**2 * const.G * Mtot / (4*np.pi**2))**(1/3)
    return a

#--------------------------------------------------------#
#.     Mock observations and detection fractions.        #
#--------------------------------------------------------#

def draw_inclinations(n, rng=None):
    """
    draw n isotropic inclinations
    """
    if rng is None:
        rng = np.random.default_rng()
        
    theta = np.arccos(1-2*rng.random(n))#np.random.rand(n))
    return theta

def solve_kepler(M, e, tol=1e-10, max_iter=100):
    """
    Vectorized Kepler solver using Newton-Raphson with safety.
    """
    M = np.asarray(M)
    e = np.asarray(e)

    # Ensure shapes are compatible
    if M.shape != e.shape:
        e = np.full_like(M, e)

    E = M.copy()  # initial guess

    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        f_prime = 1 - e * np.cos(E)
        delta = f / f_prime

        # Protect against divide-by-zero or overflow
        delta = np.where(np.isfinite(delta), delta, 0.0)
        E_new = E - delta

        # Convergence mask
        if np.all(np.abs(delta) < tol):
            break
        E = E_new

    # Final sanity check
    E = np.where(np.isfinite(E), E, np.nan)
    return E

def true_anomaly(E, e):
    """
    Convert eccentric anomaly E to true anomaly nu safely.
    """
    sqrt_1_plus_e = np.sqrt(np.clip(1 + e, 0, None))
    sqrt_1_minus_e = np.sqrt(np.clip(1 - e, 0, None))

    sin_E2 = np.sin(E / 2)
    cos_E2 = np.cos(E / 2)

    # Replace NaNs with 0 to avoid propagating errors
    sin_E2 = np.nan_to_num(sin_E2)
    cos_E2 = np.nan_to_num(cos_E2)

    return 2 * np.arctan2(
        sqrt_1_plus_e * sin_E2,
        sqrt_1_minus_e * cos_E2
    )

def radial_velocity(t, params):
    """
    t : times at which to compute RV
    v0 : systemic velocity (not a quantity, but in km/s)
    K : RV semiamplitude (not a quantity, but in km/s)
    w : arg of periapsis
    e : eccentricity
    phi_0 : mean anomaly phase offset
    P : period (in same units as t)
    """
    v0, K, w, phi_0, e, P = params


    M = 2 * np.pi * t / P - phi_0
    E = solve_kepler(M % (2 * np.pi), e)
    nu = true_anomaly(E, e)
    vr = v0 + K * (np.cos(nu + w) + e * np.cos(w))
    ### in another version i might add gaussian noise with sigma=.1 km/s to this. 
    ### returns radial velocity in whatever units vr and v0 are in!
    return vr

def get_obstimes(N, rng=None): ### not currently in use
    if rng is None:
        rng = np.random.default_rng()

    base = np.repeat(0, N)
    gap1 = rng.uniform(30,90, size=N)
    gap2 = rng.uniform(3*365, 5*365, size=N)

    deltaTs = np.vstack((base, gap1, gap2)).T

    obstimes_all = np.cumsum(deltaTs, axis=1) ##### THIS IS WHAT WILL GO INTO THE RV GENERATION FUNCTION!   
    return obstimes_all

def get_rvs(params_all, obstimes_all, verbose=True,
            add_noise=False, noise_level=None, rng=None):
    """
    add gaussian noise option; defaults to false
    noise level in km/s
    """

    RVs_all = []
    N=len(params_all[:,0])

    iterator = tqdm(range(N)) if verbose else range(N)
    for j in iterator:
        params = params_all[j]
        obstimes = obstimes_all[j]

        rvs = radial_velocity(obstimes, params)
        if add_noise==True:
            rvs += rng.normal(0, noise_level, size=obstimes.shape)

        RVs_all.append(rvs)


    rvs_all = np.array(RVs_all)
    return rvs_all


def get_detections(e_rv, rvs_all, v0_vals, bool_arr="undet"): #return "undet" of "det"
    """
    returns a bool array; v0_vals should be dimensionless and in km/s
    """
    mu_rvs = np.mean(rvs_all, axis=1) # vsys
    delta_vsys = np.abs(mu_rvs-v0_vals.to(u.km/u.s).value)
    resid_rvs = rvs_all - mu_rvs[:, None]

    chi2_rvs = np.sum((resid_rvs/e_rv)**2, axis=1)
    Nobs=rvs_all.shape[1] ### previously this was always set to 3 -- bug ! 
    P_chi2 = stats.chi2.sf(chi2_rvs, df=Nobs-1)
    zscore = np.max(np.abs(resid_rvs/e_rv), axis = 1) # idk what this is,

    undet = P_chi2>0.1 # detect binaires when the P value of the chi square test is <0.1 
    
    if bool_arr=='undet':
        return undet, delta_vsys
    if bool_arr=='detet':
        return ~undet, delta_vsys
    





