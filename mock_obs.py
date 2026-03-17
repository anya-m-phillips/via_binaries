##### THIS WILL BE RUn On CANNON! ! ! 

import sys
sys.path.append('/Users/anyaphillips/Desktop/harvard/research/via_binaries/scripts')

import PETAR_ANALYSIS_FUNCTIONS as paf
import petar
import numpy as np

from astropy.table import Table
from scipy.stats import binned_statistic_2d

from tqdm import tqdm

import pandas as pd

import astropy.constants as const
import astropy.units as u

### load binaries
binaries = pd.read_csv('cosmic_example_IBC.csv')
N = len(binaries)
print(N)


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

# get binary parameters
# do nominal mock observations
### stack orbital params. want v0 (km/s), K (km/s), w, phi0, e, P (day)
rng = np.random.default_rng(seed=42)

m1 = binaries['mass_1'].values * u.Msun
m2 = binaries['mass_2'].values * u.Msun
mtot = m1+m2
P = binaries['porb'].values * u.day # days
a = calc_a(P, mtot)

e = binaries['ecc'].values
i = paf.draw_inclinations(N, rng=rng)
K = calc_K(mtot, m2, a, e, P, i)
v0 = np.zeros(N)*(u.km/u.s)

w = rng.uniform(low=0, high=2*np.pi, size=N)
phi0 = rng.uniform(low=0, high=1, size=N)

params = np.array([
    v0.to(u.km/u.s).value,
    K.to(u.km/u.s).value, 
    w,
    phi0,
    e,
    P.to(u.day).value
]).T

def get_obstime(N, DT1, DT2):
    """
    the exact observing cadence version of the above. 
    i.e., observations will be separated by exactly DT1 and DT2.
    """
    base = np.repeat(0, N)
    gap1 = np.repeat(DT1, N)
    gap2 = np.repeat(DT2, N)

    deltaTs = np.vstack((base, gap1, gap2)).T
    obstimes_all = np.cumsum(deltaTs, axis=1) ##### THIS IS WHAT WILL GO INTO THE RV GENERATION FUNCTION!   
    return obstimes_all



#------------------------------------------------#
#           MAIN PROGRAM                         #
#------------------------------------------------#


### RUN THE SAMPLER
rng = np.random.default_rng(seed=42)


#### loop through variable first visit lengths. 
dt1_vals  = np.arange(5, 105, 5) # 20 options 
dt = 30
dt2_vals = np.arange(30, 5*365+dt, dt)

detection_fractions = []
for dt1_val in tqdm(dt1_vals):
    detection_fractions_this_dt1 = []
    for dt2_val in dt2_vals:
        obstimes = get_obstime(N=N, DT1=dt1_val, DT2=dt2_val)
        rvs = paf.get_rvs(params, obstimes, verbose=False)
        detected, delta_vsys = paf.get_detections(
                    0.1, # km/s
                    rvs,
                    v0, 
                    bool_arr='detet'
                )
        detection_fraction = len(P[detected])/len(P)
        detection_fractions_this_dt1.append(detection_fraction)
    detection_fractions.append(detection_fractions_this_dt1)

# save a numpy text file
detection_fraction_array = np.array(detection_fractions)
np.savetxt("detection_fractions.txt", detection_fraction_array)
