##### THIS WILL BE RUn On CANNON! ! ! 
print("importing packages...")
import sys
#sys.path.append('/Users/anyaphillips/Desktop/harvard/research/via_binaries/scripts')
sys.path.append('/n/home02/amphillips/via_binaries/scripts')
import functions as paf
import numpy as np
import argparse


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
erv_viamock = 0.11823167210684976 # from Fe/H=-2, Teff=4700K, Gmag=17, exposure time = 0.66 effective hours. 

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
#                MAIN PROGRAM                    #
#------------------------------------------------#
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt1", type=int)
    parser.add_argument("--filename", type=str)
    args = parser.parse_args()

dt1_val = args.dt1

dt2_min, dt2_max, dt2_step = 1, 2*365, 1
dt2_vals = np.arange(dt2_min, dt2_max+dt2_step, dt2_step)

#### print the start/stop/step values for plotting later... 
# print("dt1_min, dt1_max, dt1_step", dt1_min, dt1_max, dt1_step)
print("dt2_min, dt2_max, dt2_step", dt2_min, dt2_max, dt2_step)

detection_fractions = []
# for k, dt1_val in enumerate(dt1_vals):
#     print("iteration %i / %i"%(k, len(dt1_vals)+1))
#     detection_fractions_this_dt1 = []
for dt2_val in tqdm(dt2_vals):
    obstimes = get_obstime(N=N, DT1=dt1_val, DT2=dt2_val)
    rvs = paf.get_rvs(params, obstimes, verbose=False, 
                      add_noise=True, noise_level=erv_viamock, rng=rng)
    detected, delta_vsys = paf.get_detections(
                erv_viamock, # km/s
                rvs,
                v0, 
                bool_arr='detet'
            )
    detection_fraction = len(P[detected])/len(P)
    detection_fractions.append(detection_fraction)

# save a numpy text file
detection_fraction_array = np.array(detection_fractions)
np.savetxt(args.filename+".txt", detection_fraction_array)
