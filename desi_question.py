# %%
repo_path = '/Users/anyaphillips/Desktop/harvard/research/via_binaries'
import sys
sys.path.append(repo_path+'/scripts')
sys.path.append('/Users/anyaphillips/Downloads/software/viamock/')

import functions as paf

from mock_obs_par import calc_a, calc_K
import petar
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('vedant')
%matplotlib inline
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from astropy.table import Table
from scipy.stats import binned_statistic_2d

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm


from tqdm import tqdm

import pandas as pd

from astropy.table import Table
import astropy.constants as const
import astropy.units as u
from scipy.stats import binned_statistic
from matplotlib.gridspec import GridSpecFromSubplotSpec

import matplotlib.pyplot as plt
# %config InlineBackend.figure_format='retina'

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap

import viamock

lm_colors, hm_colors, simcolors = paf.define_simcolors()
time_cmap = paf.define_time_cmap()
# %%
# from desi MWS slack/vedant -- what epoch-to-epoch RV differences 
# would we expect night-to-night for different binary fractions. is is 500 m/s ? is it color-dependent? 
# %%
binaries = pd.read_csv('UPDATED_detection_fractions_dense/cosmic_example_IBC.csv')
N = len(binaries)
print(N)
# %%
# recover single masses using the binfrac column of the cosmic table
single_masses = binaries['mass_1'].values
fbins = binaries['binfrac'].values

pick_binary = np.zeros(len(fbins), dtype=bool)
for k in tqdm(range(len(fbins))):
    uu = np.random.uniform(size=1)
    fbin = fbins[k]
    if uu<fbin:
        pick_binary[k]=True

fig, ax = plt.subplots()
ax.hist(np.log10(single_masses[~pick_binary]), bins=np.linspace(np.log10(0.08), 2, 100))
ax.hist(np.log10(binaries['mass_1'][pick_binary]), bins=np.linspace(np.log10(0.08), 2, 100))
# %%
# cool, now generate RV curves for everyone

dT = 1000 #* u.day

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

def get_obstime(N, dT):
    """
    the exact observing cadence version of the above. 
    i.e., observations will be separated by exactly DT1 and DT2.
    """
    base = np.repeat(0, N)
    gap1 = np.repeat(dT, N)
    # gap2 = np.repeat(DT2, N)

    deltaTs = np.vstack((base, gap1)).T
    obstimes_all = np.cumsum(deltaTs, axis=1) ##### THIS IS WHAT WILL GO INTO THE RV GENERATION FUNCTION!   
    return obstimes_all


obstimes = get_obstime(len(binaries['mass_1'][pick_binary]), dT)

rvs = paf.get_rvs(params[pick_binary], obstimes, verbose=False)
### ADD RV NOISE HERE
sigma_rv = 0.1 # km/s
rv_noise = np.random.normal(0, 0.1, rvs.shape)
rvs_binary_noised = rvs + rv_noise

rvs_single = np.zeros((len(single_masses[~pick_binary]),2))
rv_single_noise = np.random.normal(0, 0.1, rvs_single.shape)
rvs_single_noised = rvs_single + rv_single_noise
rvs_noised = np.vstack((rvs_binary_noised, rvs_single_noised))

drv = rvs_noised[:,1] - rvs_noised[:,0]
# drv_single = rvs_single_
n_binary = len(binaries['mass_1'][pick_binary])


# bins = np.linspace(-4, 2.5, 100)
bins = np.logspace(-4, 3, 100)
fig, ax = plt.subplots()
ax.hist(np.abs(drv), color='k',
        alpha=0.2, bins=bins,label='all')

ax.hist(np.abs(drv[:n_binary]), color='tomato',
        histtype='step', lw=3, bins=bins, label='binaries')
ax.hist(np.abs(drv[n_binary:]), color='k',
        histtype='step', bins=bins, lw=3, label='single stars')

ax.legend(loc='upper right', fontsize=15)

ax.axvline(0.5, c='k', ls='--', lw=2)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel(r'$|\Delta v_r|~[\rm km~s^{-1}]$')
# ax.set_ylabel(r'')
# %%
# binaries.columns