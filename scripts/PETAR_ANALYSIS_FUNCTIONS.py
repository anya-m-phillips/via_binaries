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
import gala.coordinates as gc
import gala.dynamics as gd
import gala.potential as gp
from gala.dynamics import mockstream as ms

from gala.units import galactic

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


def define_paths():
    ph = "/n/holystore01/LABS/conroy_lab/Lab/amphillips/finished_grid/"

    path0 = ph+"optimized_dense_sims/0_circ_rvir0.75_lm/" # unfinished
    # path0 = "/n/netscratch/conroy_lab/Lab/amphillips/small_grid/optimize_dense_sims_testing/0_circ_rvir0.75_lm/"
    path1 = ph+"optimized_dense_sims/1_circ_rvir0.75_hm/"  # done
    path2 = ph+"2_circ_rvir1.5_lm/"
    path3 = ph+"3_circ_rvir1.5_hm/"
    path4 = ph+"4_circ_rvir3_lm/"
    path5 = ph+"5_circ_rvir3_hm/"
    path6 = ph+"6_circ_rvir6_lm/"
    path7 = ph+"7_circ_rvir6_hm/"

    path8 = ph+"optimized_dense_sims/8_gd1_rvir0.75_lm/"  # done
    path9 = ph+"optimized_dense_sims/9_gd1_rvir0.75_hm/" # done
    # path10 = ph+"10_gd1_rvir1.5_lm/" #<-- this sim has particles deleted; unphysical
    path10 = "/n/holystore01/LABS/conroy_lab/Lab/amphillips/finished_grid/10_copies_successful/0/" # <--sucsessful sim w no particle deletions. 

    path11 = ph+"11_gd1_rvir1.5_hm/"
    path12 = ph+"12_gd1_rvir3_lm/"
    path13 = ph+"13_gd1_rvir3_hm/"
    path14 = ph+"14_gd1_rvir6_lm/"
    path15 = ph+"15_gd1_rvir6_hm/"

    path16 = ph+"optimized_dense_sims/16_pal5_rvir0.75_lm/" # done. 
    path17 = ph+"optimized_dense_sims/17_pal5_rvir0.75_hm/" # done. 
    path18 = ph+"18_pal5_rvir1.5_lm/"
    path19 = ph+"19_pal5_rvir1.5_hm/"
    path20 = ph+"20_pal5_rvir3_lm/"
    path21 = ph+"21_pal5_rvir3_hm/"
    path22 = ph+"22_pal5_rvir6_lm/"
    path23 = ph+"23_pal5_rvir6_hm/"

    paths = [path0,path1,path2,path3,path4,path5,path6,path7,
             path8,path9,path10,path11,path12,path13,path14,path15,
             path16,path17,path18,path19,path20,path21,path22,path23]

    return paths


def define_init_displacements():
    """
    returns a list of initial displacements in kpc, km/s
    for the main grid of petar simulations
    """
    id_circ=np.array([20.0,0.0,0.0,0.0,197.61111175113643,0.0])

    init_pos = np.array([-14680.175269960177, -18049.00342055592, -3355.258112269282]) * u.pc.to(u.kpc)
    init_vel = np.array([126.94017915235732, -33.56076783776999, 112.93136486685516]) # km/s
    id_gd1 = np.concatenate([init_pos, init_vel])


    init_pos_pal5 = np.array([13536.852506889449, -3328.304203161837, 5478.0923546972035]) * u.pc.to(u.kpc)
    init_vel_pal5 = np.array([119.95920838380918, -89.0209982268969, -88.31131556786855]) # km/s
    id_pal5 = np.concatenate([init_pos_pal5, init_vel_pal5])


    init_displacements = [id_circ]*8 + [id_gd1]*8 + [id_pal5]*8

    return init_displacements

def define_apocenters():
    """
    these numbers come from the notebook
    1_table_and_context.ipynb
    """
    circ_apocenter = 20 # kpc
    gd1_apocenter = 27.58821821 # from Gala integration
    pal5_apocenter =  18.55466194 # from Gala integration
    apocenters = [circ_apocenter] * 8 + [gd1_apocenter] * 8 + [pal5_apocenter] * 8
    return apocenters


def get_tdis_tplot(paths):
    ##### dissolution and plotting times
    dissolution_times = []
    plotting_times_peri = []
    plotting_times_apo = []

    circ_bools = [True]*8 + [False]*16
    for n, path in enumerate(paths):
        circ_bool = circ_bools[n]
        if n==10: # manually do sim 10, which has time snaps only every 10 Myr
            tidal = load_tidal(path)
            bound_times = tidal.time[tidal.n>=100]
            tdis = int(max(bound_times))

            core = petar.Core(interrupt_mode='bse', external_mode='galpy')
            core.loadtxt(path+"data.core")
            x,y,z = core.pos.T
            r = np.sqrt(x**2 + y**2 + z**2)

            i_dis = int(tdis/10)
            r_times_to_check = tidal.time[i_dis-50:i_dis].astype(int) 
            r_inds_to_check = r_times_to_check / 10
            r_inds_to_check = r_inds_to_check.astype(int) 
            rs_to_check = r[r_inds_to_check]

            min_inds, min_vals = find_minima(rs_to_check)
            times_at_minimum = r_times_to_check[min_inds]
            inds_at_minimum = r_inds_to_check[min_inds] 

            max_inds, max_vals = find_maxima(rs_to_check)
            times_at_maximum = r_times_to_check[max_inds]
            inds_at_maximum = r_inds_to_check[max_inds]

            tplot_p = int(max(times_at_minimum))
            tplot_a = int(max(times_at_maximum))     


        else:
            tdis, tplot_p, tplot_a = find_dissolution_plotting_times(path, circ=circ_bool)
        # print("found times...appending...")
        dissolution_times.append(tdis)
        plotting_times_peri.append(tplot_p)
        plotting_times_apo.append(tplot_a)
        print(n, tdis, tplot_p, tplot_a) 

    return dissolution_times, plotting_times_peri, plotting_times_apo


def calc_P(a, Mtot):
    """
    period from semimajor axis, make the inputs astropy unit quantities
    """
    P = 2*np.pi * np.sqrt((a**3)/(const.G * Mtot))
    return P


def calc_amplitude(binary,inclination=np.pi/2):
    """
    calculate the velocity amplitude
    in m/s for a petar binary object assuming some inclination
    (sin(i) = 1)
    """
    Mtot = (binary.mass * u.Msun)#.value
    M2 = min([binary.p1.mass, binary.p2.mass]) * u.Msun

    a = (binary.semi * u.pc).to(u.m)#.value
    e = binary.ecc
    P = calc_P(a, Mtot).to(u.s)

    a1 = (M2/Mtot) * a

    num = 2*np.pi*a1.value *np.sin(inclination)# nominally this would include sin(i)
    denom = P.value * np.sqrt(1-(e**2))
    K = num/denom # velocity semiamplitude - multiply by 2 for full amplitude.
    return K #2*K

def prog_position(init_displacement, i):
        """
        integrate the progenitor orbit up to this point
        init displacement should be in kpc, **km/s (not with astropy units)
        using Gala and the Bovy2014 milky way potential.
        returns (x, y, z, vx, vy, vz) in kpc, km/s
        """
        pos0 = init_displacement[:3] * u.kpc 
        vel0 = init_displacement[3:] * u.km/u.s
        w0 = gd.PhaseSpacePosition(pos0, vel0)
        mwp = gp.BovyMWPotential2014(units=galactic)
        H = gp.Hamiltonian(mwp)
        int_time = i
        dt = 1
        orbit = H.integrate_orbit(w0, dt=dt, n_steps = int_time/dt)

        x = orbit.pos.x[-1].to(u.kpc)
        y = orbit.pos.y[-1].to(u.kpc)
        z = orbit.pos.z[-1].to(u.kpc)
        vx = orbit.vel.d_x[-1].to(u.km/u.s)
        vy = orbit.vel.d_y[-1].to(u.km/u.s)
        vz = orbit.vel.d_z[-1].to(u.km/u.s)
        return np.array([x.value, y.value, z.value, vx.value, vy.value, vz.value])

def load_particle(path, i, interrupt="bse",file_naming_convention="every integer"):
    """
    CAUTION! 
    works only for loading from
    an all particles file, e.g.,
    data.0
    for data.0.binary, use petar.Binary()
    and for data.0.single, use petar.Particle() 
    without skiprows=1 in .loadtxt()
    """
    if file_naming_convention=="every integer":
        file = path+"data."+str(i)
    if file_naming_convention=="every 10":
        file = path+"data."+str(int(i/10))
    particle = petar.Particle(interrupt_mode=interrupt, external_mode='galpy')
    particle.loadtxt(file, skiprows=1)
    return particle

def load_core(path, interrupt='bse'):
    core = petar.Core(interrupt_mode=interrupt, external_mode='galpy')
    core.loadtxt(path+"data.core")
    return core

def load_tidal(path):
    tidal = petar.external.Tidal()
    tidal.loadtxt(path+"data.tidal")
    return tidal

def fix_core_vel(core):
    """
    differentiate the core position to get the "correct" core velocity. 
    FOR NOW, ONLY WORKS WHEN THE OUTPUT CADENCE IS 1 MYR!!

    returns corrected core velocity (w/o astropy units) in pc/Myr
    """
    core_x, core_y, core_z = core.pos.T * u.pc
    # core_vx, core_vy, core_vz = core.vel.T * u.pc/u.Myr
    times = core.time * u.Myr
    core_dx_dt = np.gradient(core_x, times) # it's giving rigorous. 
    core_dy_dt = np.gradient(core_y, times)
    core_dz_dt = np.gradient(core_z, times)

    core_vel_corrected = np.array([
        core_dx_dt.to(u.pc/u.Myr).value, core_dy_dt.to(u.pc/u.Myr).value, core_dz_dt.to(u.pc/u.Myr).value
    ]).T   

    return core_vel_corrected

def is_dissolved(path, i, threshold=100, interrupt="bse"):
    """
    check whether the simulated cluster located at path
    has dissolved at timestep i
    dissolved is defined as fewer than [threshold] stars
    within the tidal radius.
    """
    # load tidal data
    tidal = load_tidal(path)
    n_bound = int(tidal.n[i])
    
    if n_bound<threshold:
        return True
    else:
        return False

def load_coords_v2(path, i, core,
                    interrupt="bse",
                    tdis_estimate = 3000, 
                    load_all=True, load_singles=True, load_binaries=True,
                    correct_core_vel=True):
    """
    load stream frame coordinates using jake's package
    path is a path to petar output files directory
    i is the output number we care about
    return all particles and return all single particles lets you return those 
    petar objects as well as their stream coordinates
    returns: [all_particles], binaries, [singles], coords, single_coords, binary_coords
    """

    file = path+"data."+str(i)

    dis = is_dissolved(path, i, interrupt=interrupt) # threshold==100

    if dis==False:
        ### if the cluster is not dissolved, use the current core position to do the xform
        if correct_core_vel==True:
            corrected_core_vel = fix_core_vel(core) # avoid spurious
        if correct_core_vel==False:
            corrected_core_vel = core.vel

        core_pos = core.pos[i] * u.pc.to(u.kpc) # dimensionless core position in kpc
        core_vel = corrected_core_vel[i] * (u.pc/u.Myr).to(u.kpc/u.Myr) # dimensionless core velocity in kpc/myr
        core_w_jake_units = np.hstack([core_pos, core_vel])
        prog_w_jake_units = core_w_jake_units


    if dis==True: # if dissolved, option to use progenitor or use last core position
        # if the cluster _is_ dissolved, integrate the core from estimated dissolution
        # time to now
        if tdis_estimate is None:
            raise BaseException("ENTER AN ESTIMATED DISSOLUTION TIME!")


        core_pos = core.pos[tdis_estimate] * u.pc.to(u.kpc) # dimensionless core position in kpc
        if correct_core_vel==True:
            corrected_core_vel = fix_core_vel(core) # avoid spurious
        if correct_core_vel==False:
            corrected_core_vel = core.vel

        core_vel = corrected_core_vel[tdis_estimate] * (u.pc/u.Myr).to(u.km/u.s) # dimensionless core velocity in kpc/myr

        # use prog_position function to integrate from final core position to now
        ### note prog_position requires dimensionless init_displacement in kpc, km/s
        new_init_displacement = np.hstack([core_pos, core_vel])
        time_to_integrate = int(i-tdis_estimate)
        prog_w = prog_position(init_displacement=new_init_displacement, i=time_to_integrate)

        # transform to jake units
        prog_pos = prog_w[:3] * (u.kpc)
        prog_vel = prog_w[3:] * (u.km/u.s)
        # concatenate. 
        prog_w_jake_units = np.hstack([
            prog_pos.value, prog_vel.to(u.kpc/u.Myr).value
        ]) 



    if load_all==True:
        # load all particles, put in the galactocentric frame, get stream coordinates.
        header = petar.PeTarDataHeader(path+'data.'+str(i), external_mode='galpy')
        all_particles = petar.Particle(interrupt_mode=interrupt, external_mode='galpy')

        all_particles.loadtxt(file, skiprows=1)
        CM_pos = (header.pos_offset*u.pc).to(u.kpc)
        CM_vel = (header.vel_offset*(u.pc/u.Myr)).to(u.kpc/u.Myr)
        CM_w = np.hstack([CM_pos.value, CM_vel.value])


        all_pos = (all_particles.pos*u.pc).to(u.kpc) + CM_pos
        all_vel = (all_particles.vel*(u.pc/u.Myr)).to(u.kpc/u.Myr) + CM_vel
        all_w = np.hstack([all_pos.value, all_vel.value])


        stream_obj = StreamFrame(sim_coords = all_w, prog_sim_coord = prog_w_jake_units)#CM_w)
        coords = stream_obj.GetStreamFrame()

    # load binaries
    if load_binaries==True:
        binary_file = path+"data."+str(i)+".binary"
        binaries = petar.Binary(member_particle_type=petar.Particle,
                                G=petar.G_MSUN_PC_MYR, interrupt_mode=interrupt,
                                external_mode="galpy")

        binaries.loadtxt(binary_file)#, skiprows=1)

    if load_singles==True:
        # load singles
        single_file = path+"data."+str(i)+".single"
        singles = petar.Particle(interrupt_mode=interrupt,
                                        external_mode='galpy')

        singles.loadtxt(single_file)#, skiprows=1)

    core_data = petar.Core(interrupt_mode=interrupt, external_mode='galpy')
    
    core_data.loadtxt(path+'data.core')
    core_pos = (core_data.pos[i]*u.pc).to(u.kpc)
    core_vel = (core_data.vel[i]*(u.pc/u.Myr)).to(u.kpc/u.Myr) # **not corrected !
    core_w = np.hstack([core_pos.value, core_vel.value])

    if load_singles==True:
        # put singles in the galactocentric frame
        single_pos = (singles.pos*u.pc).to(u.kpc) + core_pos
        single_vel = (singles.vel*(u.pc/u.Myr)).to(u.kpc/u.Myr) + core_vel
        single_w = np.hstack([single_pos.value, single_vel.value])
        single_stream_obj = StreamFrame(sim_coords = single_w, prog_sim_coord = prog_w_jake_units)#core_w)
        single_coords = single_stream_obj.GetStreamFrame()


    if load_binaries==True:
        # put binaries in the galactocentric frame
        binary_pos = (binaries.pos*u.pc).to(u.kpc) + core_pos
        binary_vel = (binaries.vel*(u.pc/u.Myr)).to(u.kpc/u.Myr) + core_vel
        binary_w = np.hstack([binary_pos.value, binary_vel.value])
        binary_stream_obj = StreamFrame(sim_coords = binary_w, prog_sim_coord = prog_w_jake_units)#core_w)
        binary_coords = binary_stream_obj.GetStreamFrame()

    
    particle_data = []
    streamframe_data = []

    if load_all==True:
        particle_data.append(all_particles)
        streamframe_data.append(coords)

    if load_singles==True:
        particle_data.append(singles)
        streamframe_data.append(single_coords)

    if load_binaries==True:
        particle_data.append(binaries)
        streamframe_data.append(binary_coords)

    return particle_data, streamframe_data

def CM_to_galcen_frame(path, particles, i):
    header = petar.PeTarDataHeader(path+"data."+str(i), external_mode='galpy')
    CM_pos = header.pos_offset*u.pc
    CM_vel = (header.vel_offset*(u.pc/u.Myr)).to(u.km/u.s)
    # add center of mass position / move to galctocentric frame
    all_pos = (particles.pos*u.pc) + CM_pos
    all_vel = (particles.vel*(u.pc/u.Myr)).to(u.km/u.s) + CM_vel
    
    return all_pos.to(u.pc), all_vel.to(u.km/u.s)

def core_to_galcen_frame(path, particles, i, core=None, interrupt="bse"):
    pos = particles.pos*u.pc
    vel = (particles.vel*(u.pc/u.Myr)).to(u.km/u.s)
    
    if core is None:
        core = petar.Core(interrupt_mode=interrupt, external="galpy")
        core.loadtxt(path+"data.core")

    core_pos = core.pos[i] * u.pc
    core_vel = (core.vel[i] * (u.pc/u.Myr)).to(u.km/u.s)
    
    pos+=core_pos
    vel+=core_vel
    return pos.to(u.pc), vel.to(u.km/u.s)

def xform_to_core_frame(path, particles, i, interrupt="bse", file_naming_convention="every integer"):
    """
    pass a petar.Particles object and the corresponding i sim timestep
    returns w (phase space position in core frame)
    *** i is the desired simulation time not the file index ***
    """
    if file_naming_convention=="every integer":
        header = petar.PeTarDataHeader(path+"data."+str(i), external_mode='galpy')
        index = i
    if file_naming_convention=="every 10":
        header = petar.PeTarDataHeader(path+"data."+str(int(i/10)), external_mode='galpy')
        index = int(i/10)
    

    CM_pos = header.pos_offset*u.pc
    CM_vel = (header.vel_offset*(u.pc/u.Myr)).to(u.km/u.s)
    # add center of mass position / move to galctocentric frame
    all_pos = (particles.pos*u.pc) + CM_pos
    all_vel = (particles.vel*(u.pc/u.Myr)).to(u.km/u.s) + CM_vel

    # subtract core position / move to core frame
    core = petar.Core(interrupt_mode=interrupt, external="galpy")
    core.loadtxt(path+"data.core")
    
    core_pos = core.pos[index] *u.pc
    core_vel = (core.vel[index] * (u.pc/u.Myr)).to(u.km/u.s)

    pos_rel = all_pos - core_pos
    vel_rel = all_vel - core_vel
    
    xrel, yrel, zrel = pos_rel.T
    rrel = np.sqrt(xrel**2 + yrel**2 + zrel**2)
    
    return pos_rel.to(u.pc), vel_rel.to(u.km/u.s), rrel.to(u.pc)

def clip_outside_rtid(path, particles, i, interrupt="bse", in_core_frame=True,
                      file_naming_convention="every integer"):
    if not in_core_frame:
        pos_rel, vel_rel, rrel = xform_to_core_frame(path, particles, i, interrupt,
                                                    file_naming_convention)
        
    if in_core_frame:
        pos_rel, vel_rel = particles.pos*u.pc, particles.vel*(u.km/u.s)
        x, y, z = pos_rel.T
        rrel = np.sqrt(x**2 + y**2 + z**2)
        
    tidal = load_tidal(path)

    if file_naming_convention=="every integer":
        index=i
    if file_naming_convention=="every 10":
        index = int(i/10)

    rtid = tidal.rtid[index] * u.pc
    
    clip = rrel<=rtid
    return rrel, clip
    

def compute_cluster_structure(path, i_list, init_displacement, interrupt="bse", save=False, savepath=None, density_shell_width=0.15):
    """
    i_list, init_displacement should be np arrays
    path should be a string.
    return an array of half mass radii over i_list
    array of nbounds over i_list
    dissolution time (last timestep at which nbound >100 stars
    """
    hmrs = []
    nbounds = []
    mass_bounds = []
    t_rhs = []
    t_dyns = []
    
    tidal = petar.external.Tidal()
    tidal.loadtxt(path+"data.tidal")
    rtid = tidal.rtid * u.pc
    for j in tqdm(range(len(i_list))):
        i = i_list[j]
        file = path+"data."+str(i)
        # if bse==True:
        all_particles = petar.Particle(interrupt_mode=interrupt, external_mode='galpy')
        all_particles.loadtxt(file, skiprows=1)
        
        
        pos_rel, vel_rel, rrel = xform_to_core_frame(path, all_particles, i, interrupt=interrupt)
        in_rtid = rrel.to(u.pc) <= rtid[i]
        
        nbound = len(rrel[in_rtid])
        nbounds.append(nbound)
        
        mass_bound = np.sum(all_particles.mass[in_rtid] * u.Msun)
        mass_bounds.append(mass_bound.to(u.Msun).value)
        
        if nbound<=100:
            hmrs.append(0.)
            t_rhs.append(0.)
            
        else:
            hmr = calculate_half_mass_radius(all_particles.mass[in_rtid]*u.Msun, rrel[in_rtid].to(u.pc)) * u.pc
            hmrs.append(hmr.to(u.pc).value)
                        
            shell_width = density_shell_width*u.pc
            inner = hmr-shell_width
            outer = hmr+shell_width
            in_shell = (rrel>inner) & (rrel<outer)
            
            shell_volume = (4*np.pi/3) * (outer**3 - inner**3)
            shell_mass = np.sum(all_particles.mass[in_shell]) * u.Msun
            shell_density = shell_mass / shell_volume
            
            # print(rho_avg.decompose().unit)
            
            t_cross = (const.G * shell_density)**(-1/2)
            t_dyns.append(t_cross.to(u.Myr).value)
            # print(t_cross.decompose().unit)
            t_rh = ((0.1*nbound)/(np.log(nbound))) * t_cross
            t_rhs.append(t_rh.to(u.Myr).value)
       
    hmrs = np.array(hmrs)
    nbounds = np.array(nbounds)
    mass_bounds = np.array(mass_bounds) # in msun
    t_rhs = np.array(t_rhs)
    t_dyns = np.array(t_dyns)
    # indices at which nbounds>=100:
    bound_indices = i_list[nbounds>=100]
    t_dis = np.array([max(bound_indices)])
    
    if save==True:
        np.savez(savepath,
                 i_list = i_list,
                 hmrs = hmrs,
                 nbounds = nbounds,
                 mass_bounds = mass_bounds,
                 t_rhs = t_rhs,
                 t_dis = t_dis,
                 allow_pickle=True)
    
    return hmrs, nbounds, mass_bounds, t_rhs, t_dyns, t_dis



def calc_dispersion(x):
    x2 = x**2
    meanx = np.mean(x)
    meanx2 = np.mean(x2)
    sig2 = meanx2 - (meanx**2)
    return np.sqrt(sig2)

def calc_dispersion_with_3sig_clip(x):
    x2 = x**2
    meanx = np.mean(x)
    meanx2 = np.mean(x2)
    sig2 = meanx2 - (meanx**2)
    sig = np.sqrt(sig2)
    
    clip = np.abs(x-meanx)<=3*sig
    x_new = x[clip]
    x2_new = x_new**2
    meanx_new = np.mean(x_new)
    meanx2_new = np.mean(x2_new)
    sig2_new = meanx2_new - (meanx_new**2)
    return np.sqrt(sig2_new)


def straighten_stream_binmeds(phi1, y, trim_criteria, 
                              nbins=200, sigma_kernel=3):
    inMW, trim = trim_criteria
    phi1_trimmed = phi1[inMW][trim]
    y_trimmed = y[inMW][trim]
    ### get the fit: 
    bins = np.linspace(min(phi1_trimmed), max(phi1_trimmed), nbins+1)
    bin_centers = 0.5*(bins[:-1]+bins[1:])
    
    med, _, _ = binned_statistic(x=phi1_trimmed, values=y_trimmed,
                                  statistic='median', bins=bins)

    med_smooth = gaussian_filter1d(med, sigma_kernel)
    baseline = np.interp(phi1, bin_centers, med_smooth)
    y_smooth = y - baseline

    return phi1, y_smooth



def straighten_stream_polynomial(phi1,y, degree=5, show_plot=False, return_poly_fn=False,
                                trim_criteria=None):
    """
    straighten a streamframe object's phi1/phi2 coordinates 
    by fitting a high-order polynomial
    trim criteria optional, how to trim the stream before the polyfit to avoid sensitivity to outliers. 
        * should be a list, [inMW, trim]
    """
    inMW, trim = trim_criteria
    def make_polynomial(order):
        # Create the argument list: a0, a1, ..., an
        arg_list = ", ".join([f"a{i}" for i in range(order + 1)])

        # Build the function definition string
        func_str = f"def poly(x, {arg_list}):\n"
        func_str += "    return " + " + ".join([f"a{i}*x**{i}" for i in range(order + 1)]) + "\n"

        # Local namespace for exec
        local_ns = {}
        exec(func_str, {}, local_ns)
        return local_ns['poly']
    
    poly=make_polynomial(order=degree)

    # next fit a polynomial and subtract to straighten out the stream using only the trimmed data. 
    fit, cov = curve_fit(f=poly, xdata=phi1[inMW][trim], ydata=y[inMW][trim])
    new_y = y - poly(phi1, *fit)

    if show_plot==True:
        fig, [ax1,ax2] = plt.subplots(2,1,figsize=[10,20])
        x = np.linspace(min(phi1), max(phi1))
        y = poly(x, *fit)
        ax1.scatter(phi1, y, s=.1)
        ax1.plot(x, y) 
                                    
        ax2.scatter(phi1, new_y, s=.1)
        ax2.set_aspect('equal')

    if return_poly_fn==False:
        return phi1, new_y
    if return_poly_fn==True:
        return phi1, new_y, poly, fit



def trim_coords_percentile(coords, low, high, apo):
    """
    assumes apo is in kpc
    """
    inMW = coords['r'] <= 1.5*apo#.to(u.kpc).value # first remove high-velocity stars...
    
    trim_keys = ['phi1', 'phi2']#, 'r']

    # NEXT clip in-MW stars. 
    criteria = []
    for key in trim_keys:
        key_low, key_high = np.percentile(coords[key][inMW], q=[low, high])
        key_crit = (coords[key][inMW]<=key_high) & (coords[key][inMW]>=key_low)
        criteria.append(key_crit)
        # print(key, key_low, key_high)

    trim_criteria = criteria[0] & criteria[1] #& criteria [2]

    return inMW, trim_criteria

# concatenate single and binary coordinates. 
def concat_sb_coords(single_coords, binary_coords):
    keys = ['phi1', 'phi2', 'pm_phi1','pm_phi2','r', 'vr']
    combined = {
        key:np.concatenate([single_coords[key], binary_coords[key]]) for key in keys
    }
    return combined


def straighten_stream_orbit_interp(coords, yval, core, i, 
                                   trim_criteria=None,
                                   Dt_start=375, # Myr
                                   return_orbit_chunk=False, correct_core_vel=True,
                                   verbose=False):
    """
    coords: a dictionary from streamframe script
    yval: a string indexing the y coordinate of the stream coord (x assumed phi1)
          OR a list of strings, e.g., ['r', 'vr'] in which case the all straightened coords in the list
          will be returned. 
    core: a petar.Core() object
    i: a time at which the straightening occurs (to get the core position)
    Dt: a number of Myr to integrate forward/backward to get the prog orbit from time i
    """
    if trim_criteria is None:
        phi1_min_data = min(coords['phi1'])
        phi1_max_data = max(coords['phi1'])

    else:
        inMW, trim = trim_criteria # trim_criteria should be a tuple or list.

        # get range in phi1 of data (to ensure the interpolant covers all of it)
        phi1_min_data = min(coords['phi1'][inMW][trim])
        phi1_max_data = max(coords['phi1'][inMW][trim])

    # get present-day core position and velocity to set up the integration. 
    core_pos = core.pos[i] * u.pc

    if correct_core_vel==True:
        corrected_core_vel = fix_core_vel(core)
    if correct_core_vel==False:
        corrected_core_vel = core.vel
    
    core_vel = corrected_core_vel[i] * u.pc/u.Myr
    core_w0_sf = np.hstack([core_pos.to(u.kpc).value, core_vel.to(u.kpc/u.Myr).value])
    
    # do the gala orbit:
    core_w0 = gd.PhaseSpacePosition(core_pos, core_vel)
    mwp = gp.BovyMWPotential2014(units=galactic)
    H = gp.Hamiltonian(mwp)


    Dt = Dt_start
    Dt_step = 50
    last_Dt_step=Dt_step
    last_case=None

    max_iters = 100 # return a failure if this takes >100 iterations... 
    iter = 0
    while iter<max_iters: # ensure that there are no large jumps where the orbit circles around in phi1
        orbit_forward = H.integrate_orbit(core_w0, dt=1, n_steps = Dt)
        orbit_backward = H.integrate_orbit(core_w0, dt=-1, n_steps = Dt)

        # concatenate orbit forward/backward info into something that StreamFrame can use...
        orbit_pos_f = np.array([
            orbit_forward.pos.x.to(u.kpc).value,
            orbit_forward.pos.y.to(u.kpc).value,
                orbit_forward.pos.z.to(u.kpc).value
        ]).T
        orbit_pos_b = np.array([
            orbit_backward.pos.x.to(u.kpc).value,
            orbit_backward.pos.y.to(u.kpc).value,
                orbit_backward.pos.z.to(u.kpc).value
        ]).T[::-1]
        orbit_pos = np.vstack([orbit_pos_b, orbit_pos_f])

        orbit_vel_f = np.array([
            orbit_forward.vel.d_x.to(u.kpc/u.Myr).value,
            orbit_forward.vel.d_y.to(u.kpc/u.Myr).value,
                orbit_forward.vel.d_z.to(u.kpc/u.Myr).value
        ]).T
        orbit_vel_b = np.array([
            orbit_backward.vel.d_x.to(u.kpc/u.Myr).value,
            orbit_backward.vel.d_y.to(u.kpc/u.Myr).value,
                orbit_backward.vel.d_z.to(u.kpc/u.Myr).value
        ]).T[::-1]
        orbit_vel = np.vstack([orbit_vel_b, orbit_vel_f])

        orbit_w = np.hstack([orbit_pos, orbit_vel])
        orbit_streamobj = StreamFrame(sim_coords=orbit_w, prog_sim_coord=core_w0_sf)
        orbit_coords = orbit_streamobj.GetStreamFrame()   

        # ensure that the range in orbit coords captures that of the data. 
        max_phi1_orbit = max(orbit_coords['phi1'])
        min_phi1_orbit = min(orbit_coords['phi1'])
        orbit_range_danger_zone = np.max(np.abs(orbit_coords['phi1']))
        
        dphi1_max_orbit = np.max(np.abs(np.diff(orbit_coords['phi1'])))

        #### CASE BOOLS:
        covered = (max_phi1_orbit >= phi1_max_data) and (min_phi1_orbit <= phi1_min_data)
        wrap_like = (dphi1_max_orbit > 100)   # or whatever threshold makes sense after unwrapping      
        
        if covered and (not wrap_like): # perfect ! amazing !
            if verbose:
                print("OK")

            case="PERFECT. RIGOROUS, EVEN."
            break


        if wrap_like: ## avoid wrap issues first and foremost!!! START TAKING THE TIMESTEP DOWN
            case="wrap_issue"
            Dt=max(1, Dt-Dt_step)

        else: # if not a wrap issue, try increasing the coverage
            case="need_more_coverage"
            Dt+=Dt_step


        # halve step only if we flipped case
        if (last_case is not None) and (case != last_case):
            last_Dt_step = Dt_step
            Dt_step = max(1, Dt_step // 2)       

        if verbose:
            print(f"{case}: Dt={Dt} step={Dt_step} covered={covered} wrap_like={wrap_like} dphi1_max={dphi1_max_orbit:.2f}")

        # stop refinement if step is tiny but we still can't satisfy both
        if Dt_step == 1 and last_Dt_step == 1 and wrap_like: ### OKAY. IF WE ARE PING-PONGING AND STILL WRAP LIKE, CUT THINGS OFF!!!
            # best we can do near boundary; accept this Dt but warn
            orbit_coords = last_orbit_coords
            if verbose:
                print("Warning: step==1 and still wrap-like; returning best attempt (before wrap started).")
            break

        last_case = case
        last_orbit_coords = orbit_coords
        iter += 1



    if type(yval)==str: # when I request only one string, only return that to me.
        def interp_orbit(phi1):
            return np.interp(phi1, orbit_coords['phi1'], orbit_coords[yval]) 
        y_new = coords[yval] - interp_orbit(coords['phi1'])

    else:
        interp_orbit = []
        y_new = []
        for y in yval:
                
            def interp_orbit_fn(phi1):
                return np.interp(phi1, orbit_coords['phi1'], orbit_coords[y])
            interp_orbit.append(interp_orbit_fn)
            y_new_now = coords[y]-interp_orbit_fn(coords['phi1']) 
            y_new.append(y_new_now)

    if return_orbit_chunk==True:
        return y_new, interp_orbit, orbit_coords, orbit_w, iter
    
    else:
        return y_new, interp_orbit, iter


def get_ICRS_coords(path, i, interrupt="bse"):
    """
    return ICRS coordinates object
    provide path to PeTar data.# files and i
    """
    # load coordinates
    file = path+'data.'+str(i)
    header = petar.PeTarDataHeader(path+'data.'+str(i), external_mode='galpy')
    all_particles = petar.Particle(interrupt_mode=interrupt, external_mode='galpy')
    all_particles.loadtxt(file, skiprows=1)
    pos = all_particles.pos * u.pc
    vel = all_particles.vel * (u.pc/u.Myr)
    CM_pos = header.pos_offset*u.pc
    CM_vel = header.vel_offset * (u.pc/u.Myr)
 
    # transform out of CM frame
    pos+=CM_pos
    vel+=CM_vel

    # create a cartesian representation/differential
    rep = CartesianRepresentation(pos.T)
    rep_vel = CartesianDifferential(vel.T)
    rep = rep.with_differentials(rep_vel)

    # create coords in galactocentric frame
    galcen_frame = Galactocentric()
    coords_galcen = SkyCoord(rep, frame=galcen_frame)

    # transform to ICRS
    coords_ICRS = coords_galcen.transform_to(ICRS())
    return coords_ICRS

def get_streamcoords_from_ICRS(ICRS_coords, try_every=1):
    """
    get phi1, phi2 from transforming to ICRS coordinates, fitting a great circle coordinate frame
    then fitting a quadratic, cubic, or spline and subtracting it. 
    for finding stream widths; progenitor will not necessarily be centered.

    returns gc_frame_best, xformed_coords
    """
    obfun_best = np.inf
    gc_frame_best = None
    point1_best = None
    point2_best = None

    point1 = SkyCoord(
        ra = np.nanmedian(ICRS_coords.ra),
        dec = np.nanmedian(ICRS_coords.dec),
        frame='icrs'
    )
    
    print("iterating to find best frame to minimize phi2")
    for i in tqdm(range(0, len(ICRS_coords), try_every)):
        point2 = ICRS_coords[i]

        try:
            gc_frame = gc.GreatCircleICRSFrame.from_endpoints(point1, point2)
            stars_sc_gc = ICRS_coords.transform_to(gc_frame)
            obfun = np.sum(stars_sc_gc.phi2.value**2)

            if obfun < obfun_best:
                obfun_best = obfun # update objective function
                gc_frame_best = gc_frame
                point2_best = point2

        except Exception as e:
            continue # skipped failed point2s
    
    xformed_coords = ICRS_coords.transform_to(gc_frame_best)
    return gc_frame_best, xformed_coords



def calculate_half_mass_radius(ms, rs):
    """
    this is for checking the half mass radius
    for sanity purposes
    units of ms don't matter, will return r_h in whatever units "rs" is in
    """
    radii = rs
    masses = ms

    # Sort stars by radius
    sorted_indices = np.argsort(radii)
    sorted_radii = np.array(radii)[sorted_indices]
    sorted_masses = np.array(masses)[sorted_indices]

    # Compute cumulative mass
    cumulative_mass = np.cumsum(sorted_masses)

    # Total mass
    total_mass = np.sum(masses)

    # Find the half-mass radius
    half_mass = total_mass / 2
    half_mass_radius_index = np.searchsorted(cumulative_mass, half_mass)
    half_mass_radius = sorted_radii[half_mass_radius_index]
    # print("half mass radius = %.3f pc"%half_mass_radius)
    return half_mass_radius 


    
def find_minima(arr):
    min_indeces = []
    min_values = []
    for i,a in enumerate(arr):
        # print(i)
        if i==0 or i==int(len(arr)-1):
            # print("first/last value, skipping")
            continue
        if (arr[i-1]>a) & (arr[i+1]>a):
            # print("appending a local minimum")
            min_indeces.append(i)
            min_values.append(a)

    return np.array(min_indeces), np.array(min_values)

def find_maxima(arr):
    max_indeces = []
    max_values = []
    for i, a in enumerate(arr):
        if i==0 or i==int(len(arr)-1):
            continue
        if (arr[i-1]<a) & (arr[i+1]<a):
            max_indeces.append(i)
            max_values.append(a)

    return np.array(max_indeces), np.array(max_values)

def find_dissolution_plotting_times(path, interrupt='bse', circ=False):
    # find galactocentric radius at each timestep
    core = petar.Core(interrupt_mode=interrupt, external='galpy')
    core.loadtxt(path+'data.core')
    x,y,z = core.pos.T
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # find dissolution time
    tidal = load_tidal(path)
    bound_times = tidal.time[tidal.n>=100]
    tdis = int(max(bound_times))
    
    if circ==True:
        t_plot_peri = tdis
        t_plot_apo = tdis

    else:    
        # find last peri/apocenter before dissolution
        r_times_to_check = tidal.time[tdis-500:tdis].astype(int) # check last 500 Myr for a min/max r
        rs_to_check = r[r_times_to_check]
        min_inds, min_vals = find_minima(rs_to_check)
        times_at_minimum = r_times_to_check[min_inds]
        max_inds, max_vals = find_maxima(rs_to_check)
        times_at_maximum = r_times_to_check[max_inds]
        
        
        t_plot_peri = int(max(times_at_minimum))
        t_plot_apo = int(max(times_at_maximum))
    
    return tdis, t_plot_peri, t_plot_apo
    

## for uncertainties in the binary fraction. 
def Prob_of_frac(k, N, parr=np.linspace(0, 1, 1000)): # uniform prior??
    """
    return the probability mass function for p given k successes in N trials
    """
    return parr, np.array([binom.pmf(k, N, p) for p in parr]) # probability mass function

def percentile(p, Prob, percentiles=[16,50,84]):
    """
    return percentile values of the binomial dist???
    """
    cdf = np.cumsum(Prob)
    cdf /= cdf.max()
    cdf *= 100 
    values = np.interp(percentiles, cdf, p)
    return values


def p16(data):
    return np.percentile(data, 16)
def p84(data):
    return np.percentile(data, 84)

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

def get_orbital_params(paths, times, n, rng=None):
    """
    retrieve orbital parameters from path paths[n] at time times[n]
    returns an array of params
    [v0 (km/s), K (km/s), w, phi0, e, P (day)]
    where w, phi1 are uniformly sampled and an isotropic inclination is picked to compute k
    """
    i = times[n]
    path = paths[n]

    core = petar.Core(interrupt_mode='bse', external_mode='galpy')
    core.loadtxt(path+"data.core")
    core_pos = core.pos[i] * u.pc.to(u.kpc) # dimensionless core position in kpc
    core_vel = core.vel[i] * (u.pc/u.Myr).to(u.kpc/u.Myr) # dimensionless core velocity in kpc/myr
    core_w_jake_units = np.hstack([core_pos, core_vel])

    binaries = petar.Binary(member_particle_type=petar.Particle, G=petar.G_MSUN_PC_MYR,
                            interrupt_mode='bse', external_mode='galpy')
    binaries.loadtxt(path+"data.%i.binary"%i)#, skiprows=1)
    binary_pos = binaries.pos*u.pc.to(u.kpc)+core_pos
    binary_vel = binaries.vel*(u.pc/u.Myr).to(u.kpc/u.Myr)+core_vel
    binary_w = np.hstack([binary_pos, binary_vel])



    binary_stream_obj = StreamFrame(sim_coords=binary_w, prog_sim_coord=core_w_jake_units)
    binary_coords = binary_stream_obj.GetStreamFrame()

    ###### collect everything we need
    a_vals = binaries.semi*u.pc
    Mtot_vals = binaries.mass*u.Msun
    P_vals = calc_P(a_vals, Mtot_vals)

    if rng is None:
        rng = np.random.default_rng()

    inclinations = draw_inclinations(len(a_vals), rng)

    K_vals = np.array([calc_amplitude(binary, i) for binary, i in zip(binaries, inclinations)]) * u.m/u.s

    e_vals = binaries.ecc

    phi1_vals = binary_coords['phi1'] # degrees
    v0_vals = (binary_coords['vr']*u.km/u.s).to(u.m/u.s) # extract systemic velocities.



    ### draw arguments of periapsis and mean anomaly phase offsets ?? 

    w_vals = rng.uniform(low=0, high=2*np.pi, size=len(a_vals))
    phi0_vals = rng.uniform(low=0, high=1, size=len(a_vals))
    
    params_all = np.array([v0_vals.to(u.km/u.s).value, 
                           K_vals.to(u.km/u.s).value,
                           w_vals,
                           phi0_vals,
                           e_vals,
                           P_vals.to(u.day).value]
                          ).T
    
    return params_all, phi1_vals

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

def get_obstimes(N_obsm, N, rng=None):
    dt_min = 30
    dt_max = 3*360

    if rng is None:
        rng = np.random.default_rng()

    base = np.repeat(0, N)
    gap1 = rng.uniform(30,90, size=N)
    gap2 = rng.uniform(3*365, 5*365, size=N)

    deltaTs = np.vstack((base, gap1, gap2)).T

    obstimes_all = np.cumsum(deltaTs, axis=1) ##### THIS IS WHAT WILL GO INTO THE RV GENERATION FUNCTION!   
    return obstimes_all

def get_rvs(params_all, obstimes_all, verbose=True):

    RVs_all = []
    N=len(params_all[:,0])

    if verbose==True:
        for j in tqdm(range(N)):
            params = params_all[j]
            obstimes = obstimes_all[j]

            rvs = radial_velocity(obstimes, params)
            RVs_all.append(rvs)
    if verbose==False:
        for j in range(N):
            params = params_all[j]
            obstimes = obstimes_all[j]

            rvs = radial_velocity(obstimes, params)
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
    Nobs=3
    P_chi2 = stats.chi2.sf(chi2_rvs, df=Nobs-1)
    zscore = np.max(np.abs(resid_rvs/e_rv), axis = 1) # idk what this is,

    undet = P_chi2>0.1 # detect binaires when the P value of the chi square test is <0.1 
    
    if bool_arr=='undet':
        return undet, delta_vsys
    if bool_arr=='detet':
        return ~undet, delta_vsys
    








def gen_kroupa_IMF(m_max, N):
    m0, m1, m2 = (0.08, 0.5, m_max) #* u.Msun
    if m_max<0.5:
        print("maximum mass <0.5 Msun not supported sry")
        return 
    
    # define normalization coefficients
    X1 = m0**(-0.3) - m1**(-0.3)
    X2 = m1**(-1.3) - m2**(-1.3)

    A1 = 1/((X1/0.3) + m1*(X2/1.3))
    A2 = m1/((X1/0.3) + m1*(X2/1.3))

    def C1(m):
        return A1 * (m0**(-0.3) - m**(-0.3))/0.3
    # print(C1(m1))
    def C2(m): # don't need actually
        return (A1 * (m0**(-0.3) - m1**(-0.3))/0.3) + (A2 * (m1**(-1.3) - m**(-1.3))/1.3)
    # print(C1(m1)==C2(m1), "should be equal")

    masses = []
    for i in range(N):
        u_rand = np.random.rand()
        if u_rand < C1(m1):
            m = (m0**(-0.3) - (0.3/A1)*u_rand)**(-1/0.3)
            masses.append(m)
        if u_rand >= C1(m1):
            m = (m1**(-1.3) - (1.3/A2)*(u_rand - (A1*X1)/0.3))**(-1/1.3)
            masses.append(m)
        
    return masses



#------------------------------------------#
#    stuff for gala particle spray streams #
#------------------------------------------#
def gen_stream(time, init_displacement, Mprog, release_every=5, b=0.75 * u.pc, 
               save=False, output_every=None,
               output_filename=None,
               streamtype='chen' # or 'streakline'
               ):
    
    pos0 = init_displacement[:3]*u.kpc
    vel0 = init_displacement[3:]*(u.km/u.s)

    prog_w0 = gd.PhaseSpacePosition(pos0, vel0)
    if streamtype=='chen':
        print("running a chen-type particle spray stream")
        df = ms.ChenStreamDF(lead=True, trail=True) # or fardal or something
        prog_mass = Mprog
        prog_pot = gp.PlummerPotential(m=prog_mass,
                                       b=b, units=galactic)


    if streamtype=='streakline':
        print("running a streakline (0-dispersion) stream")
        df = ms.StreaklineStreamDF(lead=True, trail=True)
        prog_mass = Mprog
        prog_pot=None

    

    mwp = gp.BovyMWPotential2014(units=galactic)
    H = gp.Hamiltonian(mwp)

    gen = ms.MockStreamGenerator(df, H, progenitor_potential=prog_pot)
    if save==False:
        stream, prog = gen.run(prog_w0, prog_mass, dt=1*u.Myr, n_steps=int(time),
                                release_every=release_every,
                                progress=True)
    if save==True:
        stream, prog = gen.run(prog_w0, prog_mass, dt=1*u.Myr, n_steps = int(time),
                               release_every=release_every, 
                               output_every=output_every,
                               output_filename=output_filename, overwrite=True,
                               progress=True)

    return stream, prog


def get_streamcoords_ms_streamfile(streamfile, i, init_displacement, output_every=5):
    """
    assumes that i is a TIME in Myr
    """
    
    prog_w = prog_position(init_displacement, i=i)
    prog_pos = prog_w[:3] * (u.kpc)
    prog_vel = prog_w[3:] * (u.km/u.s)
    # concatenate. 
    prog_w_jake_units = np.hstack([
        prog_pos.value, prog_vel.to(u.kpc/u.Myr).value
    ])  



    index = int(i/output_every)
    x, y, z = streamfile['stream']['pos'][:,index] # kpc
    vx, vy, vz = streamfile['stream']['vel'][:,index] # kpc/Myr ? 

    all_pos = np.array([
        x, y, z
    ]).T # should be saved in kpc; i.e., no unit gymnastics necessary.
    all_vel = np.array([
        vx, vy, vz
    ]).T # saved be returned in kpc/myr; i.e., no unit gymnastics necessary.

    all_w = np.hstack([all_pos, all_vel])

    stream_obj = StreamFrame(sim_coords = all_w, prog_sim_coord = prog_w_jake_units)
    coords = stream_obj.GetStreamFrame()

    return coords


def get_streamcoords_ms(stream, i, init_displacement):
    """
    stream coordinates for mock streams at the final time from a stream object from gala. 
    """
    prog_w = prog_position(init_displacement, i=i)
    prog_pos = prog_w[:3] * (u.kpc)
    prog_vel = prog_w[3:] * (u.km/u.s)
    # concatenate. 
    prog_w_jake_units = np.hstack([
        prog_pos.value, prog_vel.to(u.kpc/u.Myr).value
    ])  

    all_pos = np.array([
        stream.pos.x.to(u.kpc).value, stream.pos.y.to(u.kpc).value, stream.pos.z.to(u.kpc).value
    ]).T

    all_vel = np.array([
        stream.vel.d_x.to(u.kpc/u.Myr).value, stream.vel.d_y.to(u.kpc/u.Myr).value, stream.vel.d_z.to(u.kpc/u.Myr).value
    ]).T

    all_w = np.hstack([all_pos, all_vel])

    stream_obj = StreamFrame(sim_coords = all_w, prog_sim_coord = prog_w_jake_units)
    coords = stream_obj.GetStreamFrame()

    return coords