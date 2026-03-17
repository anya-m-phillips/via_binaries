import numpy as np
# import jax
# jax.config.update("jax_enable_x64", True)
# from functools import partial


class StreamFrame:
    """
    Input units must be kpc, kpc/Myr.
    sim_coords is Nx6
    prog_sim_coord is length 6 array
    Output units are deg, mas/yr, km/s
    """
    def __init__(self, sim_coords, prog_sim_coord):
        self.sim_coords = sim_coords
        self.prog_sim_coord = prog_sim_coord
        self.prog_L_sim = np.cross(prog_sim_coord[:3], prog_sim_coord[3:])
        
        self.xs_hat = prog_sim_coord[:3]/np.sqrt(np.sum(prog_sim_coord[:3]**2))
        self.zs_hat = self.prog_L_sim / np.sqrt(np.sum(self.prog_L_sim**2))
        self.ys_hat = np.cross(self.zs_hat, self.xs_hat)
        
        ## This can all be done in a single matmul, but keeping seperated for now...
        ## Get positions and stream-aligned cartesian frame
        self.xs = np.sum(sim_coords[:,:3]*self.xs_hat[None,:],axis=1)
        self.ys = np.sum(sim_coords[:,:3]*self.ys_hat[None,:],axis=1)
        self.zs = np.sum(sim_coords[:,:3]*self.zs_hat[None,:],axis=1)
        self.rs = np.sqrt(self.xs**2 + self.ys**2 + self.zs**2)
        
        self.xyz_s = np.vstack([self.xs,self.ys,self.zs]).T
        
        
        ## Get velocities in stream-aligned cartesian frame
        ## Make all velocities relative to the progenitor
        DeltaV = sim_coords[:,3:] - prog_sim_coord[None,3:]
        self.vxs = np.sum(DeltaV*self.xs_hat[None,:],axis=1)
        self.vys = np.sum(DeltaV*self.ys_hat[None,:],axis=1)
        self.vzs = np.sum(DeltaV*self.zs_hat[None,:],axis=1)
        
        self.vxyz_s = np.vstack([self.vxs,self.vys,self.vzs]).T
        
    # @partial(jax.jit,static_argnums=(0))
    def GetStreamFrame(self):
        phi1 = np.arctan2(self.ys,self.xs)
        phi2 = np.arcsin(self.zs/self.rs)
        dist = self.rs
        
        rs_hat = self.xyz_s/dist[:,None]
        phi1_hat = -np.sin(phi1[:,None])*np.cos(phi2[:,None])*self.xs_hat[None,:] + np.cos(phi1[:,None])*np.cos(phi2[:,None])*self.ys_hat[None,:]
        phi2_hat = -np.cos(phi1[:,None])*np.sin(phi2[:,None])*self.xs_hat[None,:] - np.sin(phi1[:,None])*np.sin(phi2[:,None])*self.ys_hat[None,:] + np.cos(phi2[:,None])*self.zs_hat[None,:]
        
        
        vr = np.sum(self.vxyz_s*rs_hat,axis=1) 
        vr = vr ##- vr.mean() #TODO: should subtract prog velocity, not 
        vphi1 = np.sum(self.vxyz_s*phi1_hat,axis=1)
        vphi2 = np.sum(self.vxyz_s*phi2_hat,axis=1)
        
        pm_phi1 = vphi1/dist #cosphi2*phi1hat
        pm_phi2 = vphi2/dist
        
        rad_per_Myr_to_mas_per_yr = (2.0626481e8)*(1e-6)
        kpc_per_Myr_to_km_per_s = 977.79222
        stream_frame_coords = {'phi1':np.rad2deg(phi1), 
                               'phi2':np.rad2deg(phi2), 
                               'pm_phi1':pm_phi1*rad_per_Myr_to_mas_per_yr, 
                               'pm_phi2':pm_phi2*rad_per_Myr_to_mas_per_yr, 
                               'r':dist, 
                               'vr':vr*kpc_per_Myr_to_km_per_s}
        return stream_frame_coords
       
        
        
        
        
        
        