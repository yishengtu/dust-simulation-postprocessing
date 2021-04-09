import numpy as np
from numba import njit, jit, prange, float64, int64, types, config
from athena_read import athdf
from scipy.interpolate import interp1d, interp2d
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import colors
import os


kb = 1.380649 * 10 ** (-16)  # Boltzmann constant
mH = 1.6733 * 10 ** (-24)    # mass of neutral hydrogen
h = 6.6261 * 10 ** (-27)     # plank's constant
sigma = 5.6704 * 10 ** (-5)  # Stefan-Boltzmann constant
a = 7.5646 * 10 ** (-15)     # radiation constant
m_sun = 1.989 * 10 ** 33     # solar mass
AU = 1.496 * 10 ** 13        # 1AU
year = 31536000              # 1 year = 365 days * 24 hrs / day * 3600 sec / hrs
l_sun = 3.839 * 10 ** 33     # 1 solar luminosity
G = 6.674 * 10 ** (-8)       # gravitational constant
c = 2.99792458 * 10 ** 10    # light speed

mu = 2.33
gamma = 5.0/3.0


@njit
def arctan(x, z):
    if z == 0:
        theta = 0
    elif x > 0 and z >= 0:  # 1st quarter
        theta = np.arctan(x / z)
    elif x > 0 and z < 0:  # 4th quarter
        theta = np.pi + np.arctan(x / z)
    elif x < 0 and z < 0:  # 3rd quarter
        theta = np.pi + np.arctan(x / z)
    else:  # parx < 0 and parz > 0:      # 2nd quarter
        theta = 2 * np.pi + np.arctan(x / z)
    return theta


@njit
def interp(newx, oldx, oldy):
    newy = np.zeros(len(newx))
    for loc in range(0, len(newx), 1):
        x_ind = np.searchsorted(oldx, newx[loc])
        if x_ind >= len(oldx):
            newy[loc] = oldy[-1] + (newx[loc] - oldx[-1]) * (oldy[-1] - oldy[-2]) / (oldx[-1] - oldx[- 2])
        elif x_ind <= 0:
            newy[loc] = oldy[0] + (newx[loc] - oldx[0]) * (oldy[1] - oldy[0]) / (oldx[1] - oldx[0])
        else:
            newy[loc] = oldy[x_ind] + (newx[loc] - oldx[x_ind]) * (oldy[x_ind] - oldy[x_ind-1]) / (oldx[x_ind] - oldx[x_ind - 1])
    return newy


class Util:
    @staticmethod
    def read_input_file(fn):
        f = open(fn, 'r')
        d = {}
        for line in f.readlines():
            split_line = line.split('#')
            key, val = split_line[0].split('=')
            key = key.strip()
            d.update({key: val})
        return d

    @staticmethod
    def write_restart_file_npy(fn, grain_list, ot, ini_time_in_athena, frame_number, hydro_x, hydro_y, hydro_rho):
        np.savez(fn, grain_list=grain_list, time_now=ot, ini_time_in_athena=ini_time_in_athena, frame_number=frame_number,
                 hydro_xmesh = hydro_x, hydro_ymesh = hydro_y, hydro_rhomesh = hydro_rho)

    @staticmethod
    def read_restart_file_npy(fn):
        x = np.load(fn, mmap_mode='r')
        print ('#RESTART# reading ' + fn)
        time_now = x['time_now']
        ini_time_in_athena = x['ini_time_in_athena']
        frame_number = x['frame_number']
        grain_list = x['grain_list']
        hydro_xmesh = x['hydro_xmesh']
        hydro_ymesh = x['hydro_ymesh']
        hydro_rhomesh = x['hydro_rhomesh']
        return time_now, ini_time_in_athena, frame_number, grain_list, hydro_xmesh, hydro_ymesh, hydro_rhomesh

    @staticmethod
    def choosenumber(num):
        if num < 10:
            return '0000' + str(num)
        elif num < 100:
            return '000' + str(num)
        elif num < 1000:
            return '00' + str(num)
        elif num < 10000:
            return '0' + str(num)
        elif num < 100000:
            return str(num)
        else:
            raise ValueError

    @staticmethod
    def makeplot(grain_list, x_mesh, y_mesh, rho_mesh, fn_root_pars, frame_number, index_of_mesh, time_in_sim, ot, size_of_marker = 1, show = False):
        length_scale = 10 ** 4 * AU
        grain_x_list, grain_y_list, grain_z_list, grain_vx_list, grain_vy_list, grain_vz_list, s_list, rho_list, partype_list = np.transpose(grain_list)
        grain_x_list = [x / length_scale for x in grain_x_list]
        grain_y_list = [y / length_scale for y in grain_y_list]
        grain_z_list = [z / length_scale for z in grain_z_list]
        grain_xy_list = [np.sqrt(x ** 2 + y ** 2) for x, y in zip(grain_x_list, grain_y_list)]

        fig = plt.figure()
        fig.set_size_inches(20, 6)
        dim = (10, 27)
        ax0 = plt.subplot2grid(dim, (0, 0), colspan=15, rowspan=10)
        ax1 = plt.subplot2grid(dim, (0, 17), colspan=10, rowspan=10)
        c0 = ax0.pcolor(np.array(x_mesh) / (1e4 * AU),
                        np.array(y_mesh) / (1e4 * AU),
                        rho_mesh[index_of_mesh], cmap='rainbow', norm=colors.LogNorm(1e-19, 1e-13))  # , edgecolors='k')
        c1 = ax1.pcolor(np.array(x_mesh) / (1e4 * AU),
                        np.array(y_mesh) / (1e4 * AU),
                        rho_mesh[index_of_mesh], cmap='rainbow', norm=colors.LogNorm(1e-19, 1e-13))  # , edgecolors='k')
        # c1 = ax1.pcolor(orig_sim.x_mesh, orig_sim.y_mesh, orig_sim.cgs_rho_mesh, cmap='rainbow', norm=colors.LogNorm(1e-19, 1e-13)) #, edgecolors='k')
        ax0.scatter(grain_xy_list, grain_z_list, color='black', marker='o', alpha=0.8, s=size_of_marker)
        ax1.scatter(grain_xy_list, grain_z_list, color='black', marker='o', alpha=0.8, s=size_of_marker)
        # print ('1')
        # print (grain_x_list)
        # print (grain_y_list)
        # print (grain_z_list)
        # print ('2')

        ax0.set_xlim(0, 0.021)
        ax0.set_ylim(-0.007, 0.007)
        ax0.set_aspect('equal')
        # ax1.set_xlim(0, 0.04)
        # ax1.set_ylim(-0.02, 0.02)
        ax0.set_title('time from start: ' + str(round(time_in_sim*1e6, 2)) + ' year')
        ax1.set_title('time from restart: ' + str(round(ot / year, 2)) + ' year')
        ax1.set_aspect('equal')
        axins = inset_axes(ax0,
                           width="5%",  # width = 5% of parent_bbox width
                           height="100%",  # height : 100%
                           loc='lower left',
                           bbox_to_anchor=(1.02, 0., 1, 1),
                           bbox_transform=ax0.transAxes,
                           borderpad=0,
                           )
        fig.colorbar(c0, cax=axins)
        axins = inset_axes(ax1,
                           width="5%",  # width = 5% of parent_bbox width
                           height="100%",  # height : 100%
                           loc='lower left',
                           bbox_to_anchor=(1.02, 0., 1, 1),
                           bbox_transform=ax1.transAxes,
                           borderpad=0,
                           )
        fig.colorbar(c1, cax=axins)
        if show:
            plt.show()
        else:
            plt.savefig(fn_root_pars + 'frame' + Util.choosenumber(frame_number) + '.png', bbox_inches='tight')
        plt.close()


class OutputAthdf():
    def __init__(self, fn):
        print('reading ' + fn)
        full_data = athdf(fn)
        self.hydro_time = full_data['Time']

        # setup location meshes and find data location
        r_list, theta_list, phi_list = full_data['x1f'], full_data['x2f'], full_data['x3f']
        dataloc_r_list = [(r_list[i] + r_list[i + 1]) / 2.0 for i in range(0, len(r_list) - 1, 1)]
        cgs_dataloc_r_list = [x * 10 ** 4 * AU for x in dataloc_r_list]
        dataloc_theta_list = [(theta_list[i] + theta_list[i + 1]) / 2.0 for i in range(0, len(theta_list) - 1, 1)]

        rho_mesh = full_data['rho'][0]
        pres_mesh = full_data['press'][0]
        Phi_mesh = full_data['Phi'][0]  # I guess the Phi is gravitational potential
        vr_mesh = full_data['vel1'][0]
        vtheta_mesh = full_data['vel2'][0]
        vphi_mesh = full_data['vel3'][0]

        cgs_rho_mesh = [[x * m_sun / (10 ** 4 * AU) ** 3 for x in y] for y in rho_mesh]
        cgs_pres_mesh = [[x * 1.33687 * 10 ** (-11) for x in y] for y in pres_mesh]
        cgs_Phi_mesh = [[x * (10 ** 4 * AU) ** 2 / (10 ** 6 * year) ** 2 for x in Phi_list] for Phi_list in Phi_mesh]
        cgs_vr_mesh = [[x * 4743.78 for x in y] for y in vr_mesh]
        cgs_vtheta_mesh = [[x * 4743.78 for x in y] for y in vtheta_mesh]
        cgs_vphi_mesh = [[x * 4743.78 for x in y] for y in vphi_mesh]
        cgs_temp_mesh = [[self.__temperature(rho, pres) for rho, pres in zip(rho_list, pres_list)] for rho_list, pres_list in zip(cgs_rho_mesh, cgs_pres_mesh)]
        phi_theta_gradient_mesh = np.transpose([- np.gradient(phitheta, dataloc_theta_list) for phitheta in np.transpose(cgs_Phi_mesh)])
        grav_r_mesh = np.array([- np.gradient(phir, cgs_dataloc_r_list) for phir in cgs_Phi_mesh])
        grav_theta_mesh = np.array([[dtheta * 1.0 / r for dtheta, r in zip(dt_l, cgs_dataloc_r_list)] for dt_l in phi_theta_gradient_mesh])
        grav_phi_mesh = np.array([[0.0 for r in cgs_dataloc_r_list] for t in dataloc_theta_list])

        self.rho_fun = interp2d(x=cgs_dataloc_r_list, y=dataloc_theta_list, z=cgs_rho_mesh, kind='cubic')
        self.pres_fun = interp2d(x=cgs_dataloc_r_list, y=dataloc_theta_list, z=cgs_pres_mesh, kind='cubic')
        self.Phi_fun = interp2d(x=cgs_dataloc_r_list, y=dataloc_theta_list, z=cgs_Phi_mesh, kind='cubic')
        self.vr_fun = interp2d(x=cgs_dataloc_r_list, y=dataloc_theta_list, z=cgs_vr_mesh, kind='cubic')
        self.vtheta_fun = interp2d(x=cgs_dataloc_r_list, y=dataloc_theta_list, z=cgs_vtheta_mesh, kind='cubic')
        self.vphi_fun = interp2d(x=cgs_dataloc_r_list, y=dataloc_theta_list, z=cgs_vphi_mesh, kind='cubic')
        self.temp_fun = interp2d(x=cgs_dataloc_r_list, y=dataloc_theta_list, z=cgs_temp_mesh, kind='cubic')
        self.grav_r_fun = interp2d(x=cgs_dataloc_r_list, y=dataloc_theta_list, z=grav_r_mesh, kind='cubic')
        self.grav_theta_fun = interp2d(x=cgs_dataloc_r_list, y=dataloc_theta_list, z=grav_theta_mesh, kind='cubic')
        self.grav_phi_fun = interp2d(x=cgs_dataloc_r_list, y=dataloc_theta_list, z=grav_phi_mesh, kind='cubic')

    def __temperature(self, rho, pres):
        return pres * mu * mH / (rho * kb)

    def __internal_energy_volume(self, rho, pres):
        return pres / (gamma - 1.0)

    def __internal_energy_mass(self, rho, pres):
        return 1.0 / (gamma - 1.0) * pres / rho

    def __sound_speed(self, rho, pres):
        return (gamma * pres / rho) ** 0.5


class SphericalPolarCoord():
    '''

        self.rho_mesh_fun =        np.array([[interp1d(self.hydro_time_list, info, kind=kind, fill_value='extrapolate') for info in info_l] for info_l in self.rho_mesh_with_time])
        self.temp_mesh_fun =       np.array([[interp1d(self.hydro_time_list, info, kind=kind, fill_value='extrapolate') for info in info_l] for info_l in self.temp_mesh_with_time])
        self.vr_mesh_fun =         np.array([[interp1d(self.hydro_time_list, info, kind=kind, fill_value='extrapolate') for info in info_l] for info_l in self.vr_mesh_with_time])
        self.vtheta_mesh_fun =     np.array([[interp1d(self.hydro_time_list, info, kind=kind, fill_value='extrapolate') for info in info_l] for info_l in self.vtheta_mesh_with_time])
        self.vphi_mesh_fun =       np.array([[interp1d(self.hydro_time_list, info, kind=kind, fill_value='extrapolate') for info in info_l] for info_l in self.vphi_mesh_with_time])
        self.grav_r_mesh_fun =     np.array([[interp1d(self.hydro_time_list, info, kind=kind, fill_value='extrapolate') for info in info_l] for info_l in self.grav_r_mesh_with_time])
        self.grav_theta_mesh_fun = np.array([[interp1d(self.hydro_time_list, info, kind=kind, fill_value='extrapolate') for info in info_l] for info_l in self.grav_theta_mesh_with_time])
        self.grav_phi_mesh_fun =   np.array([[interp1d(self.hydro_time_list, info, kind=kind, fill_value='extrapolate') for info in info_l] for info_l in self.grav_phi_mesh_with_time])

    def set_hydro_with_time(self, new_time_list):
        rho_mesh =        np.array([[func(new_time_list) for func in func_l] for func_l in self.rho_mesh_fun])
        temp_mesh =       np.array([[func(new_time_list) for func in func_l] for func_l in self.temp_mesh_fun])
        vr_mesh =         np.array([[func(new_time_list) for func in func_l] for func_l in self.vr_mesh_fun])
        vtheta_mesh =     np.array([[func(new_time_list) for func in func_l] for func_l in self.vtheta_mesh_fun])
        vphi_mesh =       np.array([[func(new_time_list) for func in func_l] for func_l in self.vphi_mesh_fun])
        grav_r_mesh =     np.array([[func(new_time_list) for func in func_l] for func_l in self.grav_r_mesh_fun])
        grav_theta_mesh = np.array([[func(new_time_list) for func in func_l] for func_l in self.grav_theta_mesh_fun])
        grav_phi_mesh =   np.array([[func(new_time_list) for func in func_l] for func_l in self.grav_phi_mesh_fun])
    '''
    def __init__(self, rmin, rmax, nr, thetamin, thetamax, ntheta):
        self.mu = mu
        self.gamma = gamma

        self.rmin, self.rmax, self.nr, self.thetamin, self.thetamax, self.ntheta = rmin, rmax, nr, thetamin, thetamax, ntheta

        self.r_list = np.logspace(np.log10(rmin), np.log10(rmax), nr + 1)
        self.theta_list = np.linspace(thetamin, thetamax, ntheta + 1)

        self.dataloc_r_list = [(self.r_list[i] + self.r_list[i + 1]) / 2.0 for i in range(0, len(self.r_list) - 1, 1)]
        self.dataloc_theta_list = [(self.theta_list[i] + self.theta_list[i + 1]) / 2.0 for i in range(0, len(self.theta_list) - 1, 1)]

        self.dr_list = np.array([self.r_list[i + 1] - self.r_list[i] for i in range(0, len(self.r_list) - 1, 1)])
        self.dtheta_list = np.array([self.theta_list[i + 1] - self.theta_list[i] for i in range(0, len(self.theta_list) - 1, 1)])

        self.y_mesh = np.array([[r * np.cos(theta) for r in self.r_list] for theta in self.theta_list])
        self.x_mesh = np.array([[r * np.sin(theta) for r in self.r_list] for theta in self.theta_list])

        self.rho_mesh = None
        self.temp_mesh = None
        self.vr_mesh = None
        self.vtheta_mesh = None
        self.vphi_mesh = None

        self.grav_r_mesh = None
        self.grav_theta_mesh = None
        self.grav_phi_mesh = None

        self.hydro_time_list = []
        self.rho_mesh_with_time = [[[] for r in self.dataloc_r_list] for theta in self.dataloc_theta_list]  # initialize
        self.temp_mesh_with_time = [[[] for r in self.dataloc_r_list] for theta in self.dataloc_theta_list]  # initialize
        self.vr_mesh_with_time = [[[] for r in self.dataloc_r_list] for theta in self.dataloc_theta_list]  # initialize
        self.vtheta_mesh_with_time = [[[] for r in self.dataloc_r_list] for theta in self.dataloc_theta_list]  # initialize
        self.vphi_mesh_with_time = [[[] for r in self.dataloc_r_list] for theta in self.dataloc_theta_list]  # initialize
        self.grav_r_mesh_with_time = [[[] for r in self.dataloc_r_list] for theta in self.dataloc_theta_list]  # initialize
        self.grav_theta_mesh_with_time = [[[] for r in self.dataloc_r_list] for theta in self.dataloc_theta_list]  # initialize
        self.grav_phi_mesh_with_time = [[[] for r in self.dataloc_r_list] for theta in self.dataloc_theta_list]  # initialize

    def set_hydro(self, outputathdf_obj):
        self.rho_mesh = np.array([[outputathdf_obj.rho_fun(r, theta)[0] for r in self.dataloc_r_list] for theta in self.dataloc_theta_list])
        self.temp_mesh = np.array([[outputathdf_obj.temp_fun(r, theta)[0] for r in self.dataloc_r_list] for theta in self.dataloc_theta_list])
        self.vr_mesh = np.array([[outputathdf_obj.vr_fun(r, theta)[0] for r in self.dataloc_r_list] for theta in self.dataloc_theta_list])
        self.vtheta_mesh = np.array([[outputathdf_obj.vtheta_fun(r, theta)[0] for r in self.dataloc_r_list] for theta in self.dataloc_theta_list])
        self.vphi_mesh = np.array([[outputathdf_obj.vphi_fun(r, theta)[0] for r in self.dataloc_r_list] for theta in self.dataloc_theta_list])
        Phi_mesh = np.array([[outputathdf_obj.Phi_fun(r, theta)[0] for r in self.dataloc_r_list] for theta in self.dataloc_theta_list])

        self.grav_r_mesh = np.array([- np.gradient(phir, self.dataloc_r_list) for phir in Phi_mesh])
        phi_theta_gradient_mesh = np.transpose([- np.gradient(phitheta, self.dataloc_theta_list) for phitheta in np.transpose(Phi_mesh)])
        self.grav_theta_mesh = np.array([[dtheta * 1.0 / r for dtheta, r in zip(dt_l, self.dataloc_r_list)] for dt_l in phi_theta_gradient_mesh])
        self.grav_phi_mesh = np.array([[0.0 for r in self.dataloc_r_list] for t in self.dataloc_theta_list])

    def setup_hydro_with_time(self, athdf_obj_list):
        hydro_time_list = []
        rho_mesh_with_time = [[[] for r in self.dataloc_r_list] for theta in self.dataloc_theta_list]  # initialize
        temp_mesh_with_time = [[[] for r in self.dataloc_r_list] for theta in self.dataloc_theta_list]  # initialize
        vr_mesh_with_time = [[[] for r in self.dataloc_r_list] for theta in self.dataloc_theta_list]  # initialize
        vtheta_mesh_with_time = [[[] for r in self.dataloc_r_list] for theta in self.dataloc_theta_list]  # initialize
        vphi_mesh_with_time = [[[] for r in self.dataloc_r_list] for theta in self.dataloc_theta_list]  # initialize
        grav_r_mesh_with_time = [[[] for r in self.dataloc_r_list] for theta in self.dataloc_theta_list]  # initialize
        grav_theta_mesh_with_time = [[[] for r in self.dataloc_r_list] for theta in self.dataloc_theta_list]  # initialize
        grav_phi_mesh_with_time = [[[] for r in self.dataloc_r_list] for theta in self.dataloc_theta_list]  # initialize
        for athdf_obj in athdf_obj_list:
            hydro_time_list.append(athdf_obj.hydro_time)        # time
            for ri in range(0, len(self.dataloc_r_list), 1):
                for thetai in range(0, len(self.dataloc_theta_list), 1):
                    r, theta = self.dataloc_r_list[ri], self.dataloc_theta_list[thetai]
                    # append data
                    rho_mesh_with_time[thetai][ri].append(athdf_obj.rho_fun(r, theta)[0])
                    temp_mesh_with_time[thetai][ri].append(athdf_obj.temp_fun(r, theta)[0])
                    vr_mesh_with_time[thetai][ri].append(athdf_obj.vr_fun(r, theta)[0])
                    vtheta_mesh_with_time[thetai][ri].append(athdf_obj.vtheta_fun(r, theta)[0])
                    vphi_mesh_with_time[thetai][ri].append(athdf_obj.vphi_fun(r, theta)[0])

                    grav_r_mesh_with_time[thetai][ri].append(athdf_obj.grav_r_fun(r, theta)[0])
                    grav_theta_mesh_with_time[thetai][ri].append(athdf_obj.grav_theta_fun(r, theta)[0])
                    grav_phi_mesh_with_time[thetai][ri].append(athdf_obj.grav_phi_fun(r, theta)[0])

        self.hydro_time_list = np.array(hydro_time_list)
        self.rho_mesh_with_time = np.array(rho_mesh_with_time)
        self.temp_mesh_with_time = np.array(temp_mesh_with_time)
        self.vr_mesh_with_time = np.array(vr_mesh_with_time)
        self.vtheta_mesh_with_time = np.array(vtheta_mesh_with_time)
        self.vphi_mesh_with_time = np.array(vphi_mesh_with_time)
        self.grav_r_mesh_with_time = np.array(grav_r_mesh_with_time)
        self.grav_theta_mesh_with_time = np.array(grav_theta_mesh_with_time)
        self.grav_phi_mesh_with_time = np.array(grav_phi_mesh_with_time)

    def set_hydro_with_time(self, new_time_list):
        rho_mesh  = set_hydro_with_time_docker(new_time_list, self.hydro_time_list, self.rho_mesh_with_time)
        temp_mesh = set_hydro_with_time_docker(new_time_list, self.hydro_time_list, self.temp_mesh_with_time)
        vr_mesh   = set_hydro_with_time_docker(new_time_list, self.hydro_time_list, self.vr_mesh_with_time)
        vtheta_mesh = set_hydro_with_time_docker(new_time_list, self.hydro_time_list, self.vtheta_mesh_with_time)
        vphi_mesh   = set_hydro_with_time_docker(new_time_list, self.hydro_time_list, self.vphi_mesh_with_time)
        grav_r_mesh = set_hydro_with_time_docker(new_time_list, self.hydro_time_list, self.grav_r_mesh_with_time)
        grav_theta_mesh = set_hydro_with_time_docker(new_time_list, self.hydro_time_list, self.grav_theta_mesh_with_time)
        grav_phi_mesh   = set_hydro_with_time_docker(new_time_list, self.hydro_time_list, self.grav_phi_mesh_with_time)

        self.rho_mesh = reorder_matrix(rho_mesh)
        self.temp_mesh = reorder_matrix(temp_mesh)
        self.vr_mesh = reorder_matrix(vr_mesh)
        self.vtheta_mesh = reorder_matrix(vtheta_mesh)
        self.vphi_mesh = reorder_matrix(vphi_mesh)
        self.grav_r_mesh = reorder_matrix(grav_r_mesh)
        self.grav_theta_mesh = reorder_matrix(grav_theta_mesh)
        self.grav_phi_mesh = reorder_matrix(grav_phi_mesh)


@njit(parallel=True)
def set_hydro_with_time_docker(new_time, hydro_time, hydro_mesh):
    new_mesh = np.zeros((len(hydro_mesh), len(hydro_mesh[0]), len(new_time)))
    for ii in prange(0, len(hydro_mesh), 1):
        for jj in range(0, len(hydro_mesh[0]), 1):
            new_mesh[ii, jj] = interp(new_time, hydro_time, hydro_mesh[ii, jj])
    return new_mesh


@njit(parallel=True)
def reorder_matrix(old_matrix):
    t_i, t_j, t_k = len(old_matrix), len(old_matrix[0]), len(old_matrix[0][0])
    m_new = np.zeros((t_k, t_i, t_j))
    for i in prange(0, t_i, 1):
        for j in range(0, t_j, 1):
            for k in range(0, t_k, 1):
                m_new[k, i, j] = old_matrix[i, j, k]
    return m_new


@njit
def TSC(x, y, z, r_list, dr_list, theta_list, dtheta_list):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = arctan(abs(np.sqrt(x ** 2 + y ** 2)), z)
    i, j = int(np.searchsorted(r_list, r)) - 1, int(np.searchsorted(theta_list, theta)) - 1
    x1 = (r - r_list[i]) / dr_list[min(i, len(dr_list)-1)]
    x2 = (theta - theta_list[j]) / dtheta_list[j]
    wx1l, wx1r = 0.5 * (1 - x1) ** 2, 0.5 * x1 ** 2
    wx1m = 1 - (wx1l + wx1r)
    wx2l, wx2r = 0.5 * (1 - x2) ** 2, 0.5 * x2 ** 2
    wx2m = 1 - (wx2l + wx2r)
    wx1 = np.array([wx1l, wx1m, wx1r]).reshape(3, 1)
    wx2 = np.array([wx2l, wx2m, wx2r]).reshape(1, 3)
    w = wx1 * wx2

    return w, i, j


@njit# (locals={'w': types.float64[:,:], 'i': int64, 'j': int64, 'mesh': types.pyobject[:,:], 'nr': int64, 'ntheta': int64})
def apply_TSC(w, i, j, mesh, nr, ntheta):
    # return w[0, 0] * mesh[j - 1, i - 1] + w[0, 1] * mesh[j, i - 1] + w[0, 2] * mesh[j + 1, i - 1] + \
    #        w[1, 0] * mesh[j - 1, i]     + w[1, 1] * mesh[j, i]     + w[1, 2] * mesh[j + 1, i] + \
    #        w[2, 0] * mesh[j - 1, i + 1] + w[2, 1] * mesh[j, i - 1] + w[2, 2] * mesh[j + 1, i + 1]
    summed_quan = 0.0
    for yi in range(-1, 2, 1):             # theta
        index_y = j + yi
        if index_y < 0:
            index_y = 0
        elif index_y > ntheta - 2:
            index_y = ntheta - 1
        for xi in range(-1, 2, 1):         # r
            index_x = i + xi
            if index_x < 0:
                index_x = 0
            elif index_x > nr - 2:
                index_x = nr - 1
            summed_quan += mesh[index_y, index_x] * w[xi+1, yi+1]
    return summed_quan


@njit(parallel=True)
def move(grain_list, min_stoppingtime, min_dt, dt_max_over_min, ot, output_dt,
         r_list, dr_list, nr, theta_list, dtheta_list, ntheta,
         vr_mesh, vtheta_mesh, vphi_mesh, rho_mesh, temp_mesh, grav_r_mesh, grav_theta_mesh, grav_phi_mesh):
    for ind in prange(0, len(grain_list), 1):
        grain_time_now = ot
        counter = 0     # to avoid numerical issues with int(t/dt)
        while grain_time_now < ot + output_dt:
            x, y, z, vx, vy, vz, s, rho, particletype = grain_list[ind]
            w, i, j = TSC(x, y, z, r_list, dr_list, theta_list, dtheta_list)
            gas_vr = apply_TSC(w, i, j, vr_mesh[counter], nr, ntheta)
            gas_vtheta = apply_TSC(w, i, j, vtheta_mesh[counter], nr, ntheta)
            gas_vphi = apply_TSC(w, i, j, vphi_mesh[counter], nr, ntheta)
            gas_rho = apply_TSC(w, i, j, rho_mesh[counter], nr, ntheta)
            gas_T = apply_TSC(w, i, j, temp_mesh[counter], nr, ntheta)
            grav_r = apply_TSC(w, i, j, grav_r_mesh[counter], nr, ntheta)
            grav_theta = apply_TSC(w, i, j, grav_theta_mesh[counter], nr, ntheta)
            grav_phi = apply_TSC(w, i, j, grav_phi_mesh[counter], nr, ntheta)
            r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            theta = np.arccos(z / r)
            phi = arctan(y, x)
            gas_vx = gas_vr * np.sin(theta) * np.cos(phi) + gas_vtheta * np.cos(theta) * np.cos(phi) - gas_vphi * np.sin(phi)
            gas_vy = gas_vr * np.sin(theta) * np.sin(phi) + gas_vtheta * np.cos(theta) * np.sin(phi) + gas_vphi * np.cos(phi)
            gas_vz = gas_vr * np.cos(theta) - gas_vtheta * np.sin(theta)
            if particletype < 0.5:
                grav_x = (grav_r * np.sin(theta) * np.cos(phi) + grav_theta * np.cos(theta) * np.cos(phi) - grav_phi * np.sin(phi))
                grav_y = (grav_r * np.sin(theta) * np.sin(phi) + grav_theta * np.cos(theta) * np.sin(phi) + grav_phi * np.cos(phi))
                grav_z = (grav_r * np.cos(theta) - grav_theta * np.sin(theta))

                cs = np.sqrt(gamma * kb * gas_T / (mu * mH))
                dvx, dvy, dvz = gas_vx - vx, gas_vy - vy, gas_vz - vz

                ts = max(min_stoppingtime, (rho * s) / (gas_rho * cs))  # stopping time
                # adaptive time step, speed up simulation when stopping time is very large.
                ts_over_dt = min(len(rho_mesh) - counter, int(ts/min_dt), dt_max_over_min)    # set a maximum stopping time to avoid overshooting, and make sure all grain have the same last frame time
                if ts_over_dt > 1:
                    dt_adj = min_dt * ts_over_dt
                    counter += 1 * ts_over_dt
                else:
                    dt_adj = min_dt
                    counter += 1
                ax_drag = 1.0 / ts * dvx + grav_x
                ay_drag = 1.0 / ts * dvy + grav_y
                az_drag = 1.0 / ts * dvz + grav_z

                grain_list[ind, 0] += vx * dt_adj + 0.5 * ax_drag * dt_adj ** 2
                grain_list[ind, 1] += vy * dt_adj + 0.5 * ay_drag * dt_adj ** 2
                grain_list[ind, 2] += vz * dt_adj + 0.5 * az_drag * dt_adj ** 2
                grain_list[ind, 3] += ax_drag * dt_adj
                grain_list[ind, 4] += ay_drag * dt_adj
                grain_list[ind, 5] += az_drag * dt_adj

                grain_time_now += dt_adj
            elif 0.5 <= particletype < 1.5:
                grain_list[ind, 0] += vx * min_dt        # x
                grain_list[ind, 1] += vy * min_dt        # y
                grain_list[ind, 2] += vz * min_dt        # z
                grain_list[ind, 3] = gas_vx          # vx
                grain_list[ind, 4] = gas_vy          # vy
                grain_list[ind, 5] = gas_vz          # vz

                grain_time_now += dt_max_over_min
            else:
                grain_list[len(grain_list) + 100]   # only here to raise an error, since type>1.5 is not implimanted.

        # grain_list[ind] = np.array([x, y, z, vx, vy, vz, s, rho])
    return grain_list


def init_grain_square():
    grain_init_loc_list = []
    for x in np.linspace(50 * AU, 250 * AU, 100):
        for z in np.linspace(-100 * AU, 100 * AU, 100):
            y = 0
            grain_init_loc_list.append([x, y, z])
    return grain_init_loc_list


def init_grain(npar, x1min, x1max, nx1, x2min, x2max, nx2, rho_fun):
    i, di = np.linspace(0.0, 1.0, nx1 + 1, retstep=True)
    x1f = x1min * (x1max / x1min) ** i

    j, dj = np.linspace(0.0, 1.0, nx2 + 1, retstep=True)
    x2f = x2min + (x2max - x2min) * j
    n = len(x1f)
    m = len(x2f)

    x1loc = [(x1f[i+1]+x1f[i])/2.0 for i in range(0, len(x1f)-1, 1)]
    x2loc = [(x2f[i+1]+x2f[i])/2.0 for i in range(0, len(x2f)-1, 1)]
    rho = [[rho_fun(x1, x2)[0] for x2 in x2loc] for x1 in x1loc]
    # first, compute the mass of each cell and the total mass
    x1fcube = x1f**3
    cosx2f = np.cos(x2f)
    x1fcube_ = x1fcube.reshape(n, 1)
    cosx2f_ = cosx2f.reshape(1, m)
    dvol = (2 * np.pi / 3 * (x1fcube_[1:, :] - x1fcube_[:-1, :]) * (cosx2f_[:, :-1] - cosx2f_[:, 1:]))
    dm = rho * dvol
    mtot = np.sum(dm)

    # the first random variable, uniform from 0 to 1, decide the radius
    # since the ordering of particles is not important, pick one such that xi is increasing
    xi = (np.arange(npar) + 0.5) / npar

    # can use random instead, but sorting is slow, pick one
    # xi = np.random.rand(npar)
    # xi.sort()

    # the second random variable, also uniform from 0 to 1, decide the theta
    # do not sort this since the ordering is picked above
    zeta = np.random.rand(npar)

    # the output
    x1p = np.empty(npar)
    x2p = np.empty(npar)

    # now, loop through the hydro shells since xi is sorted and the particles have increasing radii
    p = 0   # particle index
    mr = 0
    for i in range(1, n):
        mc = np.zeros(m)
        mc[1:] = np.cumsum(dm[i-1, :])  # cumulative mass in the theta direction of the shell
        ml, mr = mr, mr + mc[-1]        # enclosed mass up to the left and right boundary of the shell
        # print (mr)

        # do the following for all particles within this shell
        while p < npar and xi[p] * mtot < mr:
            # interpolate between ml and mr to get the radius, mass proportional to r^3 so use x1fcube instead of x1f
            x1p[p] = (x1fcube[i-1] + (x1fcube[i] - x1fcube[i-1]) / (mr - ml) * (xi[p] * mtot - ml))**(1/3)
            # find the index in the theta direction and then interpolate, mass roportional to cos(theta)
            j = np.searchsorted(mc, zeta[p] * mc[-1])
            x2p[p] = np.arccos(cosx2f[j-1] + (cosx2f[j] - cosx2f[j-1]) / (mc[j] - mc[j-1]) * (zeta[p] * mc[-1] - mc[j-1]))
            p += 1

    x1r = x1p * np.sin(x2p)     # convert from spherical to cartesian
    x2r = x1p * np.cos(x2p)     # convert from spherical to cartesian
    return np.transpose([x1r, np.zeros(len(x1p)), x2r])


