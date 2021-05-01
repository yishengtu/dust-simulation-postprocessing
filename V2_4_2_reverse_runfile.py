from V2_4_2_reverse import kb, mH, h, sigma, a, m_sun, AU, year, l_sun, G, c, mu, gamma
from V2_4_2_reverse import Util, OutputAthdf, SphericalPolarCoord
from V2_4_2_reverse import arctan, TSC, apply_TSC, move #, init_grain
import numpy as np
import time
from numba import config
import os


def init_grain(rmin, rmax, nr, thetamin, thetamax, ntheta):
    grain_loc_list = np.zeros((nr * ntheta, 3))
    r_lst = np.logspace(np.log10(rmin), np.log10(rmax), nr, endpoint=True)
    theta_lst = np.linspace(thetamin, thetamax, ntheta, endpoint=True)
    for ri in range(0, len(r_lst), 1):
        for thetai in range(0, len(theta_lst), 1):
            grain_loc_list[ri * ntheta + thetai, 0] = r_lst[ri] * np.sin(theta_lst[thetai])
            grain_loc_list[ri * ntheta + thetai, 2] = r_lst[ri] * np.cos(theta_lst[thetai])

    return grain_loc_list


if __name__ == '__main__':
    start_time = time.time()
    # fn_root_athdf = '/scratch/yt2cr/athena_rundir/test_sim4/small_hole_multiSpecies_morepar/out/'
    # fn_root_pars = '/scratch/yt2cr/vortex/simulation_hydro/test1/'
    # orig_sim = OutputAthdf(fn=fn_root_athdf + 'athdf/HotCore.out1.00120.athdf')
    # # dust_par = OutputTracer(fn_root=fn_root_athdf + 'tracer/', fn='HotCore.out2.00120.npy', coord='cartesian', nspecies=1)
    # hist_inf = OutputHistory(fn=fn_root_athdf + 'HotCore.hst')
    # input_params = FileFunctions.read_input_file(fn=fn_root_pars + 'input.dat')

    # fn_root_pars = './'
    fn_root_pars = 'H:\\StarFormation\\test_sim6_3\\0.1l_more_frames\\01_dust_reverse\\'
    input_params = Util.read_input_file(fn=fn_root_pars + 'input.dat')
    s =                float(input_params["s"])
    rho =              float(input_params["rho"])
    time_scale =       float(input_params["time_unit"])
    length_scale =     float(input_params["length_unit"])
    dt_min =           float(input_params["dt_min"]) * time_scale           # minimum dt in simulation, also determines hydro frame rate. If tracer particles this will be the dt.
    dt_max =           float(input_params["dt_max"]) * time_scale           # maximum dt in simulation.
    output_dt =        float(input_params["output_dt"]) * time_scale
    T =                float(input_params["total_time"]) * time_scale
    min_stoppingtime = float(input_params["min_stopping_time"]) * time_scale
    particle_type =    int(input_params["particle_type"])
    fill_value =       int(input_params["fill_value"])
    num_core =         int(input_params["num_core"])
    start_file =       str(input_params["start_fn"]).strip()
    restart =          bool(int(input_params["restart"]))
    restart_file =     str(input_params["restart_fn"]).strip()
    orig_sim_dir =     str(input_params["hydro_dir"]).strip()
    num_hydro_load =   int(input_params["num_hydro_load"])
    par_rmin =         float(input_params["par_rmin"]) * length_scale
    par_rmax =         float(input_params["par_rmax"]) * length_scale
    par_thetamin =     float(input_params["par_thetamin"])
    par_thetamax =     float(input_params["par_thetamax"])
    par_nr =           int(input_params["par_nr"])
    par_ntheta =       int(input_params["par_ntheta"])

    dt_max_over_min = max(1, int(dt_max/dt_min))

    if output_dt%dt_min != 0.0:
        raise ValueError('output_dt must be divisible by dt')

    config.NUMBA_NUM_THREADS = num_core
    print ('number of cores using = ' + str(config.NUMBA_NUM_THREADS), flush=True)
    grid = SphericalPolarCoord()

    # setup a pseudo-hydro background for initialization
    athdf_output_fn_list = os.listdir(orig_sim_dir)
    athdf_output_fn_list.sort()      # sort so once read = True, it reads all frames after restart_file
    athdf_output_fn_list.reverse()   # reverse the list so running backwards

    # print (athdf_output_fn_list)
    athdf_obj_list = []
    while len(athdf_output_fn_list) > 0 :
        if start_file not in athdf_output_fn_list[0]:
            athdf_output_fn_list.pop(0)
        else:
            break
    athdf_obj_list = [OutputAthdf(orig_sim_dir + athdf_output_fn_list[0]), OutputAthdf(orig_sim_dir + athdf_output_fn_list[1])]
    ini_time_in_athena = athdf_obj_list[1].hydro_time         # initial time in starting frame of athena

    nr = len(athdf_obj_list[0].cgs_r_list) - 1
    ntheta = len(athdf_obj_list[0].theta_list) - 1
    grid.setup_grid(rmin=min(athdf_obj_list[0].cgs_r_list),
                    rmax=max(athdf_obj_list[0].cgs_r_list),
                    nr=nr,
                    thetamin=min(athdf_obj_list[0].theta_list),
                    thetamax=max(athdf_obj_list[0].theta_list),
                    ntheta=ntheta
                    )
    if restart:
        ot, ini_time_in_athena_check, frame_number, grain_list, hydro_x, hydro_y, hydro_rho, hydro_temp = Util.read_restart_file_npy(fn_root_pars + restart_file)    # read in particle locations
        if ini_time_in_athena != ini_time_in_athena_check:
            raise ValueError('In input.dat start_fn must be the same as the frame used when first started the simulation. Initial time in athena is ' + str(ini_time_in_athena_check))
        time_in_sim = (ot / (1.0e6 * year)) + ini_time_in_athena                   # in athena computation unit
        last_hydro_time = time_in_sim
    else:
        # setup a pseudo-hydro background for initialization
        last_hydro_time = athdf_obj_list[1].hydro_time                  # second to last hydro time

        # store all athena outputs in the object so that later grid.set_hydro_with_time() can be called to set hydro information within the specified time range
        grid.setup_hydro_with_time(athdf_obj_list=athdf_obj_list)       # read in and store info for later use
        # Initialize particle locations and speeds
        ot = 0
        frame_number = 0
        time_in_sim = ini_time_in_athena
        grid.set_hydro_with_time(new_time_list=np.array([athdf_obj_list[-1].hydro_time, athdf_obj_list[-2].hydro_time]), fill_value=fill_value)
        grain_init_loc_list = init_grain(par_rmin, par_rmax, par_nr, par_thetamin, par_thetamax, par_ntheta)
        grain_list = []
        grainID = 0
        for grain_init_loc in grain_init_loc_list:
            x, y, z = grain_init_loc
            r, theta = np.sqrt(x ** 2 + z ** 2), arctan(x, z)
            # grain_list.append(Grain(x=x, y=0, z=z, vx=vx, vy=vy, vz=vz, s=1e-5, rho=3, istracer=False))
            w, i, j = TSC(x, y, z, grid.r_list, grid.dr_list, grid.theta_list, grid.dtheta_list)
            gas_vr = apply_TSC(w, i, j, grid.vr_mesh[0], nr, ntheta)
            gas_vtheta = apply_TSC(w, i, j, grid.vtheta_mesh[0], nr, ntheta)
            gas_vphi = apply_TSC(w, i, j, grid.vphi_mesh[0], nr, ntheta)
            gas_rho = apply_TSC(w, i, j, grid.rho_mesh[0], nr, ntheta)
            gas_T = apply_TSC(w, i, j, grid.temp_mesh[0], nr, ntheta)
            grav_r = apply_TSC(w, i, j, grid.grav_r_mesh[0], nr, ntheta)
            grav_theta = apply_TSC(w, i, j, grid.grav_theta_mesh[0], nr, ntheta)
            grav_phi = apply_TSC(w, i, j, grid.grav_phi_mesh[0], nr, ntheta)
            r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            theta = np.arccos(z / r)
            phi = arctan(y, x)
            gas_vx = gas_vr * np.sin(theta) * np.cos(phi) + gas_vtheta * np.cos(theta) * np.cos(phi) - gas_vphi * np.sin(phi)
            gas_vy = gas_vr * np.sin(theta) * np.sin(phi) + gas_vtheta * np.cos(theta) * np.sin(phi) + gas_vphi * np.cos(phi)
            gas_vz = gas_vr * np.cos(theta) - gas_vtheta * np.sin(theta)

            new_grain = [x, y, z, gas_vx, gas_vy, gas_vz, s, rho, particle_type, grainID]
            grain_list.append(new_grain)
            grainID += 1
        grain_list=np.array(grain_list)
        Util.makeplot(grain_list, grid.x_mesh, grid.y_mesh, grid.rho_mesh, fn_root_pars, frame_number,  index_of_mesh=0, time_in_sim=time_in_sim, ot=ot, size_of_marker=0.1)
        Util.write_restart_file_npy(fn_root_pars + 'frame' + Util.choosenumber(frame_number) + '', grain_list=grain_list, ot=ot, ini_time_in_athena=ini_time_in_athena, frame_number=frame_number,
                                    hydro_x=grid.x_mesh, hydro_y=grid.y_mesh, hydro_rho=grid.rho_mesh[0], hydro_temp=grid.temp_mesh[0])
    print ('Initialization complete, entering main loop.....', flush=True)
    first_round = True
    while ot < T:
        reset_hydro = False
        while (ini_time_in_athena - (ot + output_dt) / (1e6 * year) < last_hydro_time or first_round) and len(athdf_output_fn_list) != 1:
            athdf_obj_list = []
            first_round = False
            reset_hydro = True
            count = 0
            while count < num_hydro_load:
                athdf_output_fn = athdf_output_fn_list[0]       # because we pop the read file
                athdf_obj_list.append(OutputAthdf(fn=orig_sim_dir + athdf_output_fn))
                if count < num_hydro_load - 1:
                    athdf_output_fn_list.pop(0)
                count += 1
            last_hydro_time = athdf_obj_list[-1].hydro_time
        if reset_hydro:
            print ("processing hydro.....", flush=True)
            grid.setup_hydro_with_time(athdf_obj_list=athdf_obj_list) # process info needed in the next few loops
        # new_time_list must be matched with ot + dt.
        # set hydro here so each time looping around only need to index the hydro instead of calculating (not compatible with numba)
        new_time_list = np.linspace(ini_time_in_athena - ot / (1e6 * year), ini_time_in_athena - (ot + output_dt) / (1e6 * year), int(output_dt/dt_min) + 1)[:-1]

        grid.set_hydro_with_time(new_time_list=new_time_list, fill_value=fill_value) # only process info needed in the next loop

        time_passed = time.time() - start_time
        hour = int(time_passed / 3600)
        minute = int((time_passed - 3600 * hour) / 60)
        second = int(time_passed - 3600 * hour - 60 * minute)
        print ('Set hydro completed. The number of stored frames is ' + str(len(grid.rho_mesh)) + '     number of grains in domain is ' + str(grain_list.shape[0]) +  '   Elapsed time: ' +
               '{0:4d} hour, {1:2d} min, {2:2d} sec'.format(hour, minute, second), flush=True)
        grain_list = move(grain_list, min_stoppingtime, dt_min, dt_max_over_min, ot, output_dt,
                          grid.r_list, grid.dr_list, nr, grid.theta_list, grid.dtheta_list, ntheta,
                          grid.vr_mesh, grid.vtheta_mesh, grid.vphi_mesh, grid.rho_mesh, grid.temp_mesh, grid.grav_r_mesh, grid.grav_theta_mesh, grid.grav_phi_mesh)
        ot += output_dt
        frame_number += 1
        time_in_sim += output_dt / (1e6 * year)

        # remove grains that are out of simulation domain
        i = 0
        while i < len(grain_list):
            xx, yy, zz = grain_list[i, 0], grain_list[i, 1], grain_list[i, 2]
            r = np.sqrt(xx ** 2 + yy ** 2 + zz ** 2)
            if r < min(grid.r_list) or r > max(grid.r_list):
                # print (r/AU, hydro_grid.rmin/AU, i)
                grain_list = np.delete(grain_list, i, axis=0)
            else:
                i += 1
        time_passed = time.time() - start_time
        hour = int(time_passed / 3600)
        minute = int((time_passed - 3600 * hour) / 60)
        second = int(time_passed - 3600 * hour - 60 * minute)
        print('frame: ' + '{:4d}'.format(frame_number) + '     ' +
              'simulation time: ' + '{:6d}'.format(int(ini_time_in_athena*1e6)) + ' + ' + '{:6d}'.format(int(ot / year)) + ' years' + '      ' + 'elapsed time: ' +
              '{0:4d} hour, {1:2d} min, {2:2d} sec'.format(hour, minute, second), flush=True)
        restart = False
        Util.makeplot(grain_list, grid.x_mesh, grid.y_mesh, grid.rho_mesh, fn_root_pars, frame_number, index_of_mesh=-1, time_in_sim=time_in_sim, ot=ot, size_of_marker=0.1)

        Util.write_restart_file_npy(fn_root_pars + 'frame' + Util.choosenumber(frame_number) + '', grain_list=grain_list, ot=ot, ini_time_in_athena=ini_time_in_athena, frame_number=frame_number,
                                    hydro_x=grid.x_mesh, hydro_y=grid.y_mesh, hydro_rho=grid.rho_mesh[-1], hydro_temp=grid.temp_mesh[-1])