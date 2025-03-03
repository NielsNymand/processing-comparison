import numpy as np
import matplotlib.pyplot as plt
import h5py
import plot_velmap as pv

def power(x):
    return 20*np.log10(np.abs(x))

path2data = '/media/niels/NielsSSD/data/UWB_raw/raw_h5'

fn_dict = {
    'coh2_2'        : 'coh2_2/20220701_091615_UWB_Greenland_2022.h5',
    'coh2_4'        : 'coh4_2/20220701_091615_UWB_Greenland_2022.h5',
    'coh4_4'        : 'coh4_4/20220701_091615_UWB_Greenland_2022.h5',
    'coh4_4_hanning': 'coh4_4_hanning/20220701_091615_UWB_Greenland_2022.h5',
    'coh4_4_hamming': 'coh4_4_hamming/20220701_091615_UWB_Greenland_2022.h5',
}


integrators = ['2','6']
reference = 'coh2_2'
trace_nr_reference = 1000 # coh2_2
data_dict = {}
for file in fn_dict:
    data_dict[file] = {}
    with h5py.File(f'{path2data}/{fn_dict[file]}','r') as f:
        Channel = f['channel_0']
        for integr in integrators:
            data_dict[file][integr] = {}
            Integrator = Channel[f'Integrator_{integr}']
#            print(Integrator.keys())
            if file == reference:            
                PPS_counter_reference = Integrator['PPS_Counters'][trace_nr_reference]
                trace_nr = trace_nr_reference
            
            if file != reference:
                PPS_counters = Integrator['PPS_Counters'][:]
                trace_nr = np.argmin( np.abs(PPS_counters - PPS_counter_reference) )
                
            data_dict[file][integr]['trace'] = Integrator['Chirps'][:,trace_nr]
            data_dict[file][integr]['PPS_counter'] = Integrator['PPS_Counters'][trace_nr]
            data_dict[file][integr]['lon'] = Integrator['lon'][trace_nr]
            data_dict[file][integr]['lat'] = Integrator['lat'][trace_nr]
            data_dict[file][integr]['time']= Integrator['_time'][:]
            data_dict[file][integr]['trace_nr'] = trace_nr



# plot_velmap
proj_polar,proj_cart = pv._create_projections(cen_lon=-45)
gps_fig = plt.figure(figsize=(12,12))
ax = plt.axes(projection=proj_polar)

# ax,proj_polar,proj_cart,im = plot_velmap(gps_fig,ax,vmin_max=(.1,1000),zoom='north',ticks=[1,10,100,1000],colorbar_pos=(-0.04,0.02,0.6),show_margin=False,
#                                          show_gridlines=False,n_lon_ticks=20,lx=4,ly=4,
#                                          cmap='nice',interpolation='gaussian',show_terrain=True,show_ocean=False,dy=(0,0))
# ax.axis('off')

ax,proj_polar,proj_cart,im = pv.plot_velmap(gps_fig,ax,vmin_max=(.1,5000),zoom='egrip',ticks=[1,10,100,1000],colorbar_pos=(-0.1,0.02,0.3),show_margin=True,
                                            show_gridlines=True,n_lon_ticks=10,lx=1,ly=1,colorbar=False,plot_flow_lines=True,flow_line_dx=1e4,quiver=False,flowline_arrow=False,
                                            cmap='nice1',interpolation='gaussian',show_terrain=False,show_ocean=False,dy=(0,0),smooth_vel=True,flowline_lw=0.4)
xy = proj_polar.transform_points(proj_cart,np.array([-pv.east_grip_pos[0]]) ,np.array([pv.east_grip_pos[1]]) )
ax.plot(xy[0,0],xy[0,1],'.',markersize=10,label='EastGRIP')
for integr in integrators:
    fig,axs = plt.subplots(1,len(data_dict)+1,sharey=True,sharex=True)

    t1,t2 = 1e-6,45e-6
    colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:cyan']
    for i,file in enumerate(data_dict):
        axs[i].plot(power(data_dict[file][integr]['trace']),data_dict[file][integr]['time'],color=colors[i] )
        axs[i].grid()
        axs[-1].plot(power(data_dict[file][integr]['trace']),data_dict[file][integr]['time'],color=colors[i] )
        if integr == integrators[0]:
            xy = proj_polar.transform_points(proj_cart,np.array([data_dict[file][integr]['lon']]) ,np.array([data_dict[file][integr]['lat']]) )
            ax.plot(xy[0,0],xy[0,1],'.',color=colors[i],alpha=0.5,markersize=15-i*2)


    axs[-1].grid()
    axs[0].set_ylim(t2,t1)


#print(f"trace nr: \n coh4_4: {data_dict['coh4_4']['0']['trace_nr']}\n coh4_4_hanning: {data_dict['coh4_4_hanning']['0']['trace_nr']}\n coh4_4_hamming: {data_dict['coh4_4_hamming']['0']['trace_nr']} ")

plt.show()