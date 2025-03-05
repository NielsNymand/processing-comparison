import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import signal,fft,fftpack
from UoAPy import radar_plotting  as rp
import copy
from scipy.ndimage import gaussian_filter1d
### 3 different approaches
# f1 = fc - B/2
# f2 = fc + B/2
# fc = (f1+f2)/2
### #1: keep fc fixed and change B to B_eff
#       f1_eff = fc - B_eff/2
#       f2_eff = fc + B_eff/2

### #2: Keep f1 fixed and change B to B_eff
#       - the efffective fc decreases for decreasing B
#       f2_new = f1 + B_eff = fc - B/2 + B_eff
#       fc_eff = (f1+f2_new)/2 = (2fc - B + B_eff)/2 = fc-B/2 + B_eff/2 = f1 + B_eff/2

### 3: Keep f2 fixed and change B to B_eff
#       - The effective fc increases for decreasing B
#       f1_new = f2 - B_eff = fc+B/2 - B_eff
#       fc_eff = (f1_new+f2)/2 = (2*f2 - B_eff)/2 = fc + B/2 - B_eff/2 = f2 - B_eff/2

### (4): 1,2 and 3 are special cases of letting f1 or f2 change continously
#       - The effective fc increases with increasing x and B_eff
#       f1_new = f1 + x
#       f2_new = f1_new + B_eff = f1+B_eff + x 
#       fc_eff = (f1_new+f2_new)/2 = f1 + B_eff/2 + x
#       for  (B-B_eff) >= x >= 0

def power(x):
    return 20*np.log10(np.abs(x))




def read_h5(filename,integrators,trace_nr=10000,chunk = '0000-0029',cable_delay_us = 0,read_all_chirps=False):

    data_dict = {}

    integrators = ['Integrator_0','Integrator_2']

#    trace_nr = 10000
    with h5py.File(filename,'r') as f:
            Chunk   = f[chunk]
            print(Chunk.keys())
            Channel = Chunk['subChan_0']
            print(Channel.keys())
            for integr in integrators:
                data_dict[integr] = {}
                Integrator = Channel[integr]
    #            print(channel[f'Integrator_{integr}'].keys())

                PPS_counter_reference = Integrator['PPS_Counters'][trace_nr]
                trace_nr = trace_nr
                if read_all_chirps:
                       data_dict[integr]['Chirps'] = Integrator['Chirps'][::10,:]
                data_dict[integr]['trace'] = np.sum(Integrator['Chirps'][:,trace_nr-8:trace_nr+8],axis=1)
                data_dict[integr]['PPS_counter'] = Integrator['PPS_Counters'][trace_nr]
                data_dict[integr]['lon'] = Integrator['lon'][trace_nr]
                data_dict[integr]['lat'] = Integrator['lat'][trace_nr]
                data_dict[integr]['time']= Integrator['_time'][:] -cable_delay_us*1e-6
                data_dict[integr]['trace_nr'] = trace_nr

    return data_dict

def plot_signal(data_dict,title='',radar_par = {},plot_complex=False,TWT_lim=(None,None),ylim_signal=(None,None)):
    keys = data_dict.keys()
    if plot_complex:
        fig,axs = plt.subplots(len(keys),2,sharex=True,sharey='col')
    if not plot_complex:
        fig,axs = plt.subplots(len(keys),2,sharex='col',sharey='col')
    axs = axs.flatten()
    # plot raw taces
    for i,key in enumerate(keys):
        trace = data_dict[key]['trace']
        time   = data_dict[key]['time']
        axs[i*2].plot(time*1e6,power(trace) )
        axs[i*2].set_ylabel('Signal power (dB)')

        if not plot_complex:
            signal_fft,freq_axis = rp.calc_spectrum(trace,radar_par['fs'])
            axs[i*2+1].plot(freq_axis/1e6,fftpack.fftshift(signal_fft) )
            axs[i*2+1].set_ylabel('Spectrum power (dB)')

        if plot_complex:
            axs[i*2+1].plot(time*1e6,np.angle(trace) )
            axs[i*2+1].set_ylabel('Signal phase (rad)')
            

        axs[i*2+1].yaxis.set_label_position("right")
        axs[i*2+1].yaxis.tick_right()
        
        axs[i*2].set_title(key.replace('_',' '))
        axs[i*2+1].set_title(key.replace('_',' '))
    [ax.grid() for ax in axs]
    axs[(len(keys)-1)*2].set_xlabel('TWT (us)')
    axs[(len(keys)-1)*2].set_xlim(TWT_lim)
    axs[(len(keys)-1)*2].set_ylim(ylim_signal)
    if not plot_complex:
        axs[(len(keys)-1)*2+1].set_xlabel('Frequency (MHz)')
    if plot_complex:
        axs[(len(keys)-1)*2+1].set_xlabel('TWT (us)')

    fig.suptitle(title)

# do frequency shift
def frequency_shift_trace(trace,time,freq_shift=0.0):
    shift_function = np.exp(1j * 2 * np.pi * freq_shift * time, dtype=trace.dtype)
    return trace * shift_function

def frequency_shift(data_dict,freq_shift):
    for integr in data_dict:
        trace = data_dict[integr]['trace']
        time   = data_dict[integr]['time'] 
        data_dict[integr]['trace'] = frequency_shift_trace(trace,time,freq_shift=freq_shift)


# lowpass filter
def lowpass(data_dict,bandwidth=300e6,radar_par={}):
    sos = signal.butter(50,bandwidth/radar_par['fs'],output='sos')
#    sos = signal.butter(50,[B/3,B/2],fs=fs,output='sos',btype='bandpass')
    for integr in data_dict:
        data_dict[integr]['trace'] = signal.sosfiltfilt(sos,data_dict[integr]['trace'])

def bandpass(data_dict,bandwidth,freq_shift=0,radar_par={}):
    frequency_shift(data_dict,freq_shift)
    lowpass(data_dict,bandwidth=bandwidth,radar_par=radar_par)
    frequency_shift(data_dict,-freq_shift)


def generate_chirp_radar(radar_par={}):
    '''The generated chirp is independent of the effective bandwidth. It will be truncated later'''

    chirp_time = np.arange(0,  radar_par['chirp_length']* radar_par['fs'] - 1) / radar_par['fs']
    reference_chirp = np.exp(
            -1j* 2* np.pi* ((radar_par['B'] / 2) * chirp_time - radar_par['B'] / radar_par['chirp_length'] / 2 * chirp_time ** 2)
        )

    return reference_chirp,chirp_time

def bandwidths_for_perfect_alignment(radar_par):
    ''' The 300 MHz chirp has real value of 1 at t = T/2. The phase at t = T/2 is not independent of T and B.
        This mean that if the reduced bandwidth is not chosen carefully it will not actually match the transmitted one.
        It turns out that the following conditions needs to be true, BT/8 = n, where n is an integer. 
        The reduced bandwidth, and thereby pulse length, is b = r*B and T_red = r*T. b*T_red/8 = r^2 *BT/8 = n.
        Possible reduction factors, r, therefor has to be chosen as r = sqrt(8n/(BT)) for n <= BT/8
        THIS DOES NOT SEEM TO ACTUALLY BE A PROBLEM. 
         '''
    
    n = radar_par['B']*radar_par['chirp_length']/8
    ns = np.arange(1,n)
    reduction_factors = np.sqrt(8*ns/(radar_par['B']*radar_par['chirp_length']) )

    return reduction_factors

def generate_chirp_modified(bandwidth,radar_par={}):
    '''Generate a chirp of abitrary bandwidth'''
    chirp_length =  radar_par['chirp_length'] * bandwidth/radar_par['B'] 
    chirp_time   = np.arange(0,  chirp_length* radar_par['fs'] - 1) / radar_par['fs']
    reference_chirp = np.exp(
            -1j* 2* np.pi* ((bandwidth / 2) * chirp_time - bandwidth / chirp_length / 2 * chirp_time ** 2)
        )
    return reference_chirp,chirp_time


def pulse_compress(data_dict,bandwidth=300e6,approach='0',apply_bandpass=False,radar_par={}):
    ''' 
        Pulse compress signal. Truncating the radar transmitted chirp (10 us and 300MHz bandwidth) to reduce the bandwidth
            bandwidth: Bandwidth in Hz of chirp, default 300 MHz
            approach : The method for pulse compression. see top of script. Default is a normal pulse compression utilizing the entire chirp transmitted by the radar system
     '''
    import scipy.signal.windows as scisigW
#    reference_window = scisigW.hamming(len(reference_chirp))

    for integr in integrators:
        trace = data_dict[integr]['trace']
        time  = data_dict[integr]['time']
        chirp,chirp_time = generate_chirp_radar(radar_par=radar_par)
        N_chirp = len(chirp)
        time_shift = 0.0 # For approach 3 we need to apply a time shift as the pulse compressed peaks are not aligned with the start of the chirp
        if approach == '0': # normal pulse compression without a truncated chirp
            chirp_truncated = chirp
            data_dict[integr]['bandwidth']= str(int(radar_par['B']/1e6) )
        
        if approach =='1': 
            N_truncated = int(N_chirp/2 * bandwidth/radar_par['B']  )
            N_midpoint  = np.argmin( np.abs(chirp_time - radar_par['chirp_length']/2) )
            print(N_midpoint,radar_par['chirp_length'],chirp_time[-1])
#            print('min time:', chirp_time[N_midpoint])
            chirp_truncated = chirp[N_midpoint-N_truncated:N_midpoint+N_truncated]
            
            time_shift = -(N_midpoint-N_truncated)/radar_par['fs'] #-chirp_length * (1-bandwidth/B)
            data_dict[integr]['bandwidth']= str(int(bandwidth/1e6))

            if apply_bandpass:
                bandpass(data_dict,bandwidth,freq_shift=0.0,radar_par=radar_par)

        if approach == '2':
            N_truncated = int(N_chirp * bandwidth/radar_par['B'] )
            chirp_truncated = chirp[0:N_truncated]
            data_dict[integr]['bandwidth']= str(int(bandwidth/1e6))
            if apply_bandpass:
                bandpass(data_dict,bandwidth,freq_shift=-(bandwidth-radar_par['B'])/2,radar_par=radar_par)

        if approach == '3':
            N_truncated = int(N_chirp * bandwidth/radar_par['B'] )
            chirp_truncated = chirp[::-1][0:N_truncated][::-1]
            data_dict[integr]['bandwidth']= str(int(bandwidth/1e6))
            time_shift = -(N_chirp-N_truncated)/radar_par['fs']#-chirp_length * (1-bandwidth/B)
            if apply_bandpass:
                bandpass(data_dict,bandwidth,freq_shift=(bandwidth-radar_par['B'])/2,radar_par=radar_par)


        print(len(chirp_truncated))
        data_dict[integr]['approach'] = approach
        data_dict[integr]['bandpass'] = apply_bandpass


        hamming_window = scisigW.hamming(len(chirp_truncated))
        chirp_truncated *= hamming_window

#        chirp = chirp[N_chirp//3:int(N_chirp*2/3)]


        N = len(trace)
        nfft = 2 ** (
            int(np.log2(max([len(chirp_truncated), N]))) + 1
        )  # array length to be a power of 2 for imporoved efficiency of fft.
        reference_freq = np.conj(fftpack.fftshift(fft.fft(chirp_truncated, n=nfft)))  # spectrum of chirp

        N_out = N - len(chirp_truncated) + 1
#        data_out = np.zeros(N_out, dtype=trace)  # output data array
        data_freq = (
                    fftpack.fftshift(fft.fft(trace, n=nfft)) * reference_freq
                )  # cross correlation of chirp and data in frequency domain
        data_t = fft.ifft(fftpack.ifftshift(data_freq))  # transform back to time domain


        data_dict[integr]['trace'] = data_t[0:N_out]  # cropped back to correct size
        data_dict[integr]['time']  = time[0:N_out] + time_shift
        if approach == '3' or approach=='1':
            data_dict[integr]['trace'] = data_dict[integr]['trace'][N_truncated:]
            data_dict[integr]['time']  = data_dict[integr]['time' ][N_truncated:]
#        if approach == '1':

def pulse_compress_alternative(data_dict,bandwidth=300e6,approach='0',apply_bandpass=False,radar_par={}):
    ''' 
        Pulse compress signal. Frequency shifting and generating reference chirp of lower bandwidth
            bandwidth: Bandwidth in Hz of chirp, default 300 MHz
            approach : The method for pulse compression. see top of script. Default is a normal pulse compression utilizing the entire chirp transmitted by the radar system
     '''
    import scipy.signal.windows as scisigW
#    reference_window = scisigW.hamming(len(reference_chirp))

    for integr in integrators:
        trace = data_dict[integr]['trace']
        time  = data_dict[integr]['time']

        time_shift = 0.0 # For approach 3 we need to apply a time shift as the pulse compressed peaks are not aligned with the start of the chirp
        if approach == '0': # normal pulse compression without a truncated chirp
            chirp,chirp_time = generate_chirp_radar(radar_par=radar_par)
            N_chirp = len(chirp)
            data_dict[integr]['bandwidth']= str(int(radar_par['B']/1e6) )

        if approach !='0':
            chirp,chirp_time = generate_chirp_modified(bandwidth,radar_par=radar_par)
            data_dict[integr]['bandwidth']= str(int(bandwidth/1e6))
            data_dict[integr]['alt'] = True

        if approach =='1': # centre frequency of trace is already correct
            time_shift = -radar_par['chirp_length']* 0.5* (1 - bandwidth/radar_par['B']) 

            if apply_bandpass:
                bandpass(data_dict,bandwidth,freq_shift=0.0,radar_par=radar_par)

        if approach == '2':
            trace = frequency_shift_trace(trace,time,freq_shift= -0.5*(-radar_par['B']+bandwidth) )

            if apply_bandpass:
                bandpass(data_dict,bandwidth,freq_shift=-(bandwidth-radar_par['B'])/2,radar_par=radar_par)

        if approach == '3':
            trace = frequency_shift_trace(trace,time,freq_shift= -0.5*( radar_par['B']-bandwidth) )
            #time_shift = -(N_chirp-N_truncated)/radar_par['fs']#-chirp_length * (1-bandwidth/B)
            time_shift = -radar_par['chirp_length'] *(1-bandwidth/radar_par['B'])
            if apply_bandpass:
                bandpass(data_dict,bandwidth,freq_shift=(bandwidth-radar_par['B'])/2,radar_par=radar_par)



        data_dict[integr]['approach'] = approach
        data_dict[integr]['bandpass'] = apply_bandpass


        hamming_window = scisigW.hamming(len(chirp))
        chirp *= hamming_window

#        chirp = chirp[N_chirp//3:int(N_chirp*2/3)]


        N = len(trace)
        nfft = 2 ** (
            int(np.log2(max([len(chirp), N]))) + 1
        )  # array length to be a power of 2 for imporoved efficiency of fft.
        reference_freq = np.conj(fftpack.fftshift(fft.fft(chirp, n=nfft)))  # spectrum of chirp

        N_out = N - len(chirp) + 1
#        data_out = np.zeros(N_out, dtype=trace)  # output data array
        data_freq = (
                    fftpack.fftshift(fft.fft(trace, n=nfft)) * reference_freq
                )  # cross correlation of chirp and data in frequency domain
        data_t = fft.ifft(fftpack.ifftshift(data_freq))  # transform back to time domain


        data_dict[integr]['trace'] = data_t[0:N_out]  # cropped back to correct size
        data_dict[integr]['time']  = time[0:N_out] + time_shift
        # if approach == '3' or approach=='1':
        #     data_dict[integr]['trace'] = data_dict[integr]['trace'][N_truncated:]
        #     data_dict[integr]['time']  = data_dict[integr]['time' ][N_truncated:]
#        if approach == '1':



def combine_results(data2combine):
    combined_dict = {}

    for key in data2combine[0].keys():
        combined_dict[key] = {}
        for d_dict in data2combine:
            if d_dict[key]['bandpass']:
                add_str = '(BP)'
            if 'alt' in d_dict[key]:
                add_str = '(alt)'
            else:
                add_str = ''
            combined_dict[key]['approach: ' + d_dict[key]['approach'] + ', B = ' + d_dict[key]['bandwidth'] +'MHz' + f' {add_str}'] = d_dict[key]
#        pulse_compress_dict[key]['approach: ' + data_dict_approach_2[key]['approach'] + '- B = ' + data_dict_approach_2[key]['bandwidth']] = data_dict_approach_2[key]

    return combined_dict

def basic_processing(data_dict,radar_par,plot_steps=False):

    if plot_steps:
        plot_signal(data_dict,radar_par=radar_par,title='Raw signal',plot_complex=False)
    freq_shift = radar_par['fs']/2.0 - radar_par['fc']
    frequency_shift(data_dict,freq_shift)
    if plot_steps:
        plot_signal(data_dict,radar_par=radar_par,title='Frequency shift',plot_complex=False)
    
    lowpass(data_dict,bandwidth=radar_par['B'],radar_par=radar_par)
    if plot_steps:
        plot_signal(data_dict,radar_par=radar_par,title='Lowpass filtered',plot_complex=False)



if __name__ == '__main__':

    path2data = '/media/niels/NielsSSD/data/UWB_raw/raw_h5'
    fn = '20220701_091615_UWB_Greenland_2022_ch0.h5'
    
    integrators = ['Integrator_0','Integrator_2']
    cable_delay_us = 7.23957 + 2.0 * 0.008786 + 2.0 * 0.00443307 + 0.19757 # time spent in cables/ arena offset

    data_dict=read_h5(f'{path2data}/{fn}',integrators,trace_nr=10000,chunk='0000-0029',cable_delay_us=cable_delay_us)

    radar_par = {
        'fs': 500e6,
        'B' : 300e6,
        'fc': 330e6,
        'chirp_length': 10e-6,
    }

    basic_processing(data_dict,radar_par,plot_steps=False)
    

    data_dict_approach_1 = copy.deepcopy(data_dict)
    data_dict_approach_2 = copy.deepcopy(data_dict)
    data_dict_approach_3 = copy.deepcopy(data_dict)
    data_dict_approach_1bp = copy.deepcopy(data_dict)
    data_dict_approach_2bp = copy.deepcopy(data_dict)
    data_dict_approach_3bp = copy.deepcopy(data_dict)

    print(bandwidths_for_perfect_alignment(radar_par)*300)


    pulse_compress(data_dict,radar_par=radar_par)
    pulse_compress_alternative(data_dict_approach_1bp,120e6,approach='1',apply_bandpass=False,radar_par=radar_par)
    pulse_compress_alternative(data_dict_approach_2bp,120e6,approach='2',apply_bandpass=False,radar_par=radar_par)
    pulse_compress_alternative(data_dict_approach_3bp,120e6,approach='3',apply_bandpass=False,radar_par=radar_par)

    pulse_compress_alternative(data_dict_approach_1,120e6,approach='1',apply_bandpass=False,radar_par=radar_par)
    pulse_compress_alternative(data_dict_approach_2,120e6,approach='2',apply_bandpass=False,radar_par=radar_par)
    pulse_compress_alternative(data_dict_approach_3,120e6,approach='3',apply_bandpass=False,radar_par=radar_par)



    # frequency shift (independent of bandwidth)


    dicts = [data_dict,data_dict_approach_1,data_dict_approach_2,data_dict_approach_3]
    pulse_compress_dict = combine_results(dicts)

#    plot_signal(pulse_compress_dict['Integrator_0'],radar_par=radar_par,title='Integrator 0',plot_complex=False,TWT_lim=(22,27),ylim_signal=(40,140))
    plot_signal(pulse_compress_dict['Integrator_2'],radar_par=radar_par,title='Integrator 2',plot_complex=True,TWT_lim=(22,27),ylim_signal=(40,140))


    dicts = [data_dict,data_dict_approach_1bp,data_dict_approach_2bp,data_dict_approach_3bp]
    pulse_compress_dict = combine_results(dicts)

#    plot_signal(pulse_compress_dict['Integrator_0'],radar_par=radar_par,title='Integrator 0',plot_complex=False,TWT_lim=(22,27),ylim_signal=(40,140))
    plot_signal(pulse_compress_dict['Integrator_2'],radar_par=radar_par,title='Integrator 2',plot_complex=True,TWT_lim=(22,27),ylim_signal=(40,140))






    # dicts = [data_dict_approach_1,data_dict_approach_1bp,data_dict_approach_2,data_dict_approach_2bp,data_dict_approach_3,data_dict_approach_3bp]
    # pulse_compress_dict = combine_results(dicts)

    # plot_signal(pulse_compress_dict['Integrator_0'],radar_par=radar_par,title='Integrator 0',plot_complex=False)
    # plot_signal(pulse_compress_dict['Integrator_2'],radar_par=radar_par,title='Integrator 2',plot_complex=True)



# plot pulse compressede
# plot_signal(data_dict,title=f'pulse compressed')
# plot_signal(data_dict_approach_2,title=f'pulse compressed')
#sos = signal.butter(50,B/5/fs,output='sos')
# for integr in integrators:
#      data_dict[integr]['trace'] = signal.sosfiltfilt(sos,data_dict[integr]['trace'])

# plot_signal(data_dict)

#    result.frequency_shift += self.freq_shift


#def processing_approach_2(data_dict,bandwidth):
    
# frequency shift



plt.show()

