import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as scisig

def power(x):
    return 20*np.log10(np.abs(x))

fs = 2500e6
def generate_chirp(chirp_length=10e-6,fs=500e6,bandwidth=300e6, plot_chirp=False, plot_params={}):

    chirp_time = np.arange(0, chirp_length * fs - 1) / fs
    reference_chirp = np.exp(
        -1j
        * 2
        * np.pi
        * ((bandwidth / 2) * chirp_time - bandwidth / chirp_length / 2 * chirp_time ** 2)
    )
    return reference_chirp,chirp_time

chirp,time = generate_chirp(fs=fs)

hanning_window = scisig.windows.tukey(len(chirp), 1)
tukey_window_005 = scisig.windows.tukey(len(chirp), 0.05)
hamming_window = scisig.windows.hamming(len(chirp))

chirp_hanning = chirp * hanning_window
chirp_005 = chirp * tukey_window_005
chirp_hamming = chirp * hamming_window

pc_hanning = scisig.correlate(chirp,chirp_hanning,mode='full')
pc_hamming = scisig.correlate(chirp,chirp_hamming,mode='full')
pc_005 = scisig.correlate(chirp,chirp_005,mode='full')

lags = scisig.correlation_lags(len(chirp),len(chirp_005),mode='full') / fs



plt.plot(lags*1e6,power(pc_hanning/np.max(pc_005)),label='hanning')
plt.plot(lags*1e6,power(pc_005/np.max(pc_005)),label='5% Tukey')
plt.plot(lags*1e6,power(pc_hamming/np.max(pc_005)),label='hamming')

plt.xlim(-.1,.1)
plt.ylim(-150,0)
plt.legend()

#plt.figure()


plt.show()
