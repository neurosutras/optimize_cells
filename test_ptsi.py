from simple_network_utils import *
import numpy as np
from scipy.signal import butter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('plot', nargs='?', type=int, default=1,
                            help='1 to plot, 0 ow')
args = parser.parse_args()

filter_band = [4., 10.]
dur = 1000.
dt = 1.
time_x = np.arange(dur) / 1000.
time = np.arange(0., dur, dt)
sampling_rate = 1000. / dt
order = 15.

fm = 1.
fc = 7.
x0 = 5.
carrier = np.sin(2.0 * np.pi * fc * time_x)
carrier -= np.min(carrier)

print "For all signals:"
print "carrier freq=%i, mod freq=%i, amp=1 (Hz)" % (fc, fm)

nyq = 0.5 * sampling_rate
normalized_band = np.divide(filter_band, nyq)
sos = butter(order, normalized_band, analog=False, btype='band', output='sos')

filt_sig, env, env_ratio, centroid_freq, freq_tuning_index = \
    get_bandpass_filtered_signal_stats(carrier, time, sos, filter_band, plot=True, verbose=True)
print "baseline:", freq_tuning_index


"""FM"""
def FM(time, sos, filter_band, time_x, x0, fm, fc, m):
    modulator = np.sin(2.0 * np.pi * fm * time_x) * m
    signal = np.sin(2. * np.pi * (fc * time_x + modulator)) + x0
    signal -= np.min(signal)
    filt_sig, env, env_ratio, centroid_freq, freq_tuning_index = \
                get_bandpass_filtered_signal_stats(signal, time, sos, filter_band, plot=True, verbose=True)
    print "FM; mod index = %.1f: %f" % (m, freq_tuning_index)
    return signal, freq_tuning_index

m=.4
signal, freq_tuning_index = FM(time, sos, filter_band, time_x, x0, fm, fc, m)
plt.plot(time, signal)
plt.title('FM; carrier freq %i, mod freq %i, mod index %.1f' % (fc, fm, m))
plt.ylabel('Amp')
plt.xlabel('Time (ms)')
if args.plot: plt.show()

signal, freq_tuning_index = FM(time, sos, filter_band, time_x, x0, fm, fc, m=.6)
signal, freq_tuning_index = FM(time, sos, filter_band, time_x, x0, fm, fc, m=1.2)

"""adding carrier + mod signal"""
m=.4
modulator = np.sin(2. * np.pi * fm * time) * m
signal = modulator + carrier + x0
signal -= np.min(signal)
plt.plot(time, signal)
plt.title('Adding signals; carrier freq %i, mod freq %i, mod index %.1f' % (fc, fm, m))
if args.plot: plt.show()
filt_sig, env, env_ratio, centroid_freq, freq_tuning_index = \
        get_bandpass_filtered_signal_stats(signal, time, sos, filter_band, verbose=True)
print "addition:", freq_tuning_index

"""amp mod"""
def AM(time, sos, filter_band, time_x, x0, fm, carrier, m):
    modulator = np.sin(2.0 * np.pi * fm * time_x) * m
    signal = (1 + modulator) * carrier + x0
    signal -= np.min(signal)
    filt_sig, env, env_ratio, centroid_freq, freq_tuning_index = \
            get_bandpass_filtered_signal_stats(signal, time, sos, filter_band, verbose=True)
    print "AM; mod index = %.1f: %f" %(m, freq_tuning_index)
    return signal, freq_tuning_index

m=.4
signal, freq_tuning_index = AM(time, sos, filter_band, time_x, x0, fm, carrier, m)
plt.plot(time, signal)
plt.title('AM; carrier freq %i, mod freq %i, mod index %.1f' %(fc, fm, m))
if args.plot: plt.show()

signal, freq_tuning_index = AM(time, sos, filter_band, time_x, x0, fm, carrier, m=.6)
signal, freq_tuning_index = AM(time, sos, filter_band, time_x, x0, fm, carrier, m=1.2)

"""discrete noise packets"""
def discrete(time, filter_band, sos, carrier, x0,  p, noise_amp):
    noise_p = np.random.rand(len(time), 1)
    noise = np.zeros((len(time),))
    noise[np.where(noise_p <= p)[0]] = 1.
    
    discrete_noise = noise * noise_amp
    signal = carrier + discrete_noise + x0
    signal -= np.min(signal)
    filt_sig, env, env_ratio, centroid_freq, freq_tuning_index = \
            get_bandpass_filtered_signal_stats(signal, time, sos, filter_band, verbose=True)
    print "discrete noise; probability = %.1f, amp = %.1f: %f" %(p, noise_amp, freq_tuning_index)
    return signal, freq_tuning_index

p=.2
noise_amp=.1
signal, freq_tuning_index = discrete(time, filter_band, sos, carrier, x0, p, noise_amp)
plt.plot(time, signal)
plt.title('Packet noise; probability %.1f, noise amp %.1f' % (p, noise_amp))
if args.plot: plt.show()

signal, freq_tuning_index = discrete(time, filter_band, sos, carrier, x0, p=.8, noise_amp=.1)
signal, freq_tuning_index = discrete(time, filter_band, sos, carrier, x0, p=.8, noise_amp=.8)

"""uniform noise"""
def uni(sos, time, filter_band, carrier, x0, mu):
    signal = carrier + np.random.rand(len(time)) * mu + x0
    signal -= np.min(signal)
    filt_sig, env, env_ratio, centroid_freq, freq_tuning_index = \
            get_bandpass_filtered_signal_stats(signal, time, sos, filter_band, verbose=True)
    print "uniform noise around %.1f: %f" % (mu, freq_tuning_index)
    return signal, freq_tuning_index
mu = .1
signal, freq_tuning_index = uni(sos, time, filter_band, carrier, x0, mu)
plt.plot(time, signal)
plt.title("Uniform noise around %.1f" % (mu))
if args.plot: plt.show()

signal, freq_tuning_index = uni(sos, time, filter_band, carrier, x0, mu=.4)
signal, freq_tuning_index = uni(sos, time, filter_band, carrier, x0, mu=.6)

"""normal noise"""
def normal(sos, time, filter_band, carrier, x0, mu):
    signal = carrier + np.random.normal(scale=.1, size=len(time)) + x0
    signal -= np.min(signal)
    filt_sig, env, env_ratio, centroid_freq, freq_tuning_index = \
            get_bandpass_filtered_signal_stats(signal, time, sos, filter_band, verbose=True)
    print "Normal noise around %.1f: %f" % (mu, freq_tuning_index)
    return signal, freq_tuning_index
mu = .1
signal, freq_tuning_index = normal(sos, time, filter_band, carrier, x0, mu)
plt.plot(time, signal)
plt.title("Normal noise around %.1f" %(mu))
if args.plot: plt.show()

normal(sos, time, filter_band, carrier, x0, mu=.4)
normal(sos, time, filter_band, carrier, x0, mu=.6)
