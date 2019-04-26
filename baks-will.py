from scipy.special import gamma
import numpy as np
import time as time_module


def baks(spike_times, time, a=4., b=0.8):
    """Bayesian Adaptive Kernel Smoother (BAKS)
    adapted from https://github.com/nurahmadi/BAKS

    Parameters
    ----------
    spike_times: np.ndarray
        event spike times
    time: np.ndarray
        time at which the firing rate is estimated
    a: float
        shape parameter (alpha)
    b: float
        scale parameter (beta)


    Returns
    -------

    firing_rate: np.ndarray
    h: float

    """

    n = len(spike_times)

    b = n**(b)

    sumnum = 0.
    sumdenom = 0.
    spike_times /= 1000
    time /= 1000
    current_time = time_module.time()
    for i in range(n):
        numerator = (((time - spike_times[i])**2.)/2. + 1./b)**(-a)
        denominator = (((time - spike_times[i])**2.)/2. + 1./b)**(-a - 0.5)
        sumnum += numerator
        sumdenom += denominator
    print("Time after iteration: " + str(time_module.time() - current_time))
    current_time = time_module.time()
    h = (gamma(a)/gamma(a+0.5))*(sumnum/sumdenom)
    print("Time after h calculation: " + str(time_module.time() - current_time))
    current_time = time_module.time()
    firing_rate = np.zeros(time.shape)
    for j in range(n):
        k = (1./(np.sqrt(2.*np.pi)*h))*np.exp(-((time - spike_times[j])**2.)/(2.*h**2.))
        firing_rate = firing_rate + k
    print("Time after k iteration: " + str(time_module.time() - current_time))

    return firing_rate, h
