from scipy.special import gamma
from baks import baks
import numpy as np
import matplotlib.pyplot as plt

tstop = 3000
binned_dt = 1.

spike_times = np.array([100, 105, 110, 115, 118, 121, 124, 126, 128, 130, 131, 132, 133, 134, 135, 137, 139, 141, 144, 147, 150, 154, 158, 162, 167, 172])
spike_times += 1000
spike_times = np.array([100])
t = np.arange(0., tstop + binned_dt, binned_dt)
smoothed, _ = baks(spike_times, t)
print(smoothed)
plt.plot(range(len(smoothed)), smoothed)
plt.show()
