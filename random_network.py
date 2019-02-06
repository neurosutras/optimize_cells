from mpi4py import MPI
from neuron import h
import numpy as np
import random
import sys, time
import scipy.signal as signal
import matplotlib.pyplot as plt
# for h.lambda_f
h.load_file('stdlib.hoc')
# for h.stdinit
h.load_file('stdrun.hoc')


# adopted from ring_network.py and ring_cell.py
# populations: FF; I, E
NUM_POP = 3


class Network(object):

    def __init__(self, ncell, delay, pc, tstop, dt=None, e2e_prob=.05, e2i_prob=.05, i2i_prob=.05, i2e_prob=.05,
                 ff2i_weight=1., ff2e_weight=2., e2e_weight=1., e2i_weight=1., i2i_weight=.5, i2e_weight=.5,
                 ff_meanfreq=100, ff_frac_active=.8, ff2i_prob=.5, ff2e_prob=.5, std_dict=None, tau_E=2., tau_I=5.,
                 connection_seed=0, spikes_seed=1):
        """

        :param ncell:
        :param delay:
        :param pc:
        :param tstop:
        :param dt:
        :param e2e_prob:
        :param e2i_prob:
        :param i2i_prob:
        :param i2e_prob:
        :param ff2i_weight:
        :param ff2e_weight:
        :param e2e_weight:
        :param e2i_weight:
        :param i2i_weight:
        :param i2e_weight:
        :param ff_meanfreq:
        :param ff_frac_active:
        :param ff2i_prob:
        :param ff2e_prob:
        :param std_dict:
        :param tau_E:
        :param tau_I:
        :param connection_seed: int
        :param spikes_seed: int
        """
        self.pc = pc
        self.delay = delay
        self.ncell = int(ncell)
        self.FF_firing = {}
        self.ff_frac_active = ff_frac_active
        self.tau_E = tau_E
        self.tau_I = tau_I
        self.prob_dict = {'e2e': e2e_prob, 'e2i': e2i_prob, 'i2i': i2i_prob, 'i2e': i2e_prob, 'ff2i': ff2i_prob,
                          'ff2e': ff2e_prob}
        self.weight_dict = {'ff2i': ff2i_weight, 'ff2e': ff2e_weight, 'e2e': e2e_weight, 'e2i': e2i_weight,
                            'i2i': i2i_weight, 'i2e': i2e_weight}
        self.weight_std_dict = std_dict
        self.index_dict = {'ff2i': ((0, ncell), (ncell, ncell * 2)),  # exclusive [x, y)
                           'ff2e': ((0, ncell), (ncell * 2, ncell * NUM_POP)),
                           'e2e': ((ncell, ncell * 2), (ncell, ncell * 2)),
                           'e2i': ((ncell, ncell * 2), (ncell * 2, ncell * NUM_POP)),
                           'i2i': ((ncell * 2, ncell * NUM_POP), (ncell * 2, ncell * NUM_POP)),
                           'i2e': ((ncell * 2, ncell * NUM_POP), (ncell, ncell * 2))}
        self.tstop = tstop
        self.ff_meanfreq = ff_meanfreq
        self.event_rec = {}

        self.local_random = random.Random()
        self.connection_seed = connection_seed
        self.spikes_seed = spikes_seed

        self.mknetwork(self.ncell)
        #self.mkstim(self.ncell)
        self.voltage_record(dt)
        self.spike_record()
        self.pydicts = {}

    def mknetwork(self, ncell):
        self.mkcells(ncell)
        self.connectcells(ncell)

    def mkcells(self, ncell):
        rank = int(self.pc.id())
        nhost = int(self.pc.nhost())
        self.cells = []
        self.gids = []
        for i in range(rank, ncell * NUM_POP, nhost):
            if i < ncell:
                self.local_random.seed(self.spikes_seed + i)
                cell = FFCell(self.tstop, self.ff_meanfreq, self.ff_frac_active, self, i,
                              local_random=self.local_random)
            else: 
                if i not in list(range(ncell * 2, ncell * 3)):
                    cell_type = 'RS'
                else:
                    cell_type = 'FS'
                cell = IzhiCell(self.tau_E, self.tau_I, cell_type)
            self.cells.append(cell)
            self.gids.append(i)
            self.pc.set_gid2node(i, rank)
            nc = cell.connect2target(None)
            self.pc.cell(i, nc)
            test = self.pc.gid2cell(i)
            if not cell.is_art():
                rec = h.Vector()
                nc.record(rec)
                self.event_rec[i] = rec

    def createpairs(self, prob, input_indices, output_indices):
        pair_list = []
        for i in input_indices:
            for o in output_indices:
                if self.local_random.random() <= prob:
                    pair_list.append((i, o))
        for elem in pair_list:
            x, y = elem
            if x == y: pair_list.remove(elem)
        return pair_list

    def get_cell_type(self, gid):
        if gid < self.ncell:
            return None
        elif gid < self.ncell * 2:
            return 'FS'
        else:
            return 'RS'

    def connectcells(self, ncell):
        rank = int(self.pc.id())
        nhost = int(self.pc.nhost())
        self.local_random.seed(self.connection_seed + rank)
        self.ncdict = {}  # not efficient but demonstrates use of pc.gid_exists

        for connection in ['ff2i', 'ff2e', 'e2e', 'e2i', 'i2i', 'i2e']:

            mu = self.weight_dict[connection]
            if self.weight_std_dict[connection] >= 2. / 3. / np.sqrt(2.):
                print 'network.connectcells: connection: %s; reducing std to avoid negative weights: %.2f' % \
                      (connection, self.weight_std_dict[connection])
                self.weight_std_dict[connection] = 2. / 3. / np.sqrt(2.)
            std_factor = self.weight_std_dict[connection]

            indices = self.index_dict[connection]
            inp = indices[0]
            out = indices[1]
            pair_list = self.createpairs(self.prob_dict[connection], list(range(inp[0], inp[1])),
                                         list(range(out[0], out[1])))
            for pair in pair_list:
                presyn_gid = pair[0]
                target_gid = pair[1]
                if self.pc.gid_exists(target_gid):
                    target = self.pc.gid2cell(target_gid)
                    """pre_type = self.get_cell_type(presyn_gid)
                    target_type = self.get_cell_type(target_gid)
                    if pre_type is None: #E
                        syn = target.synlist[0]
                        std = self.weight_std_dict['ff']
                        if target_type == 'RS':
                            mu = self.weight_dict['ff2e']
                        else:
                            mu = self.weight_dict['ff2i']
                    elif pre_type == 'RS': #E
                        syn = target.synlist[0]
                        std = self.weight_std_dict['E']
                        if target_type == 'RS':
                            mu = self.weight_dict['e2e']
                        else:
                            mu = self.weight_dict['e2i']
                    else: #I
                        syn = target.synlist[1]
                        std = self.weight_std_dict['I']
                        if target_type == 'RS':
                            mu = self.weight_dict['i2e']
                        else:
                            mu = self.weight_dict['i2i']"""
                    if connection[-1] == 'e':
                        syn = target.synlist[0]
                    else:
                        syn = target.synlist[1]
                    nc = self.pc.gid_connect(presyn_gid, syn)
                    nc.delay = self.delay
                    weight = self.local_random.gauss(mu, mu * std_factor)
                    while weight < 0.: weight = self.local_random.gauss(mu, mu * std_factor)
                    nc.weight[0] = weight
                    self.ncdict[pair] = nc

    # Instrumentation - stimulation and recording
    """def mkstim(self, ncell):
        self.ns = h.NetStim()
        self.ns.number = 100
        self.ns.start = 0
        for i in range(ncell):
            if not self.pc.gid_exists(i):  # or random.random() >= .3:  # stimulate only 30% of FF
                continue
            self.nc = h.NetCon(self.ns, self.pc.gid2cell(i).synlist[0])
            self.nc.delay = 0
            self.nc.weight[0] = 2
            self.ncdict[('stim', i)] = self.nc
            #print self.ncdict"""

    def spike_record(self):
        self.spike_tvec = {}
        self.spike_idvec = {}
        for i, gid in enumerate(self.gids):
            if self.cells[i].is_art(): continue
            tvec = h.Vector()
            idvec = h.Vector()
            nc = self.cells[i].connect2target(None)
            self.pc.spike_record(nc.srcgid(), tvec, idvec)# Alternatively, could use nc.record(tvec)
            self.spike_tvec[gid] = tvec
            self.spike_idvec[gid] = idvec

    """def event_record(self, dt = None):
        for i, cell in enumerate(self. cells):
            if cell.is_art(): continue
            nc = h.NetCon(cell.sec(.5)._ref_v, None)
            rec = h.Vector()
            nc.record(rec)
            self.event_rec[i] = rec"""

    def voltage_record(self, dt=None):
        self.voltage_tvec = {}
        self.voltage_recvec = {}
        if dt is None:
            self.dt = h.dt
        else:
            self.dt = dt
        for i, cell in enumerate(self.cells):
            if cell.is_art(): continue
            tvec = h.Vector()
            tvec.record(
                h._ref_t)  # dt is not accepted as an argument to this function in the PC environment -- may need to turn on cvode?
            rec = h.Vector()
            rec.record(getattr(cell.sec(.5), '_ref_v'))  # dt is not accepted as an argument
            self.voltage_tvec[self.gids[i]] = tvec
            self.voltage_recvec[self.gids[i]] = rec

    def vecdict_to_pydict(self, vecdict):
        this_pydict = dict()
        for key, value in vecdict.iteritems():
            this_pydict[key] = np.array(value)
        return this_pydict

    """def compute_isi(self, vecdict):  # vecdict is an event dict
        self.ratedict = {}
        self.peakdict = {}
        for key, vec in vecdict.iteritems():
            isivec = h.Vector()
            if len(vec) > 1:
                isivec.deriv(vec, 1, 1)
                rate = 1. / (isivec.mean() / 1000)
                self.ratedict[key] = rate
                if isivec.min() > 0:
                    self.peakdict[key] = 1. / (isivec.min() / 1000)"""

    def py_compute_isi(self, vecdict):
        self.rate_dict = {}
        self.peak_dict = {}
        for key, vec in vecdict.iteritems():
            if len(vec) > 1:
                isi = []
                for i in range(len(vec) - 1):
                    isi.append(vec[i + 1] - vec[i])
                isi = np.array(isi)
                rate = 1. / (isi.mean() / 1000.)
                self.rate_dict[key] = rate
                self.peak_dict[key] = 1. / (isi.min() / 1000.)

    
    def summation(self, vecdict, dt=.025):
        """
        TODO: operate on pydicts
        :param vecdict:
        :param dt:
        :return:
        """
        size = self.tstop + 2  #* (1 / dt) + 1
        self.osc_E = h.Vector(size)
        self.osc_I = h.Vector(size)
        for key, vec in vecdict.iteritems():
            cell_type = self.get_cell_type(key)
            binned = vec.histogram(0, self.tstop, 1)
            # for i in binned: print i
            if cell_type == 'RS':  #E
                self.osc_E.add(binned)
            elif cell_type == 'FS':  #F
                self.osc_I.add(binned)

    def py_summation(self, vecdict, dt=.025):
        size = self.tstop
        self.E_sum = np.zeros(size)
        self.I_sum = np.zeros(size)
        for i in range(self.ncell, self.ncell * NUM_POP):
            cell_type = self.get_cell_type(i)
            vec = map(int, vecdict[i])
            t = np.arange(0., self.tstop, 1.)
            spike_indexes = [np.where(t >= time)[0][0] for time in vec]
            spike_count = np.zeros_like(t)
            spike_count[spike_indexes] = 1.
            if cell_type == 'RS':
                self.E_sum = np.add(self.E_sum, spike_count)
            else:
                self.I_sum = np.add(self.I_sum, spike_count)

    def compute_peak_osc_freq(self, osc, freq, plot=False):
        sub_osc = osc[int(len(osc) / 6):]
        filtered = smooth_peaks(sub_osc)
        if freq == 'theta':
            widths = np.arange(50, 200)
        else:
            widths = np.arange(5, 50)
        peak_loc = signal.find_peaks_cwt(filtered, widths)
        tmp = h.Vector()
        x = [i for i in range(len(sub_osc) + 100)]
        if len(peak_loc) > 1:
            tmp.deriv(h.Vector(peak_loc), 1, 1)
            peak = 1 / (tmp.min() / 1000.)
            if plot:
                plt.plot(x, filtered, '-gD', markevery=peak_loc)
                plt.show()
        else:
            peak = 0.
        return peak

    def compute_pop_firing_rates(self, lower, upper, rate_dict, peak_dict):
        uncounted = 0
        mean = 0;
        max_firing = 0
        for i in range(lower, upper):
            if i not in rate_dict:
                uncounted += 1
                continue
            mean += rate_dict[i]
            max_firing += peak_dict[i]
        if self.ncell - uncounted != 0:
            mean = mean / float(self.ncell - uncounted)
            max_firing = max_firing / float(self.ncell - uncounted)

        return mean, max_firing

    def plot_voltage_trace(self, vecdict, all_events, dt=.025):
        down_dt = 1.
        ms_step = int(down_dt / dt)
        for i in range(self.ncell * 2, self.ncell * NUM_POP):
            ms_rec = []
            for j, v in enumerate(vecdict[i]):
                if j % ms_step == 0: ms_rec.append(v)
            ev = [int(event/down_dt) for event in all_events[i]]
            plt.plot(range(len(ms_rec)), ms_rec, '-gD', markevery=ev)
            plt.title('v trace')
            plt.show()

    def plot_cell_activity(self, all_spikes_dict):
        import seaborn as sns

        hm = np.zeros((self.ncell * NUM_POP, self.tstop))
        for key, val in all_spikes_dict.iteritems():
            x = np.zeros(self.tstop)
            for t in val: x[int(t)] = 1
            smoothed = boxcar(x)
            hm[key] = smoothed
        sns.heatmap(hm)
        plt.show()

    def plot_smoothing(self, gauss_E):
        plt.plot(range(len(gauss_E)), gauss_E)
        plt.title('gauss smoothing - E pop')
        plt.show()
        plt.plot(range(len(self.E_sum)), self.E_sum)
        plt.title('spike counts - E pop')
        plt.show()

    def plot_bands(self, theta_E, gamma_E):
        plt.plot(range(len(theta_E)), theta_E)
        plt.title('theta E')
        plt.show()
        plt.plot(range(len(gamma_E)), gamma_E)
        plt.title('gamma E')
        plt.show()

    def get_bands_of_interest(self, plot):
        gauss_E = gauss(self.E_sum, 1., self.tstop)
        gauss_I = gauss(self.I_sum, 1., self.tstop)

        filter_dt = 1.  # ms
        window_len = int(2000. / filter_dt)
        theta_band = [5., 10.]
        theta_E, theta_I = filter_band(gauss_E, gauss_I, window_len, theta_band)
        window_len = int(200. / filter_dt)
        gamma_band = [30., 100.]
        gamma_E, gamma_I = filter_band(gauss_E, gauss_I, window_len, gamma_band)

        if plot:
            self.plot_smoothing(gauss_E)
            self.plot_bands(theta_E, gamma_E)

        return theta_E, theta_I, gamma_E, gamma_I

    """def remake_syn(self):
        if int(self.pc.id() == 0):
            for pair, nc in self.ncdict.iteritems():
                nc.weight[0] = 0.
                self.ncdict.pop(pair)
                #print pair
        self.connectcells(self.ncell)"""


class IzhiCell(object):
    # derived from modelDB
    def __init__(self, tau_E, tau_I, type='RS'):  # RS = excit or FS = inhib
        self.type = type
        self.sec = h.Section(cell=self)
        self.sec.L, self.sec.diam, self.sec.cm = 10, 10, 31.831
        self.izh = h.Izhi2007b(.5, sec=self.sec)
        self.vinit = -60
        self.sec(0.5).v = self.vinit
        self.sec.insert('pas')

        if type == 'RS': self.izh.a = .1
        if type == 'FS': self.izh.a = .02

        self.synapses(tau_E, tau_I)

    def __del__(self):
        # print 'delete ', self
        pass

    # from Ball_Stick
    def synapses(self, tau_E, tau_I):
        synlist = []
        s = h.ExpSyn(self.sec(0.8))  # E
        s.tau = tau_E
        synlist.append(s)
        s = h.ExpSyn(self.sec(0.1))  # I1
        s.tau = tau_I
        s.e = -80
        synlist.append(s)

        self.synlist = synlist

    # also from Ball Stick
    def connect2target(self, target):
        nc = h.NetCon(self.sec(1)._ref_v, target, sec=self.sec)
        nc.threshold = 10
        return nc

    def is_art(self):
        return 0


class FFCell(object):
    def __init__(self, tstop, mean_freq, frac_active, network, gid, local_random=None):
        if local_random is None:
            local_random = random.Random()
        self.pp = h.VecStim()
        #tstop in ms and mean_rate in Hz
        spikes = get_inhom_poisson_spike_times_by_thinning([mean_freq, mean_freq], [0, tstop], dt=0.025,
                                                           generator=local_random)
        vec = h.Vector(spikes)
        """self.pp.play(vec)
        print spikes"""
        if local_random.random() <= frac_active:  #vec = h.Vector([5, 200])
            self.pp.play(vec)
            network.FF_firing[gid] = np.array(spikes)
        else:
            network.FF_firing[gid] = []

    def connect2target(self, target):
        nc = h.NetCon(self.pp, target)
        return nc

    def is_art(self):
        return 1


def gauss(spikes, dt, tstop, filter_duration=100):
    filter_duration = filter_duration  # ms
    filter_t = np.arange(-filter_duration, filter_duration, dt)
    sigma = filter_duration / 3. / np.sqrt(2.)
    gaussian_filter = np.exp(-(filter_t / sigma) ** 2.)
    gaussian_filter /= np.trapz(gaussian_filter, dx=dt / 1000)

    signal = np.convolve(spikes, gaussian_filter)
    signal_t = np.arange(0., len(signal) * dt, dt)
    signal = signal[int(filter_duration / dt):][:len(spikes)]

    return signal


def filter_band(E, I, window_len, band):
    filt = signal.firwin(window_len, band, nyq=1000. / 2., pass_zero=False)
    E_band = signal.filtfilt(filt, [1.], E, padtype='even', padlen=window_len)
    I_band = signal.filtfilt(filt, [1.], I, padtype='even', padlen=window_len)

    return E_band, I_band


def smooth_peaks(x, window_len=101):
    """smoothing to denoise peaks"""
    window = signal.general_gaussian(window_len, p=1, sig=20)
    filtered = signal.fftconvolve(window, x)
    filtered = (np.average(x) / np.average(filtered)) * filtered
    filtered = np.roll(filtered, -25)
    return filtered


def boxcar(x, window_len=101):
    x = np.array(x)
    y = np.zeros((len(x) + window_len,))
    x = np.append(x[:window_len], x)
    for i in range(len(x)):
        y[i] = y[i - 1] + x[i] - x[i - window_len]

    """for i in range(window_len):
        y[i] = x[i]
    for i in range(window_len, len(x)):
        y[i] = y[i - 1] + x[i] - x[i - window_len]"""
    y = y * 1 / float(window_len)
    return y[window_len:]


def run_network(network, pc, comm, tstop, dt=.025, plot=False):
    pc.set_maxstep(10)
    h.stdinit()
    pc.psolve(tstop)
    nhost = int(pc.nhost())
    # hoc vec to np
    py_spike_dict = network.vecdict_to_pydict(network.spike_tvec)
    py_V_dict = network.vecdict_to_pydict(network.voltage_recvec)
    gauss_firing_rates = {}
    for key, val in py_spike_dict.iteritems():
        if len(val) > 0:
            val = map(int, val)
            t = np.arange(0., tstop, 1.)
            spike_indexes = [np.where(t >= time)[0][0] for time in val]
            spike_count = np.zeros_like(t)
            spike_count[spike_indexes] = 1.

            smoothed = gauss(spike_count, 1., tstop)
            gauss_firing_rates[key] = smoothed
        else:
            gauss_firing_rates[key] = []
    

    all_spikes_dict = comm.gather(py_spike_dict, root=0)
    all_V_dict = comm.gather(py_V_dict, root=0)
    ff_spikes = comm.gather(network.FF_firing, root=0)
    gauss_firing_rates = comm.gather(gauss_firing_rates, root=0)
    if comm.rank == 0:
        gauss_firing_rates = {key: val for fire_dict in gauss_firing_rates for key, val in fire_dict.iteritems()}
        widths = np.arange(50, 200)
        rate_dict = {}
        peak_dict = {}
        for key, val in gauss_firing_rates.iteritems():
            if len(val) > 0:
                peak_loc = signal.find_peaks_cwt(val, widths)
                c_sum = 0
                peak = float("-inf")
                for loc in peak_loc:
                    c_sum += val[loc]
                    peak = max(peak, val[loc])
                rate_dict[key] = c_sum / float(len(peak_loc))
                peak_dict[key] = peak

        all_V_dict = {key: val for V_dict in all_V_dict for key, val in V_dict.iteritems()}
        all_spikes_dict = {key: val for spike_dict in all_spikes_dict for key, val in spike_dict.iteritems()}
        for ff_dict in ff_spikes:
            for key, val in ff_dict.iteritems():
                all_spikes_dict[key] = val

    """
    #old method is being commented out. kept for comparison purposes. new method (not using
    hoc vec.deriv()) seems more biologically accurate
    network.py_compute_isi(py_spike_dict)
    rate_dicts = comm.gather(network.rate_dict, root=0)
    peak_dicts = comm.gather(network.peak_dict, root=0)
    if comm.rank == 0:
        rate_dicts = {key: val for r_dict in rate_dicts for key, val in r_dict.iteritems()}
        peak_dicts = {key: val for p_dict in peak_dicts for key, val in p_dict.iteritems()}#
        print "old method", rate_dicts
        print "old method", peak_dicts"""

    if comm.rank == 0:
        network.py_summation(all_spikes_dict)
        #x = range(len(E_test))
        if plot:
            network.plot_voltage_trace(all_V_dict, all_spikes_dict, dt)
            network.plot_cell_activity(all_spikes_dict)
        #window_len = min(int(2000./down_dt), len(down_t) - 1)
        E_mean, E_max = network.compute_pop_firing_rates(network.ncell, network.ncell * 2, rate_dict, peak_dict)
        I_mean, I_max = network.compute_pop_firing_rates(network.ncell * 2, network.ncell * NUM_POP, rate_dict,
                                                         peak_dict)

        network.py_summation(all_spikes_dict)
        theta_E, theta_I, gamma_E, gamma_I = network.get_bands_of_interest(plot)
        peak_theta_osc_E = network.compute_peak_osc_freq(theta_E, 'theta')
        peak_theta_osc_I = network.compute_peak_osc_freq(theta_I, 'theta')
        peak_gamma_osc_E = network.compute_peak_osc_freq(gamma_E, 'gamma')
        peak_gamma_osc_I = network.compute_peak_osc_freq(gamma_I, 'gamma')
        # rec = {key: value for dict in all_dicts for key, value in dict['rec'].iteritems()}
        return {'E_mean_rate': E_mean, 'E_peak_rate': E_max, 'I_mean_rate': I_mean, "I_peak_rate": I_max,
                'peak_theta_osc_E': peak_theta_osc_E, 'peak_theta_osc_I': peak_theta_osc_I,
                'peak_gamma_osc_E': peak_gamma_osc_E, 'peak_gamma_osc_I': peak_gamma_osc_I, 'event': all_spikes_dict,
                'osc_E': network.E_sum}


"""from dentate > stgen.py. temporary. personal issues with importing dentate -S"""
def get_inhom_poisson_spike_times_by_thinning(rate, t, dt=0.02, refractory=3., generator=None):
    if generator is None:
        generator = random.Random()
    interp_t = np.arange(t[0], t[-1] + dt, dt)
    interp_rate = np.interp(interp_t, t, rate)
    interp_rate /= 1000.
    non_zero = np.where(interp_rate > 0.)[0]
    interp_rate[non_zero] = 1. / (1. / interp_rate[non_zero] - refractory)
    spike_times = []
    max_rate = np.max(interp_rate)
    if not max_rate > 0.:
        return spike_times
    i = 0
    ISI_memory = 0.
    while i < len(interp_t):
        x = generator.random()
        if x > 0.:
            ISI = -np.log(x) / max_rate
            i += int(ISI / dt)
            ISI_memory += ISI
            if (i < len(interp_t)) and (generator.random() <= interp_rate[i] / max_rate) and ISI_memory >= 0.:
                spike_times.append(interp_t[i])
                ISI_memory = -refractory
    return spike_times
