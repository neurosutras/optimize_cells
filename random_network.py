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
NUM_POP = 3


"""FF ; I ; E"""


# =================== network class
class Network(object):

    def __init__(self, ncell, delay, pc, tstop, dt=None, e2e_prob=.05, e2i_prob=.05, \
                 i2i_prob=.05, i2e_prob=.05, ff2i_weight=1., ff2e_weight=2., e2e_weight=1., e2i_weight=1., \
                 i2i_weight=.5, i2e_weight=.5, ff_meanfreq=100, ff_frac_active=.8, ff2i_prob=.5, ff2e_prob=.5, \
                 std_dict=None, tau_E=2., tau_I=5., local_random=None):
        self.pc = pc
        self.delay = delay
        self.ncell = int(ncell)
        self.FF_firing = {}
        self.ff_frac_active = ff_frac_active
        self.tau_E = tau_E
        self.tau_I = tau_I
        self.prob_dict = {'e2e': e2e_prob, 'e2i': e2i_prob, 'i2i': i2i_prob, 'i2e': i2e_prob, \
                          'ff2i': ff2i_prob, 'ff2e': ff2e_prob}
        self.weight_dict = {'ff2i': ff2i_weight, 'ff2e': ff2e_weight, 'e2e': e2e_weight, 'e2i': e2i_weight, \
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

        if local_random is None:
            self.random = random.Random()
        else:
            self.random = random

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
                cell = FFCell(self.tstop, self.ff_meanfreq, self.ff_frac_active, self, i, local_random=self.random)
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
                if self.random.random() <= prob:
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
        self.ncdict = {}  # not efficient but demonstrates use of pc.gid_exists

        for connection in ['ff2i', 'ff2e', 'e2e', 'e2i', 'i2i', 'i2e']:

            mu = self.weight_dict[connection]
            std_factor = self.weight_std_dict[connection]
            if std_factor >= 2. / 3. / np.sqrt(2.):
                print 'network.connectcells: connection: %s; reducing std to avoid negative weights: %.2f' % \
                      (connection, self.weight_std_dict[connection])
                self.weight_std_dict[connection] = 2. / 3. / np.sqrt(2.)

            indices = self.index_dict[connection]
            inp = indices[0]
            out = indices[1]
            pair_list = self.createpairs(self.prob_dict[connection], list(range(inp[0], \
                                                                                inp[1])), list(range(out[0], out[1])))
            std_factor = self.weight_std_dict[connection]
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
                    weight = self.random.gauss(mu, mu * std_factor)
                    while weight < 0.: weight = self.random.gauss(mu, mu * std_factor)
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

    def compute_isi(self, vecdict):  # vecdict is an event dict
        """
        TODO: Operate on pydicts instead of vecdicts
        TODO: Use gauss_smooth to get rates instead
        :param vecdict:
        :return:
        """
        self.ratedict = {}
        self.peakdict = {}
        for key, vec in vecdict.iteritems():
            isivec = h.Vector()
            if len(vec) > 1:
                isivec.deriv(vec, 1, 1)
                rate = 1. / (isivec.mean() / 1000)
                self.ratedict[key] = rate
                if isivec.min() > 0:
                    self.peakdict[key] = 1. / (isivec.min() / 1000)

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
            peak = -1
        return peak

    def remake_syn(self):
        if int(self.pc.id() == 0):
            for pair, nc in self.ncdict.iteritems():
                nc.weight[0] = 0.
                self.ncdict.pop(pair)
                #print pair
        self.connectcells(self.ncell)


def gauss(spikes, dt, tstop, filter_duration=100):
    filter_duration = filter_duration  # ms
    filter_t = np.arange(-filter_duration, filter_duration, dt)
    sigma = filter_duration / 3. / np.sqrt(2.)
    gaussian_filter = np.exp(-(filter_t / sigma) ** 2.)
    gaussian_filter /= np.trapz(gaussian_filter, dx=dt / 1000)

    signal = np.convolve(spikes, gaussian_filter)
    signal_t = np.arange(0., len(signal) * dt, dt)
    signal = signal[int(filter_duration / dt):][:len(spikes)]
    print signal

    return signal


"""smoothing to denoise peaks"""


def smooth_peaks(x, window_len=101):
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

    """py_spike_dict = network.vecdict_to_pydict(network.spike_tvec)
    all_spikes_dict = comm.gather(py_spike_dict, root=0)
    if comm.rank == 0:
        all_spikes_dict = {key: val for this_dict in all_spikes_dict for key, val in this_dict.iteritems()}
        print all_spikes_dict.keys()
    else:
        if all_spikes_dict is not None:
            print 'something went wrong'"""

    """
    1) Convert hoc vectors to python arrays (spikes and voltages)
    2) Compute firing rates by smoothing, on each rank
    3) Process anything that can be computed locally on each rank.
    4) Collect spikes to rank 0 for population analysis.
    5) Collect anything that rank 0 needs to do a final population analysis.
    6) return feature_dict
    """

    all_events = pc.py_alltoall([network.spike_tvec for _ in range(nhost)])  # list
    all_events = {key : val for this_dict in all_events for key, val in this_dict.iteritems()} #collapse list into dict
    network.compute_isi(all_events)
    ff_events = pc.py_alltoall([network.FF_firing for _ in range(nhost)])
    ff_events = {key: val for dict in ff_events for key, val in dict.iteritems()}
    # Use MPI Gather instead:
    rate_dicts = pc.py_alltoall([network.ratedict for _ in range(nhost)])
    peak_dicts = pc.py_alltoall([network.peakdict for _ in range(nhost)])
    processed_rd = {key: val for dict in rate_dicts for key, val in dict.iteritems()}
    processed_p = {key: val for dict in peak_dicts for key, val in dict.iteritems()}
    # all_dicts = pc.py_alltoall([network.voltage_recvec for i in range(nhost)])
    """network.vecdict_to_pydict(network.voltage_recvec, 'rec')
    test = pc.py_alltoall([network.pydicts for i in range(nhost)])"""
    tmp = pc.py_alltoall([network.voltage_recvec for i in range(nhost)])
    if int(pc.id()) == 0:
        """rec = {key: val for dict in test for key, val in dict['rec'].iteritems()}
        for i in range(network.ncell * 2, network.ncell * NUM_POP):
            recv = rec[i]
            recv2 = []
            for j, v in enumerate(recv):
                if j % 40 == 0: recv2.append(v)
            ev = all_events[i]
            x = range(len(recv2))
            plt.plot(x, recv2, '-gD', markevery=ev)
            plt.title('v trace')
            plt.show()"""

        network.summation(all_events)

        hm = np.zeros((network.ncell * NUM_POP, network.tstop + 1))
        for i in range(network.ncell):
            x = np.zeros((1, network.tstop + 1))
            for t in ff_events[i]:
                x[0][int(t)] = 1
            smoothed = boxcar(x[0])
            hm[i] = smoothed
        for i in range(network.ncell, network.ncell * 3):
            x = np.zeros((1, network.tstop + 1))
            for t in all_events[i]:
                x[0][int(t)] = 1
            smoothed = boxcar(x[0])
            hm[i] = smoothed
        if plot:
            import seaborn as sns
            sns.heatmap(hm)
            plt.show()
            #sns.plt.show()
        peak_voltage = 35.
        """peak_voltage = float("-inf") #print list(rec.keys())
        if network.ncell in list(rec.keys()):
            # use np.max, look at more cells
            for i, x in enumerate(rec[network.ncell]):
                if i % 500 == 0:
                    peak_voltage = max(peak_voltage, x)"""

        E_mean = 0
        I_mean = 0
        I_max = 0
        E_max = 0
        uncounted = 0

        for i in range(network.ncell, network.ncell * 2):
            if i not in processed_p:
                uncounted += 1
                continue
            E_mean += processed_rd[i]
            E_max += processed_p[i]
        if network.ncell - uncounted != 0:
            E_mean = E_mean / float(network.ncell - uncounted)
            E_max = E_max / float(network.ncell - uncounted)
        uncounted = 0
        for i in range(network.ncell * 2, network.ncell * 3):
            if i not in processed_p:
                uncounted = 0
                continue
            I_mean += processed_rd[i]
            I_max += processed_p[i]
        if network.ncell - uncounted != 0:
            I_mean = I_mean / float(network.ncell - uncounted)
            I_max = I_max / float(network.ncell - uncounted)

        # t = {key: value for dict in all_dicts for key, value in dict['t'].iteritems()}
        # temp until smoothing gets sorted
        gauss_E = gauss(network.osc_E, 1., network.tstop)
        gauss_I = gauss(network.osc_I, 1., network.tstop)
        # x = range(len(E_test))
        if plot:
            plt.plot(x, gauss_E)
            plt.title('gauss smoothing - E pop')
            plt.show()
            plt.plot(range(len(network.osc_E)), network.osc_E)
            plt.title('spike counts - E pop')
            plt.show()

        #window_len = min(int(2000./down_dt), len(down_t) - 1)
        filter_dt = 1.  # ms
        window_len = int(2000. / filter_dt)
        theta_filter = signal.firwin(window_len, [5., 10.], nyq=1000. / 2., pass_zero=False)
        theta_E = signal.filtfilt(theta_filter, [1.], gauss_E, padtype='even', padlen=window_len)
        theta_I = signal.filtfilt(theta_filter, [1.], gauss_I, padtype='even', padlen=window_len)
        if plot:
            plt.plot([i for i in range(len(theta_E))], theta_E)
            plt.title('theta')
            plt.show()

        #window_len = min(int(100./down_dt), len(down_t) - 1)
        window_len = int(200. / filter_dt)
        gamma_filter = signal.firwin(window_len, [30., 100.], nyq=1000. / 2., pass_zero=False)
        gamma_E = signal.filtfilt(gamma_filter, [1.], gauss_E, padtype='even', padlen=window_len)
        gamma_I = signal.filtfilt(gamma_filter, [1.], gauss_I, padtype='even', padlen=window_len)
        if plot:
            plt.plot([i for i in range(len(gamma_E))], gamma_E)
            plt.title('gamma')
            plt.show()

        peak_theta_osc_E = network.compute_peak_osc_freq(theta_E, 'theta')
        peak_theta_osc_I = network.compute_peak_osc_freq(theta_I, 'theta')
        peak_gamma_osc_E = network.compute_peak_osc_freq(gamma_E, 'gamma')
        peak_gamma_osc_I = network.compute_peak_osc_freq(gamma_I, 'gamma')


        # rec = {key: value for dict in all_dicts for key, value in dict['rec'].iteritems()}
        return {'E_mean_rate': E_mean, 'E_peak_rate': E_max, 'I_mean_rate': I_mean, "I_peak_rate": I_max, \
                'peak': peak_voltage, 'peak_theta_osc_E': peak_theta_osc_E, 'peak_theta_osc_I': \
                    peak_theta_osc_I, 'peak_gamma_osc_E': peak_gamma_osc_E, 'peak_gamma_osc_I': peak_gamma_osc_I, \
                "event": all_events, 'osc_E': network.osc_E}


# ==================== cell class                                                                                                                                                                                                                                                                                                                                                        # single cell

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
        self.pp = h.VecStim()
        #tstop in ms and mean_rate in s
        spikes = get_inhom_poisson_spike_times_by_thinning([mean_freq, mean_freq], [0, tstop], dt=0.025,
                                                           generator=local_random)
        vec = h.Vector(spikes)
        """self.pp.play(vec)
        print spikes"""
        if random.random() <= frac_active:  #vec = h.Vector([5, 200])
            self.pp.play(vec)
            network.FF_firing[gid] = spikes
        else:
            network.FF_firing[gid] = []

    def connect2target(self, target):
        nc = h.NetCon(self.pp, target)
        return nc

    def is_art(self):
        return 1


"""from dentate > stgen.py. temporary. personal issues with importing dentate -S"""


def get_inhom_poisson_spike_times_by_thinning(rate, t, dt=0.02, refractory=3., generator=None):
    if generator is None:
        generator = random
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
