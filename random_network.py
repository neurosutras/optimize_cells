from mpi4py import MPI
from neuron import h
import numpy as np
import random
import sys, time
import scipy.signal as signal
import matplotlib.pyplot as plt
np.seterr(divide='print')
# for h.lambda_f
h.load_file('stdlib.hoc')
# for h.stdinit
h.load_file('stdrun.hoc')


class Network(object):

    def __init__(self, FF_ncell, E_ncell, I_ncell, delay, pc, tstop, dt=0.025, e2e_prob=.05, e2i_prob=.05, i2i_prob=.05,
                 i2e_prob=.05, ff2i_weight=1., ff2e_weight=2., e2e_weight=1., e2i_weight=1., i2i_weight=.5,
                 i2e_weight=.5, ff_meanfreq=100, ff_frac_active=.8, ff2i_prob=.5, ff2e_prob=.5, std_dict=None, tau_E=2.,
                 tau_I=5., connection_seed=0, spikes_seed=1):
        """

        :param FF_ncell: int, number of cells in FF population
        :param delay:
        :param pc: ParallelContext object
        :param tstop: int, duration of sim
        :param dt: float, timestep in ms. default is .025
        :param e2e_prob: float, connection probability for E cell -> E cell
        :param e2i_prob: E cell -> I cell
        :param i2i_prob: I -> I
        :param i2e_prob: I -> E
        :param ff2i_weight: float, weight for NetCon object for connection between FF cells -> I cells
        :param ff2e_weight: FF -> E
        :param e2e_weight: E -> E
        :param e2i_weight: E -> I
        :param i2i_weight: I -> I
        :param i2e_weight: I -> E
        :param ff_meanfreq: int, mean frequency (lambda) for FF cells. spikes modeled as a poisson
        :param ff_frac_active: float, fraction active of FF cells
        :param ff2i_prob: float, connection probability for FF cell -> I cell
        :param ff2e_prob: FF -> E
        :param std_dict: dict, standard dev of weights for each projection (e.g., I->E='i2e') relative to the mean
        :param tau_E: float, tau decay for excitatory synapses
        :param tau_I: float, tau decay for inhib synapses
        :param connection_seed: int
        :param spikes_seed: int
        """
        self.pc = pc
        self.delay = delay
        self.tstop = tstop
        if dt is None:
            dt = h.dt
        self.dt = dt
        self.FF_ncell = int(FF_ncell)
        self.E_ncell = int(E_ncell)
        self.I_ncell = int(I_ncell)
        self.FF_firing = {}
        self.ff_frac_active = ff_frac_active
        self.tau_E = tau_E
        self.tau_I = tau_I
        self.prob_dict = {'e2e': e2e_prob, 'e2i': e2i_prob, 'i2i': i2i_prob, 'i2e': i2e_prob, 'ff2i': ff2i_prob,
                          'ff2e': ff2e_prob}
        self.weight_dict = {'ff2i': ff2i_weight, 'ff2e': ff2e_weight, 'e2e': e2e_weight, 'e2i': e2i_weight,
                            'i2i': i2i_weight, 'i2e': i2e_weight}
        self.weight_std_dict = std_dict
        self.total_cells = FF_ncell + I_ncell + E_ncell
        self.cell_index = {'FF': (0, FF_ncell), 'I': (FF_ncell, FF_ncell + I_ncell),
                           'E': (FF_ncell + I_ncell, self.total_cells)}
        self.connectivity_index_dict = {'ff2i': (self.cell_index['FF'], self.cell_index['I']),  # exclusive [x, y)
                                        'ff2e': (self.cell_index['FF'], self.cell_index['E']),
                                        'e2e': (self.cell_index['E'], self.cell_index['E']),
                                        'e2i': (self.cell_index['E'], self.cell_index['I']),
                                        'i2i': (self.cell_index['I'], self.cell_index['I']),
                                        'i2e': (self.cell_index['I'], self.cell_index['E'])}

        self.ff_meanfreq = ff_meanfreq
        self.event_rec = {}

        self.local_random = random.Random()
        self.connection_seed = connection_seed
        self.spikes_seed = spikes_seed

        self.mknetwork()
        self.voltage_record()
        self.spike_record()
        self.pydicts = {}

    def mknetwork(self):
        self.mkcells()
        self.check_cell_type_correct()
        self.connectcells()

    def mkcells(self):
        rank = int(self.pc.id())
        nhost = int(self.pc.nhost())
        self.cells = []
        self.gids = []
        for i in range(rank, self.total_cells, nhost):
            if i < self.FF_ncell:
                self.local_random.seed(self.spikes_seed + i)
                cell = FFCell(self.tstop, self.ff_meanfreq, self.ff_frac_active, self, i,
                              local_random=self.local_random)
            else:
                if i in range(self.cell_index['E'][0], self.cell_index['E'][1]):
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

    # def createpairs(self, prob, input_indices, output_indices):
    #     pair_list = []
    #     for i in input_indices:
    #         for o in output_indices:
    #             if self.local_random.random() <= prob:
    #                 pair_list.append((i, o))
    #     for elem in pair_list:
    #         x, y = elem
    #         if x == y: pair_list.remove(elem)
    #     return pair_list

    def get_cell_type(self, gid):
        if gid in range(self.cell_index['FF'][1]):
            return None
        elif gid in range(self.cell_index['I'][0], self.cell_index['I'][1]):
            return 'FS'
        else:
            return 'RS'

    def connectcells(self):
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

            indices = self.connectivity_index_dict[connection]
            inp = indices[0]
            out = indices[1]
            rank_subset_gids = [i for i in self.gids if i in range(inp[0], inp[1])]
            for presyn_gid in rank_subset_gids:
                for target_gid in range(out[0], out[1]):
                    if presyn_gid == target_gid:
                        continue
                    if self.local_random.random() <= self.prob_dict[connection]:
                        if self.pc.gid_exists(target_gid):
                            target = self.pc.gid2cell(target_gid)
                            if connection[0] == 'i':
                                syn = target.synlist[1]
                            else:
                                syn = target.synlist[0]
                            nc = self.pc.gid_connect(presyn_gid, syn)
                            nc.delay = self.delay
                            weight = self.local_random.gauss(mu, mu * std_factor)
                            while weight < 0.: weight = self.local_random.gauss(mu, mu * std_factor)
                            nc.weight[0] = weight
                            self.ncdict[(presyn_gid, target_gid)] = nc


    # Instrumentation - stimulation and recordi
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

    def voltage_record(self):
        self.voltage_tvec = {}
        self.voltage_recvec = {}
        for i, cell in enumerate(self.cells):
            if cell.is_art(): continue
            tvec = h.Vector()
            tvec.record(h._ref_t)
            rec = h.Vector()
            rec.record(getattr(cell.sec(.5), '_ref_v'))
            self.voltage_tvec[self.gids[i]] = tvec
            self.voltage_recvec[self.gids[i]] = rec

    def vecdict_to_pydict(self, vecdict):
        this_pydict = dict()
        for key, value in vecdict.iteritems():
            this_pydict[key] = np.array(value)
        return this_pydict

    def summation(self, vecdict, dt=.025):
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

    def py_summation(self, spikes_dict, t):
        self.E_sum = np.zeros_like(t)
        self.I_sum = np.zeros_like(t)
        self.FF_sum = np.zeros_like(t)
        for i in range(self.total_cells):
            cell_type = self.get_cell_type(i)
            if len(spikes_dict[i]) > 0:
                binned_spikes = get_binned_spike_train(spikes_dict[i], t)
                if cell_type == 'RS':
                    self.E_sum = np.add(self.E_sum, binned_spikes)
                elif cell_type == 'FS':
                    self.I_sum = np.add(self.I_sum, binned_spikes)
                else:
                    self.FF_sum = np.add(self.FF_sum, binned_spikes)

    def run(self):
        self.pc.set_maxstep(10)
        h.stdinit()
        h.dt = self.dt
        self.pc.psolve(self.tstop)

    def compute_mean_max_firing_per_cell(self, gauss_firing_rates):
        rate_dict = {}
        peak_dict = {}
        for key, val in gauss_firing_rates.iteritems():
            if len(val) > 0:
                rate_dict[key] = np.mean(val)
                peak_dict[key] = np.max(val)
        return rate_dict, peak_dict

    def compute_pop_firing_features(self, bounds, rate_dict, peak_dict):
        uncounted = 0
        mean = 0
        max_firing = 0
        for i in range(bounds[0], bounds[1]):
            if i not in rate_dict:
                uncounted += 1
                continue
            mean += rate_dict[i]
            max_firing += peak_dict[i]
        ncell = bounds[1] - bounds[0]
        if ncell - uncounted != 0:
            mean = mean / float(ncell - uncounted)
            max_firing = max_firing / float(ncell - uncounted)

        return mean, max_firing

    def plot_voltage_trace(self, vecdict, all_events, dt=.025):
        down_dt = 1.
        ms_step = int(down_dt / dt)
        for i in range(self.cell_index['I'][0], self.cell_index['I'][1]):
            ms_rec = []
            for j, v in enumerate(vecdict[i]):
                if j % ms_step == 0: ms_rec.append(v)
            ev = [int(event/down_dt) for event in all_events[i]]
            plt.plot(range(len(ms_rec)), ms_rec, '-gD', markevery=ev)
            plt.title('v trace' + str(i))
            plt.show()

    def plot_cell_activity(self, gauss_firing_rates, t):
        import seaborn as sns

        counter = 0
        wrap = 0
        populations = ['FF', 'I', 'E']
        ncell = [self.FF_ncell, self.I_ncell, self.E_ncell]
        last_idx = [self.FF_ncell - 1, self.cell_index['I'][1] - 1, self.cell_index['E'][1] - 1]
        hm = np.zeros((ncell[counter], len(t)))
        for key, val in gauss_firing_rates.iteritems():
            if len(gauss_firing_rates[key]) != 0:
                hm[wrap] = gauss_firing_rates[key]
            wrap += 1
            if key == last_idx[counter]:
                sns.heatmap(hm)
                plt.title(populations[counter])
                plt.show()
                counter += 1
                wrap = 0
                if counter < len(ncell): hm = np.zeros((ncell[counter], len(t)))

    def plot_smoothing(self, gauss_E):
        plt.plot(range(len(gauss_E)), gauss_E)
        plt.title('gauss smoothing - E pop')
        plt.show()
        plt.plot(range(len(self.E_sum)), self.E_sum)
        plt.title('spike counts - E pop')
        plt.show()

    def plot_bands(self, theta_E, gamma_E, gauss_E, theta_FF, gamma_FF, gauss_FF):
        plt.plot(range(len(theta_E)), theta_E, label="theta E")
        gauss_E = np.subtract(gauss_E, np.mean(gauss_E))
        plt.plot(range(len(gauss_E)), gauss_E, label="smoothed spike rate")
        plt.legend(loc=1)
        plt.title('theta E')
        plt.show()
        plt.plot(range(len(gamma_E)), gamma_E, label="gamma E")
        plt.plot(range(len(gauss_E)), gauss_E, label="smothed spike rate")
        plt.legend(loc=1)
        plt.title('gamma E')
        plt.show()

        plt.plot(range(len(theta_FF)), theta_FF, label="theta FF")
        gauss_FF = np.subtract(gauss_FF, np.mean(gauss_FF))
        plt.plot(range(len(gauss_FF)), gauss_FF, label="smoothed spike rate")
        plt.legend(loc=1)
        plt.title('theta FF')
        plt.show()
        plt.plot(range(len(gamma_FF)), gamma_FF, label="gamma FF")
        plt.plot(range(len(gauss_FF)), gauss_FF, label="smothed spike rate")
        plt.legend(loc=1)
        plt.title('gamma FF')
        plt.show()

    def get_bands_of_interest(self, binned_dt, plot):
        gauss_E = gauss(self.E_sum, binned_dt)
        gauss_I = gauss(self.I_sum, binned_dt)
        gauss_FF = gauss(self.FF_sum, binned_dt)

        filter_dt = 1.  # ms
        window_len = int(2000. / filter_dt)
        theta_band = [5., 10.]
        theta_E, theta_I, theta_FF = filter_band(gauss_E, gauss_I, gauss_FF, window_len, theta_band)
        window_len = int(200. / filter_dt)
        gamma_band = [30., 100.]
        gamma_E, gamma_I, gamma_FF = filter_band(gauss_E, gauss_I, gauss_FF, window_len, gamma_band)

        if plot:
            self.plot_smoothing(gauss_E)
            self.plot_bands(theta_E, gamma_E, gauss_E, theta_FF, gamma_FF, gauss_FF)

        return theta_E, theta_I, gamma_E, gamma_I

    def event_dict_to_firing_rate(self, spike_dict, binned_dt=1.):
        gauss_firing_rates = {}
        t = np.arange(0., self.tstop + binned_dt, binned_dt)
        for key, val in spike_dict.iteritems():
            if len(val) > 0:
                spikes_t = get_binned_spike_train(val, t)
                smoothed = gauss(spikes_t, binned_dt)
                gauss_firing_rates[key] = smoothed
            else:
                gauss_firing_rates[key] = np.zeros_like(t)
        return gauss_firing_rates

    def get_active_pop_stats(self, firing_rates_dict, t, threshold=1., plot=False):
        frac_active = {}
        mean_firing_active = {}

        for population in self.cell_index:
            frac_active[population], mean_firing_active[population] = \
                self.compute_active_pop_stats(firing_rates_dict, t, threshold, self.cell_index[population])

        if plot:
            for population in self.cell_index:
                fig, axes = plt.subplots(1, 2)
                axes[0].plot(t, frac_active[population])
                axes[0].set_title('Fraction active cells')
                axes[1].plot(t, mean_firing_active[population])
                axes[1].set_title('Mean firing rate of active cells')
                fig.suptitle('Population: %s' % population)
                plt.show()
        return frac_active, mean_firing_active

    def compute_active_pop_stats(self, firing_rates_dict, t, threshold, bounds):
        frac_active = np.zeros_like(t)
        mean_firing_active = np.zeros_like(frac_active)
        for j in range(bounds[0], bounds[1]):
            active_indexes = np.where(firing_rates_dict[j] >= threshold)[0]
            if np.any(active_indexes):
                frac_active[active_indexes] += 1.
                mean_firing_active[active_indexes] += firing_rates_dict[j][active_indexes]
        active_indexes = np.where(frac_active > 0)[0]
        if np.any(active_indexes):
            mean_firing_active[active_indexes] = np.divide(mean_firing_active[active_indexes],
                                                           frac_active[active_indexes])

        frac_active = np.divide(frac_active, float(bounds[1] - bounds[0]))

        return frac_active, mean_firing_active

    def compute_pop_firing(self, firing_rates_dict, bounds):
        pop_rate = [0] * self.tstop
        for i in range(bounds[0], bounds[1]):
            if len(firing_rates_dict[i]) == 0: continue
            for j in range(self.tstop):
                if firing_rates_dict[i][j] > 1.: pop_rate[j] += firing_rates_dict[i][j]
        return pop_rate

    def compute_envelope_ratio(self, band, pop_rate, t, label=None, plot=False):
        hilb_transform = np.abs(signal.hilbert(band))
        mean_envelope = np.mean(hilb_transform)
        mean_rate = np.mean(pop_rate)
        if mean_rate > 0.:
            ratio = mean_envelope / mean_rate
            if plot:
                plt.plot(t, hilb_transform, label='hilbert transform')
                plt.plot(t, pop_rate, label='pop firing rate')
                plt.axhline(y=np.mean(hilb_transform), color='red')
                plt.axhline(y=np.mean(pop_rate), color='red')
                plt.legend(loc=1)
                if label is None:
                    label = 'ratio: %.3E' % ratio
                else:
                    label += ' ratio: %.3E' % ratio
                plt.title(label)
                plt.show()
        else:
            ratio = 0.
        return ratio

    def convert_ncdict_to_weights(self):
        connections = {}
        for pair, nc in self.ncdict.iteritems():
            connections[pair] = nc.weight[0]
        return connections

    def print_connections(self, connections):
        connections_per_cell = {}
        for pair, weight in connections.iteritems():
            pre, post = pair
            if pre not in connections_per_cell:
                connections_per_cell[pre] = [post]
            else:
                li = connections_per_cell[pre]
                li.append(post)
                connections_per_cell[pre] = li
        print "connections and weights: \n", connections
        print "connections by presynaptic cell: \n", connections_per_cell

    def plot_adj_matrix(self, connections):
        import seaborn as sns
        if self.total_cells > 100: return
        """source = row; target = col"""
        matrixmap = np.zeros((self.total_cells, self.total_cells))
        for pair, weight in connections.iteritems():
            source, target = pair
            matrixmap[source][target] = weight
        ax = sns.heatmap(matrixmap)
        ax.hlines([self.FF_ncell - 1, self.FF_ncell + self.I_ncell - 1], color='white', *ax.get_xlim())
        ax.vlines([self.FF_ncell - 1, self.FF_ncell + self.I_ncell - 1], color='white', *ax.get_ylim())
        plt.show()

    def check_cell_type_correct(self):
        for cell in self.gids:
            cell_type = self.get_cell_type(cell)
            if cell_type is None and isinstance(self.pc.gid2cell(cell), IzhiCell):
                print cell, " is not a FF cell but a ", type(self.pc.gid2cell(cell))
            elif cell_type in ['FS', 'RS'] and not isinstance(self.pc.gid2cell(cell), IzhiCell):
                print cell, " is not a Izhi cell but a ", type(self.pc.gid2cell(cell))
            elif cell_type is 'FS' and self.pc.gid2cell(cell).izh.a != .02:
                print cell, " is not a inhibitory Izhi cell but an excitatory one"
            elif cell_type is 'RS' and self.pc.gid2cell(cell).izh.a != .1:
                print cell, " is not an excitatory Izhi cell but an inhibitory one"



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


def gauss(spikes, dt, filter_duration=100.):
    pad_len = min(2 * int(filter_duration/dt), len(spikes))
    filter_duration = filter_duration  # ms
    filter_t = np.arange(-filter_duration, filter_duration, dt)
    sigma = filter_duration / 3. / np.sqrt(2.)
    gaussian_filter = np.exp(-(filter_t / sigma) ** 2.)
    gaussian_filter /= np.trapz(gaussian_filter, dx=dt / 1000.)

    mirror_beginning = spikes[:pad_len][::-1]
    mirror_end = spikes[-pad_len:][::-1]
    modified_spikes = np.append(np.append(mirror_beginning, spikes), mirror_end)

    signal = np.convolve(modified_spikes, gaussian_filter)
    signal_t = np.arange(0., len(signal) * dt, dt)
    signal = signal[int(filter_duration / dt) + pad_len:][:len(spikes)]
    return signal


def filter_band(E, I, FF, window_len, band):
    filt = signal.firwin(window_len, band, nyq=1000. / 2., pass_zero=False)
    E_band = signal.filtfilt(filt, [1.], E, padtype='even', padlen=window_len)
    I_band = signal.filtfilt(filt, [1.], I, padtype='even', padlen=window_len)
    FF_band = signal.filtfilt(filt, [1.], FF, padtype='even', padlen=window_len)

    return E_band, I_band, FF_band


def peak_from_spectrogram(freq, title='not specified', dt=1., plot=False):
    freq, density = signal.periodogram(freq, 1000. / dt)
    if plot:
        plt.plot(freq, density)
        plt.title(title)
        plt.show()
    max_dens = np.max(density)
    loc = np.where(density == max_dens)
    if len(loc[0]) > 0:
        peak_idx = loc[0][0]
        return freq[peak_idx]
    return 0.


def get_binned_spike_train(spikes, t):
    binned_spikes = np.zeros_like(t)
    if len(spikes) > 0:
        spike_idx = [np.where(t >= spike_time)[0][0] for spike_time in spikes]
        binned_spikes[spike_idx] = 1.
    return binned_spikes


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
