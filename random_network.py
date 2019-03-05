from mpi4py import MPI
from neuron import h
import numpy as np
import random
import sys, time
import scipy.signal
import matplotlib.pyplot as plt
import seaborn as sns
from baks import baks
# for h.lambda_f
h.load_file('stdlib.hoc')
# for h.stdinit
h.load_file('stdrun.hoc')


class Network(object):

    def __init__(self, FF_ncell, E_ncell, I_ncell, delay, pc, tstop, dt=0.025, e2e_prob=.05, e2i_prob=.05, i2i_prob=.05,
                 i2e_prob=.05, ff2i_weight=1., ff2e_weight=2., e2e_weight=1., e2i_weight=1., i2i_weight=.5,
                 i2e_weight=.5, ff_meanfreq=100, ff_frac_active=.8, ff2i_prob=.5, ff2e_prob=.5, std_dict=None, tau_E=2.,
                 tau_I=5., connection_seed=0, spikes_seed=1, plot_ncells=5):
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
        :param plot_ncells: int, how many cells to sample (per population) for plotting voltage trace and firing rate
        """
        self.npop = 3  # number of populations - 3 for FF, inhib, excit cells
        self.pc = pc
        self.delay = delay
        self.tstop = tstop
        if dt is None:
            dt = h.dt
        self.dt = dt
        self.FF_ncell = int(FF_ncell)
        self.E_ncell = int(E_ncell)
        self.I_ncell = int(I_ncell)

        self.FF_spikes_dict = {}
        self.ff_frac_active = ff_frac_active
        self.ff_meanfreq = ff_meanfreq

        self.tau_E = tau_E
        self.tau_I = tau_I

        self.prob_dict = {'e2e': e2e_prob, 'e2i': e2i_prob, 'i2i': i2i_prob, 'i2e': i2e_prob, 'ff2i': ff2i_prob,
                          'ff2e': ff2e_prob}
        self.weight_dict = {'ff2i': ff2i_weight, 'ff2e': ff2e_weight, 'e2e': e2e_weight, 'e2i': e2i_weight,
                            'i2i': i2i_weight, 'i2e': i2e_weight}
        self.weight_std_dict = std_dict
        self.total_cells = FF_ncell + I_ncell + E_ncell
        self.plot_ncells = plot_ncells
        self.cell_index = {'FF': (0, FF_ncell), 'I': (FF_ncell, FF_ncell + I_ncell),  # [x, y)
                           'E': (FF_ncell + I_ncell, self.total_cells)}
        self.connectivity_index_dict = {'ff2i': (self.cell_index['FF'], self.cell_index['I']),  # exclusive [x, y)
                                        'ff2e': (self.cell_index['FF'], self.cell_index['E']),
                                        'e2e': (self.cell_index['E'], self.cell_index['E']),
                                        'e2i': (self.cell_index['E'], self.cell_index['I']),
                                        'i2i': (self.cell_index['I'], self.cell_index['I']),
                                        'i2e': (self.cell_index['I'], self.cell_index['E'])}

        self.local_random = random.Random()
        self.connection_seed = connection_seed
        self.spikes_seed = spikes_seed

        self.mknetwork()
        self.voltage_record()
        self.spike_record()

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

    def get_cell_type(self, gid):
        if gid in range(self.cell_index['FF'][1]):
            return None
        elif gid in range(self.cell_index['I'][0], self.cell_index['I'][1]):
            return 'FS'
        else:
            return 'RS'

    def get_weight(self, connection):
        """
        want to reduce std if std is problematic, i.e, makes it possible to sample a negative weight
        :param connection: str
        """
        mu = self.weight_dict[connection]
        if self.weight_std_dict[connection] >= 2. / 3. / np.sqrt(2.):
            print 'network.connectcells: connection: %s; reducing std to avoid negative weights: %.2f' % \
                  (connection, self.weight_std_dict[connection])
            self.weight_std_dict[connection] = 2. / 3. / np.sqrt(2.)
        std_factor = self.weight_std_dict[connection]
        return mu, std_factor

    def connectcells(self):
        """
        if the pre-synaptic neuron is E/FF - connect to 1st synapse (excitatory), otherwise connect to
        2nd synapse (inhibitory) -- see IzhiCell
        connections based on connection probability dict and weight of connection is based on a gaussian distribution

        restrictions: cells cannot connect with themselves, weights cannot be negative, and the presynaptic cell
        needs to live on this rank. (also, FF cells should not connect with other FF cells.)
        """
        rank = int(self.pc.id())
        self.local_random.seed(self.connection_seed + rank)
        self.ncdict = {}

        for connection in ['ff2i', 'ff2e', 'e2e', 'e2i', 'i2i', 'i2e']:
            mu, std_factor = self.get_weight(connection)
            pre_gids, post_gids = self.connectivity_index_dict[connection]
            rank_subset_gids = [i for i in self.gids if i in range(post_gids[0], post_gids[1])]

            for presyn_gid in range(pre_gids[0], pre_gids[1]):
                for target_gid in rank_subset_gids:
                    if presyn_gid == target_gid:
                        continue
                    if self.local_random.random() <= self.prob_dict[connection]:
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
            self.pc.spike_record(nc.srcgid(), tvec, idvec)  # Alternatively, could use nc.record(tvec)
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

    def convert_hoc_vec_dict(self, hoc_vec_dict):
        this_array_dict = dict()
        for key, value in hoc_vec_dict.iteritems():
            this_array_dict[key] = np.array(value)
        return this_array_dict

    def get_spikes_dict(self):
        return self.convert_hoc_vec_dict(self.spike_tvec)

    def get_voltage_rec_dict(self):
        return self.convert_hoc_vec_dict(self.voltage_recvec)

    def summation(self, spikes_dict, t):
        """
        sums up spikes per population
        :param spikes_dict: dict s.t. key = gid, val = array of spike times
        :param t: array
        """
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

    def compute_mean_max_firing_per_cell(self, smoothed_firing_rates):
        """
        computes mean and max firing rate (Hz) per cell.
        :return: dict of rates, and dict of peaks s.t. key = gid, val = scalar
        """
        rate_dict = {}
        peak_dict = {}
        for key, val in smoothed_firing_rates.iteritems():
            if len(val) > 0:
                rate_dict[key] = np.mean(val)
                peak_dict[key] = np.max(val)
        return rate_dict, peak_dict

    def compute_pop_firing_features(self, bounds, rate_dict, peak_dict):
        """
        from dictionaries containing the mean rate and peaks of each cell, returns
        a value for the mean and peak firing rate (Hz) for the specified population
        :param bounds: a tuple representing the contiguous range of gids associated with the population
        :param rate_dict: dict, key = gid, val = scalar
        :param peak_dict: dict, key = gid, val = scalar
        """
        count = 0;
        mean = 0;
        max_firing = 0
        for i in range(bounds[0], bounds[1]):
            if i not in rate_dict: continue
            mean += rate_dict[i]
            max_firing += peak_dict[i]
            count += 1
        if count != 0:
            mean = mean / float(count)
            max_firing = max_firing / float(ncell - uncounted)

        return mean, max_firing

    def sample_cells_for_plotting(self):
        """
        for plot_voltage_trace. sample a number of cells from the E and I populations to be plotted
        :return: list of gids
        """
        I_sample = range(self.cell_index['I'][0], self.cell_index['I'][1])
        E_sample = range(self.cell_index['E'][0], self.cell_index['E'][1])
        if self.I_ncell > self.plot_ncells:
            I_sample = self.local_random.sample(I_sample, self.plot_ncells)
        if self.E_ncell > self.plot_ncells:
            E_sample = self.local_random.sample(E_sample, self.plot_ncells)
        return I_sample + E_sample

    def plot_voltage_trace(self, v_dict, spikes_dict, dt=.025):
        """
        plots voltage trace for a random subset of E and I cells. downsamples to 1 ms for plotting.
        markers represent spikes
        """
        down_dt = 1.
        ms_step = int(down_dt / dt)
        sampled_cells = self.sample_cells_for_plotting()
        for i in sampled_cells:
            ms_rec = []
            for j, v in enumerate(v_dict[i]):
                if j % ms_step == 0: ms_rec.append(v)
            ev = [int(event / down_dt) for event in spikes_dict[i]]
            fig = plt.figure()
            plt.plot(range(len(ms_rec)), ms_rec, '-gD', markevery=ev)
            plt.title('v trace ' + self.get_cell_type(i) + str(i))
            fig.show()
        plt.show()

    def plot_population_firing_rates(self, firing_rates, t):
        """
        relies on the fact that populations have continuous gids. go through all the gids, and when
        we've hit an end of a population, plot the colormap

        :param firing_rates: dict of array
        :param t: array
        """
        wrap = 0
        counter = 0
        populations = ['FF', 'I', 'E']
        ncell = [self.FF_ncell, self.I_ncell, self.E_ncell]
        last_idx = [self.FF_ncell - 1, self.cell_index['I'][1] - 1, self.cell_index['E'][1] - 1]
        hm = np.zeros((ncell[counter], len(t)))
        for key in range(self.total_cells):
            if len(firing_rates[key]) != 0:
                hm[wrap] = firing_rates[key]
            wrap += 1
            if key == last_idx[counter]:
                sns.heatmap(hm)
                plt.title(populations[counter])
                plt.show()
                wrap = 0
                counter += 1
                if key < len(ncell): hm = np.zeros((ncell[counter], len(t)))

    def plot_smoothing(self, gauss_E):
        self.plot_two_traces(gauss_E, self.E_sum, 'smoothed spike rate vs. population spiking - E')

    def plot_two_traces(self, one, two, title):
        plt.plot(range(len(one)), one)
        plt.plot(range(len(two)), two)
        plt.title(title)
        plt.show()

    def plot_bands(self, t, theta_E, gamma_E, input_E, theta_FF, gamma_FF, input_FF):
        """
        plots filtered bands vs. their input
        :param t: array
        :param theta_E: array
        :param gamma_E: array
        :param input_E: array
        :param theta_FF: array
        :param gamma_FF: array
        :param input_FF: array
        """
        input_E = np.subtract(input_E, np.mean(input_E))
        plt.plot(t, input_E, label="input rate")
        plt.plot(t, theta_E, label="theta")
        plt.legend(loc=1)
        plt.title('theta E')
        plt.show()
        plt.plot(t, input_E, label="input rate")
        plt.plot(t, gamma_E, label="gamma")
        plt.legend(loc=1)
        plt.title('gamma E')
        plt.show()

        input_FF = np.subtract(input_FF, np.mean(input_FF))
        plt.plot(t, input_FF, label="input rate")
        plt.plot(t, theta_FF, label="theta")
        plt.legend(loc=1)
        plt.title('theta FF')
        plt.show()
        plt.plot(t, input_FF, label="input rate")
        plt.plot(t, gamma_FF, label="gamma")
        plt.legend(loc=1)
        plt.title('gamma FF')
        plt.show()

    """def get_bands_of_interest(self, t, filter_dt, plot=False):
        # gauss_E = gauss(self.E_sum, binned_dt)
        #  gauss_I = gauss(self.I_sum, binned_dt)
        #  gauss_FF = gauss(self.FF_sum, binned_dt)
        # gauss_E, _ = baks(self.E_sum, t, self.baks_alpha, self.baks_beta)
        # gauss_I, _ = baks(self.I_sum, t, self.baks_alpha, self.baks_beta)
        # gauss_FF, _ = baks(self.FF_sum, t, self.baks_alpha, self.baks_beta)

        window_len = int(2000. / filter_dt)
        theta_band = [5., 10.]
        #theta_E, theta_I, theta_FF = filter_band(gauss_E, gauss_I, gauss_FF, window_len, theta_band)
        theta_E, theta_I, theta_FF = filter_band(self.E_sum, self.I_sum, self.FF_sum, window_len, theta_band, filter_dt)
        window_len = int(200. / filter_dt)
        gamma_band = [30., 100.]
        #gamma_E, gamma_I, gamma_FF = filter_band(gauss_E, gauss_I, gauss_FF, window_len, gamma_band)
        gamma_E, gamma_I, gamma_FF = filter_band(self.E_sum, self.I_sum, self.FF_sum, window_len, gamma_band, filter_dt)

        if plot:
            # self.plot_smoothing(gauss_E)
            # self.plot_bands(theta_E, gamma_E, gauss_E, theta_FF, gamma_FF, gauss_FF)
            self.plot_bands(t, theta_E, gamma_E, self.E_sum, theta_FF, gamma_FF, self.FF_sum)

        return theta_E, theta_I, gamma_E, gamma_I"""

    def get_active_pop_stats(self, firing_rates_dict, t, threshold=1., plot=False):
        """
        get mean firing for active populations over simulation. plot if needed.
        :param threshold: Hz, firing rate threshold above which a cell is considered "active"
        """
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
        """
        compute frac active and mean firing for active cells
        :param threshold: Hz, firing rate threshold above which a cell is considered "active"
        :param bounds: gid range for population
        """
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
        """
        from individual cell firing rates, compute population firing rate
        :param firing_rates_dict: dict; key = gid, val = array of firing rates
        :param bounds: range of gids for population
        """
        pop_rate = [0] * self.tstop
        for i in range(bounds[0], bounds[1]):
            if len(firing_rates_dict[i]) == 0: continue
            for j in range(self.tstop):
                if firing_rates_dict[i][j] > 1.: pop_rate[j] += firing_rates_dict[i][j]
        return pop_rate

    """def compute_envelope_ratio(self, band, pop_rate, t, label=None, plot=False):
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
        return ratio"""

    def plot_envelope_ratio(self, t, hilb_transform, pop_rate, label, ratio):
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

    def calculate_envelope_ratio(self, pop_rate, band, pad_len):
        hilb_transform = np.abs(scipy.signal.hilbert(band))[pad_len:][:-pad_len]
        mean_envelope = np.mean(hilb_transform)
        mean_rate = np.mean(pop_rate)
        if mean_rate > 0:
            ratio = mean_envelope / mean_rate
        else:
            ratio = 0.
        return ratio, hilb_transform

    def get_bands_of_interest(self, filter_dt, basic_rate_E, basic_rate_I, basic_rate_FF, t, plot=False):
        pad_len_theta = window_len = int(2000. / filter_dt)
        theta_band = [5., 10.]
        theta_E, theta_I, theta_FF = untruncated_filter_band(basic_rate_E, basic_rate_I, basic_rate_FF,
                                                             window_len, theta_band, pad_len_theta, filter_dt)

        pad_len_gamma = window_len = int(200. / filter_dt)
        gamma_band = [30., 100.]
        gamma_E, gamma_I, gamma_FF = untruncated_filter_band(basic_rate_E, basic_rate_I, basic_rate_FF,
                                                             window_len, gamma_band, pad_len_gamma, filter_dt)
        if plot:
            self.plot_bands(t, theta_E[pad_len_theta:][:len(t)], gamma_E[pad_len_gamma:][:len(t)],
                            basic_rate_E, theta_FF[pad_len_theta:][:len(t)],
                            gamma_FF[pad_len_gamma:][:len(t)], basic_rate_FF)

        return theta_E, theta_I, gamma_E, gamma_I, pad_len_theta, pad_len_gamma

    def get_envelope_ratio(self, pop_rates, t, filter_dt, label=None, plot=False):
        basic_rate_E = self.count_to_rate_basic(self.E_sum, self.E_ncell)
        basic_rate_I = self.count_to_rate_basic(self.I_sum, self.I_ncell)
        basic_rate_FF = self.count_to_rate_basic(self.FF_sum, self.FF_ncell)

        theta_E, theta_I, gamma_E, gamma_I, pad_len_theta, pad_len_gamma = self.get_bands_of_interest(filter_dt,
                                                                                                      basic_rate_E,
                                                                                                      basic_rate_I,
                                                                                                      basic_rate_FF, t,
                                                                                                      plot)

        ratios = {}
        bands = {'theta_I': theta_I, 'gamma_I': gamma_I, 'theta_E': theta_E, 'gamma_E': gamma_E}
        truncated_bands = {}
        for label, band in bands.iteritems():
            if label[-1] == 'I':
                pop_rate = pop_rates[1]
            else:
                pop_rate = pop_rates[2]
            if label.find('theta') != -1:
                pad_len = pad_len_theta
            else:
                pad_len = pad_len_gamma
            ratio, hilb_transform = self.calculate_envelope_ratio(pop_rate, band, pad_len)
            ratios[label] = ratio
            truncated_bands[label] = band[pad_len:][:len(t)]

            if plot:
                self.plot_envelope_ratio(t, hilb_transform, pop_rate, label, ratio)

        return ratios, truncated_bands

    def convert_ncdict_to_weights(self):
        """
        can't collapse NCs on each rank onto the master rank. instead, keep track of
        connections and their weights to be gathered by the master rank.
        """
        connections = {}
        for pair, nc in self.ncdict.iteritems():
            connections[pair] = nc.weight[0]
        return connections

    def print_connections(self, connections):
        """
        prints ALL of the connections, first in (pre, post) : weight format, then
        pre : [post, ..., post] form.
        """
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
        """
        plots connections in a matrix map. color represents weight of connection
        :param connections: dict, key = (pre, post), val = weight
        """
        if self.total_cells > 100: return
        #source = row; target = col
        matrixmap = np.zeros((self.total_cells, self.total_cells))
        for pair, weight in connections.iteritems():
            source, target = pair
            matrixmap[source][target] = weight
        ax = sns.heatmap(matrixmap)
        ax.hlines([self.FF_ncell, self.FF_ncell + self.I_ncell], color='white', *ax.get_xlim())
        ax.vlines([self.FF_ncell, self.FF_ncell + self.I_ncell], color='white', *ax.get_ylim())
        plt.show()

        FF_matrixmap = matrixmap[:self.FF_ncell]
        ax = sns.heatmap(FF_matrixmap)
        ax.vlines([self.FF_ncell, self.FF_ncell + self.I_ncell], color='white', *ax.get_ylim())
        plt.show()

    def count_to_rate_basic(self, spike_sums, ncell, dt=1.):
        """converts spike count (summed over population) to average instaneous firing rate for one cell per timestep."""
        factor = dt / 1000.
        rate = np.divide(np.divide(spike_sums, float(ncell)), factor)
        return rate

    def check_cell_type_correct(self):
        """
        goes over each cell and checks whether or not the cell is the cell type it is supposed to be
        (e.g., cells with gids in a certain range must be FF cells/Hoc object)
        """
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
        if local_random.random() <= frac_active:
            self.pp.play(vec)
            network.FF_spikes_dict[gid] = np.array(spikes)
        else:
            network.FF_spikes_dict[gid] = []

    def connect2target(self, target):
        nc = h.NetCon(self.pp, target)
        return nc

    def is_art(self):
        return 1


def infer_firing_rates(spike_times_dict, t, alpha, beta, pad_dur, plot=False):
    """

    :param spike_times_dict: dict of array
    :param t: array
    :param baks_alpha: float
    :param baks_beta: float
    :param pad_dur: float
    :param plot: bool
    :return: dict of array
    """
    inferred_firing_rates = {}
    for gid, spike_train in spike_times_dict.iteritems():
        if len(spike_train) > 0:
            # spikes_t = get_binned_spike_train(val, t)
            # smoothed = gauss(spikes_t, binned_dt)
            smoothed = padded_baks(spike_train, t, alpha=alpha, beta=beta, pad_dur=pad_dur)
            """if plot:
                fig = plt.figure()
                plt.plot(spike_train, np.ones_like(spike_train), 'k.')
                plt.plot(t, smoothed)
                plt.title('Inferred firing rate - cell: %i' % gid)
                fig.show()"""
            inferred_firing_rates[gid] = smoothed
        else:
            inferred_firing_rates[gid] = np.zeros_like(t)
    return inferred_firing_rates


def plot_inferred_rates(inferred_firing_rates, spike_times_dict, t, network):
    sampled_gids = network.sample_cells_for_plotting()
    sample_per_pop = network.plot_ncells

    fig = plt.figure()
    fig.suptitle('inferred spike rates')
    for i, gid in enumerate(sampled_gids):
        smoothed = inferred_firing_rates[gid]
        spike_train = spike_times_dict[gid]
        ax = fig.add_subplot(2, sample_per_pop, i + 1)
        ax.plot(spike_train, np.ones_like(spike_train), 'k.')
        ax.set_title("cell " + str(gid))
        if network.get_cell_type(gid) == 'RS':
            color = 'red'
        else:
            color = 'blue'
        ax.plot(t, smoothed, color=color)
    plt.show()  # clear stack


def padded_baks(spike_times, t, alpha, beta, pad_dur=500.):
    """
    Expects spike times in ms. Uses mirroring to pad the edges to avoid edge artifacts. Converts ms to ms for baks
    filtering, then returns the properly truncated estimated firing rate.
    :param spike_times: array
    :param t: array
    :param baks_alpha: float
    :param baks_beta: float
    :param pad_dur: float (ms)
    :return: array
    """
    dt = t[1] - t[0]
    pad_dur = min(pad_dur, len(t)*dt)
    pad_len = int(pad_dur/dt)
    padded_spike_times = np.array(spike_times)
    r_pad_indexes = np.where((spike_times > t[0]) & (spike_times <= t[pad_len]))[0]
    if np.any(r_pad_indexes):
        r_pad_spike_times = np.add(t[0], np.subtract(t[0], spike_times[r_pad_indexes])[::-1])
        padded_spike_times = np.append(r_pad_spike_times, padded_spike_times)
    l_pad_indexes = np.where((spike_times >= t[-pad_len]) & (spike_times < t[-1]))[0]
    if np.any(l_pad_indexes):
        l_pad_spike_times = np.add(t[-1]+dt, np.subtract(t[-1]+dt, spike_times[l_pad_indexes])[::-1])
        padded_spike_times = np.append(padded_spike_times, l_pad_spike_times)
    padded_t = np.concatenate((np.arange(-pad_dur, 0., dt), t, np.arange(t[-1] + dt, t[-1] + pad_dur + dt / 2., dt)))
    padded_rate, h = baks(padded_spike_times/1000., padded_t/1000., alpha, beta)

    return padded_rate[pad_len:-pad_len]


def gauss(spikes, dt, filter_duration=100.):
    """gaussian convolving to transform spike counts to firing rate.. mirror padding. """
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


def mirror_signal(signal, pad_len):
    """np.fliplr hates python 2.7"""
    mirror_beginning = signal[:pad_len][::-1]
    mirror_end = signal[-pad_len:][::-1]
    modified_signal = np.append(np.append(mirror_beginning, signal), mirror_end)

    return modified_signal


def untruncated_filter_band(E, I, FF, window_len, band, padlen=250, dt=1.):
    """from input, filter for certain frequencies"""
    E_mir = mirror_signal(E, padlen)
    I_mir = mirror_signal(I, padlen)
    FF_mir = mirror_signal(FF, padlen)

    filt = scipy.signal.firwin(window_len, band, nyq=1000. / 2. / dt, pass_zero=False)
    E_band = scipy.signal.filtfilt(filt, [1.], E_mir, padtype=None, padlen=0)
    I_band = scipy.signal.filtfilt(filt, [1.], I_mir, padtype=None, padlen=0)
    FF_band = scipy.signal.filtfilt(filt, [1.], FF_mir, padtype=None, padlen=0)

    return E_band, I_band, FF_band


"""def plot_things(E_mir, E_band, transform, abs_envelope):
    x = range(len(E_mir))
    plt.plot(range(len(E_mir)), E_mir, label="mirrored signal")
    plt.plot(x, E_band, label="theta E", color="black")
    plt.plot(x, transform, label="hilb transform")
    plt.plot(x, abs_envelope, label="envelope")
    plt.legend()
    plt.show()"""


def peak_from_spectrogram(freq, title='not specified', dt=1., plot=False):
    """return the most dense frequency in a certain band (gamma, theta) based on the spectrogram"""
    freq, density = scipy.signal.periodogram(freq, 1000. / dt)
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


def prune_and_shift_spikes(spike_dict, throwaway):
    """
    cells equilibriate during the first chunk of the simulation. not useful for analysis, so throw away
    then shift spikes (in ms) so that the "starting point" is 0 ms
    :param throwaway: int - how many ms of the simulation to throw away
    """
    prune_dict = {}
    for key, val in spike_dict.iteritems():
        val = np.array(val)
        idx = np.where(val <= throwaway)[0]
        new = np.subtract(np.delete(val, idx), throwaway)
        prune_dict[key] = new
    return prune_dict


def prune_voltages(v_dict, dt, throwaway):
    """for analysis, only keep voltage recordings after a certain point; discard first bit."""
    prune_dict = {}
    toss = int(throwaway * (1 / dt))
    for key, val in v_dict.iteritems():
        prune_dict[key] = val[toss:]
    return prune_dict


def get_binned_spike_train(spikes, t):
    """
    convert spike times to a binned binary spike train
    :param spikes: array, elements are spike times in ms
    """
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
