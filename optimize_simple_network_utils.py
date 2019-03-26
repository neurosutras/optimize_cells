from nested.utils import *
from neuron import h
import seaborn as sns
from baks import baks


# for h.lambda_f
h.load_file('stdlib.hoc')
# for h.stdinit
h.load_file('stdrun.hoc')


class Network(object):

    def __init__(self, pc, pop_sizes, pop_gid_ranges, pop_cell_types, connection_syn_types, prob_connection,
                 connection_weights, connection_weight_sigma_factors, connection_kinetics, input_pop_mean_rates,
                 input_pop_fraction_active, tstop=1000., equilibrate=250., dt=0.025, delay=1., connection_seed=0,
                 spikes_seed=100000, verbose=1, debug=False):
        """

        :param pc: ParallelContext object
        :param pop_sizes: dict of int: cell population sizes
        :param pop_gid_ranges: dict of tuple of int: start and stop indexes; gid range of each cell population
        :param pop_cell_types: dict of str: cell_type of each cell population
        :param connection_syn_types: dict of str: synaptic connection type for each presynaptic population
        :param prob_connection: nested dict of float: connection probabilities between cell populations
        :param connection_weights: nested dict of float: mean strengths of each connection type
        :param connection_weight_sigma_factors: nested dict of float: variances of connection strengths, normalized to
                                                mean
        :param connection_kinetics: nested dict of float: synaptic decay kinetics (ms)
        :param input_pop_mean_rates: dict of float: mean firing rate of each input population (Hz)
        :param input_pop_fraction_active: dict of float: fraction of each input population chosen to be active
        :param tstop: int: simulation duration (ms)
        :param equilibrate: float: simulation equilibration duration (ms)
        :param dt: float: simulation timestep (ms)
        :param delay: float: netcon synaptic delay (ms)
        :param connection_seed: int: random seed for reproducible connections
        :param spikes_seed: int: random seed for reproducible input spike trains
        :param verbose: int: level for verbose print statements
        :param debug: bool: turn on for extra tests
        """
        self.pc = pc
        self.delay = delay
        self.tstop = tstop
        self.equilibrate = equilibrate
        if dt is None:
            dt = h.dt
        self.dt = dt
        self.verbose = verbose
        self.debug = debug

        self.pop_sizes = pop_sizes
        self.total_cells = np.sum(self.pop_sizes.values())

        self.pop_gid_ranges = pop_gid_ranges
        self.pop_cell_types = pop_cell_types
        self.connection_syn_types = connection_syn_types
        self.prob_connection = prob_connection
        self.connection_weights = connection_weights
        self.connection_weight_sigma_factors = connection_weight_sigma_factors
        self.connection_kinetics = connection_kinetics

        self.spikes_dict = defaultdict(dict)
        self.input_pop_mean_rates = input_pop_mean_rates
        self.input_pop_fraction_active = input_pop_fraction_active

        self.local_random = random.Random()
        self.connection_seed = connection_seed
        self.spikes_seed = spikes_seed

        self.mkcells()
        if self.debug:
            self.check_cell_type_correct()
        self.connectcells()
        if not self.debug:
            self.voltage_record()
            self.spike_record()

    def mkcells(self):
        rank = int(self.pc.id())
        nhost = int(self.pc.nhost())
        self.cells = defaultdict(dict)

        for pop_name, (gid_start, gid_stop) in self.pop_gid_ranges.iteritems():
            cell_type = self.pop_cell_types[pop_name]
            for i, gid in enumerate(xrange(gid_start, gid_stop)):
                # round-robin distribution of cells across MPI ranks
                if i % nhost == rank:
                    if self.debug:
                        print('mkcells: rank: %i got %s gid: %i' % (rank, pop_name, gid))
                    if cell_type == 'input':
                        cell = FFCell()
                        self.local_random.seed(self.spikes_seed + gid)
                        if self.local_random.random() <= self.input_pop_fraction_active[pop_name]:
                            this_spike_train = get_inhom_poisson_spike_times_by_thinning(
                                [self.input_pop_mean_rates[pop_name], self.input_pop_mean_rates[pop_name]],
                                [0., float(self.tstop)], dt=self.dt, generator=self.local_random)
                        else:
                            this_spike_train = []
                        vec = h.Vector(this_spike_train)
                        cell.pp.play(vec)
                        self.spikes_dict[pop_name][gid] = np.array(this_spike_train)
                    elif cell_type in ['RS', 'FS']:
                        cell = IzhiCell(tau_E=self.connection_kinetics[pop_name]['E'],
                                        tau_I=self.connection_kinetics[pop_name]['I'], type=cell_type)
                    self.cells[pop_name][gid] = cell
                    self.pc.set_gid2node(gid, rank)
                    nc = cell.connect2target(None)
                    self.pc.cell(gid, nc)

    def get_weight(self, connection):
        """
        want to reduce std if std is problematic, i.e, makes it possible to sample a negative weight.
        for use in connectcells()
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
        Consult to prob_connections dict to determine connections. Consult connection_weights and
        connection_weight_sigma_factor to determine synaptic strength.
        Restrictions: 1) cells cannot connect with themselves
                      2) weights cannot be negative
        """
        self.ncdict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for target_pop_name in self.prob_connection:
            for target_gid in self.cells[target_pop_name]:
                self.local_random.seed(self.connection_seed + target_gid)
                target_cell = self.cells[target_pop_name][target_gid]
                for source_pop_name in self.prob_connection[target_pop_name]:
                    this_prob_connection = self.prob_connection[target_pop_name][source_pop_name]
                    this_syn_type = self.connection_syn_types[source_pop_name]
                    start_gid = self.pop_gid_ranges[source_pop_name][0]
                    stop_gid = self.pop_gid_ranges[source_pop_name][1]
                    mu = self.connection_weights[target_pop_name][source_pop_name]
                    sigma_factor = self.connection_weight_sigma_factors[target_pop_name][source_pop_name]
                    for source_gid in xrange(start_gid, stop_gid):
                        if source_gid == target_gid:
                            continue
                        if self.local_random.random() <= this_prob_connection:
                            this_syn = target_cell.syns[this_syn_type]
                            this_nc = self.pc.gid_connect(source_gid, this_syn)
                            this_nc.delay = self.delay
                            this_weight = self.local_random.gauss(mu, mu * sigma_factor)
                            while this_weight < 0.: this_weight = self.local_random.gauss(mu, mu * sigma_factor)
                            this_nc.weight[0] = this_weight
                            self.ncdict[target_pop_name][target_gid][source_pop_name][source_gid] = this_nc
                if self.debug:
                    print('%s gid: %i: %s' % (target_pop_name, target_gid, self.ncdict[target_pop_name][target_gid]))

    def run(self):
        self.pc.set_maxstep(10)
        h.stdinit()
        h.dt = self.dt
        self.pc.psolve(self.tstop)

    # Instrumentation - stimulation and recording
    def spike_record(self):
        for i, gid in enumerate(self.gids):
            if self.cells[i].is_art(): continue
            tvec = h.Vector()
            nc = self.cells[i].connect2target(None)
            nc.record(tvec)
            self.spikes_rec_dict[gid] = tvec

    def voltage_record(self):
        self.voltage_tvec = h.Vector()
        self.voltage_tvec.record(h._ref_t)
        self.voltage_recvec = {}
        for i, cell in enumerate(self.cells):
            if cell.is_art(): continue
            rec = h.Vector()
            rec.record(getattr(cell.sec(.5), '_ref_v'))
            self.voltage_recvec[self.gids[i]] = rec

    def get_spikes_dict(self):
        spikes_dict = dict()
        for gid, spike_train in self.spikes_rec_dict.iteritems():
            spike_train_array = np.array(spike_train)
            indexes = np.where(spike_train_array >= self.equilibrate)[0]
            if np.any(indexes):
                spike_train_array = np.subtract(spike_train_array[indexes], self.equilibrate)
            spikes_dict[gid] = spike_train_array
        return spikes_dict

    def get_voltage_rec_dict(self):
        tvec_array = np.array(self.voltage_tvec)
        start_index = np.where(tvec_array >= self.equilibrate)[0][0]
        voltage_rec_dict = dict()
        for gid, recvec in self.voltage_recvec.iteritems():
            voltage_rec_dict[gid] = np.array(recvec)[start_index:]
        tvec_array = np.subtract(tvec_array[start_index:], self.equilibrate)
        return voltage_rec_dict, tvec_array

    # --Firing rate and spiking calculations
    def spike_summation(self, spikes_dict, t):
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
        count = 0
        mean = 0
        max_firing = 0
        for i in range(bounds[0], bounds[1]):
            if i not in rate_dict: continue
            mean += rate_dict[i]
            max_firing += peak_dict[i]
            count += 1
        if count != 0:
            mean = mean / float(count)
            max_firing = max_firing / float(count)

        return mean, max_firing

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
        :return: two arrays of size t. each element of frac_active = fraction of cells active at time bin,
        each element of mean_firing_active = firing rate (Hz) of active cells only in each time bin
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

    def count_to_rate_basic(self, spike_sums, ncell, dt=1.):
        """converts spike count (summed over population) to average instaneous firing rate for one cell per timestep."""
        factor = dt / 1000.
        rate = np.divide(np.divide(spike_sums, float(ncell)), factor)
        return rate

    # --Theta/gamma stuff
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

    def calculate_envelope_ratio(self, pop_rate, band, pad_len):
        """mean of the hilbert transform over the mean of the firing rate"""
        hilb_transform = np.abs(scipy.signal.hilbert(band))[pad_len:][:-pad_len]
        mean_envelope = np.mean(hilb_transform)
        mean_rate = np.mean(pop_rate)
        if mean_rate > 0:
            ratio = mean_envelope / mean_rate
        else:
            ratio = 0.
        return ratio, hilb_transform

    def get_bands_of_interest(self, filter_dt, basic_rate_E, basic_rate_I, basic_rate_FF, t, plot=False):
        """
        pad length = window length to get rid of edge effects
        :param: filter_dict: float, ms
        :param: basic_rate_E: array, instantaneous firing rate for E population (divided by number of E cells)
        :param: basic_rate_I: array
        :param: basic_rate_E: array
        :param: t, array
        :param: plot
        """
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

    def get_envelope_ratio(self, pop_rates, t, filter_dt, plot=False):
        """
        gets the fluctation in population firing rate based on theta/gamma
        :param pop_rates: list of arrays in the order: FF pop rate, I pop rate, E pop rate
        :param t: array
        :param filter_dt: float, ms
        :param plot:
        :return: ratios = a dict; truncated_bands; both keyed by strings like 'theta_I', 'gamma_E', etc
        """
        basic_rate_E = self.count_to_rate_basic(self.E_sum, self.E_ncell)
        basic_rate_I = self.count_to_rate_basic(self.I_sum, self.I_ncell)
        basic_rate_FF = self.count_to_rate_basic(self.FF_sum, self.FF_ncell)

        theta_E, theta_I, gamma_E, gamma_I, pad_len_theta, pad_len_gamma = self.get_bands_of_interest(filter_dt,
                                                    basic_rate_E, basic_rate_I, basic_rate_FF, t, plot)

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

    # --Plotting
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
        plt.figure()
        ax = sns.heatmap(matrixmap)
        ax.hlines([self.FF_ncell, self.FF_ncell + self.I_ncell], color='white', *ax.get_xlim())
        ax.vlines([self.FF_ncell, self.FF_ncell + self.I_ncell], color='white', *ax.get_ylim())
        plt.show()

        FF_matrixmap = matrixmap[:self.FF_ncell]
        plt.figure()
        ax = sns.heatmap(FF_matrixmap)
        ax.vlines([self.FF_ncell, self.FF_ncell + self.I_ncell], color='white', *ax.get_ylim())
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
                plt.figure()
                sns.heatmap(hm)
                plt.title(populations[counter])
                plt.show()
                wrap = 0
                counter += 1
                if key < len(ncell): hm = np.zeros((ncell[counter], len(t)))

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

    def plot_bands(self, t, theta_E, gamma_E, input_E, theta_FF, gamma_FF, input_FF):
        """
        plots filtered bands vs. their input (FF and E)
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

    # --Checks and print-outs
    def check_cell_type_correct(self):
        """
        goes over each cell and checks whether or not the cell is the cell type it is supposed to be
        (e.g., cells with gids in a certain range must be FF cells/Hoc object)
        """
        for pop_name in self.cells:
            target_cell_type = self.pop_cell_types[pop_name]
            for gid in self.cells[pop_name]:
                this_cell = self.cells[pop_name][gid]
                if isinstance(this_cell, FFCell):
                    if target_cell_type != 'input':
                        found_cell_type = type(this_cell)
                        raise RuntimeError('check_cell_type_correct: %s gid: %i is cell_type: %s, not FFCell' %
                                           (pop_name, gid, found_cell_type))
                elif isinstance(this_cell, IzhiCell):
                    if target_cell_type != this_cell.type:
                        raise RuntimeError('check_cell_type_correct: %s gid: %i is IzhiCell type: %s, not %s' %
                                           (pop_name, gid, this_cell.type, target_cell_type))
                    if (target_cell_type == 'RS' and this_cell.izh.a != .1) or \
                            (target_cell_type == 'FS' and this_cell.izh.a != .02):
                        raise RuntimeError('check_cell_type_correct: %s gid: %i; IzhiCell type: %s not configured '
                                           'properly' % (pop_name, gid, this_cell.type))
                else:
                    raise RuntimeError('check_cell_type_correct: %s gid: %i is an unknown type: %s' %
                                       (pop_name, gid, type(this_cell)))

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


class IzhiCell(object):
    # derived from modelDB
    def __init__(self, tau_E, tau_I, type='RS'):
        self.type = type
        self.sec = h.Section(cell=self)
        self.sec.L, self.sec.diam, self.sec.cm = 10., 10., 31.831
        self.izh = h.Izhi2007b(.5, sec=self.sec)
        self.vinit = -60
        self.sec(0.5).v = self.vinit
        self.sec.insert('pas')

        # RS = excit or FS = inhib
        if type == 'RS': self.izh.a = .1
        if type == 'FS': self.izh.a = .02

        self.mksyns(tau_E, tau_I)

    def __del__(self):
        pass

    def mksyns(self, tau_E, tau_I):
        self.syns = dict()
        s = h.ExpSyn(self.sec(0.5))
        s.tau = tau_E
        s.e = 0.
        self.syns['E'] = s
        s = h.ExpSyn(self.sec(0.5))
        s.tau = tau_I
        s.e = -80.
        self.syns['I'] = s

    def connect2target(self, target):
        nc = h.NetCon(self.sec(1)._ref_v, target, sec=self.sec)
        nc.threshold = 10
        return nc

    def is_art(self):
        return 0


class FFCell(object):
    def __init__(self):
        self.pp = h.VecStim()

    def connect2target(self, target):
        nc = h.NetCon(self.pp, target)
        return nc

    def is_art(self):
        return 1


def infer_firing_rates(spike_trains_dict, t, alpha, beta, pad_dur):
    """

    :param spike_trains_dict: dict of array
    :param t: array
    :param alpha: float
    :param beta: float
    :param pad_dur: float
    :return: dict of array
    """
    inferred_firing_rates = {}
    for gid, spike_train in spike_trains_dict.iteritems():
        if len(spike_train) > 0:
            smoothed = padded_baks(spike_train, t, alpha=alpha, beta=beta, pad_dur=pad_dur)
        else:
            smoothed = np.zeros_like(t)
        inferred_firing_rates[gid] = smoothed

    return inferred_firing_rates


def find_nearest(arr, tt):
    arr = arr[arr > tt[0]]
    arr = arr[arr < tt[-1]]
    return np.searchsorted(tt, arr)


def padded_baks(spike_times, t, alpha, beta, pad_dur=500.):
    """
    Expects spike times in ms. Uses mirroring to pad the edges to avoid edge artifacts. Converts ms to ms for baks
    filtering, then returns the properly truncated estimated firing rate.
    :param spike_times: array
    :param t: array
    :param alpha: float
    :param beta: float
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


"""def gauss(spikes, dt, filter_duration=100.):
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
    signal = signal[int(filter_duration / dt) + pad_len:][:len(spikes)]
    return signal"""


def untruncated_filter_band(E, I, FF, window_len, band, padlen=250, dt=1.):
    """from input, filter for certain frequencies. untruncated because we run a hilbert transform on the bands
    and we want to reduce edge effects"""
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


def mirror_signal(signal, pad_len):
    """np.fliplr hates python 2.7"""
    mirror_beginning = signal[:pad_len][::-1]
    mirror_end = signal[-pad_len:][::-1]
    modified_signal = np.append(np.append(mirror_beginning, signal), mirror_end)
    return modified_signal


# from dentate > stgen.py. temporary. personal issues with importing dentate -S
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


def plot_inferred_spike_rates(gid_range_dict, spike_trains_dict, inferred_firing_rates, t,
                              active_rate_threshold=1., cells_per_pop=4):
    """

    :param gid_range_dict: dict: {pop_name (str): tuple of int}
    :param spike_trains_dict: dict of array
    :param inferred_firing_rates: dict of array
    :param t: array
    :param gid_sample_dict: dict: {pop_name (str): list of gid (int)}
    :param active_rate_threshold: float
    :param cells_per_pop: int
    """
    num_pops = len(gid_range_dict)
    fig, axes = plt.subplots(num_pops, cells_per_pop, sharex=True)
    for j in xrange(cells_per_pop):
        axes[num_pops-1][j].set_xlabel('Time (ms)')
    for i in xrange(num_pops):
        axes[i][0].set_ylabel('Firing rate (Hz)')

    for row, pop_name in enumerate(gid_range_dict):
        active_gid_range = []
        for gid in range(*gid_range_dict[pop_name]):
            inferred_rate = inferred_firing_rates[gid]
            if np.max(inferred_rate) >= active_rate_threshold:
                active_gid_range.append(gid)
        gid_sample = random.sample(active_gid_range, min(len(active_gid_range), cells_per_pop))
        for col, gid in enumerate(gid_sample):
            inferred_rate = inferred_firing_rates[gid]
            spike_train = spike_trains_dict[gid]
            binned_spike_indexes = find_nearest(spike_train, t)
            axes[row][col].plot(t, inferred_rate, label='Inferred rate')
            axes[row][col].plot(t[binned_spike_indexes], np.ones(len(binned_spike_indexes)), 'k.', label='Spikes')
            axes[row][col].set_title('%s cell: %i' % (pop_name, gid))

    axes[0][0].legend(loc='best')
    clean_axes(axes)
    fig.suptitle('Inferred spike rates')
    fig.tight_layout()
    fig.show()


def plot_voltage_traces(gid_range_dict, voltage_rec_dict, t, cells_per_pop=8, pop_names=None):
    """

    :param gid_range_dict: dict: {pop_name (str): tuple of int}
    :param voltage_rec_dict: dict of array
    :param t: array
    :param cells_per_pop: int
    :param pop_names: list of str
    """
    if pop_names is None:
        pop_names = [pop_name for pop_name in gid_range_dict if pop_name not in ['FF']]
    num_pops = len(pop_names)
    num_rows = 2 * num_pops
    num_cols = cells_per_pop / 2
    fig, axes = plt.subplots(num_rows, num_cols, sharex=True)
    for j in xrange(num_cols):
        axes[num_rows-1][j].set_xlabel('Time (ms)')
    for i in xrange(num_rows):
        axes[i][0].set_ylabel('Voltage (mV)')

    for i, pop_name in enumerate(pop_names):
        this_gid_range = range(*gid_range_dict[pop_name])
        gid_sample = random.sample(this_gid_range, min(len(this_gid_range), cells_per_pop))
        for j, gid in enumerate(gid_sample):
            row = (i * 2) + (j / num_cols)
            col = j % num_cols
            rec = voltage_rec_dict[gid]
            axes[row][col].plot(t, rec)
            axes[row][col].set_title('%s cell: %i' % (pop_name, gid))

    axes[0][0].legend(loc='best')
    clean_axes(axes)
    fig.suptitle('Voltage recordings')
    fig.tight_layout()
    fig.show()