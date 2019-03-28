from nested.utils import *
from neuron import h
from baks import baks
from scipy.signal import butter, sosfiltfilt, sosfreqz, hilbert, periodogram


# for h.lambda_f
h.load_file('stdlib.hoc')
# for h.stdinit
h.load_file('stdrun.hoc')


class Network(object):

    def __init__(self, pc, pop_sizes, pop_gid_ranges, pop_cell_types, connection_syn_types, prob_connection,
                 connection_weights_mean, connection_weight_sigma_factors, connection_kinetics, input_pop_mean_rates,
                 tstop=1000., equilibrate=250., dt=0.025, delay=1., connection_seed=0, spikes_seed=100000, verbose=1,
                 debug=False):
        """

        :param pc: ParallelContext object
        :param pop_sizes: dict of int: cell population sizes
        :param pop_gid_ranges: dict of tuple of int: start and stop indexes; gid range of each cell population
        :param pop_cell_types: dict of str: cell_type of each cell population
        :param connection_syn_types: dict of str: synaptic connection type for each presynaptic population
        :param prob_connection: nested dict of float: connection probabilities between cell populations
        :param connection_weights_mean: nested dict of float: mean strengths of each connection type
        :param connection_weight_sigma_factors: nested dict of float: variances of connection strengths, normalized to
                                                mean
        :param connection_kinetics: nested dict of float: synaptic decay kinetics (ms)
        :param input_pop_mean_rates: dict of float: mean firing rate of each input population (Hz)
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
        self.connection_weights_mean = connection_weights_mean
        self.connection_weight_sigma_factors = connection_weight_sigma_factors
        self.connection_kinetics = connection_kinetics

        self.spikes_dict = defaultdict(dict)
        self.input_pop_mean_rates = input_pop_mean_rates

        self.local_random = random.Random()
        self.connection_seed = connection_seed
        self.spikes_seed = spikes_seed

        self.mkcells()
        if self.debug:
            self.verify_cell_types()
        self.connectcells()
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
                    if self.verbose > 1:
                        print('mkcells: rank: %i got %s gid: %i' % (rank, pop_name, gid))
                    if cell_type == 'input':
                        cell = FFCell()
                        self.local_random.seed(self.spikes_seed + gid)
                        this_spike_train = get_inhom_poisson_spike_times_by_thinning(
                            [self.input_pop_mean_rates[pop_name], self.input_pop_mean_rates[pop_name]],
                            [0., float(self.tstop)], dt=self.dt, generator=self.local_random)
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

    def verify_cell_types(self):
        """
        Double-checks each created cell has the intended cell type.
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

    def connectcells(self):
        """
        Consult prob_connections dict to assign connections. Consult connection_weights_mean and
        connection_weight_sigma_factor to assign synaptic weights.
        Restrictions: 1) Cells do not form connections with self
                      2) Weights cannot be negative
        """
        rank = int(self.pc.id())
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
                    mu = self.connection_weights_mean[target_pop_name][source_pop_name]
                    sigma_factor = self.connection_weight_sigma_factors[target_pop_name][source_pop_name]
                    if sigma_factor >= 2. / 3. / np.sqrt(2.):
                        orig_sigma_factor = sigma_factor
                        sigma_factor = 2. / 3. / np.sqrt(2.)
                        print('network.connectcells: %s: %s connection; reducing weight_sigma_factor to avoid negative '
                              'weights; orig: %.2f, new: %.2f' %
                              (source_pop_name, target_pop_name, orig_sigma_factor, sigma_factor))
                    for source_gid in xrange(start_gid, stop_gid):
                        if source_gid == target_gid:
                            continue
                        if self.local_random.random() <= this_prob_connection:
                            this_syn = target_cell.syns[this_syn_type]
                            this_nc = self.pc.gid_connect(source_gid, this_syn)
                            this_nc.delay = self.delay
                            this_weight = self.local_random.gauss(mu, mu * sigma_factor)
                            while this_weight < 0.:
                                this_weight = self.local_random.gauss(mu, mu * sigma_factor)
                            this_nc.weight[0] = this_weight
                            self.ncdict[target_pop_name][target_gid][source_pop_name][source_gid] = this_nc
                    if self.verbose > 1:
                        print('rank: %i; target: %s gid: %i: source: %s; num_syns: %i' %
                              (rank, target_pop_name, target_gid, source_pop_name,
                               len(self.ncdict[target_pop_name][target_gid][source_pop_name])))

    # Instrumentation - stimulation and recording
    def spike_record(self):
        for pop_name in self.cells:
            for gid, cell in self.cells[pop_name].iteritems():
                if cell.is_art(): continue
                tvec = h.Vector()
                nc = cell.connect2target(None)
                nc.record(tvec)
                self.spikes_dict[pop_name][gid] = tvec

    def voltage_record(self):
        self.voltage_tvec = h.Vector()
        self.voltage_tvec.record(h._ref_t)
        self.voltage_recvec = defaultdict(dict)
        for pop_name in self.cells:
            for gid, cell in self.cells[pop_name].iteritems():
                if cell.is_art(): continue
                rec = h.Vector()
                rec.record(getattr(cell.sec(.5), '_ref_v'))
                self.voltage_recvec[pop_name][gid] = rec

    def run(self):
        self.pc.set_maxstep(10)
        h.stdinit()
        h.dt = self.dt
        self.pc.psolve(self.tstop)

    def get_spikes_dict(self):
        spikes_dict = dict()
        for pop_name in self.spikes_dict:
            spikes_dict[pop_name] = dict()
            for gid, spike_train in self.spikes_dict[pop_name].iteritems():
                spike_train_array = np.array(spike_train)
                indexes = np.where(spike_train_array >= self.equilibrate)[0]
                if np.any(indexes):
                    spike_train_array = np.subtract(spike_train_array[indexes], self.equilibrate)
                spikes_dict[pop_name][gid] = spike_train_array
        return spikes_dict

    def get_voltage_rec_dict(self):
        tvec_array = np.array(self.voltage_tvec)
        start_index = np.where(tvec_array >= self.equilibrate)[0][0]
        voltage_rec_dict = dict()
        for pop_name in self.voltage_recvec:
            voltage_rec_dict[pop_name] = dict()
            for gid, recvec in self.voltage_recvec[pop_name].iteritems():
                voltage_rec_dict[pop_name][gid] = np.array(recvec)[start_index:]
        tvec_array = np.subtract(tvec_array[start_index:], self.equilibrate)
        return voltage_rec_dict, tvec_array

    def get_connection_weights(self):
        """
        Collate connection weights. Assign a value of zero to non-connected cells.
        """
        weights = dict()
        for target_pop_name in self.ncdict:
            weights[target_pop_name] = dict()
            for target_gid in self.ncdict[target_pop_name]:
                weights[target_pop_name][target_gid] = dict()
                for source_pop_name in self.prob_connection[target_pop_name]:
                    weights[target_pop_name][target_gid][source_pop_name] = dict()
                    start_gid = self.pop_gid_ranges[source_pop_name][0]
                    stop_gid = self.pop_gid_ranges[source_pop_name][1]
                    for source_gid in xrange(start_gid, stop_gid):
                        if source_gid in self.ncdict[target_pop_name][target_gid][source_pop_name]:
                            weights[target_pop_name][target_gid][source_pop_name][source_gid] = \
                                self.ncdict[target_pop_name][target_gid][source_pop_name][source_gid].weight[0]
                        else:
                            weights[target_pop_name][target_gid][source_pop_name][source_gid] = 0.

        return weights


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


def infer_firing_rates(spike_trains_dict, t, alpha, beta, pad_dur):
    """

    :param spike_trains_dict: nested dict: {pop_name: {gid: array} }
    :param t: array
    :param alpha: float
    :param beta: float
    :param pad_dur: float
    :return: dict of array
    """
    inferred_firing_rates = defaultdict(dict)
    for pop_name in spike_trains_dict:
        for gid, spike_train in spike_trains_dict[pop_name].iteritems():
            if len(spike_train) > 0:
                smoothed = padded_baks(spike_train, t, alpha=alpha, beta=beta, pad_dur=pad_dur)
            else:
                smoothed = np.zeros_like(t)
            inferred_firing_rates[pop_name][gid] = smoothed

    return inferred_firing_rates


def find_nearest(arr, tt):
    arr = arr[arr > tt[0]]
    arr = arr[arr < tt[-1]]
    return np.searchsorted(tt, arr)


def padded_baks(spike_times, t, alpha, beta, pad_dur=500.):
    """
    Expects spike times in ms. Uses mirroring to pad the edges to avoid edge artifacts. Converts ms to sec for baks
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


def get_binned_spike_count(spike_times, t):
    """
    Convert spike times to a binned binary spike train
    :param spike_times: array (ms)
    :param t: array (ms)
    :return: array
    """
    binned_spikes = np.zeros_like(t)
    if len(spike_times) > 0:
        spike_indexes = [np.where(t >= spike_time)[0][0] for spike_time in spike_times]
        binned_spikes[spike_indexes] = 1.
    return binned_spikes


def get_pop_activity_stats(spikes_dict, firing_rates_dict, t, threshold=1., plot=False):
    """
    Calculate firing rate statistics for each cell population.
    :param spikes_dict: nested dict of array
    :param firing_rates_dict: nested dict of array
    :param t: array
    :param threshold: firing rate threshold for "active" cells: float (Hz)
    :param plot: bool
    :return: tuple of dict
    """
    dt = t[1] - t[0]
    mean_rate_dict = defaultdict(dict)
    peak_rate_dict = defaultdict(dict)
    mean_rate_active_cells_dict = dict()
    pop_fraction_active_dict = dict()
    binned_spike_count_dict = defaultdict(dict)
    mean_rate_from_spike_count_dict = dict()

    for pop_name in spikes_dict:
        this_active_cell_count = np.zeros_like(t)
        this_summed_rate_active_cells = np.zeros_like(t)
        for gid in spikes_dict[pop_name]:
            this_firing_rate = firing_rates_dict[pop_name][gid]
            mean_rate_dict[pop_name][gid] = np.mean(this_firing_rate)
            peak_rate_dict[pop_name][gid] = np.max(this_firing_rate)
            binned_spike_count_dict[pop_name][gid] = get_binned_spike_count(spikes_dict[pop_name][gid], t)
            active_indexes = np.where(this_firing_rate >= threshold)[0]
            if np.any(active_indexes):
                this_active_cell_count[active_indexes] += 1.
                this_summed_rate_active_cells[active_indexes] += this_firing_rate[active_indexes]

        active_indexes = np.where(this_active_cell_count > 0.)[0]
        if np.any(active_indexes):
            mean_rate_active_cells_dict[pop_name] = np.array(this_summed_rate_active_cells)
            mean_rate_active_cells_dict[pop_name][active_indexes] = \
                np.divide(this_summed_rate_active_cells[active_indexes], this_active_cell_count[active_indexes])
        else:
            mean_rate_active_cells_dict[pop_name] = np.zeros_like(t)
        pop_fraction_active_dict[pop_name] = np.divide(this_active_cell_count, len(spikes_dict[pop_name]))
        mean_rate_from_spike_count_dict[pop_name] = \
            np.divide(np.mean(binned_spike_count_dict[pop_name].values(), axis=0), dt / 1000.)

    if plot:
        for pop_name in pop_fraction_active_dict:
            fig, axes = plt.subplots(1, 2)
            axes[0].plot(t, pop_fraction_active_dict[pop_name])
            axes[0].set_title('Active fraction of cell population')
            axes[1].plot(t, mean_rate_active_cells_dict[pop_name])
            axes[1].set_title('Mean firing rate of active cells')
            clean_axes(axes)
            fig.suptitle('Population: %s' % pop_name)
            fig.tight_layout()
            fig.subplots_adjust(top=0.9)
            fig.show()

    return mean_rate_dict, peak_rate_dict, mean_rate_active_cells_dict, pop_fraction_active_dict, \
           binned_spike_count_dict, mean_rate_from_spike_count_dict


def get_butter_bandpass_filter(filter_band, sampling_rate, order, filter_label='', plot=False):
    """

    :param filter_band: list of float
    :param sampling_rate: float
    :param order: int
    :param filter_label: str
    :param plot: bool
    :return: array
    """
    nyq = 0.5 * sampling_rate
    normalized_band = np.divide(filter_band, nyq)
    sos = butter(order, normalized_band, analog=False, btype='band', output='sos')
    if plot:
        fig = plt.figure()
        w, h = sosfreqz(sos, worN=2000)
        plt.plot((sampling_rate * 0.5 / np.pi) * w, abs(h))
        plt.plot([0, 0.5 * sampling_rate], [np.sqrt(0.5), np.sqrt(0.5)], '--', label='sqrt(0.5)')
        plt.title('%s bandpass filter (%.1f:%.1f Hz), Order: %i' %
                  (filter_label, min(filter_band), max(filter_band), order))
        plt.xlabel('Frequency (Hz)')
        plt.xlim(0., min(nyq, 2. * max(filter_band)))
        plt.ylabel('Gain')
        plt.grid(True)
        fig.show()

    return sos


def get_bandpass_filtered_signal_stats(signal, t, sos, filter_band, signal_label='', filter_label='', pad=True,
                                 pad_len=None, plot=False):
    """

    :param signal: array
    :param t: array (ms)
    :param sos: array
    :param filter_band: list of float (Hz)
    :param signal_label: str
    :param filter_label: str
    :param pad: bool
    :param pad_len: int
    :param plot: bool
    :return: tuple of array
    """
    if pad and pad_len is None:
        dt = t[1] - t[0]  # ms
        pad_dur = min(10. * 1000. / np.min(filter_band), len(t) * dt)  # ms
        pad_len = min(int(pad_dur / dt), len(t) - 1)
    if pad:
        padded_signal = get_mirror_padded_signal(signal, pad_len)
    else:
        padded_signal = np.array(signal)
    filtered_padded_signal = sosfiltfilt(sos, padded_signal)
    filtered_signal = filtered_padded_signal[pad_len:-pad_len]
    padded_envelope = np.abs(hilbert(filtered_padded_signal))
    envelope = padded_envelope[pad_len:-pad_len]
    f, power = periodogram(filtered_signal, fs=1000./dt)
    centroid_freq = f[get_center_of_mass_index(power)]

    mean_envelope = np.mean(envelope)
    mean_signal = np.mean(signal)
    envelope_ratio = mean_envelope / mean_signal

    if plot:
        fig, axes = plt.subplots(2,2)
        axes[0][0].plot(t, np.subtract(signal, np.mean(signal)), c='k', label='Original signal')
        axes[0][0].plot(t, filtered_signal, c='r', label='Filtered signal')
        axes[0][1].plot(t, signal, label='Original signal', c='grey', alpha=0.5, zorder=1)
        axes[0][1].plot(t, np.ones_like(t) * mean_signal, c='k', zorder=0)
        axes[0][1].plot(t, envelope, label='Envelope amplitude', c='r', alpha=0.5, zorder=1)
        axes[0][1].plot(t, np.ones_like(t) * mean_envelope, c='m', zorder=0)
        box = axes[0][0].get_position()
        axes[0][0].set_position([box.x0, box.y0, box.width, box.height * 0.8])
        axes[0][0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), frameon=False, framealpha=0.5)
        box = axes[0][1].get_position()
        axes[0][1].set_position([box.x0, box.y0, box.width, box.height * 0.8])
        axes[0][1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), frameon=False, framealpha=0.5)

        axes[1][0].plot(f, power, c='k')
        axes[1][0].set_xlabel('Frequency (Hz)')
        axes[1][0].set_ylabel('Spectral density (units$^{2}$/Hz)')
        axes[1][0].set_xlim(min(filter_band)/2., max(filter_band) * 1.5)

        clean_axes(axes)
        fig.suptitle('%s\n%s bandpass filter (%.1f:%.1f Hz); Envelope ratio: %.3f; Centroid freq: %.3f Hz' %
                          (signal_label, filter_label, min(filter_band), max(filter_band), envelope_ratio,
                           centroid_freq))
        fig.tight_layout()
        fig.subplots_adjust(top=0.8)
        fig.show()

    return filtered_signal, envelope, envelope_ratio, centroid_freq


def get_pop_bandpass_filtered_signal_stats(signal_dict, t, filter_band_dict, order=15, plot=False):
    """

    :param signal_dict: array
    :param t: array (ms)
    :param filter_band_dict: dict: {filter_label (str): list of float (Hz) }
    :param order: int
    :param plot: bool
    :return: array
    """
    dt = t[1] - t[0]  # ms
    sampling_rate = 1000. / dt  # Hz
    filtered_signal_dict = {}
    envelope_dict = {}
    envelope_ratio_dict = {}
    centroid_freq_dict = {}
    for filter_label, filter_band in filter_band_dict.iteritems():
        filtered_signal_dict[filter_label] = {}
        envelope_dict[filter_label] = {}
        envelope_ratio_dict[filter_label] = {}
        centroid_freq_dict[filter_label] = {}
        sos = get_butter_bandpass_filter(filter_band, sampling_rate, filter_label=filter_label, order=order, plot=plot)
        for pop_name in signal_dict:
            signal = signal_dict[pop_name]
            filtered_signal_dict[filter_label][pop_name], envelope_dict[filter_label][pop_name], \
            envelope_ratio_dict[filter_label][pop_name], centroid_freq_dict[filter_label][pop_name] = \
                get_bandpass_filtered_signal_stats(signal, t, sos, filter_band,
                                                   signal_label='Population: %s' % pop_name,
                                                   filter_label=filter_label, plot=plot)

    return filtered_signal_dict, envelope_dict, envelope_ratio_dict, centroid_freq_dict


def plot_heatmap_from_matrix(data, xticks=None, xtick_labels=None, yticks=None, ytick_labels=None, ax=None,
                             cbar_kw={}, cbar_label="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        xticks : A list or array of length <=M with xtick locations
        xtick_labels : A list or array of length <=M with xtick labels
        yticks : A list or array of length <=N with ytick locations
        ytick_labels : A list or array of length <=N with ytick labels
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbar_label  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom")

    if xticks is not None:
        ax.set_xticks(xticks)
    if xtick_labels is not None:
        ax.set_xticklabels(xtick_labels)
    if yticks is not None:
        ax.set_yticks(yticks)
    if ytick_labels is not None:
        ax.set_yticklabels(ytick_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")


def get_center_of_mass_index(signal, subtract_min=True):
    """
    Return the index of the center of mass of a signal, or None if the signal is mean zero. By default searches for
    area above the signal minimum.
    :param signal:
    :param subtract_min: bool
    :return: int
    """
    if subtract_min:
        this_signal = np.subtract(signal, np.min(signal))
    else:
        this_signal = np.array(signal)
    cumsum = np.cumsum(this_signal)
    if cumsum[-1] == 0.:
        return None
    normalized_cumsum = cumsum / cumsum[-1]
    return np.argwhere(normalized_cumsum >= 0.5)[0][0]


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


def get_mirror_padded_signal(signal, pad_len):
    """
    Pads the ends of the signal by mirroring without duplicating the end points.
    :param signal: array
    :param pad_len: int
    :return: array
    """
    mirror_beginning = signal[:pad_len][::-1]
    mirror_end = signal[-pad_len:][::-1]
    padded_signal = np.concatenate((mirror_beginning, signal, mirror_end))
    return padded_signal


def get_mirror_padded_time_series(t, pad_len):
    """
    Pads the ends of a time series by mirroring without duplicating the end points.
    :param t: array
    :param pad_len: int
    :return: array
    """
    dt = t[1] - t[0]
    t_end = len(t) * dt
    padded_t = np.concatenate((
        np.subtract(t[0] - dt, t[:pad_len])[::-1], t, np.add(t[-1], np.subtract(t_end, t[-pad_len:])[::-1])))
    return padded_t


def plot_inferred_spike_rates(spikes_dict, firing_rates_dict, t, active_rate_threshold=1., rows=3, cols=4,
                              pop_names=None):
    """

    :param spikes_dict: dict of array
    :param firing_rates_dict: dict of array
    :param t: array
    :param active_rate_threshold: float
    :param rows: int
    :param cols: int
    :param pop_names: list of str
    """
    if pop_names is None:
        pop_names = list(spikes_dict.keys())
    for pop_name in pop_names:
        fig, axes = plt.subplots(rows, cols, sharex=True)
        for j in xrange(cols):
            axes[rows-1][j].set_xlabel('Time (ms)')
        for i in xrange(rows):
            axes[i][0].set_ylabel('Firing rate (Hz)')
        active_gid_range = []
        for gid, rate in firing_rates_dict[pop_name].iteritems():
            if np.max(rate) >= active_rate_threshold:
                active_gid_range.append(gid)
        gid_sample = random.sample(active_gid_range, min(len(active_gid_range), rows * cols))
        for i, gid in enumerate(gid_sample):
            inferred_rate = firing_rates_dict[pop_name][gid]
            spike_train = spikes_dict[pop_name][gid]
            binned_spike_indexes = find_nearest(spike_train, t)
            row = i / cols
            col = i % cols
            axes[row][col].plot(t, inferred_rate, label='Inferred firing rate')
            axes[row][col].plot(t[binned_spike_indexes], np.ones(len(binned_spike_indexes)), 'k.', label='Spikes')
            axes[row][col].set_title('gid: %i' % gid)

        axes[0][0].legend(loc='best')
        clean_axes(axes)
        fig.suptitle('Inferred spike rates: %s population' % pop_name)
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        fig.show()


def plot_voltage_traces(voltage_rec_dict, t, rows=3, cols=4, pop_names=None):
    """

    :param voltage_rec_dict: dict of array
    :param t: array
    :param cells_per_pop: int
    :param pop_names: list of str
    """
    if pop_names is None:
        pop_names = list(voltage_rec_dict.keys())
    for pop_name in pop_names:
        fig, axes = plt.subplots(rows, cols, sharex=True)
        for j in xrange(cols):
            axes[rows - 1][j].set_xlabel('Time (ms)')
        for i in xrange(rows):
            axes[i][0].set_ylabel('Voltage (mV)')
        this_gid_range = list(voltage_rec_dict[pop_name].keys())
        gid_sample = random.sample(this_gid_range, min(len(this_gid_range), rows * cols))
        for i, gid in enumerate(gid_sample):
            rec = voltage_rec_dict[pop_name][gid]
            row = i / cols
            col = i % cols
            axes[row][col].plot(t, rec)
            axes[row][col].set_title('gid: %i' % gid)
        axes[0][0].legend(loc='best')
        clean_axes(axes)
        fig.suptitle('Voltage recordings: %s population' % pop_name)
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        fig.show()


def plot_weight_matrix(connection_weights_dict, pop_names=None):
    """
    Plots heat maps of connection strengths across all connected cell populations.
    :param connection_weights_dict: nested dict of float
    :param pop_names: list of str
    """
    if pop_names is None:
        pop_names = list(connection_weights_dict.keys())
    for target_pop_name in pop_names:
        sorted_target_gids = sorted(list(connection_weights_dict[target_pop_name].keys()))
        source_pop_list = list(connection_weights_dict[target_pop_name][sorted_target_gids[0]].keys())
        cols = len(source_pop_list)
        fig, axes = plt.subplots(1, cols, sharey=True)
        for col, source_pop_name in enumerate(source_pop_list):
            sorted_source_gids = sorted(list(
                connection_weights_dict[target_pop_name][sorted_target_gids[0]][source_pop_name].keys()))
            weight_matrix = np.empty((len(sorted_target_gids), len(sorted_source_gids)), dtype='float32')
            for i, target_gid in enumerate(sorted_target_gids):
                for j, source_gid in enumerate(sorted_source_gids):
                    weight_matrix[i][j] = \
                        connection_weights_dict[target_pop_name][target_gid][source_pop_name][source_gid]
            y_interval = len(sorted_target_gids) / 10 + 1
            yticks = range(0, len(sorted_target_gids), y_interval)
            ylabels = np.array(sorted_target_gids)[yticks]
            x_interval = len(sorted_source_gids) / 10 + 1
            xticks = range(0, len(sorted_source_gids), x_interval)
            xlabels = np.array(sorted_source_gids)[xticks]
            plot_heatmap_from_matrix(weight_matrix, xticks=xticks, xtick_labels=xlabels, yticks=yticks,
                                     ytick_labels=ylabels, ax=axes[col], aspect='auto')
            axes[col].set_xlabel('Source: %s' % source_pop_name)
            axes[0].set_ylabel('Target: %s' % target_pop_name)
        clean_axes(axes)
        fig.suptitle('Connection weights onto %s population' % target_pop_name, )
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        fig.show()


def plot_firing_rate_heatmaps(firing_rates_dict, t, pop_names=None):
    """

    :param firing_rates_dict: dict of array
    :param t: array
    :param pop_names: list of str
    """
    if pop_names is None:
        pop_names = list(firing_rates_dict.keys())
    for pop_name in pop_names:
        fig, axes = plt.subplots()
        sorted_gids = sorted(list(firing_rates_dict[pop_name].keys()))
        rate_matrix = np.empty((len(sorted_gids), len(t)), dtype='float32')
        for i, gid in enumerate(sorted_gids):
            rate_matrix[i][:] = firing_rates_dict[pop_name][gid]
        y_interval = len(sorted_gids) / 10 + 1
        yticks = range(0, len(sorted_gids), y_interval)
        ylabels = np.array(sorted_gids)[yticks]
        dt = t[1] - t[0]
        x_interval = int(1000. / dt)
        xticks = range(0, len(t), x_interval)
        xlabels = np.array(t)[xticks].astype('int32')
        plot_heatmap_from_matrix(rate_matrix, xticks=xticks, xtick_labels=xlabels, yticks=yticks,
                                 ytick_labels=ylabels, ax=axes, aspect='auto')
        axes.set_xlabel('Time (ms)')
        axes.set_ylabel('Firing rate (Hz)')
        axes.set_title('Firing rate: %s population' % pop_name)
        clean_axes(axes)
        fig.tight_layout()
        fig.show()