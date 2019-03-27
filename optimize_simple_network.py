from nested.optimize_utils import *
from optimize_simple_network_utils import *
import click
import time


context = Context()


@click.command()
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/optimize_simple_network_config.yaml')
@click.option("--export", is_flag=True)
@click.option("--output-dir", type=str, default='data')
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default=None)
@click.option("--interactive", is_flag=True)
@click.option("--verbose", type=int, default=2)
@click.option("--plot", is_flag=True)
@click.option("--debug", is_flag=True)
def main(config_file_path, export, output_dir, export_file_path, label, interactive, verbose, plot, debug):
    """

    :param config_file_path: str (path)
    :param export: bool
    :param output_dir: str
    :param export_file_path: str
    :param label: str
    :param interactive: bool
    :param verbose: int
    :param plot: bool
    :param debug: bool
    """
    # requires a global variable context: :class:'Context'

    context.update(locals())
    comm = MPI.COMM_WORLD

    from nested.parallel import ParallelContextInterface
    context.interface = ParallelContextInterface(procs_per_worker=comm.size)
    context.interface.start(disp=True)
    context.interface.ensure_controller()
    context.interface.apply(config_optimize_interactive, __file__, config_file_path=config_file_path,
                            output_dir=output_dir, export=export, export_file_path=export_file_path, label=label,
                            disp=verbose > 0, verbose=verbose, plot=plot)
    sys.stdout.flush()
    features = context.interface.execute(compute_features, context.x0_array, context.export)
    sys.stdout.flush()
    if not debug:
        features, objectives = context.interface.execute(get_objectives, features)
        sys.stdout.flush()
        print 'params:'
        pprint.pprint(context.x0_dict)
        print 'features:'
        pprint.pprint(features)
        print 'objectives:'
        pprint.pprint(objectives)
        sys.stdout.flush()
    context.update(locals())

    if not interactive:
        context.interface.stop()


def config_worker():
    """

    """
    if 'plot' not in context():
        context.plot = False
    if 'verbose' not in context():
        context.verbose = 1
    if 'debug' not in context():
        context.debug = False
    init_context()
    context.pc = h.ParallelContext()


def init_context():
    pop_sizes = {'FF': 100, 'E': 10, 'I': 10}
    prev_gid = 0
    pop_gid_ranges = dict()
    for pop_name in pop_sizes:
        next_gid = prev_gid + pop_sizes[pop_name]
        pop_gid_ranges[pop_name] = (prev_gid, next_gid)
        prev_gid += pop_sizes[pop_name]
    pop_cell_types = {'FF': 'input', 'E': 'RS', 'I': 'FS'}
    # {'postsynaptic population': {'presynaptic population': float} }
    prob_connection = defaultdict(dict)
    connection_weights_mean = defaultdict(dict)
    connection_weight_sigma_factors = defaultdict(dict)
    connection_syn_types = {'FF': 'E', 'E': 'E', 'I': 'I'}  # {'presynaptic population': syn_type}
    connection_kinetics = defaultdict(dict)
    input_pop_mean_rates = defaultdict(dict)
    input_pop_fraction_active = defaultdict(dict)
    if 'FF_mean_rate' not in context():
        raise RuntimeError('optimize_simple_network: missing required kwarg: FF_mean_rate')

    delay = 1.  # ms
    equilibrate = 250.  # ms
    tstop = 3000 + equilibrate  # ms
    dt = 0.025
    binned_dt = 1.  # ms
    filter_dt = 1.  # ms
    active_rate_threshold = 1.  # Hz
    baks_alpha = 4.7725100028345535
    baks_beta = 0.41969058927343522
    baks_pad_dur = 1000.  # ms
    context.update(locals())


def update_context(x, local_context=None):
    """

    :param x: array
    :param local_context: :class:'Context'
    """
    if local_context is None:
        local_context = context
    x_dict = param_array_to_dict(x, context.param_names)
    local_context.prob_connection['E']['FF'] = x_dict['FF_E_prob_connection']
    local_context.prob_connection['E']['E'] = x_dict['E_E_prob_connection']
    local_context.prob_connection['E']['I'] = x_dict['I_E_prob_connection']
    local_context.prob_connection['I']['FF'] = x_dict['FF_I_prob_connection']
    local_context.prob_connection['I']['E'] = x_dict['E_I_prob_connection']
    local_context.prob_connection['I']['I'] = x_dict['I_I_prob_connection']

    local_context.connection_kinetics['E']['E'] = x_dict['tau_E']
    local_context.connection_kinetics['E']['I'] = x_dict['tau_I']
    local_context.connection_kinetics['I']['E'] = x_dict['tau_E']
    local_context.connection_kinetics['I']['I'] = x_dict['tau_I']

    local_context.connection_weights_mean['E']['FF'] = x_dict['FF_E_mean_weight']
    local_context.connection_weights_mean['E']['E'] = x_dict['E_E_mean_weight']
    local_context.connection_weights_mean['E']['I'] = x_dict['I_E_mean_weight']
    local_context.connection_weights_mean['I']['FF'] = x_dict['FF_I_mean_weight']
    local_context.connection_weights_mean['I']['E'] = x_dict['E_I_mean_weight']
    local_context.connection_weights_mean['I']['I'] = x_dict['I_I_mean_weight']

    local_context.connection_weight_sigma_factors['E']['FF'] = x_dict['FF_E_weight_sigma_factor']
    local_context.connection_weight_sigma_factors['E']['E'] = x_dict['E_E_weight_sigma_factor']
    local_context.connection_weight_sigma_factors['E']['I'] = x_dict['I_E_weight_sigma_factor']
    local_context.connection_weight_sigma_factors['I']['FF'] = x_dict['FF_I_weight_sigma_factor']
    local_context.connection_weight_sigma_factors['I']['E'] = x_dict['E_I_weight_sigma_factor']
    local_context.connection_weight_sigma_factors['I']['I'] = x_dict['I_I_weight_sigma_factor']
    
    local_context.input_pop_mean_rates['FF'] = local_context.FF_mean_rate
    local_context.input_pop_fraction_active['FF'] = x_dict['FF_frac_active']


def analyze_network_output(network, export=False, plot=False):
    """

    :param network: :class:'Network'
    :param export: bool
    :param plot: bool
    :return: dict
    """
    binned_t = np.arange(0., context.tstop + context.binned_dt - context.equilibrate, context.binned_dt)
    spikes_dict = network.get_spikes_dict()
    voltage_rec_dict, rec_t = network.get_voltage_rec_dict()
    firing_rates_dict = infer_firing_rates(spikes_dict, binned_t, alpha=context.baks_alpha, beta=context.baks_beta,
                                               pad_dur=context.baks_pad_dur)
    connection_weights_dict = network.get_connection_weights()

    spikes_dict = context.comm.gather(spikes_dict, root=0)
    voltage_rec_dict = context.comm.gather(voltage_rec_dict, root=0)
    firing_rates_dict = context.comm.gather(firing_rates_dict, root=0)
    connection_weights_dict = context.comm.gather(connection_weights_dict, root=0)
    if context.comm.rank == 0:
        spikes_dict = merge_list_of_dict(spikes_dict)
        voltage_rec_dict = merge_list_of_dict(voltage_rec_dict)
        firing_rates_dict = merge_list_of_dict(firing_rates_dict)
        connection_weights_dict = merge_list_of_dict(connection_weights_dict)
        mean_rate_dict, peak_rate_dict, mean_rate_active_cells_dict, pop_fraction_active_dict, \
        binned_spike_count_dict, pop_binned_spike_count_dict = \
            get_pop_activity_stats(spikes_dict, firing_rates_dict, binned_t, threshold=context.active_rate_threshold,
                                   plot=plot)

        if plot:
            plot_inferred_spike_rates(spikes_dict, firing_rates_dict, binned_t, context.active_rate_threshold)
            plot_voltage_traces(voltage_rec_dict, rec_t)
            if context.debug:
                context.update(locals())
                return dict()
            network.plot_adj_matrix(connection_weights_dict)
            network.plot_population_firing_rates(firing_rates_dict, binned_t)

        E_mean, E_max = network.compute_pop_firing_features(network.cell_index['E'], rate_dict, peak_dict)
        I_mean, I_max = network.compute_pop_firing_features(network.cell_index['I'], rate_dict, peak_dict)

        I_pop_rate = mean_firing_active['I']
        E_pop_rate = mean_firing_active['E']
        FF_pop_rate = mean_firing_active['FF']
        pop_rates = [FF_pop_rate, I_pop_rate, E_pop_rate]
        ratios, bands = network.get_envelope_ratio(pop_rates, binned_t, context.filter_dt, plot=plot)

        theta_E = bands['theta_E']; gamma_E = bands['gamma_E']
        theta_I = bands['theta_I']; gamma_I = bands['gamma_I']
        theta_E_ratio = ratios['theta_E']; gamma_E_ratio = ratios['gamma_E']
        theta_I_ratio = ratios['theta_I']; gamma_I_ratio = ratios['gamma_I']

        peak_theta_freq_E = peak_from_spectrogram(theta_E, 'theta E', context.filter_dt, plot)
        peak_theta_freq_I = peak_from_spectrogram(theta_I, 'theta I', context.filter_dt, plot)
        peak_gamma_freq_E = peak_from_spectrogram(gamma_E, 'gamma E', context.filter_dt, plot)
        peak_gamma_freq_I = peak_from_spectrogram(gamma_I, 'gamma I', context.filter_dt, plot)

        context.update(locals())

        return {'E_mean_rate': E_mean, 'E_peak_rate': E_max, 'I_mean_rate': I_mean, "I_peak_rate": I_max,
                'peak_theta_freq_E': peak_theta_freq_E, 'peak_theta_freq_I': peak_theta_freq_I,
                'peak_gamma_freq_E': peak_gamma_freq_E, 'peak_gamma_freq_I': peak_gamma_freq_I,
                'E_frac_active': np.mean(frac_active['E']), 'I_frac_active': np.mean(frac_active['I']),
                'FF_frac_active': np.mean(frac_active['FF']), 'theta_E_envelope_ratio': theta_E_ratio,
                'theta_I_envelope_ratio': theta_I_ratio, 'gamma_E_envelope_ratio': gamma_E_ratio,
                'gamma_I_envelope_ratio': gamma_I_ratio}


def compute_features(x, export=False):
    """

    :param x: array
    :param export: bool
    :return: dict
    """
    update_source_contexts(x, context)
    context.pc.gid_clear()
    start_time = time.time()
    context.network = Network(pc=context.pc, pop_sizes=context.pop_sizes, pop_gid_ranges=context.pop_gid_ranges,
                              pop_cell_types=context.pop_cell_types, connection_syn_types=context.connection_syn_types,
                              prob_connection=context.prob_connection,
                              connection_weights_mean=context.connection_weights_mean,
                              connection_weight_sigma_factors=context.connection_weight_sigma_factors,
                              input_pop_mean_rates=context.input_pop_mean_rates,
                              input_pop_fraction_active=context.input_pop_fraction_active,
                              connection_kinetics=context.connection_kinetics, tstop=context.tstop,
                              equilibrate=context.equilibrate, dt=context.dt, delay=context.delay,
                              connection_seed=context.connection_seed, spikes_seed=context.spikes_seed,
                              verbose=context.verbose, debug=context.debug)
    if context.disp and int(context.pc.id()) == 0:
        print('NETWORK BUILD RUNTIME: %.2f s' % (time.time() - start_time))
    current_time = time.time()
    context.network.run()
    if int(context.pc.id()) == 0 and context.disp:
        print('NETWORK SIMULATION RUNTIME: %.2f s' % (time.time() - current_time))
    current_time = time.time()
    results = analyze_network_output(context.network, export=export, plot=context.plot)
    if int(context.pc.id()) == 0:
        if context.disp:
            print('NETWORK ANALYSIS RUNTIME: %.2f s' % (time.time() - current_time))
        if results is None:
            return dict()
        return results


def get_objectives(features, export=False):
    """

    :param features: dict
    :param export: bool
    :return: tuple of dict
    """
    if int(context.pc.id()) == 0:
        objectives = {}
        for feature_name in ['E_peak_rate', 'I_peak_rate', 'E_mean_rate', 'I_mean_rate', 'peak_theta_freq_E',
                             'peak_theta_freq_I', 'peak_gamma_freq_E', 'peak_gamma_freq_I', 'E_frac_active',
                             'I_frac_active', 'theta_E_envelope_ratio', 'theta_I_envelope_ratio',
                             'gamma_E_envelope_ratio', 'gamma_I_envelope_ratio']:
            objective_name = feature_name
            objectives[objective_name] = ((context.target_val[objective_name] - features[feature_name]) /
                                                  context.target_range[objective_name]) ** 2.
        return features, objectives


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
