from nested.optimize_utils import *
from simple_network_utils import *
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
    time.sleep(1.)
    features = context.interface.execute(compute_features, context.x0_array, context.export)
    sys.stdout.flush()
    time.sleep(1.)
    if not context.debug:
        features, objectives = context.interface.execute(get_objectives, features)
        sys.stdout.flush()
        time.sleep(1.)
        print 'params:'
        pprint.pprint(context.x0_dict)
        print 'features:'
        pprint.pprint(features)
        print 'objectives:'
        pprint.pprint(objectives)
        sys.stdout.flush()
        time.sleep(1.)
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
    else:
        context.verbose = int(context.verbose)
    if 'debug' not in context():
        context.debug = False
    init_context()
    context.pc = h.ParallelContext()


def init_context():
    pop_sizes = {'FF': 1000, 'E': 100, 'I': 100}
    pop_gid_ranges = get_pop_gid_ranges(pop_sizes)
    pop_cell_types = {'FF': 'input', 'E': 'IB', 'I': 'FS'}
    # {'postsynaptic population': {'presynaptic population': float} }
    prob_connection = defaultdict(dict)
    connection_weights_mean = defaultdict(dict)
    connection_weight_sigma_factors = defaultdict(dict)
    connection_syn_types = {'FF': 'E', 'E': 'E', 'I': 'I'}  # {'presynaptic population': syn_type}

    syn_mech_params = defaultdict(lambda: defaultdict(dict))
    syn_mech_params['I']['FF']['g_unit'] = 0.0001925
    syn_mech_params['I']['E']['g_unit'] = 0.0001925
    syn_mech_params['I']['I']['g_unit'] = 0.0001925
    syn_mech_params['E']['FF']['g_unit'] = 0.0005275
    syn_mech_params['E']['E']['g_unit'] = 0.0005275
    syn_mech_params['E']['I']['g_unit'] = 0.0005275

    input_pop_mean_rates = defaultdict(dict)
    input_pop_fraction_active = defaultdict(dict)
    if 'FF_mean_rate' not in context():
        raise RuntimeError('optimize_simple_network: missing required kwarg: FF_mean_rate')

    nsyn = 100
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
    filter_bands = {'Theta': [4., 10.], 'Gamma': [30., 100.]}
    context.update(locals())


def update_context(x, local_context=None):
    """

    :param x: array
    :param local_context: :class:'Context'
    """
    if local_context is None:
        local_context = context
    x_dict = param_array_to_dict(x, context.param_names)
    local_context.prob_connection['E']['FF'] = x_dict['E_FF_prob_connection']
    local_context.prob_connection['E']['E'] = x_dict['E_E_prob_connection']
    local_context.prob_connection['E']['I'] = x_dict['E_I_prob_connection']
    local_context.prob_connection['I']['FF'] = x_dict['I_FF_prob_connection']
    local_context.prob_connection['I']['E'] = x_dict['I_E_prob_connection']
    local_context.prob_connection['I']['I'] = x_dict['I_I_prob_connection']

    local_context.syn_mech_params['E']['FF']['tau_offset'] = x_dict['E_E_tau_offset']
    local_context.syn_mech_params['E']['E']['tau_offset'] = x_dict['E_E_tau_offset']
    local_context.syn_mech_params['E']['I']['tau_offset'] = x_dict['E_I_tau_offset']
    local_context.syn_mech_params['I']['FF']['tau_offset'] = x_dict['I_E_tau_offset']
    local_context.syn_mech_params['I']['E']['tau_offset'] = x_dict['I_E_tau_offset']
    local_context.syn_mech_params['I']['I']['tau_offset'] = x_dict['I_I_tau_offset']

    local_context.connection_weights_mean['E']['FF'] = x_dict['E_FF_mean_weight']
    local_context.connection_weights_mean['E']['E'] = x_dict['E_E_mean_weight']
    local_context.connection_weights_mean['E']['I'] = x_dict['E_I_mean_weight']
    local_context.connection_weights_mean['I']['FF'] = x_dict['I_FF_mean_weight']
    local_context.connection_weights_mean['I']['E'] = x_dict['I_E_mean_weight']
    local_context.connection_weights_mean['I']['I'] = x_dict['I_I_mean_weight']

    local_context.connection_weight_sigma_factors['E']['FF'] = x_dict['E_FF_weight_sigma_factor']
    local_context.connection_weight_sigma_factors['E']['E'] = x_dict['E_E_weight_sigma_factor']
    local_context.connection_weight_sigma_factors['E']['I'] = x_dict['E_I_weight_sigma_factor']
    local_context.connection_weight_sigma_factors['I']['FF'] = x_dict['I_FF_weight_sigma_factor']
    local_context.connection_weight_sigma_factors['I']['E'] = x_dict['I_E_weight_sigma_factor']
    local_context.connection_weight_sigma_factors['I']['I'] = x_dict['I_I_weight_sigma_factor']

    local_context.syn_proportion = x_dict['syn_proportion']
    local_context.input_pop_mean_rates['FF'] = local_context.FF_mean_rate


def analyze_network_output(network, export=False, plot=False):
    """

    :param network: :class:'SimpleNetwork'
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
        binned_spike_count_dict, mean_rate_from_spike_count_dict = \
            get_pop_activity_stats(spikes_dict, firing_rates_dict, binned_t, threshold=context.active_rate_threshold,
                                   plot=plot)

        filtered_mean_rate_dict, filter_envelope_dict, filter_envelope_ratio_dict, centroid_freq_dict = \
            get_pop_bandpass_filtered_signal_stats(mean_rate_from_spike_count_dict, binned_t, context.filter_bands,
                                                   plot=plot, verbose=context.verbose>0)

        if plot:
            plot_inferred_spike_rates(binned_spike_count_dict, firing_rates_dict, binned_t,
                                      context.active_rate_threshold)
            plot_voltage_traces(voltage_rec_dict, rec_t, spikes_dict)
            plot_weight_matrix(connection_weights_dict)
            plot_firing_rate_heatmaps(firing_rates_dict, binned_t)

        result = dict()

        result['E_mean_active_rate'] = np.mean(mean_rate_active_cells_dict['E'])
        result['I_mean_active_rate'] = np.mean(mean_rate_active_cells_dict['I'])
        result['E_peak_rate'] = np.mean(peak_rate_dict['E'].values())
        result['I_peak_rate'] = np.mean(peak_rate_dict['I'].values())
        result['E_frac_active'] = np.mean(pop_fraction_active_dict['E'])
        result['I_frac_active'] = np.mean(pop_fraction_active_dict['I'])
        result['FF_theta_envelope_ratio'] = filter_envelope_ratio_dict['Theta']['FF']
        result['E_theta_envelope_ratio'] = filter_envelope_ratio_dict['Theta']['E']
        result['I_theta_envelope_ratio'] = filter_envelope_ratio_dict['Theta']['I']
        result['FF_gamma_envelope_ratio'] = filter_envelope_ratio_dict['Gamma']['FF']
        result['E_gamma_envelope_ratio'] = filter_envelope_ratio_dict['Gamma']['E']
        result['I_gamma_envelope_ratio'] = filter_envelope_ratio_dict['Gamma']['I']
        result['E_centroid_theta_freq'] = centroid_freq_dict['Theta']['E']
        result['I_centroid_theta_freq'] = centroid_freq_dict['Theta']['I']
        result['E_centroid_gamma_freq'] = centroid_freq_dict['Gamma']['E']
        result['I_centroid_gamma_freq'] = centroid_freq_dict['Gamma']['I']

        context.update(locals())

        return result


def compute_features(x, export=False):
    """

    :param x: array
    :param export: bool
    :return: dict
    """
    update_source_contexts(x, context)
    context.pc.gid_clear()
    start_time = time.time()
    context.network = SimpleNetwork(pc=context.pc, pop_sizes=context.pop_sizes, pop_gid_ranges=context.pop_gid_ranges,
                              pop_cell_types=context.pop_cell_types, connection_syn_types=context.connection_syn_types,
                              prob_connection=context.prob_connection,
                              connection_weights_mean=context.connection_weights_mean,
                              connection_weight_sigma_factors=context.connection_weight_sigma_factors,
                              input_pop_mean_rates=context.input_pop_mean_rates,
                              syn_mech_params=context.syn_mech_params, tstop=context.tstop,
                              equilibrate=context.equilibrate, dt=context.dt, delay=context.delay,
                              nsyn=context.nsyn, syn_proportion=context.syn_proportion,
                              connection_seed=context.connection_seed, spikes_seed=context.spikes_seed,
                              verbose=context.verbose, debug=context.debug)
    if int(context.pc.id()) == 0 and context.verbose > 0:
        print('NETWORK BUILD RUNTIME: %.2f s' % (time.time() - start_time))
    current_time = time.time()
    context.network.run()
    if int(context.pc.id()) == 0 and context.verbose > 0:
        print('NETWORK SIMULATION RUNTIME: %.2f s' % (time.time() - current_time))
    current_time = time.time()
    results = analyze_network_output(context.network, export=export, plot=context.plot)
    if int(context.pc.id()) == 0:
        if context.verbose > 0:
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
        for objective_name in context.objective_names:
            objectives[objective_name] = ((context.target_val[objective_name] - features[objective_name]) /
                                          context.target_range[objective_name]) ** 2.
        return features, objectives


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
