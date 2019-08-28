from nested.parallel import *
from nested.optimize_utils import *
from simple_network_utils import *
import click
import time

context = Context()

@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True, ))
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/optimize_simple_network_gaussian_connections_lognormal_weights_gaussian_inputs_'
                      'config.yaml')
@click.option("--export", is_flag=True)
@click.option("--output-dir", type=str, default='data')
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default=None)
@click.option("--interactive", is_flag=True)
@click.option("--verbose", type=int, default=2)
@click.option("--plot", is_flag=True)
@click.option("--debug", is_flag=True)
@click.pass_context
def main(cli, config_file_path, export, output_dir, export_file_path, label, interactive, verbose, plot, debug):
    """

    :param cli: contains unrecognized args as list of str
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
    kwargs = get_unknown_click_arg_dict(cli.args)
    context.disp = verbose > 0

    if 'procs_per_worker' not in kwargs:
        kwargs['procs_per_worker'] = int(MPI.COMM_WORLD.size)

    context.interface = get_parallel_interface(source_file=__file__, source_package=__package__, **kwargs)
    context.interface.start(disp=context.disp)
    context.interface.ensure_controller()
    config_optimize_interactive(__file__, config_file_path=config_file_path, output_dir=output_dir,
                                export=export, export_file_path=export_file_path, label=label,
                                disp=context.disp, interface=context.interface, verbose=verbose, plot=plot,
                                debug=debug, **kwargs)

    features = context.interface.execute(compute_features, context.x0_array, context.export)
    sys.stdout.flush()
    time.sleep(1.)
    if len(features) > 0:
        features, objectives = context.interface.execute(get_objectives, features)
    else:
        objectives = dict()
    sys.stdout.flush()
    time.sleep(1.)
    print('params:')
    pprint.pprint(context.x0_dict)
    print('features:')
    pprint.pprint(features)
    print('objectives:')
    pprint.pprint(objectives)
    sys.stdout.flush()
    time.sleep(1.)
    if context.plot:
        context.interface.apply(plt.show)
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
    context.pc = h.ParallelContext()
    init_context()


def init_context():
    start_time = time.time()
    pop_sizes = {'FF': 100, 'E': 100, 'I': 20}
    pop_syn_counts = {'E': 100, 'I': 100}  # {'target_pop_name': int}
    pop_gid_ranges = get_pop_gid_ranges(pop_sizes)
    pop_cell_types = {'FF': 'input', 'E': 'IB', 'I': 'FS'}

    # {'target_pop_name': {'syn_type: {'source_pop_name': float} } }
    pop_syn_proportions = defaultdict(lambda: defaultdict(dict))
    connection_weights_mean = defaultdict(dict)  # {'target_pop_name': {'source_pop_name': float} }
    connection_weights_norm_sigma = defaultdict(dict)  # {'target_pop_name': {'source_pop_name': float} }
    if 'connection_weight_distribution_types' not in context() or context.connection_weight_distribution_types is None:
        context.connection_weight_distribution_types = dict()
    if 'structured_weight_params' not in context() or context.structured_weight_params is None:
        context.structured_weight_params = dict()
        structured_weights = False
    else:
        structured_weights = True
    local_np_random = np.random.RandomState()
    if any([this_input_type == 'gaussian' for this_input_type in context.input_types.values()]) or structured_weights:
        if 'tuning_seed' not in context() or context.tuning_seed is None:
            raise RuntimeError('optimize_simple_network: config_file missing required parameter: tuning_seed')
        local_np_random.seed(int(context.tuning_seed))

    syn_mech_params = defaultdict(lambda: defaultdict(dict))
    syn_mech_params['I']['FF']['g_unit'] = 0.0001925
    syn_mech_params['I']['E']['g_unit'] = 0.0001925
    syn_mech_params['I']['I']['g_unit'] = 0.0001925
    syn_mech_params['E']['FF']['g_unit'] = 0.0005275
    syn_mech_params['E']['E']['g_unit'] = 0.0005275
    syn_mech_params['E']['I']['g_unit'] = 0.0005275

    delay = 1.  # ms
    equilibrate = 250.  # ms
    duration = 3000. # ms
    tstop = duration + equilibrate  # ms
    dt = 0.025
    binned_dt = 1.  # ms
    filter_dt = 1.  # ms
    active_rate_threshold = 1.  # Hz
    baks_alpha = 4.7725100028345535
    baks_beta = 0.41969058927343522
    baks_pad_dur = 1000.  # ms
    filter_bands = {'Theta': [4., 10.], 'Gamma': [30., 100.]}

    if context.comm.rank == 0:
        input_pop_t = dict()
        input_pop_firing_rates = dict()
        tuning_peak_locs = dict()  # {'pop_name': {'gid': float} }

        for pop_name in context.input_types:
            if pop_name not in pop_cell_types or pop_cell_types[pop_name] != 'input':
                raise RuntimeError('optimize_simple_network: %s not specified as an input population' % pop_name)
            if pop_name not in input_pop_firing_rates:
                input_pop_firing_rates[pop_name] = dict()
            if pop_name not in tuning_peak_locs:
                tuning_peak_locs[pop_name] = dict()
            if context.input_types[pop_name] == 'constant':
                try:
                    this_mean_rate = context.input_mean_rates[pop_name]
                except Exception:
                    raise RuntimeError('optimize_simple_network: missing kwarg(s) required to specify %s input '
                                       'population: %s' % (context.input_types[pop_name], pop_name))
                if pop_name not in input_pop_t:
                    input_pop_t[pop_name] = [0., tstop]
                for gid in range(pop_gid_ranges[pop_name][0], pop_gid_ranges[pop_name][1]):
                    input_pop_firing_rates[pop_name][gid] = [this_mean_rate, this_mean_rate]
            elif context.input_types[pop_name] == 'gaussian':
                try:
                    this_min_rate = context.input_min_rates[pop_name]
                    this_max_rate = context.input_max_rates[pop_name]
                    this_norm_tuning_width = context.input_norm_tuning_widths[pop_name]
                except Exception:
                    raise RuntimeError('optimize_simple_network: missing kwarg(s) required to specify %s input '
                                       'population: %s' % (context.input_types[pop_name], pop_name))

                this_stim_t = np.arange(0., tstop + dt / 2., dt)
                if pop_name not in input_pop_t:
                    input_pop_t[pop_name] = this_stim_t

                this_tuning_width = duration * this_norm_tuning_width
                this_sigma = this_tuning_width / 3. / np.sqrt(2.)
                peak_locs_array = \
                    np.linspace(-0.75 * this_tuning_width, tstop + 0.75 * this_tuning_width, pop_sizes[pop_name])
                local_np_random.shuffle(peak_locs_array)
                for peak_loc, gid in zip(peak_locs_array, range(pop_gid_ranges[pop_name][0],
                                                                pop_gid_ranges[pop_name][1])):
                    tuning_peak_locs[pop_name][gid] = peak_loc
                    input_pop_firing_rates[pop_name][gid] = \
                        get_gaussian_rate(t=this_stim_t, peak_loc=peak_loc, sigma=this_sigma, min_rate=this_min_rate,
                                          max_rate=this_max_rate)

                if context.debug and context.plot:
                    fig, axes = plt.subplots()
                    for gid in range(pop_gid_ranges[pop_name][0],
                                     pop_gid_ranges[pop_name][1])[::int(pop_sizes[pop_name] / 25)]:
                        axes.plot(this_stim_t - equilibrate, input_pop_firing_rates[pop_name][gid])
                    mean_input = np.mean(list(input_pop_firing_rates[pop_name].values()), axis=0)
                    axes.plot(this_stim_t - equilibrate, mean_input, c='k', linewidth=2.)
                    axes.set_ylabel('Firing rate (Hz)')
                    axes.set_xlabel('Time (ms)')
                    clean_axes(axes)
                    fig.show()

        for target_pop_name in context.structured_weight_params:
            if target_pop_name not in tuning_peak_locs:
                tuning_peak_locs[target_pop_name] = dict()
            this_norm_tuning_width = \
                context.structured_weight_params[target_pop_name]['norm_tuning_width']
            this_tuning_width = duration * this_norm_tuning_width
            this_sigma = this_tuning_width / 3. / np.sqrt(2.)
            peak_locs_array = \
                np.linspace(-0.75 * this_tuning_width, tstop + 0.75 * this_tuning_width, pop_sizes[target_pop_name])
            local_np_random.shuffle(peak_locs_array)
            for peak_loc, target_gid in zip(peak_locs_array,
                                            range(pop_gid_ranges[target_pop_name][0],
                                                  pop_gid_ranges[target_pop_name][1])):
                tuning_peak_locs[target_pop_name][target_gid] = peak_loc
    else:
        input_pop_t = None
        input_pop_firing_rates = None
        tuning_peak_locs = None
    input_pop_t = context.comm.bcast(input_pop_t, root=0)
    input_pop_firing_rates = context.comm.bcast(input_pop_firing_rates, root=0)
    tuning_peak_locs = context.comm.bcast(tuning_peak_locs, root=0)

    if context.connectivity_type == 'gaussian':
        pop_axon_extents = {'FF': 1., 'E': 1., 'I': 1.}
        for param_name in ['spatial_dim', 'location_seed']:
            if param_name not in context():
                raise RuntimeError('optimize_simple_network: parameter required for gaussian connectivity not found: '
                                   '%s' % param_name)

        if context.comm.rank == 0:
            pop_cell_positions = dict()
            for pop_name in pop_gid_ranges:
                for gid in range(pop_gid_ranges[pop_name][0], pop_gid_ranges[pop_name][1]):
                    local_np_random.seed(int(context.location_seed) + gid)
                    if pop_name not in pop_cell_positions:
                        pop_cell_positions[pop_name] = dict()
                    pop_cell_positions[pop_name][gid] = local_np_random.uniform(-1., 1., size=context.spatial_dim)
        else:
            pop_cell_positions = None
        pop_cell_positions = context.comm.bcast(pop_cell_positions, root=0)

    context.update(locals())
    if int(context.pc.id()) == 0 and context.verbose > 0:
        print('INITIALIZATION TIME: %.2f s' % (time.time() - start_time))
        sys.stdout.flush()


def update_context(x, local_context=None):
    """

    :param x: array
    :param local_context: :class:'Context'
    """
    if local_context is None:
        local_context = context
    x_dict = param_array_to_dict(x, local_context.param_names)
    
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

    local_context.connection_weights_norm_sigma['E']['FF'] = x_dict['E_FF_weight_norm_sigma']
    local_context.connection_weights_norm_sigma['E']['E'] = x_dict['E_E_weight_norm_sigma']
    local_context.connection_weights_norm_sigma['E']['I'] = x_dict['E_I_weight_norm_sigma']
    local_context.connection_weights_norm_sigma['I']['FF'] = x_dict['I_FF_weight_norm_sigma']
    local_context.connection_weights_norm_sigma['I']['E'] = x_dict['I_E_weight_norm_sigma']
    local_context.connection_weights_norm_sigma['I']['I'] = x_dict['I_I_weight_norm_sigma']

    local_context.pop_syn_proportions['E']['E']['FF'] = x_dict['E_E_syn_proportion'] * x_dict['E_E_FF_syn_proportion']
    local_context.pop_syn_proportions['E']['E']['E'] = x_dict['E_E_syn_proportion'] * \
                                                       (1. - x_dict['E_E_FF_syn_proportion'])
    local_context.pop_syn_proportions['E']['I']['I'] = 1. - x_dict['E_E_syn_proportion']
    local_context.pop_syn_proportions['I']['E']['FF'] = x_dict['I_E_syn_proportion'] * x_dict['I_E_FF_syn_proportion']
    local_context.pop_syn_proportions['I']['E']['E'] = x_dict['I_E_syn_proportion'] * \
                                                       (1. - x_dict['I_E_FF_syn_proportion'])
    local_context.pop_syn_proportions['I']['I']['I'] = 1. - x_dict['I_E_syn_proportion']


def analyze_network_output(network, export=False, export_file_path=None, plot=False):
    """

    :param network: :class:'SimpleNetwork'
    :param export: bool
    :param export_file_path: str
    :param plot: bool
    :return: dict
    """
    rec_t = np.arange(0., context.tstop + context.dt - context.equilibrate, context.dt)
    binned_t = np.arange(0., context.tstop + context.binned_dt - context.equilibrate, context.binned_dt)
    spikes_dict = network.get_spikes_dict()
    voltage_rec_dict = network.get_voltage_rec_dict()
    voltages_exceed_threshold = check_voltages_exceed_threshold(voltage_rec_dict, context.pop_cell_types)
    firing_rates_dict = infer_firing_rates(spikes_dict, binned_t, alpha=context.baks_alpha, beta=context.baks_beta,
                                           pad_dur=context.baks_pad_dur)
    connection_weights_dict = network.get_connection_weights()
    gid_connections = context.network.get_gid_connections_dict()

    spikes_dict = context.comm.gather(spikes_dict, root=0)
    voltage_rec_dict = context.comm.gather(voltage_rec_dict, root=0)
    firing_rates_dict = context.comm.gather(firing_rates_dict, root=0)
    connection_weights_dict = context.comm.gather(connection_weights_dict, root=0)
    voltages_exceed_threshold_list = context.comm.gather(voltages_exceed_threshold, root=0)
    gid_connections = context.comm.gather(gid_connections, root=0)

    if context.comm.rank == 0:
        spikes_dict = merge_list_of_dict(spikes_dict)
        voltage_rec_dict = merge_list_of_dict(voltage_rec_dict)
        firing_rates_dict = merge_list_of_dict(firing_rates_dict)
        connection_weights_dict = merge_list_of_dict(connection_weights_dict)
        gid_connections = merge_list_of_dict(gid_connections)
        mean_rate_dict, peak_rate_dict, mean_rate_active_cells_dict, pop_fraction_active_dict, \
            binned_spike_count_dict, mean_rate_from_spike_count_dict = \
            get_pop_activity_stats(spikes_dict, firing_rates_dict, binned_t, threshold=context.active_rate_threshold,
                                   plot=plot)

        filtered_mean_rate_dict, filter_envelope_dict, filter_envelope_ratio_dict, centroid_freq_dict, \
            freq_tuning_index_dict = \
            get_pop_bandpass_filtered_signal_stats(mean_rate_from_spike_count_dict, binned_t, context.filter_bands,
                                                   plot=plot, verbose=context.verbose > 1)
        if plot:
            plot_inferred_spike_rates(binned_spike_count_dict, firing_rates_dict, binned_t,
                                      context.active_rate_threshold)
            plot_voltage_traces(voltage_rec_dict, rec_t, spikes_dict)
            plot_weight_matrix(connection_weights_dict, tuning_peak_locs=context.tuning_peak_locs)
            plot_firing_rate_heatmaps(firing_rates_dict, binned_t, tuning_peak_locs=context.tuning_peak_locs)
            if context.connectivity_type == 'gaussian':
                plot_rel_distance(context.network.pop_gid_ranges, context.pop_cell_types, context.pop_syn_proportions,
                                  context.pop_cell_positions, gid_connections, plot_from_hdf5=False)
                visualize_connections(context.network.pop_gid_ranges, context.pop_cell_types, context.pop_syn_proportions,
                                      context.pop_cell_positions, gid_connections, n=1, plot_from_hdf5=False)

        if export:
            varname_dict = {'binned_spike_count_dict' : binned_spike_count_dict,
                            'spikes_dict' : spikes_dict,
                            'firing_rates_dict' : firing_rates_dict,
                            'binned_t' : binned_t,
                            'active_rate_threshold' : context.active_rate_threshold,
                            'voltage_rec_dict' : voltage_rec_dict,
                            'rec_t' : rec_t,
                            'filter_bands' : context.filter_bands,
                            'pop_gid_ranges' : context.network.pop_gid_ranges,
                            'pop_cell_types' : context.pop_cell_types,
                            'pop_syn_proportions' : context.pop_syn_proportions,
                            'gid_connections' : gid_connections,
                            'mean_rate_from_spike_count_dict' : mean_rate_from_spike_count_dict,
                            'connection weights': connection_weights_dict,
                            'tuning_peak_locs': context.tuning_peak_locs,
                            'pop_cell_positions': context.pop_cell_positions,
                            }
            export_plot_vars(export_file_path, varname_dict)

        """
        if context.debug:
            context.update(locals())
            return dict()
        """

        result = dict()

        result['E_mean_active_rate'] = np.mean(mean_rate_active_cells_dict['E'])
        result['I_mean_active_rate'] = np.mean(mean_rate_active_cells_dict['I'])
        result['E_peak_rate'] = np.mean(list(peak_rate_dict['E'].values()))
        result['I_peak_rate'] = np.mean(list(peak_rate_dict['I'].values()))
        result['FF_frac_active'] = np.mean(pop_fraction_active_dict['FF'])
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
        result['E_theta_tuning_index'] = freq_tuning_index_dict['Theta']['E']
        result['I_theta_tuning_index'] = freq_tuning_index_dict['Theta']['I']
        result['E_gamma_tuning_index'] = freq_tuning_index_dict['Gamma']['E']
        result['I_gamma_tuning_index'] = freq_tuning_index_dict['Gamma']['I']

        if any(voltages_exceed_threshold_list):
            if context.verbose > 0:
                print('optimize_simple_network: pid: %i; model failed - mean membrane voltage in some Izhi cells '
                      'exceeds spike threshold' % os.getpid())
                sys.stdout.flush()
            result['failed'] = True

        if context.interactive:
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
    context.network = SimpleNetwork(
        pc=context.pc, pop_sizes=context.pop_sizes, pop_gid_ranges=context.pop_gid_ranges,
        pop_cell_types=context.pop_cell_types, pop_syn_counts=context.pop_syn_counts,
        pop_syn_proportions=context.pop_syn_proportions, connection_weights_mean=context.connection_weights_mean,
        connection_weights_norm_sigma=context.connection_weights_norm_sigma,
        syn_mech_params=context.syn_mech_params, input_pop_t=context.input_pop_t,
        input_pop_firing_rates=context.input_pop_firing_rates, tstop=context.tstop, equilibrate=context.equilibrate,
        dt=context.dt, delay=context.delay, spikes_seed=context.spikes_seed, verbose=context.verbose,
        debug=context.debug)

    if context.connectivity_type == 'uniform':
        context.network.connect_cells(connectivity_type=context.connectivity_type,
                                      connection_seed=context.connection_seed)
    elif context.connectivity_type == 'gaussian':
        context.network.connect_cells(connectivity_type=context.connectivity_type,
                                      connection_seed=context.connection_seed,
                                      pop_axon_extents=context.pop_axon_extents,
                                      pop_cell_positions=context.pop_cell_positions)

    context.network.assign_connection_weights(
        default_weight_distribution_type=context.default_weight_distribution_type,
        connection_weight_distribution_types=context.connection_weight_distribution_types,
        weights_seed=context.weights_seed)

    if context.structured_weights:
        context.network.structure_connection_weights(structured_weight_params=context.structured_weight_params,
                                                     tuning_peak_locs=context.tuning_peak_locs)
    """
    if context.debug:
        context.update(locals())
        return dict()
    """

    if int(context.pc.id()) == 0 and context.verbose > 0:
        print('NETWORK BUILD RUNTIME: %.2f s' % (time.time() - start_time))
        sys.stdout.flush()
    current_time = time.time()

    context.network.run()
    if int(context.pc.id()) == 0 and context.verbose > 0:
        print('NETWORK SIMULATION RUNTIME: %.2f s' % (time.time() - current_time))
        sys.stdout.flush()
    current_time = time.time()

    results = analyze_network_output(context.network, export=export, export_file_path=context.export_file_path,
                                     plot=context.plot)
    if int(context.pc.id()) == 0:
        if context.verbose > 0:
            print('NETWORK ANALYSIS RUNTIME: %.2f s' % (time.time() - current_time))
            sys.stdout.flush()
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
            if objective_name.find('tuning_index') != -1 and \
                    features[objective_name] >= context.target_val[objective_name]:
                objectives[objective_name] = 0.
            else:
                objectives[objective_name] = ((context.target_val[objective_name] - features[objective_name]) /
                                            context.target_range[objective_name]) ** 2.
        return features, objectives


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
