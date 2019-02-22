from nested.optimize_utils import *
from random_network import *
import click
import matplotlib.pyplot as plt
import time


context = Context()


@click.command()
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/optimize_random_network_config.yaml')
@click.option("--export", is_flag=True)
@click.option("--output-dir", type=str, default='data')
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default=None)
@click.option("--interactive", is_flag=True)
@click.option("--verbose", type=int, default=2)
@click.option("--plot", is_flag=True)
def main(config_file_path, export, output_dir, export_file_path, label, interactive, verbose, plot):
    """

    :param config_file_path: str (path)
    :param export: bool
    :param output_dir: str
    :param export_file_path: str
    :param label: str
    :param interactive: bool
    :param verbose: int
    :param plot: bool
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
    init_context()
    context.pc = h.ParallelContext()


def init_context():
    """
    TODO: Define each population size separately here.
    """
    FF_ncell = 12
    E_ncell = 12
    I_ncell = 12
    delay = 1.  # ms
    tstop = 3000  # ms
    dt = 0.025  # ms
    binned_dt = 1.  # ms
    filter_dt = 1.  # ms
    active_rate_threshold = 1.  # Hz
    baks_alpha = 3.714383
    baks_beta = 5.327364E-01
    context.update(locals())


def update_context(x, local_context=None):
    """

    :param x: array
    :param local_context: :class:'Context'
    """
    if local_context is None:
        local_context = context
    x_dict = param_array_to_dict(x, context.param_names)
    local_context.e2e_prob = x_dict['EE_connection_prob']
    local_context.e2i_prob = x_dict['EI_connection_prob']
    local_context.i2i_prob = x_dict['II_connection_prob']
    local_context.i2e_prob = x_dict['IE_connection_prob']
    local_context.ff2i_weight = x_dict['FF2I_connection_weight']
    local_context.ff2e_weight = x_dict['FF2E_connection_weight']
    local_context.e2e_weight = x_dict['EE_connection_weight']
    local_context.e2i_weight = x_dict['EI_connection_weight']
    local_context.i2i_weight = x_dict['II_connection_weight']
    local_context.i2e_weight = x_dict['IE_connection_weight']
    local_context.ff_meanfreq = x_dict['FF_mean_freq']
    local_context.ff_frac_active = x_dict['FF_frac_active']
    local_context.ff2i_prob = x_dict['FF2I_connection_probability']
    local_context.ff2e_prob = x_dict['FF2E_connection_probability']
    local_context.tau_E = x_dict['tau_E']
    local_context.tau_I = x_dict['tau_I']
    local_context.weight_std_factors = {'ff2e': x_dict['FF2E_weights_sigma_factor'],
                                        'ff2i': x_dict['FF2I_weights_sigma_factor'],
                                        'e2i': x_dict['EI_weights_sigma_factor'],
                                        'e2e': x_dict['EE_weights_sigma_factor'],
                                        'i2e': x_dict['IE_weights_sigma_factor'],
                                        'i2i': x_dict['II_weights_sigma_factor']}


def analyze_network_output(network, export=False, plot=False):
    """

    :param network: :class:'Network'
    :param export: bool
    :param plot: bool
    :return: dict
    """
    py_spike_dict = network.vecdict_to_pydict(network.spike_tvec)
    py_spike_dict.update(network.FF_firing)
    py_V_dict = network.vecdict_to_pydict(network.voltage_recvec)
    gauss_firing_rates = network.event_dict_to_firing_rate(py_spike_dict, context.binned_dt)
    connection_dict = network.convert_ncdict_to_weights()

    all_spikes_dict = context.comm.gather(py_spike_dict, root=0)
    all_V_dict = context.comm.gather(py_V_dict, root=0)
    gauss_firing_rates = context.comm.gather(gauss_firing_rates, root=0)
    connection_dict = context.comm.gather(connection_dict, root=0)
    if context.comm.rank == 0:
        connection_dict = {key: val for connect_dict in connection_dict for key, val in connect_dict.iteritems()}
        gauss_firing_rates = {key: val for fire_dict in gauss_firing_rates for key, val in fire_dict.iteritems()}
        all_V_dict = {key: val for V_dict in all_V_dict for key, val in V_dict.iteritems()}
        all_spikes_dict = {key: val for spike_dict in all_spikes_dict for key, val in spike_dict.iteritems()}
        rate_dict, peak_dict = network.compute_mean_max_firing_per_cell(gauss_firing_rates)

    if context.comm.rank == 0:
        binned_t = np.arange(0., context.tstop + context.binned_dt, context.binned_dt)
        frac_active, mean_firing_active = network.get_active_pop_stats(gauss_firing_rates, binned_t,
                                                                  threshold=context.active_rate_threshold, plot=plot)
        network.py_summation(all_spikes_dict, binned_t)

        if plot:
            network.plot_adj_matrix(connection_dict)
            network.plot_voltage_trace(all_V_dict, all_spikes_dict, context.dt)
            network.plot_cell_activity(gauss_firing_rates, binned_t)

        E_mean, E_max = network.compute_pop_firing_features(network.cell_index['E'], rate_dict, peak_dict)
        I_mean, I_max = network.compute_pop_firing_features(network.cell_index['I'], rate_dict, peak_dict)

        theta_E, theta_I, gamma_E, gamma_I = network.get_bands_of_interest(binned_dt=context.binned_dt, plot=plot)
        peak_theta_freq_E = peak_from_spectrogram(theta_E, 'theta E', context.filter_dt, plot)
        peak_theta_freq_I = peak_from_spectrogram(theta_I, 'theta I', context.filter_dt, plot)
        peak_gamma_freq_E = peak_from_spectrogram(gamma_E, 'gamma E', context.filter_dt, plot)
        peak_gamma_freq_I = peak_from_spectrogram(gamma_I, 'gamma I', context.filter_dt, plot)

        I_pop_rate = mean_firing_active['I']
        E_pop_rate = mean_firing_active['E']
        theta_E_ratio = network.compute_envelope_ratio(theta_E, E_pop_rate, binned_t, label='E theta', plot=plot)
        theta_I_ratio = network.compute_envelope_ratio(theta_I, I_pop_rate, binned_t, label='I theta', plot=plot)
        gamma_E_ratio = network.compute_envelope_ratio(gamma_E, E_pop_rate, binned_t, label='E gamma', plot=plot)
        gamma_I_ratio = network.compute_envelope_ratio(gamma_I, I_pop_rate, binned_t, label='I gamma', plot=plot)

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
    context.network = Network(context.FF_ncell, context.E_ncell, context.I_ncell, context.delay, context.pc,
                              tstop=context.tstop, dt=context.dt, e2e_prob=context.e2e_prob, e2i_prob=context.e2i_prob,
                              i2i_prob=context.i2i_prob, i2e_prob=context.i2e_prob, ff2i_weight=context.ff2i_weight,
                              ff2e_weight=context.ff2e_weight, e2e_weight=context.e2e_weight,
                              e2i_weight=context.e2i_weight, i2i_weight=context.i2i_weight,
                              i2e_weight=context.i2e_weight, ff_meanfreq=context.ff_meanfreq,
                               ff_frac_active=context.ff_frac_active, ff2i_prob=context.ff2i_prob,
                              ff2e_prob=context.ff2e_prob, std_dict=context.weight_std_factors, tau_E=context.tau_E,
                              tau_I=context.tau_I, connection_seed=context.connection_seed,
                              spikes_seed=context.spikes_seed)
    context.comm.barrier()

    if context.disp and int(context.pc.id()) == 0:
        print('NETWORK BUILD RUNTIME: %.2f s' % (time.time() - start_time))
    current_time = time.time()
    context.network.run()
    results = analyze_network_output(context.network, export=export, plot=context.plot)
    if int(context.pc.id()) == 0:
        if context.disp:
            print('NETWORK SIMULATION RUNTIME: %.2f s' % (time.time() - current_time))
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
