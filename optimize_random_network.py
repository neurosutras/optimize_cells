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
    context.interface.apply(config_optimize_interactive, __file__, config_file_path=config_file_path,
                            output_dir=output_dir, export=export, export_file_path=export_file_path, label=label,
                            disp=verbose > 0, verbose=verbose, plot=plot)
    context.interface.start(disp=True)
    context.interface.ensure_controller()

    sequences = [[context.x0_array]] + [[context.export]]
    primitives = context.interface.map(compute_features, *sequences)
    features = {key: value for feature_dict in primitives for key, value in feature_dict.iteritems()}
    features, objectives = get_objectives(features)
    print 'params:'
    pprint.pprint(context.x0_dict)
    print 'features:'
    pprint.pprint(features)
    print 'objectives:'
    pprint.pprint(objectives)
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
    delay = 1  # ms
    tstop = 3000  # ms
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


def compute_features(x, export=False):
    """

    :param x: array
    :param export: bool
    :return: dict
    """
    update_source_contexts(x, context)
    context.pc.gid_clear()
    start = time.time()
    context.network = Network(context.FF_ncell, context.E_ncell, context.I_ncell, context.delay, context.pc,
                              e2e_prob=context.e2e_prob, e2i_prob=context.e2i_prob, i2i_prob=context.i2i_prob,
                              i2e_prob=context.i2e_prob, ff2i_weight=context.ff2i_weight,
                              ff2e_weight=context.ff2e_weight, e2e_weight=context.e2e_weight,
                              e2i_weight=context.e2i_weight, i2i_weight=context.i2i_weight,
                              i2e_weight=context.i2e_weight, ff_meanfreq=context.ff_meanfreq,
                              tstop=context.tstop, ff_frac_active=context.ff_frac_active, ff2i_prob=context.ff2i_prob,
                              ff2e_prob=context.ff2e_prob, std_dict=context.weight_std_factors, tau_E=context.tau_E,
                              tau_I=context.tau_I, connection_seed=context.connection_seed,
                              spikes_seed=context.spikes_seed)
    context.comm.barrier()
    end = time.time()
    print("NETWORK BUILD RUNTIME: " + str(end - start))
    results = run_network(context.network, context.pc, context.comm, context.tstop, plot=context.plot)
    if int(context.pc.id()) == 0:
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
        for feature_name in ['E_peak_rate', 'I_peak_rate', 'E_mean_rate', 'I_mean_rate', 'peak_theta_osc_E',
                             'peak_theta_osc_I', 'E_frac_active', 'I_frac_active', 'theta_E_envelope_ratio',
                             'theta_I_envelope_ratio', 'gamma_E_envelope_ratio', 'gamma_I_envelope_ratio']:
            objective_name = feature_name
            if features[feature_name] == 0.:
                objectives[objective_name] = 200.
            else:
                objectives[objective_name] = ((context.target_val[objective_name] - features[feature_name]) /
                                                      context.target_range[objective_name]) ** 2.
        return features, objectives


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
