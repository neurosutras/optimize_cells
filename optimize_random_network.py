from nested.optimize_utils import *
from random_network import *
import click
import random
import matplotlib.pyplot as plt


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

    random.seed(137)
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
    # setup_network(**context.kwargs)


def init_context():
    """

    """
    ncell = 12
    delay = 1
    tstop = 3000
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
    local_context.ff_sig = x_dict['FF_weights_sigma_factor']
    local_context.i_sig = x_dict['I_weights_sigma_factor']
    local_context.e_sig = x_dict['E_weights_sigma_factor']
    local_context.tau_E = x_dict['tau_E']
    local_context.tau_I = x_dict['tau_I']


# magic nums
def compute_features(x, export=False):
    """

    :param x: array
    :param export: bool
    :return: dict
    """
    update_source_contexts(x, context)
    context.pc.gid_clear()
    context.network = Network(context.ncell, context.delay, context.pc, e2e_prob=context.e2e_prob, \
                              e2i_prob=context.e2i_prob, i2i_prob=context.i2i_prob, i2e_prob=context.i2e_prob, \
                              ff2i_weight=context.ff2i_weight, ff2e_weight=context.ff2e_weight, e2e_weight= \
                                  context.e2e_weight, e2i_weight=context.e2i_weight, i2i_weight=context.i2i_weight, \
                              i2e_weight=context.i2e_weight, ff_meanfreq=context.ff_meanfreq, tstop=context.tstop, \
                              ff_frac_active=context.ff_frac_active, ff2i_prob=context.ff2i_prob, ff2e_prob= \
                                  context.ff2e_prob, ff_sig=context.ff_sig, i_sig=context.i_sig, e_sig=context.e_sig, \
                              tau_E=context.tau_E, tau_I=context.tau_I)
    results = run_network(context.network, context.pc, context.comm, context.tstop, context.plot)
    if int(context.pc.id()) == 0:
        if results is None:
            return dict()
        context.peak_voltage = results['peak']
        results.pop('peak', None)
        results.pop('event', None)
        results.pop('osc_E')
        return results


def get_objectives(features, export=False):
    """

    :param features: dict
    :param export: bool
    :return: tuple of dict
    """
    if int(context.pc.id()) == 0:
        objectives = {}
        for feature_name in ['E_peak_rate', 'I_peak_rate', 'E_mean_rate', 'I_mean_rate', 'peak_theta_osc_E', \
                             'peak_theta_osc_I', 'frac_active']:
            objective_name = feature_name
            if features[feature_name] == 0.:
                objectives[objective_name] = (context.peak_voltage - 45.) ** 2
            else:
                objectives[objective_name] = ((context.target_val[objective_name] - features[feature_name]) /
                                                      context.target_range[objective_name]) ** 2.
        return features, objectives


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
