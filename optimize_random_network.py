from nested.optimize_utils import *
from nested.parallel import *
from random_network import *
import collections
import click


script_filename='optimize_random_network.py'

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
def main(config_file_path, export, output_dir, export_file_path, label, interactive, verbose):
    """

    :param config_file_path: str (path)
    :param export: bool
    :param output_dir: str
    :param export_file_path: str
    :param label: str
    :param interactive: bool
    :param verbose: int
    """
    # requires a global variable context: :class:'Context'

    context.update(locals())
    from mpi4py import MPI
    from neuron import h
    comm = MPI.COMM_WORLD
    context.interface = ParallelContextInterface(procs_per_worker=comm.size)
    config_interactive(context, __file__, config_file_path=config_file_path, output_dir=output_dir,
                       export_file_path=export_file_path, label=label, verbose=verbose)
    context.interface.start()

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
    init_context()
    context.pc = h.ParallelContext()
    # setup_network(**context.kwargs)


def init_context():
    """

    """
    ncell = 1
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
    local_context.e2e = x_dict['EE_connection_prob']
    local_context.e2i = x_dict['EI_connection_prob']
    local_context.i2i = x_dict['II_connection_prob']
    local_context.i2e = x_dict['IE_connection_prob']


# magic nums
def compute_features(x, export=False):
    update_source_contexts(x, context)
    context.pc.gid_clear()
    context.network = Network(context.ncell, context.delay, context.pc,
                              e2e=context.e2e, e2i=context.e2i, i2i=context.i2i, i2e=context.i2e)
    results = run_network(context.network, context.pc, context.comm)
    if int(context.pc.id()) == 0:
        if results is None:
            return dict()
        return results


def get_objectives(features):
    if int(context.pc.id()) == 0:
        objectives = {}
        for feature_name in ['E_peak_rate', 'I_peak_rate', 'E_mean_rate', 'I_mean_rate']:
            objective_name = feature_name
            objectives[objective_name] = ((context.target_val[objective_name] - features[feature_name]) /
                                                      context.target_range[objective_name]) ** 2.
        return features, objectives


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
