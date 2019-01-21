"""
Uses nested.optimize to tune synaptic weights within a simple ring network.

Requires a YAML file to specify required configuration parameters.
Requires use of a nested.parallel interface.
"""
__author__ = 'Aaron D. Milstein and Grace Ng'
from nested.optimize_utils import *
from nested.parallel import *
from ring_network import *
import collections
import click


script_filename='optimize_simple_ring.py'

context = Context()


@click.command()
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='archived/config/optimize_simple_ring_config.yaml')
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
    config_optimize_interactive(__file__, config_file_path=config_file_path, output_dir=output_dir,
                       export_file_path=export_file_path, label=label, verbose=verbose)
    context.interface.start()

    sequences = [[context.x0_array]] + [[context.export]]
    primitives = context.interface.map(compute_features_simple_ring, *sequences)
    features = {key: value for feature_dict in primitives for key, value in feature_dict.iteritems()}
    features, objectives = get_objectives_simple_ring(features)
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
    setup_network(**context.kwargs)


def init_context():
    """

    """
    ncell = 3
    delay = 1
    tstop = 100
    context.update(locals())


def report_pc_id():
    return {'pc.id_world': context.pc.id_world(), 'pc.id': context.pc.id()}


def setup_network(verbose=2, **kwargs):
    """

    :param verbose: bool
    """
    context.ring = Ring(context.ncell, context.delay, context.pc)


#Need to update this as well -- do we need both update functions?
def update_context_simple_ring(x, local_context=None):
    """

    :param x: array
    :param local_context: :class:'Context'
    """
    if local_context is None:
        local_context = context
    x_dict = param_array_to_dict(x, context.param_names)
    local_context.ring.update_syn_weight((0, 1), x_dict['n0.syn_weight n1'])
    local_context.ring.update_syn_weight((0, 2), x_dict['n0.syn_weight n2'])
    local_context.ring.update_syn_weight((1, 2), x_dict['n1.syn_weight n2'])


def compute_features_simple_ring(x, export=False):
    update_source_contexts(x, context)
    results = runring(context.ring, context.pc, context.comm)
    if int(context.pc.id()) == 0:
        if results is None:
            return dict()
        max_ind = np.argmax(np.array(results['rec'][2]))
        min_ind = np.argmin(np.array(results['rec'][2]))
        equil_index = np.floor(0.05*len(results['rec'][2]))
        vm_baseline = np.mean(np.array(results['rec'][2][:int(equil_index)]))
        processed_result = {'n2.EPSP': results['rec'][2][max_ind] - vm_baseline, 'peak_t': results['t'][2][max_ind],
                            'n2.IPSP': results['rec'][2][min_ind] - vm_baseline, 'min_t': results['t'][2][min_ind],
                            'PC id': context.pc.id_world()}
        return processed_result


def get_objectives_simple_ring(features):
    if int(context.pc.id()) == 0:
        objectives = {}
        for feature_name in ['n2.EPSP', 'n2.IPSP']:
            objective_name = feature_name
            objectives[objective_name] = ((context.target_val[objective_name] - features[feature_name]) /
                                                      context.target_range[objective_name]) ** 2.
        return features, objectives


"""
def calc_spike_count(indiv, i):
    """"""
    x = indiv['x']
    results = runring(context.ring)
    return {'pop_id': int(i), 'result_list': [{'id': context.pc.id_world()}, results]}
"""


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
