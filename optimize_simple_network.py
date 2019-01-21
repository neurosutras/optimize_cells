"""
Uses nested.optimize to tune synaptic weights within a simple ring network.

Requires a YAML file to specify required configuration parameters.
Requires use of a nested.parallel interface.
"""
__author__ = 'Aaron D. Milstein and Grace Ng'
from nested.optimize_utils import *
from nested.parallel import *
from dentate_network import *
from dentate.env import Env
import collections
import click


script_filename='optimize_simple_network.py'

context = Context()


@click.command()
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/optimize_simple_network_config.yaml')
@click.option("--export", is_flag=True)
@click.option("--output-dir", type=str, default='data')
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default=None)
@click.option("--disp", is_flag=True)
@click.option("--verbose", is_flag=True)
def main(config_file_path, export, output_dir, export_file_path, label, disp, verbose):
    """

    :param config_file_path: str (path)
    :param export: bool
    :param output_dir: str
    :param export_file_path: str
    :param label: str
    :param disp: bool
    :param verbose: bool
    """
    # requires a global variable context: :class:'Context'

    context.update(locals())
    group_size = 2
    context.interface = ParallelContextInterface(procs_per_worker=group_size)
    config_optimize_interactive(__file__, config_file_path=config_file_path, output_dir=output_dir,
                                export_file_path=export_file_path, label=label, disp=disp, verbose=verbose)
    context.interface.start(disp=disp)
    context.interface.ensure_controller()

    num_params = 1
    sequences = [[context.x0_array] * num_params] + [[context.export] * num_params]
    primitives = context.interface.map(compute_features_simple_ring, *sequences)
    features = {key: value for feature_dict in primitives for key, value in feature_dict.iteritems()}
    features, objectives = get_objectives_simple_ring(features)
    print 'params:'
    pprint.pprint(context.x0_dict)
    print 'features:'
    pprint.pprint(features)
    print 'objectives:'
    pprint.pprint(objectives)
    context.interface.stop()


def config_worker():
    """

    """
    param_indexes = {param_name: i for i, param_name in enumerate(context.param_names)}
    processed_export_file_path = context.export_file_path.replace('.hdf5', '_processed.hdf5')
    context.update(locals())
    init_context()

    pc = context.interface.pc
    context.pc = pc
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


def setup_network(verbose=False, cvode=False, daspk=False, **kwargs):
    """

    :param verbose: bool
    :param cvode: bool
    :param daspk: bool
    """
    # Set up Env according to dentate script
    context.env = Env(context.interace.comm, kwargs['network_config_file'], kwargs['template_paths'],
                      kwargs['dataset_prefix'], kwargs['results_path'], kwargs['results_id'], kwargs['node_rank_file'],
                      kwargs['io_size'], kwargs['vrecord_fraction'], kwargs['coredat'], kwargs['tstop'], kwargs['v_init'],
                      kwargs['max_walltime_hours'], kwargs['results_write_time'], kwargs['dt'], kwargs['ldbal'],
                      kwargs['lptbal'], kwargs['verbose'])
    init(context.env)


#Need to modify this for the network
def update_source_contexts(x, local_context=None):
    """

    :param x: array
    :param local_context: :class:'Context'
    """
    if local_context is None:
        local_context = context
    """
    local_context.cell.reinit_mechanisms(from_file=True)
    if not local_context.spines:
        local_context.cell.correct_g_pas_for_spines()
    """
    for update_func in local_context.update_context_funcs:
        update_func(x, local_context)


#Need to update this as well -- do we need both update functions?
def update_context_simple_ring(x, local_context=None):
    """

    :param x: array
    :param local_context: :class:'Context'
    """
    if local_context is None:
        local_context = context
    param_indexes = local_context.param_indexes
    context.ring.update_syn_weight((0, 1), x[param_indexes['n0.syn_weight n1']])
    context.ring.update_syn_weight((0, 2), x[param_indexes['n0.syn_weight n2']])
    context.ring.update_syn_weight((1, 2), x[param_indexes['n1.syn_weight n2']])


def compute_features_simple_ring(x, export=False):
    update_source_contexts(x, context)
    results = runring(context.ring, context.pc, context.interface.comm)
    if int(context.pc.id()) == 0:
        max_ind = np.argmax(np.array(results['rec'][2]))
        min_ind = np.argmin(np.array(results['rec'][2]))
        equil_index = np.floor(0.05*len(results['rec'][2]))
        vm_baseline = np.mean(np.array(results['rec'][2][:int(equil_index)]))
        processed_result = {'n2.EPSP': results['rec'][2][max_ind] - vm_baseline, 'peak_t': results['t'][2][max_ind],
                            'n2.IPSP': results['rec'][2][min_ind] - vm_baseline, 'min_t': results['t'][2][min_ind],
                            'PC id': context.pc.id_world()}
    else:
        processed_result = None
    return processed_result


def get_objectives_simple_ring(features):
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
    main(args=sys.argv[(list_find(lambda s: s.find(script_filename) != -1, sys.argv) + 1):], standalone_mode=False)