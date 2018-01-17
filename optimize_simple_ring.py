"""
Uses nested.optimize to tune synaptic weights within a simple ring network.

Requires a YAML file to specify required configuration parameters.
Requires use of a nested.parallel interface.
"""
__author__ = 'Aaron D. Milstein and Grace Ng'
from nested.optimize_utils import *
from ring_network import *
import collections
import click


script_filename='optimize_simple_ring.py'

context = Context()


@click.command()
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/optimize_simple_ring_config.yaml')
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
    config_interactive(config_file_path=config_file_path, output_dir=output_dir, export_file_path=export_file_path,
                       label=label, verbose=verbose)
    context.interface
    group_size = len(context.x0_array)
    sequences = [[context.x0_array] * group_size] + [[context.export] * group_size]
    primitives = map(compute_features_simple_ring, *sequences)
    features = {key: value for feature_dict in primitives for key, value in feature_dict.iteritems()}
    features, objectives = get_objectives_simple_ring(features)
    print 'params:'
    pprint.pprint(context.x0_dict)
    print 'features:'
    pprint.pprint(features)
    print 'objectives:'
    pprint.pprint(objectives)


def config_interactive(config_file_path=None, output_dir=None, temp_output_path=None, export_file_path=None,
                       label=None, verbose=True, **kwargs):
    """

    :param config_file_path: str (.yaml file path)
    :param output_dir: str (dir path)
    :param temp_output_path: str (.hdf5 file path)
    :param export_file_path: str (.hdf5 file path)
    :param label: str
    :param verbose: bool
    """

    if config_file_path is not None:
        context.config_file_path = config_file_path
    if 'config_file_path' not in context() or context.config_file_path is None or \
            not os.path.isfile(context.config_file_path):
        raise Exception('config_file_path specifying required parameters is missing or invalid.')
    config_dict = read_from_yaml(context.config_file_path)
    context.param_names = config_dict['param_names']
    if 'default_params' not in config_dict or config_dict['default_params'] is None:
        context.default_params = {}
    else:
        context.default_params = config_dict['default_params']
    for param in context.default_params:
        config_dict['bounds'][param] = (context.default_params[param], context.default_params[param])
    context.bounds = [config_dict['bounds'][key] for key in context.param_names]
    if 'rel_bounds' not in config_dict or config_dict['rel_bounds'] is None:
        context.rel_bounds = None
    else:
        context.rel_bounds = config_dict['rel_bounds']
    if 'x0' not in config_dict or config_dict['x0'] is None:
        context.x0 = None
    else:
        context.x0 = config_dict['x0']
        context.x0_dict = context.x0
        context.x0_array = param_dict_to_array(context.x0_dict, context.param_names)
    context.feature_names = config_dict['feature_names']
    context.objective_names = config_dict['objective_names']
    context.target_val = config_dict['target_val']
    context.target_range = config_dict['target_range']
    context.optimization_title = config_dict['optimization_title']
    context.kwargs = config_dict['kwargs']  # Extra arguments to be passed to imported sources
    context.kwargs['verbose'] = verbose
    context.update(context.kwargs)

    missing_config = []
    if 'update_context' not in config_dict or config_dict['update_context'] is None:
        missing_config.append('update_context')
    else:
        context.update_context_dict = config_dict['update_context']
    if 'get_features_stages' not in config_dict or config_dict['get_features_stages'] is None:
        missing_config.append('get_features_stages')
    else:
        context.stages = config_dict['get_features_stages']
    if 'get_objectives' not in config_dict or config_dict['get_objectives'] is None:
        missing_config.append('get_objectives')
    else:
        context.get_objectives_dict = config_dict['get_objectives']
    if missing_config:
        raise Exception('config_file at path: %s is missing the following required fields: %s' %
                        (context.config_file_path, ', '.join(str(field) for field in missing_config)))

    if label is not None:
        context.label = label
    if 'label' not in context() or context.label is None:
        label = ''
    else:
        label = '_' + context.label

    if output_dir is not None:
        context.output_dir = output_dir
    if 'output_dir' not in context():
        context.output_dir = None
    if context.output_dir is None:
        output_dir_str = ''
    else:
        output_dir_str = context.output_dir + '/'

    if temp_output_path is not None:
        context.temp_output_path = temp_output_path
    if 'temp_output_path' not in context() or context.temp_output_path is None:
        context.temp_output_path = '%s%s_pid%i_%s%s_temp_output.hdf5' % \
                                   (output_dir_str, datetime.datetime.today().strftime('%Y%m%d%H%M'), os.getpid(),
                                    context.optimization_title, label)

    if export_file_path is not None:
        context.export_file_path = export_file_path
    if 'export_file_path' not in context() or context.export_file_path is None:
        context.export_file_path = '%s%s_%s%s_interactive_exported_output.hdf5' % \
                                   (output_dir_str, datetime.datetime.today().strftime('%Y%m%d%H%M'),
                                    context.optimization_title, label)

    context.update_context_funcs = []
    for source, func_name in context.update_context_dict.iteritems():
        if source == script_filename.split('.')[0]:
            try:
                func = globals()[func_name]
                if not isinstance(func, collections.Callable):
                    raise Exception
                context.update_context_funcs.append(func)
            except:
                raise Exception('update_context function: %s not found' % func_name)
    if not context.update_context_funcs:
        raise Exception('update_context function not found')

    config_worker(context.update_context_funcs, context.param_names, context.default_params, context.target_val,
                  context.target_range, context.temp_output_path, context.export_file_path, context.output_dir,
                  context.disp, **context.kwargs)
    update_source_contexts(context.x0_array)


def config_controller(export_file_path, output_dir, **kwargs):
    """

    :param export_file_path: str (path)
    :param output_dir: str (dir)
    """
    processed_export_file_path = export_file_path.replace('.hdf5', '_processed.hdf5')
    context.update(locals())
    context.update(kwargs)
    init_context()


def config_worker(update_context_funcs, param_names, default_params, target_val, target_range, temp_output_path,
                  export_file_path, output_dur, disp, **kwargs):
    """
    :param update_context_funcs: list of function references
    :param param_names: list of str
    :param default_params: dict
    :param target_val: dict
    :param target_range: dict
    :param temp_output_path: str
    :param export_file_path: str
    :param output_dur: str (dir path)
    :param disp: bool
    """
    context.update(kwargs)
    param_indexes = {param_name: i for i, param_name in enumerate(param_names)}
    processed_export_file_path = export_file_path.replace('.hdf5', '_processed.hdf5')
    context.update(locals())
    init_context()

    pc = context.interface.pc
    context.pc = pc
    setup_network(**kwargs)


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
    context.ring = Ring(context.ncell, context.delay, context.pc)
    # context.pc.set_maxstep(10)


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
    results = runring(context.ring, context.pc)
    if rank == 0:
        max_ind = np.argmax(np.array(results['rec'][2]))
        min_ind = np.argmin(np.array(results['rec'][2]))
        equil_index = np.floor(0.05*len(results['rec'][2]))
        vm_baseline = np.mean(np.array(results['rec'][2][:int(equil_index)]))
        processed_result = {'n2.EPSP': results['rec'][2][max_ind] - vm_baseline, 'peak_t': results['t'][2][max_ind],
                            'n2.IPSP': results['rec'][2][min_ind] - vm_baseline, 'min_t': results['t'][2][min_ind],
                            'PC id': context.pc.id_world()}
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