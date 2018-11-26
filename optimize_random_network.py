from nested.optimize_utils import *
from nested.parallel import *
from random_network import *
import collections
import click


script_filename='optimize_random_network.py'

context = Context()


@click.command()
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='optimize_random_network_config.yaml')
@click.option("--export", is_flag=True)
@click.option("--output-dir", type=str, default='data')
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default=None)
@click.option("--disp", is_flag=True)
@click.option("--verbose", is_flag=True)
#keep
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
    config_interactive(context, __file__, config_file_path=config_file_path, output_dir=output_dir,
                       export_file_path=export_file_path, label=label, verbose=verbose)
    context.interface.start()
    num_params = 1
    sequences = [[context.x0_array] * num_params] + [[context.export] * num_params]
    primitives = context.interface.map(compute_features_simple_ring, *sequences)
    features = {key: value for feature_dict in primitives for key, value in feature_dict.iteritems()}
    features, objectives = get_objectives(features)
    print 'params:'
    pprint.pprint(context.x0_dict)
    print 'features:'
    pprint.pprint(features)
    print 'objectives:'
    pprint.pprint(objectives)
    context.interface.stop()


#keep
def config_worker():
    """

    """
    param_indexes = {param_name: i for i, param_name in enumerate(context.param_names)}
    context.update(locals())
    init_context()

    pc = context.interface.pc
    context.pc = pc
    setup_network(**context.kwargs)

#keep
def init_context():
    """

    """
    ncell = 3
    delay = 1
    tstop = 100
    context.update(locals())

#keep
def report_pc_id():
    return {'pc.id_world': context.pc.id_world(), 'pc.id': context.pc.id()}


def setup_network(verbose=False, cvode=False, daspk=False, **kwargs):
    """

    :param verbose: bool
    :param cvode: bool
    :param daspk: bool
    """
    context.network = Network(context.ncell, context.delay, context.pc)


def update_context(x, local_context=None):
    """

    :param x: array
    :param local_context: :class:'Context'
    """
    if local_context is None:
        local_context = context
    param_indexes = local_context.param_indexes
    context.e2e = x[param_indexes['EE_connection_prob']]
    context.e2i = x[param_indexes['EI_connection_prob']]
    context.i2i = x[param_indexes['II_connection_prob']]
    context.i2e = x[param_indexes['IE_connection_prob']]

# magic nums
def compute_features(x, export=False):
    update_source_contexts(x, context)
    context.pc.gid_clear()
    context.network = Network(context.ncell, context.delay, context.pc,
                              e2e=context.e2e, e2i=context.e2i, i2i=context.i2i, i2e=context.i2e)
    #context.network.remake_syn()
    results = run_network(context.network, context.pc, context.interface.comm)
    if int(context.pc.id()) == 0:
        processed_result = results
    else:
        processed_result = None
    return processed_result


# keep
def get_objectives(features):
    objectives = {}
    for feature_name in ['E_peak_rate', 'I_peak_rate', 'E_mean_rate', 'I_mean_rate']:
        objective_name = feature_name
        objectives[objective_name] = ((context.target_val[objective_name] - features[feature_name]) /
                                                  context.target_range[objective_name]) ** 2.
    return features, objectives


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):], standalone_mode=False)
