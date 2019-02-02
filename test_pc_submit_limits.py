"""
Uses nested.optimize to mimic number of pc.submit calls typical of optimize_DG_GC_synaptic_integration to determine if
there is an inherent limit to the total number of submissions to pc.submit.

Requires a YAML file to specify required configuration parameters.
Requires use of a nested.parallel interface.
"""
__author__ = 'Aaron D. Milstein'
from nested.optimize_utils import *
import click
import uuid


context = Context()


@click.command()
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/test_pc_submit_limits_config.yaml')
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--verbose", type=int, default=2)
@click.option("--interactive", is_flag=True)
def main(config_file_path, output_dir, verbose, interactive):
    """

    :param config_file_path: str (path)
    :param output_dir: str (path)
    :param verbose: bool
    """
    # requires a global variable context: :class:'Context'
    context.update(locals())
    disp = verbose > 0

    from nested.parallel import ParallelContextInterface
    context.interface = ParallelContextInterface()
    context.interface.apply(config_optimize_interactive, __file__, config_file_path=config_file_path,
                            output_dir=output_dir, disp=disp, verbose=verbose)

    if context.interface.global_size < 2:
        raise Exception('test_pc_submit_limits require at least 2 MPI ranks')

    context.interface.start(disp=True)
    context.interface.ensure_controller()

    features = dict()
    objectives = dict()

    # Stage 0:
    args = get_args_dynamic_unitary_EPSP_amp(context.x0_array, features)
    group_size = len(args[0])

    sequences = [[context.x0_array] * group_size] + args + [[context.export] * group_size]
    primitives = context.interface.map(compute_features_unitary_EPSP_amp, *sequences)
    this_features = filter_features_unitary_EPSP_amp(primitives, features, context.export)
    features.update(this_features)

    context.interface.apply(export_unitary_EPSP_traces)

    # Stage 1:
    args = get_args_dynamic_compound_EPSP_amp(context.x0_array, features)
    group_size = len(args[0])
    sequences = [[context.x0_array] * group_size] + args + [[context.export] * group_size]
    primitives = context.interface.map(compute_features_compound_EPSP_amp, *sequences)
    this_features = filter_features_compound_EPSP_amp(primitives, features, context.export)
    features.update(this_features)

    context.interface.apply(export_compound_EPSP_traces)

    features, objectives = get_objectives_synaptic_integration(features, context.export)

    context.update(locals())
    print 'params:'
    pprint.pprint(context.x0_dict)
    print 'features:'
    pprint.pprint(features)
    print 'objectives:'
    pprint.pprint(objectives)

    if not interactive:
        context.interface.stop()


def config_worker():
    """

    """
    init_context()


def init_context():
    """

    """
    syn_conditions = ['control', 'AP5']

    # number of branches to test temporal integration of clustered inputs
    num_random_syns = 78
    num_clustered_branches = 2
    num_syns_per_clustered_branch = 30

    clustered_branch_names = ['clustered%i' % i for i in xrange(num_clustered_branches)]

    units_per_sim = 5

    context.update(locals())
    context.temp_model_data = dict()

    context.syn_id_dict = {'random': range(num_random_syns)}
    count = num_random_syns
    for branch_name in clustered_branch_names:
        context.syn_id_dict[branch_name] = range(count, count + num_syns_per_clustered_branch)
        count += num_syns_per_clustered_branch


def update_syn_mechanisms(x, context=None):
    """

    :param x: array
    :param context: :class:'Context'
    """
    if context is None:
        raise RuntimeError('update_syn_mechanisms: missing required Context object')
    x_dict = param_array_to_dict(x, context.param_names)


def export_unitary_EPSP_traces():
    """
    Data from model simulations is temporarily stored locally on each worker. This method uses collective operations to
    export the data to disk, with one hdf5 group per model.
    Global MPI rank 0 cannot participate in global collective operations while monitoring the NEURON ParallelContext
    bulletin board, so global rank 1 serves as root for this collective write procedure.
    """
    start_time = time.time()
    description = 'unitary_EPSP_traces'
    context.temp_model_data_legend = dict()

    if context.interface.global_comm.rank == 0:
        context.interface.pc.post('merge1', context.temp_model_data)
        print 'Key counter: %i' % context.interface.key_counter
    elif context.interface.global_comm.rank == 1:
        context.interface.pc.take('merge1')
        temp_model_data_from_master = context.interface.pc.upkpyobj()
        dict_merge(context.temp_model_data, temp_model_data_from_master)

    if context.interface.global_comm.rank > 0:
        model_keys = context.temp_model_data.keys()
        model_keys = context.interface.worker_comm.gather(model_keys, root=0)
        if context.interface.worker_comm.rank == 0:
            model_keys = list(set([key for key_list in model_keys for key in key_list]))
            print 'gathered model keys: %s' % str(model_keys)
        else:
            model_keys = None
        model_keys = context.interface.worker_comm.bcast(model_keys, root=0)

        for i, model_key in enumerate(model_keys):
            group_key = str(i)
            context.temp_model_data_legend[model_key] = group_key

    del context.temp_model_data
    context.temp_model_data = dict()

    if context.interface.global_comm.rank == 0:
        context.interface.pc.take('merge2')
        description = context.interface.pc.upkpyobj()[0]
        context.interface.pc.take('merge3')
        context.temp_model_data_legend = context.interface.pc.upkpyobj()
    elif context.interface.global_comm.rank == 1:
        context.interface.pc.post('merge2', [description])
        context.interface.pc.post('merge3', context.temp_model_data_legend)

    if context.interface.global_comm.rank == 1 and context.disp:
        print 'test_pc_submit_limits: export_unitary_EPSP_traces took %.2f s' % (time.time() - start_time)


def export_compound_EPSP_traces():
    """
    Data from model simulations is temporarily stored locally on each worker. This method uses collective operations to
    export the data to disk, with one hdf5 group per model.
    """
    start_time = time.time()
    description = 'compound_EPSP_traces'

    if context.interface.global_comm.rank == 0:
        context.interface.pc.post('merge4', context.temp_model_data)
    elif context.interface.global_comm.rank == 1:
        context.interface.pc.take('merge4')
        temp_model_data_from_master = context.interface.pc.upkpyobj()
        dict_merge(context.temp_model_data, temp_model_data_from_master)

    if context.interface.global_comm.rank > 0:
        model_keys = context.temp_model_data.keys()
        model_keys = context.interface.worker_comm.gather(model_keys, root=0)
        if context.interface.worker_comm.rank == 0:
            model_keys = list(set([key for key_list in model_keys for key in key_list]))
            print 'gathered model keys: %s' % str(model_keys)
        else:
            model_keys = None
        model_keys = context.interface.worker_comm.bcast(model_keys, root=0)

    del context.temp_model_data
    context.temp_model_data = dict()

    if context.interface.global_comm.rank == 0:
        context.interface.pc.take('merge5')
        handshake = context.interface.pc.upkscalar()
    elif context.interface.global_comm.rank == 1:
        context.interface.pc.post('merge5', 0)

    if context.interface.global_comm.rank == 1 and context.disp:
        print 'test_pc_submit_limits: export_compound_EPSP_traces took %.2f s' % \
              (time.time() - start_time)


def get_args_dynamic_unitary_EPSP_amp(x, features):
    """
    A nested map operation is required to compute unitary EPSP amplitude features. The arguments to be mapped include
    a unique string key for each set of model parameters that will be used to identify temporarily stored simulation
    output.
    :param x: array
    :param features: dict
    :return: list of list
    """
    model_key = str(uuid.uuid1())

    syn_group_list = []
    syn_id_lists = []
    syn_condition_list = []
    model_key_list = []

    for syn_group in context.syn_id_dict:
        this_syn_id_chunk = context.syn_id_dict[syn_group]
        this_syn_id_lists = []
        start = 0
        while start < len(this_syn_id_chunk):
            this_syn_id_lists.append(this_syn_id_chunk[start:start + context.units_per_sim])
            start += context.units_per_sim
        num_sims = len(this_syn_id_lists)
        for syn_condition in context.syn_conditions:
            syn_id_lists.extend(this_syn_id_lists)
            syn_group_list.extend([syn_group] * num_sims)
            syn_condition_list.extend([syn_condition] * num_sims)
            model_key_list.extend([model_key] * num_sims)

    return [syn_id_lists, syn_condition_list, syn_group_list, model_key_list]


def compute_features_unitary_EPSP_amp(x, syn_ids, syn_condition, syn_group, model_key, export=False, plot=False):
    """

    :param x: array
    :param syn_ids: list of int
    :param syn_condition: str
    :param syn_group: str
    :param model_key: str
    :param export: bool
    :param plot: bool
    :return: dict
    """
    start_time = time.time()
    update_source_contexts(x, context)

    fake_data = np.random.rand(10)

    first_syn_id = syn_ids[0]

    new_model_data = {model_key: {'unitary_EPSP_traces': {syn_group: {syn_condition: {first_syn_id: fake_data}}}}}

    dict_merge(context.temp_model_data, new_model_data)
    
    result = {'model_key': model_key}

    if context.verbose > 1:
        print 'compute_features_unitary_EPSP_amp: pid: %i; took %.3f s' % (os.getpid(), time.time() - start_time)

    return result


def filter_features_unitary_EPSP_amp(primitives, current_features, export=False):
    """
    :param primitives: list of dict (each dict contains results from a single simulation)
    :param current_features: dict
    :param export: bool
    :return: dict
    """
    features = {}

    model_key = None
    for this_feature_dict in primitives:
        this_model_key = this_feature_dict['model_key']
        if model_key is None:
            model_key = this_model_key
        if this_model_key != model_key:
            raise KeyError('filter_features_unitary_EPSP_amp: mismatched model keys')

    features['model_key'] = model_key

    return features


def get_args_dynamic_compound_EPSP_amp(x, features):
    """
    A nested map operation is required to compute compound EPSP amplitude features. The arguments to be mapped include
    a unique string key for each set of model parameters that will be used to identify temporarily stored simulation
    output.
    :param x: array
    :param features: dict
    :return: list of list
    """
    syn_group_list = []
    syn_id_lists = []
    syn_condition_list = []
    model_key = features['model_key']
    model_key_list = []
    for syn_group in context.clustered_branch_names:
        this_syn_id_group = context.syn_id_dict[syn_group]
        this_syn_id_lists = []
        for i in range(len(this_syn_id_group)):
            this_syn_id_lists.append(this_syn_id_group[:i+1])
        num_sims = len(this_syn_id_lists)
        for syn_condition in context.syn_conditions:
            syn_id_lists.extend(this_syn_id_lists)
            syn_group_list.extend([syn_group] * num_sims)
            syn_condition_list.extend([syn_condition] * num_sims)
            model_key_list.extend([model_key] * num_sims)

    return [syn_id_lists, syn_condition_list, syn_group_list, model_key_list]


def compute_features_compound_EPSP_amp(x, syn_ids, syn_condition, syn_group, model_key, export=False, plot=False):
    """

    :param x: array
    :param syn_ids: list of int
    :param syn_condition: str
    :param syn_group: str
    :param model_key: str
    :param export: bool
    :param plot: bool
    :return: dict
    """
    start_time = time.time()
    update_source_contexts(x, context)

    fake_data = np.random.rand(10)

    num_syns = len(syn_ids)

    new_model_data = {model_key:
                          {'compound_EPSP_traces':
                               {syn_group:
                                    {syn_condition: {num_syns: fake_data}}}}}

    dict_merge(context.temp_model_data, new_model_data)

    result = {'model_key': model_key}

    if context.verbose > 1:
        print 'compute_features_compound_EPSP_amp: pid: %i; took %.3f s' % \
              (os.getpid(), time.time() - start_time)

    return result


def filter_features_compound_EPSP_amp(primitives, current_features, export=False):
    """
    :param primitives: list of dict (each dict contains results from a single simulation)
    :param current_features: dict
    :param export: bool
    :return: dict
    """
    features = {}
    model_key = current_features['model_key']
    for this_feature_dict in primitives:
        this_model_key = this_feature_dict['model_key']
        if this_model_key != model_key:
            raise KeyError('filter_features_compound_EPSP_amp: mismatched model keys')

    features['model_key'] = model_key

    return features


def get_expected_compound_EPSP_traces(unitary_traces_dict, syn_id_dict):
    """

    :param unitary_traces_dict: dict
    :param syn_id_dict: dict
    :return: dict of int: array
    """
    traces = {}
    baseline_len = int(context.trace_baseline / context.dt)
    unitary_len = int(context.ISI['units'] / context.dt)
    trace_len = int((context.sim_duration['clustered'] - context.equilibrate) / context.dt) + baseline_len
    for num_syns in syn_id_dict:
        traces[num_syns] = {}
        for count, syn_id in enumerate(syn_id_dict[num_syns]):
            start = baseline_len + int(count * context.ISI['clustered'] / context.dt)
            end = start + unitary_len
            for rec_name, this_trace in unitary_traces_dict[syn_id].iteritems():
                if rec_name not in traces[num_syns]:
                    traces[num_syns][rec_name] = np.zeros(trace_len)
                traces[num_syns][rec_name][start:end] += this_trace[baseline_len:]
    return traces


def get_objectives_synaptic_integration(features, export=False):
    """

    :param features: dict
    :param export: bool
    :return: tuple of dict
    """
    objectives = dict()

    features['mock_feature'] = random.random()
    objectives['mock_objective'] = random.random()

    return features, objectives


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)