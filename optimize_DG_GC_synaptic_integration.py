"""
Uses nested.optimize to tune spatiotemporal integration of AMPA + NMDA mixed EPSPs in dentate granule cell dendrites.

Requires a YAML file to specify required configuration parameters.
Requires use of a nested.parallel interface.
"""
__author__ = 'Aaron D. Milstein and Grace Ng'
from dentate.biophysics_utils import *
from nested.parallel import get_parallel_interface
from nested.optimize_utils import Context, nested_analyze_init_contexts_interactive, merge_exported_data, \
    update_source_contexts
from cell_utils import *
import uuid
import click

context = Context()


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True, ))
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/optimize_DG_GC_synaptic_integration_config.yaml')
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--export", is_flag=True)
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default=None)
@click.option("--verbose", type=int, default=1)
@click.option("--plot", is_flag=True)
@click.option("--interactive", is_flag=True)
@click.option("--debug", is_flag=True)
@click.option("--limited-branches", is_flag=True)
@click.pass_context
def main(cli, config_file_path, output_dir, export, export_file_path, label, verbose, plot, interactive, debug,
         limited_branches):
    """

    :param cli: contains unrecognized args as list of str
    :param config_file_path: str (path)
    :param output_dir: str (path)
    :param export: bool
    :param export_file_path: str
    :param label: str
    :param verbose: bool
    :param plot: bool
    :param interactive: bool
    :param debug: bool
    :param limited_branches: bool
    """
    # requires a global variable context: :class:'Context'
    context.update(locals())
    kwargs = get_unknown_click_arg_dict(cli.args)
    context.disp = verbose > 0

    context.interface = get_parallel_interface(**kwargs)
    context.interface.start(disp=context.disp)
    context.interface.ensure_controller()
    nested_analyze_init_contexts_interactive(context, config_file_path=config_file_path, output_dir=output_dir,
                                             export=export, export_file_path=export_file_path, label=label,
                                             disp=context.disp, verbose=verbose, plot=plot, debug=debug, **kwargs)

    if plot:
        from dentate.plot import plot_synaptic_attribute_distribution
        plot_synaptic_attribute_distribution(context.cell, context.env, context.NMDA_type, 'g_unit',
                                             from_mech_attrs=True, from_target_attrs=True, show=True)
        plot_synaptic_attribute_distribution(context.cell, context.env, context.NMDA_type, 'weight',
                                             from_mech_attrs=True, from_target_attrs=True, show=True)
        plot_synaptic_attribute_distribution(context.cell, context.env, context.AMPA_type, 'g_unit',
                                             from_mech_attrs=True, from_target_attrs=True, show=True)
        plot_synaptic_attribute_distribution(context.cell, context.env, context.AMPA_type, 'weight',
                                             from_mech_attrs=True, from_target_attrs=True, show=True)

    if not debug:
        run_tests()

    if not interactive:
        context.interface.stop()


def run_tests():
    model_id = 0

    features = dict()

    # Stage 0:
    args = context.interface.execute(get_args_static_unitary_EPSP_amp)
    group_size = len(args[0])
    sequences = [[context.x0_array] * group_size] + args + [[model_id] * group_size] + \
                [[context.export] * group_size] + [[context.plot] * group_size]
    primitives = context.interface.map(compute_features_unitary_EPSP_amp, *sequences)
    this_features = {key: value for feature_dict in primitives for key, value in feature_dict.items()}
    features.update(this_features)
    context.update(locals())
    context.interface.apply(export_unitary_EPSP_traces)

    # Stage 1:
    args = context.interface.execute(get_args_static_compound_EPSP_amp)
    group_size = len(args[0])
    sequences = [[context.x0_array] * group_size] + args + [[model_id] * group_size] + \
                [[context.export] * group_size] + [[context.plot] * group_size]
    primitives = context.interface.map(compute_features_compound_EPSP_amp, *sequences)
    this_features = {key: value for feature_dict in primitives for key, value in feature_dict.items()}
    features.update(this_features)
    context.update(locals())
    context.interface.apply(export_compound_EPSP_traces)
    
    # Stage 2:
    args = context.interface.execute(get_args_static_EPSP_train)
    group_size = len(args[0])
    sequences = [[context.x0_array] * group_size] + args + [[model_id] * group_size] + \
                [[context.export] * group_size] + [[context.plot] * group_size]
    primitives = context.interface.map(compute_features_EPSP_train, *sequences)
    this_features = {key: value for feature_dict in primitives for key, value in feature_dict.items()}
    features.update(this_features)
    context.update(locals())
    context.interface.apply(export_EPSP_train_traces)
    
    features, objectives = context.interface.execute(get_objectives_synaptic_integration, features, model_id,
                                                     context.export)
    
    if 'model_key' in context() and context.model_key is not None:
        model_label = context.model_key
    else:
        model_label = 'x0'

    if context.export:
        legend = {'model_labels': [model_label], 'export_keys': ['0'],
                  'source': context.config_file_path}
        merge_exported_data(context, export_file_path=context.export_file_path,
                            output_dir=context.output_dir, legend=legend, verbose=context.disp)

    sys.stdout.flush()
    print('model_id: %i; model_labels: %s' % (model_id, model_label))
    print('params:')
    print_param_dict_like_yaml(context.x0_dict)
    print('features:')
    print_param_dict_like_yaml(features)
    print('objectives:')
    print_param_dict_like_yaml(objectives)
    sys.stdout.flush()
    time.sleep(.1)

    if context.plot:
        context.interface.show()

    context.update(locals())


def config_worker():
    """

    """
    if 'plot' not in context():
        context.plot = False
    if 'debug' not in context():
        context.debug = False
    if 'limited_branches' not in context():
        context.limited_branches = False
    if 'verbose' in context():
        context.verbose = int(context.verbose)
    context.temp_model_data = dict()
    context.temp_model_data_file_path = None
    if not context_has_sim_env(context):
        build_sim_env(context, **context.kwargs)
    else:
        config_sim_env(context)


def context_has_sim_env(context):
    """

    :param context: :class:'Context
    :return: bool
    """
    return 'env' in context() and 'sim' in context() and 'cell' in context()


def init_context():
    """

    """
    dt = 0.025
    v_init = -77.
    v_active = -77.

    local_random = random.Random()

    syn_conditions = ['control', 'AP5']
    min_random_inter_syn_distance = 30.  # um

    # number of branches to test temporal integration of clustered inputs
    if 'limited_branches' in context() and context.limited_branches:
        max_syns_per_random_branch = 1
        num_clustered_branches = 1
        num_syns_per_clustered_branch = 10
    else:
        max_syns_per_random_branch = 5
        num_clustered_branches = 2
        num_syns_per_clustered_branch = 30
    min_expected_compound_EPSP_amp, max_expected_compound_EPSP_amp = 8., 12.  # mV
    clustered_branch_names = ['clustered%i' % i for i in range(num_clustered_branches)]
    unitary_group_names = ['random'] + clustered_branch_names
    
    num_syns_train = 20
    num_pulses_train = 5
    EPSP_train_locations = ['proximal', 'distal']

    ISI = {'units': 200., 'clustered': 1.1, 'train': 20.}  # inter-stimulus interval for synaptic stim (ms)
    units_per_sim = 5
    equilibrate = 250.  # time to steady-state
    stim_dur = 150.
    sim_duration = {'units': equilibrate + units_per_sim * ISI['units'],
                    'clustered': equilibrate + ISI['units'] + num_syns_per_clustered_branch * ISI['clustered'],
                    'default': equilibrate + stim_dur,
                    'train': equilibrate + ISI['units'] + (num_pulses_train - 1) * ISI['train']}
    trace_baseline = 10.
    duration = max(sim_duration.values())

    AMPA_type = 'AMPA'
    NMDA_type = 'NMDA'
    syn_mech_names = [AMPA_type, NMDA_type]

    context.update(locals())


def build_sim_env(context, verbose=2, cvode=True, daspk=True, load_edges=True, set_edge_delays=False, **kwargs):
    """

    :param context: :class:'Context'
    :param verbose: int
    :param cvode: bool
    :param daspk: bool
    """
    verbose = int(verbose)
    init_context()
    context.env = Env(comm=context.comm, verbose=verbose > 1, **kwargs)
    configure_hoc_env(context.env)
    cell = get_biophys_cell(context.env, gid=int(context.gid), pop_name=context.cell_type, load_edges=load_edges,
                            set_edge_delays=set_edge_delays, mech_file_path=context.mech_file_path)
    init_biophysics(cell, reset_cable=True, correct_cm=context.correct_for_spines,
                    correct_g_pas=context.correct_for_spines, env=context.env, verbose=verbose > 1)
    context.sim = QuickSim(context.duration, cvode=cvode, daspk=daspk, dt=context.dt, verbose=verbose > 1)
    context.spike_output_vec = h.Vector()
    cell.spike_detector.record(context.spike_output_vec)
    context.cell = cell
    config_sim_env(context)


def config_sim_env(context):
    """

    :param context: :class:'Context'
    """
    if 'previous_module' in context() and context.previous_module == __file__:
        return
    init_context()
    if 'i_holding' not in context():
        context.i_holding = defaultdict(dict)

    cell = context.cell
    env = context.env
    sim = context.sim
    if not sim.has_rec('soma'):
        sim.append_rec(cell, cell.tree.root, name='soma', loc=0.5)
    if context.v_active not in context.i_holding['soma']:
        context.i_holding['soma'][context.v_active] = 0.
    if not sim.has_rec('dend'):
        dend, dend_loc = get_thickest_dend_branch(context.cell, 100., terminal=False)
        sim.append_rec(cell, dend, name='dend', loc=dend_loc)
    if not sim.has_rec('dend_local'):
        dend = sim.get_rec('dend')['node']
        sim.append_rec(cell, dend, name='dend_local', loc=0.5)
    if 'synaptic_integration_rec_names' not in context():
        context.synaptic_integration_rec_names = ['soma', 'dend', 'dend_local']
    
    duration = context.duration

    if not sim.has_stim('holding'):
        sim.append_stim(cell, cell.tree.root, name='holding', loc=0.5, amp=0., delay=0., dur=duration)
        offset_vm('soma', context, vm_target=context.v_active, i_history=context.i_holding)

    if 'syn_id_dict' not in context():
        init_syn_mech_attrs(cell, context.env)
        context.local_random.seed(int(float(context.seed_offset)) + int(context.gid))
        syn_attrs = env.synapse_attributes
        syn_id_dict = defaultdict(list)
        # choose a random subset of synapses across all apical branches for tuning a distance-dependent AMPA-R gradient
        for branch in cell.apical:
            this_syn_ids = syn_attrs.get_filtered_syn_ids(cell.gid, syn_sections=[branch.index],
                                                          syn_types=[env.Synapse_Types['excitatory']])
            if len(this_syn_ids) > 1:
                if branch.sec.L <= context.min_random_inter_syn_distance:
                    syn_id_dict['random'].extend(context.local_random.sample(this_syn_ids, 1))
                else:
                    this_num_syns = min(len(this_syn_ids),
                                        int(branch.sec.L / context.min_random_inter_syn_distance),
                                        context.max_syns_per_random_branch)
                    syn_id_dict['random'].extend(context.local_random.sample(this_syn_ids, this_num_syns))
            elif len(this_syn_ids) > 0:
                syn_id_dict['random'].append(this_syn_ids[0])

        # choose a random subset of apical branches that contain the required number of clustered synapses to tune
        # NMDA-R properties to match target features for spatiotemporal integration
        candidate_branches = [branch for branch in cell.apical if
                              50. < get_distance_to_node(cell, cell.tree.root, branch) < 150. and
                              90. < branch.sec.L < 120.]
        if len(candidate_branches) < context.num_clustered_branches:
            candidate_branches = [branch for branch in cell.apical if
                                  50. < get_distance_to_node(cell, cell.tree.root, branch) < 150. and
                                  branch.sec.L > 90.]
        # last resort for outlier morphologies
        if len(candidate_branches) < context.num_clustered_branches:
            candidate_branches = [branch for branch in cell.apical if
                                  50. < get_distance_to_node(cell, cell.tree.root, branch) < 200. and
                                  branch.sec.L > 90.]

        context.local_random.shuffle(candidate_branches)

        candidate_clustered_branches = []
        candidate_syn_id_dict = dict()
        for branch in candidate_branches:
            this_syn_ids = syn_attrs.get_filtered_syn_ids(cell.gid, syn_sections=[branch.index],
                                                          syn_types=[env.Synapse_Types['excitatory']])
            candidate_syn_ids = []
            for syn_id in this_syn_ids:
                syn_loc = syn_attrs.syn_id_attr_dict[cell.gid][syn_id].syn_loc
                if 30. <= syn_loc * branch.sec.L <= 60.:
                    candidate_syn_ids.append(syn_id)
            if len(candidate_syn_ids) >= context.num_syns_per_clustered_branch:
                candidate_clustered_branches.append(branch)
                candidate_syn_id_dict[branch] = context.local_random.sample(candidate_syn_ids,
                                                                            context.num_syns_per_clustered_branch)

        parents = set()
        candidate_clustered_branches_from_separate_subtrees = []
        for branch in candidate_branches:
            if branch.parent not in parents:
                candidate_clustered_branches_from_separate_subtrees.append(branch)
                parents.add(branch.parent)

        if len(candidate_clustered_branches) < context.num_clustered_branches:
            raise RuntimeError('optimize_DG_GC_synaptic_integration: problem finding required number of branches that'
                               'satisfy the requirement for clustered synapses: %i/%i' %
                               (len(candidate_clustered_branches), context.num_clustered_branches))
        elif len(candidate_clustered_branches_from_separate_subtrees) >= context.num_clustered_branches:
            candidate_clustered_branches = candidate_clustered_branches_from_separate_subtrees

        for i, branch in enumerate(candidate_clustered_branches[:context.num_clustered_branches]):
            branch_key = context.clustered_branch_names[i]
            syn_id_dict[branch_key].extend(candidate_syn_id_dict[branch])
        
        
        # choose a random subset of distal apical synapses to tune NMDA-R properties to match target features for
        # temporal integration
        distal_synapses = []
        for syn_id in syn_id_dict['random']:
            syn = syn_attrs.syn_id_attr_dict[context.cell.gid][syn_id]
            node_index = syn.syn_section
            node = context.cell.tree.get_node_with_index(node_index)
            syn_loc = syn.syn_loc
            this_distance = get_distance_to_node(context.cell, context.cell.tree.root, node, loc=syn_loc)
            if this_distance > 200.:
                distal_synapses.append(syn_id)
        # last resort for outlier morphologies
        if len(distal_synapses) < context.num_syns_train:
            distal_synapses = []
            for syn_id in syn_id_dict['random']:
                syn = syn_attrs.syn_id_attr_dict[context.cell.gid][syn_id]
                node_index = syn.syn_section
                node = context.cell.tree.get_node_with_index(node_index)
                syn_loc = syn.syn_loc
                this_distance = get_distance_to_node(context.cell, context.cell.tree.root, node, loc=syn_loc)
                if this_distance > 150.:
                    distal_synapses.append(syn_id)
        context.local_random.shuffle(distal_synapses)
        syn_id_dict['distal'].extend(distal_synapses[:context.num_syns_train])
        
        proximal_synapses = []
        for syn_id in syn_id_dict['random']:
            syn = syn_attrs.syn_id_attr_dict[context.cell.gid][syn_id]
            node_index = syn.syn_section
            node = context.cell.tree.get_node_with_index(node_index)
            syn_loc = syn.syn_loc
            this_distance = get_distance_to_node(context.cell, context.cell.tree.root, node, loc=syn_loc)
            if this_distance < 200.:
                proximal_synapses.append(syn_id)
        # last resort for outlier morphologies
        if len(proximal_synapses) < context.num_syns_train:
            proximal_synapses = []
            for syn_id in syn_id_dict['random']:
                syn = syn_attrs.syn_id_attr_dict[context.cell.gid][syn_id]
                node_index = syn.syn_section
                node = context.cell.tree.get_node_with_index(node_index)
                syn_loc = syn.syn_loc
                this_distance = get_distance_to_node(context.cell, context.cell.tree.root, node, loc=syn_loc)
                if this_distance > 150.:
                    proximal_synapses.append(syn_id)
        context.local_random.shuffle(proximal_synapses)
        syn_id_dict['proximal'].extend(proximal_synapses[:context.num_syns_train])
        
        context.syn_id_dict = syn_id_dict
        unitary_syn_id_set = set()
        train_syn_id_set = set()
        syn_id_set = set()
        
        for group_key in syn_id_dict:
            syn_id_set.update(context.syn_id_dict[group_key])
            if group_key in context.unitary_group_names:
                unitary_syn_id_set.update(context.syn_id_dict[group_key])
            elif group_key in context.EPSP_train_locations:
                train_syn_id_set.update(context.syn_id_dict[group_key])
        context.syn_id_list = list(syn_id_set)
        context.unitary_syn_id_list = list(unitary_syn_id_set)
        context.train_syn_id_list = list(train_syn_id_set)

        for syn_id in context.syn_id_list:
            syn_attrs.modify_mech_attrs(context.cell.pop_name, context.cell.gid, syn_id, 'AMPA',
                                        params={'weight': 1.})
            syn_attrs.modify_mech_attrs(context.cell.pop_name, context.cell.gid, syn_id, 'NMDA',
                                        params={'weight': 1.})

        config_biophys_cell_syns(env=context.env, gid=context.cell.gid, postsyn_name=context.cell.pop_name,
                                 syn_ids=context.syn_id_list, insert=True, insert_netcons=True, insert_vecstims=True,
                                 verbose=context.verbose > 1, throw_error=False)

    context.previous_module = __file__


def update_syn_mechanisms(x, context=None):
    """

    :param x: array
    :param context: :class:'Context'
    """
    if context is None:
        raise RuntimeError('update_syn_mechanisms: missing required Context object')
    x_dict = param_array_to_dict(x, context.param_names)
    cell = context.cell
    env = context.env
    modify_syn_param(cell, env, 'apical', context.AMPA_type, param_name='g_unit', value=x_dict['AMPA.g0'],
                     filters={'syn_types': ['excitatory']}, origin='soma', slope=x_dict['AMPA.slope'],
                     tau=x_dict['AMPA.tau'], update_targets=False)
    modify_syn_param(cell, env, 'apical', context.AMPA_type, param_name='g_unit',
                     filters={'syn_types': ['excitatory']}, origin='parent',
                     origin_filters={'syn_types': ['excitatory']},
                     custom={'func': 'custom_filter_if_terminal'}, update_targets=False, append=True)
    modify_syn_param(cell, env, 'apical', context.AMPA_type, param_name='g_unit',
                     filters={'syn_types': ['excitatory'], 'layers': ['OML']}, origin='apical',
                     origin_filters={'syn_types': ['excitatory'], 'layers': ['MML']}, update_targets=False, append=True)
    modify_syn_param(cell, env, 'apical', context.NMDA_type, filters={'syn_types': ['excitatory']},
                     param_name='Kd', value=x_dict['NMDA.Kd'], update_targets=False)
    modify_syn_param(cell, env, 'apical', context.NMDA_type, filters={'syn_types': ['excitatory']},
                     param_name='gamma', value=x_dict['NMDA.gamma'], update_targets=False)
    modify_syn_param(cell, env, 'apical', context.NMDA_type, filters={'syn_types': ['excitatory']},
                     param_name='g_unit', value=x_dict['NMDA.g_unit'], update_targets=False)
    modify_syn_param(cell, env, 'apical', context.NMDA_type, filters={'syn_types': ['excitatory']},
                     param_name='vshift', value=x_dict['NMDA.vshift'], update_targets=False)
    modify_syn_param(cell, env, 'apical', context.NMDA_type, filters={'syn_types': ['excitatory']},
                     param_name='sat', value=x_dict['NMDA.sat'], update_targets=False)
    modify_syn_param(cell, env, 'apical', context.NMDA_type, filters={'syn_types': ['excitatory']},
                     param_name='f_inc', value=x_dict['NMDA.f_inc'], update_targets=False)
    modify_syn_param(cell, env, 'apical', context.NMDA_type, filters={'syn_types': ['excitatory']},
                     param_name='f_max', value=x_dict['NMDA.f_max'], update_targets=False)
    config_biophys_cell_syns(env=env, gid=cell.gid, postsyn_name=cell.pop_name, syn_ids=context.syn_id_list,
                             verbose=context.verbose > 1, throw_error=True)


def shutdown_worker():
    """

    """
    try:
        if context.temp_model_data_file is not None:
            context.temp_model_data_file.close()
        time.sleep(2.)
        if context.interface.global_comm.rank == 0:
            os.remove(context.temp_model_data_file_path)
    except Exception:
        pass


def consolidate_unitary_EPSP_traces(source_dict):
    """
    Consolidate data structures, converting 1D arrays of simulation data into 2D arrays containing data from stimulation
    of different syn_ids. Maintain organization by syn_group, syn_condition, and recording location.
    :param source_dict: nested dict
    :return: nested dict
    """
    trace_len = int((context.ISI['units'] + context.trace_baseline) / context.dt)
    target_dict = {}

    for syn_group in source_dict:
        if syn_group not in target_dict:
            target_dict[syn_group] = {}
        num_syn_ids = len(context.syn_id_dict[syn_group])
        for syn_condition in source_dict[syn_group]:
            if syn_condition not in target_dict[syn_group]:
                target_dict[syn_group][syn_condition] = {}
            for rec_name in context.synaptic_integration_rec_names:
                target_array = np.empty((num_syn_ids, trace_len))
                for i, syn_id in enumerate(context.syn_id_dict[syn_group]):
                    target_array[i,:] = source_dict[syn_group][syn_condition][syn_id][rec_name]
                target_dict[syn_group][syn_condition][rec_name] = target_array

    return target_dict


def consolidate_compound_EPSP_traces(source_dict):
    """
    Consolidate data structures, converting 1D arrays of simulation data into 2D arrays containing data from stimulation
    of different syn_ids. Maintain organization by syn_group, syn_condition, and recording location.
    :param source_dict: nested dict
    :return: nested dict
    """
    trace_len = int((context.sim_duration['clustered'] - context.equilibrate + context.trace_baseline) / context.dt)
    target_dict = {}

    for syn_group in source_dict:
        if syn_group not in target_dict:
            target_dict[syn_group] = {}
        num_syn_ids = len(context.syn_id_dict[syn_group])
        for syn_condition in source_dict[syn_group]:
            if syn_condition not in target_dict[syn_group]:
                target_dict[syn_group][syn_condition] = {}
            for rec_name in context.synaptic_integration_rec_names:
                target_array = np.empty((num_syn_ids, trace_len))
                for i in range(num_syn_ids):
                    num_syns = i + 1
                    target_array[i,:] = source_dict[syn_group][syn_condition][num_syns][rec_name]
                target_dict[syn_group][syn_condition][rec_name] = target_array

    return target_dict


def export_unitary_EPSP_traces():
    """
    Data from model simulations is temporarily stored locally on each worker. This method uses collective operations to
    export the data to disk, with one hdf5 group per model.
    Must be called via the synchronize method of nested.parallel.ParallelContextInterface.
    """
    start_time = time.time()
    description = 'unitary_EPSP_traces'
    trace_len = int((context.ISI['units'] + context.trace_baseline) / context.dt)
    context.temp_model_data_legend = dict()

    model_keys = list(context.temp_model_data.keys())
    model_keys = context.interface.global_comm.gather(model_keys, root=0)
    if context.interface.global_comm.rank == 0:
        model_keys = list(set([key for key_list in model_keys for key in key_list]))
    else:
        model_keys = None
    model_keys = context.interface.global_comm.bcast(model_keys, root=0)

    if context.temp_model_data_file_path is None:
        if context.interface.global_comm.rank == 0:
            context.temp_model_data_file_path = '%s/%s_uuid%i_%s_temp_model_data.hdf5' % \
                                                (context.output_dir,
                                                 datetime.datetime.today().strftime('%Y%m%d_%H%M'),
                                                 uuid.uuid1(),
                                                 context.optimization_title)
        context.temp_model_data_file_path = \
            context.interface.global_comm.bcast(context.temp_model_data_file_path, root=0)
        context.temp_model_data_file = h5py.File(context.temp_model_data_file_path, 'a', driver='mpio',
                                                 comm=context.interface.global_comm)

    for i, model_key in enumerate(model_keys):
        group_key = str(i)
        context.temp_model_data_legend[model_key] = group_key
        if group_key not in context.temp_model_data_file:
            context.temp_model_data_file.create_group(group_key)
        if description not in context.temp_model_data_file[group_key]:
            context.temp_model_data_file[group_key].create_group(description)
            for syn_group in context.unitary_group_names:
                context.temp_model_data_file[group_key][description].create_group(syn_group)
                num_syn_ids = len(context.syn_id_dict[syn_group])
                for syn_condition in context.syn_conditions:
                    context.temp_model_data_file[group_key][description][syn_group].create_group(syn_condition)
                    for rec_name in context.synaptic_integration_rec_names:
                        context.temp_model_data_file[group_key][description][syn_group][
                            syn_condition].create_dataset(rec_name, (num_syn_ids, trace_len), dtype='f8')

        target_rank = i % context.interface.global_comm.size
        if model_key in context.temp_model_data:
            this_temp_model_data = context.temp_model_data.pop(model_key)
        else:
            this_temp_model_data = {}
        this_temp_model_data = context.interface.global_comm.gather(this_temp_model_data, root=target_rank)
        if context.interface.global_comm.rank == target_rank:
            context.temp_model_data[model_key] = {description: {}}
            for element in this_temp_model_data:
                if element:
                    dict_merge(context.temp_model_data[model_key], element)
        context.interface.global_comm.barrier()

    for model_key in context.temp_model_data:
        context.temp_model_data[model_key][description] = \
            consolidate_unitary_EPSP_traces(context.temp_model_data[model_key][description])
        group_key = context.temp_model_data_legend[model_key]
        for syn_group in context.temp_model_data[model_key][description]:
            for syn_condition in context.temp_model_data[model_key][description][syn_group]:
                for rec_name in context.temp_model_data[model_key][description][syn_group][syn_condition]:
                    context.temp_model_data_file[group_key][description][syn_group][syn_condition][
                        rec_name][:,:] = \
                        context.temp_model_data[model_key][description][syn_group][syn_condition][rec_name]

    context.interface.global_comm.barrier()
    context.temp_model_data_file.flush()

    del context.temp_model_data
    context.temp_model_data = dict()

    sys.stdout.flush()
    time.sleep(1.)

    if context.interface.global_comm.rank == 0 and context.disp:
        print('optimize_DG_GC_synaptic_integration: export_unitary_EPSP_traces took %.2f s' %
              (time.time() - start_time))
        sys.stdout.flush()
        time.sleep(1.)


def export_compound_EPSP_traces():
    """
    Data from model simulations is temporarily stored locally on each worker. This method uses collective operations to
    export the data to disk, with one hdf5 group per model.
    Must be called via the synchronize method of nested.parallel.ParallelContextInterface.
    """
    start_time = time.time()
    description = 'compound_EPSP_traces'
    trace_len = int((context.sim_duration['clustered'] - context.equilibrate + context.trace_baseline) / context.dt)

    model_keys = list(context.temp_model_data.keys())
    model_keys = context.interface.global_comm.gather(model_keys, root=0)
    if context.interface.global_comm.rank == 0:
        model_keys = list(set([key for key_list in model_keys for key in key_list]))
    else:
        model_keys = None
    model_keys = context.interface.global_comm.bcast(model_keys, root=0)

    for i, model_key in enumerate(model_keys):
        group_key = context.temp_model_data_legend[model_key]
        if group_key not in context.temp_model_data_file:
            context.temp_model_data_file.create_group(group_key)
        if description not in context.temp_model_data_file[group_key]:
            context.temp_model_data_file[group_key].create_group(description)
            for syn_group in context.clustered_branch_names:
                context.temp_model_data_file[group_key][description].create_group(syn_group)
                num_syn_ids = len(context.syn_id_dict[syn_group])
                for syn_condition in context.syn_conditions:
                    context.temp_model_data_file[group_key][description][syn_group].create_group(syn_condition)
                    for rec_name in context.synaptic_integration_rec_names:
                        context.temp_model_data_file[group_key][description][syn_group][
                            syn_condition].create_dataset(rec_name, (num_syn_ids, trace_len), dtype='f8')

        target_rank = i % context.interface.global_comm.size
        if model_key in context.temp_model_data:
            this_temp_model_data = context.temp_model_data.pop(model_key)
        else:
            this_temp_model_data = {}
        this_temp_model_data = context.interface.global_comm.gather(this_temp_model_data, root=target_rank)
        if context.interface.global_comm.rank == target_rank:
            context.temp_model_data[model_key] = {description: {}}
            for element in this_temp_model_data:
                if element:
                    dict_merge(context.temp_model_data[model_key], element)
        context.interface.global_comm.barrier()

    for model_key in context.temp_model_data:
        context.temp_model_data[model_key][description] = \
            consolidate_compound_EPSP_traces(context.temp_model_data[model_key][description])
        group_key = context.temp_model_data_legend[model_key]
        for syn_group in context.temp_model_data[model_key][description]:
            for syn_condition in context.temp_model_data[model_key][description][syn_group]:
                for rec_name in context.temp_model_data[model_key][description][syn_group][syn_condition]:
                    context.temp_model_data_file[group_key][description][syn_group][syn_condition][
                        rec_name][:, :] = \
                        context.temp_model_data[model_key][description][syn_group][syn_condition][rec_name]

    context.interface.global_comm.barrier()
    context.temp_model_data_file.flush()

    del context.temp_model_data
    context.temp_model_data = dict()

    sys.stdout.flush()
    time.sleep(1.)

    if context.interface.global_comm.rank == 0 and context.disp:
        print('optimize_DG_GC_synaptic_integration: export_compound_EPSP_traces took %.2f s' %
              (time.time() - start_time))
        sys.stdout.flush()
        time.sleep(1.)


def export_EPSP_train_traces():
    """
    Data from model simulations is temporarily stored locally on each worker. This method uses collective operations to
    export the data to disk, with one hdf5 group per model.
    Must be called via the synchronize method of nested.parallel.ParallelContextInterface.
    """
    start_time = time.time()
    description = 'EPSP_train_traces'
    
    trace_len = int((context.sim_duration['train'] - context.equilibrate + context.trace_baseline) / context.dt)
    
    model_keys = list(context.temp_model_data.keys())
    model_keys = context.interface.global_comm.gather(model_keys, root=0)
    if context.interface.global_comm.rank == 0:
        model_keys = list(set([key for key_list in model_keys for key in key_list]))
    else:
        model_keys = None
    model_keys = context.interface.global_comm.bcast(model_keys, root=0)
    
    for i, model_key in enumerate(model_keys):
        if model_key not in context.temp_model_data_legend:
            group_key = str(i)
            context.temp_model_data_legend[model_key] = group_key
        else:
            group_key = context.temp_model_data_legend[model_key]
        if group_key not in context.temp_model_data_file:
            context.temp_model_data_file.create_group(group_key)
        if description not in context.temp_model_data_file[group_key]:
            context.temp_model_data_file[group_key].create_group(description)
            for syn_group in context.EPSP_train_locations:
                context.temp_model_data_file[group_key][description].create_group(syn_group)
                for syn_condition in context.syn_conditions:
                    context.temp_model_data_file[group_key][description][syn_group].create_group(syn_condition)
                    for rec_name in context.synaptic_integration_rec_names:
                        context.temp_model_data_file[group_key][description][syn_group][
                            syn_condition].create_dataset(rec_name, (trace_len,), dtype='f8')
        
        target_rank = i % context.interface.global_comm.size
        if model_key in context.temp_model_data:
            this_temp_model_data = context.temp_model_data.pop(model_key)
        else:
            this_temp_model_data = {}
        this_temp_model_data = context.interface.global_comm.gather(this_temp_model_data, root=target_rank)
        if context.interface.global_comm.rank == target_rank:
            context.temp_model_data[model_key] = {description: {}}
            for element in this_temp_model_data:
                if element:
                    dict_merge(context.temp_model_data[model_key], element)
        context.interface.global_comm.barrier()
    
    for model_key in context.temp_model_data:
        # context.temp_model_data[model_key][description] = \
        #     consolidate_compound_EPSP_traces(context.temp_model_data[model_key][description])
        group_key = context.temp_model_data_legend[model_key]
        for syn_group in context.temp_model_data[model_key][description]:
            for syn_condition in context.temp_model_data[model_key][description][syn_group]:
                for rec_name in context.temp_model_data[model_key][description][syn_group][syn_condition]:
                    context.temp_model_data_file[group_key][description][syn_group][syn_condition][
                        rec_name][:] = \
                        context.temp_model_data[model_key][description][syn_group][syn_condition][rec_name]
    
    context.interface.global_comm.barrier()
    context.temp_model_data_file.flush()
    
    del context.temp_model_data
    context.temp_model_data = dict()
    
    sys.stdout.flush()
    time.sleep(1.)
    
    if context.interface.global_comm.rank == 0 and context.disp:
        print('optimize_DG_GC_synaptic_integration: export_EPSP_train_traces took %.2f s' %
              (time.time() - start_time))
        sys.stdout.flush()
        time.sleep(1.)


def get_args_static_unitary_EPSP_amp():
    """
    A nested map operation is required to compute unitary EPSP amplitude features. The arguments to be mapped are the
    same (static) for each set of parameters.
    :return: list of list
    """
    syn_group_list = []
    syn_id_lists = []
    syn_condition_list = []

    for syn_group in context.unitary_group_names:
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

    return [syn_id_lists, syn_condition_list, syn_group_list]


def compute_features_unitary_EPSP_amp(x, syn_ids, syn_condition, syn_group, model_key, export=False, plot=False):
    """

    :param x: array
    :param syn_ids: list of int
    :param syn_condition: str
    :param syn_group: str
    :param model_key: int or str
    :param export: bool
    :param plot: bool
    :return: dict
    """
    start_time = time.time()
    config_sim_env(context)
    update_source_contexts(x, context)
    # zero_na(context.cell)

    dt = context.dt
    duration = context.sim_duration['units']
    ISI = context.ISI['units']
    equilibrate = context.equilibrate
    trace_baseline = context.trace_baseline

    rec_dict = context.sim.get_rec('soma')
    node = rec_dict['node']
    loc = rec_dict['loc']

    sim = context.sim
    sim.backup_state()
    sim.set_state(dt=dt, tstop=duration, cvode=False)  # cvode=True)

    sim.modify_stim('holding', node=node, loc=loc, amp=context.i_holding['soma'][context.v_active], dur=duration)

    syn_attrs = context.env.synapse_attributes
    sim.parameters = dict()
    sim.parameters['duration'] = duration
    sim.parameters['equilibrate'] = equilibrate
    sim.parameters['syn_secs'] = []
    sim.parameters['swc_types'] = []
    sim.parameters['syn_ids'] = syn_ids
    for i, syn_id in enumerate(syn_ids):
        syn_id = int(syn_id)
        spike_time = context.equilibrate + i * ISI
        for syn_name in context.syn_mech_names:
            this_nc = syn_attrs.get_netcon(context.cell.gid, syn_id, syn_name)
            this_nc.delay = 0.
            this_nc.pre().play(h.Vector([spike_time]))
            if syn_name == context.NMDA_type and syn_condition == 'AP5':
                config_syn(syn_name=syn_name, rules=syn_attrs.syn_param_rules, mech_names=syn_attrs.syn_mech_names,
                           nc=this_nc, syn=this_nc.syn(), weight=0.)  # g_unit=0.)
        syn = syn_attrs.syn_id_attr_dict[context.cell.gid][syn_id]
        node_index = syn.syn_section
        node_type = syn.swc_type
        sim.parameters['syn_secs'].append(node_index)
        sim.parameters['swc_types'].append(node_type)
        if i == 0:
            branch = context.cell.tree.get_node_with_index(node_index)
            context.sim.modify_rec('dend_local', node=branch)

    sim.run(context.v_active)

    traces_dict = defaultdict(dict)
    for i, syn_id in enumerate(syn_ids):
        start = int((equilibrate + i * ISI) / dt)
        end = start + int(ISI / dt)
        trace_start = start - int(trace_baseline / dt)
        baseline_start, baseline_end = int(start - 3. / dt), int(start - 1. / dt)
        syn_id = int(syn_id)
        for rec_name in context.synaptic_integration_rec_names:
            this_vm = np.array(context.sim.recs[rec_name]['vec'].to_python())
            baseline = np.mean(this_vm[baseline_start:baseline_end])
            this_vm = this_vm[trace_start:end] - baseline
            peak_index = np.argmax(this_vm)
            zero_index = np.where(this_vm[peak_index:] <= 0.)[0]
            if np.any(zero_index):
                this_vm[peak_index+zero_index[0]:] = 0.
            traces_dict[syn_id][rec_name] = np.array(this_vm)
        for syn_name in context.syn_mech_names:
            this_nc = syn_attrs.get_netcon(context.cell.gid, syn_id, syn_name)
            this_nc.pre().play(h.Vector())

    new_model_data = {model_key: {'unitary_EPSP_traces': {syn_group: {syn_condition: traces_dict}}}}

    dict_merge(context.temp_model_data, new_model_data)

    result = {'model_key': model_key}

    title = 'unitary_EPSP_amp'
    description = 'condition: %s, group: %s, num_syns: %i, first syn_id: %i' % \
                  (syn_condition, syn_group, len(syn_ids), syn_ids[0])
    sim.parameters['title'] = title
    sim.parameters['description'] = description

    if context.verbose > 0:
        print('compute_features_unitary_EPSP_amp: pid: %i; model_id: %s; %s: %s took %.3f s' %
              (os.getpid(), model_key, title, description, time.time() - start_time))
        sys.stdout.flush()
    if context.plot:
        context.sim.plot()

    if export:
        context.sim.export_to_file(context.temp_output_path, model_label=model_key, category=title)

    sim.restore_state()

    return result


def get_args_static_compound_EPSP_amp():
    """
    A nested map operation is required to compute compound EPSP amplitude features. The arguments to be mapped are the
    same (static) for each set of parameters.
    :return: list of list
    """
    syn_group_list = []
    syn_id_lists = []
    syn_condition_list = []
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

    return [syn_id_lists, syn_condition_list, syn_group_list]


def compute_features_compound_EPSP_amp(x, syn_ids, syn_condition, syn_group, model_key, export=False, plot=False):
    """

    :param x: array
    :param syn_ids: list of int
    :param syn_condition: str
    :param syn_group: str
    :param model_key: int or str
    :param export: bool
    :param plot: bool
    :return: dict
    """
    start_time = time.time()
    config_sim_env(context)
    update_source_contexts(x, context)
    # zero_na(context.cell)

    dt = context.dt
    duration = context.sim_duration['clustered']
    ISI = context.ISI['clustered']
    equilibrate = context.equilibrate
    trace_baseline = context.trace_baseline

    rec_dict = context.sim.get_rec('soma')
    node = rec_dict['node']
    loc = rec_dict['loc']

    sim = context.sim
    sim.backup_state()
    sim.set_state(dt=dt, tstop=duration, cvode=False)  # cvode=True)

    sim.modify_stim('holding', node=node, loc=loc, amp=context.i_holding['soma'][context.v_active], dur=duration)

    syn_attrs = context.env.synapse_attributes
    sim.parameters = dict()
    sim.parameters['duration'] = duration
    sim.parameters['equilibrate'] = equilibrate
    sim.parameters['syn_secs'] = []
    sim.parameters['swc_types'] = []
    sim.parameters['syn_ids'] = syn_ids
    for i, syn_id in enumerate(syn_ids):
        spike_time = context.equilibrate + i * ISI
        for syn_name in context.syn_mech_names:
            this_nc = syn_attrs.get_netcon(context.cell.gid, syn_id, syn_name)
            this_nc.delay = 0.
            this_nc.pre().play(h.Vector([spike_time]))
            if syn_name == context.NMDA_type and syn_condition == 'AP5':
                config_syn(syn_name=syn_name, rules=syn_attrs.syn_param_rules, mech_names=syn_attrs.syn_mech_names,
                           nc=this_nc, syn=this_nc.syn(), weight=0.)  # g_unit=0.)
        syn = syn_attrs.syn_id_attr_dict[context.cell.gid][syn_id]
        node_index = syn.syn_section
        node_type = syn.swc_type
        sim.parameters['syn_secs'].append(node_index)
        sim.parameters['swc_types'].append(node_type)
        if i == 0:
            branch = context.cell.tree.get_node_with_index(node_index)
            context.sim.modify_rec('dend_local', node=branch)

    sim.run(context.v_active)

    traces_dict = {}
    start = int(equilibrate / dt)
    trace_start = start - int(trace_baseline / dt)
    baseline_start, baseline_end = int(start - 3. / dt), int(start - 1. / dt)
    for rec_name in context.synaptic_integration_rec_names:
        this_vm = np.array(context.sim.recs[rec_name]['vec'].to_python())
        baseline = np.mean(this_vm[baseline_start:baseline_end])
        this_vm = this_vm[trace_start:] - baseline
        traces_dict[rec_name] = np.array(this_vm)
    for syn_id in syn_ids:
        for syn_name in context.syn_mech_names:
            this_nc = syn_attrs.get_netcon(context.cell.gid, syn_id, syn_name)
            this_nc.pre().play(h.Vector())

    num_syns = len(syn_ids)
    new_model_data = {model_key: {'compound_EPSP_traces': {syn_group: {syn_condition: {num_syns: traces_dict}}}}}

    dict_merge(context.temp_model_data, new_model_data)

    result = {'model_key': model_key}

    spike_times = np.array(context.cell.spike_detector.get_recordvec().to_python())
    if np.any(spike_times > equilibrate):
        result['soma_spikes'] = True

    title = 'compound_EPSP_amp'
    description = 'condition: %s, group: %s, num_syns: %i, first syn_id: %i' % \
                  (syn_condition, syn_group, len(syn_ids), syn_ids[0])
    sim.parameters['title'] = title
    sim.parameters['description'] = description

    if context.verbose > 0:
        print('compute_features_compound_EPSP_amp: pid: %i; model_id: %s; %s: %s took %.3f s' %
              (os.getpid(), model_key, title, description, time.time() - start_time))
        sys.stdout.flush()
    if plot:
        context.sim.plot()
    if export:
        context.sim.export_to_file(context.temp_output_path, model_label=model_key, category=title)
    sim.restore_state()

    return result


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
    for i in range(len(syn_id_dict)):
        num_syns = i + 1
        traces[num_syns] = {}
        for count, syn_id in enumerate(syn_id_dict[:num_syns]):
            start = baseline_len + int(count * context.ISI['clustered'] / context.dt)
            end = start + unitary_len
            for rec_name, this_trace in unitary_traces_dict[syn_id].items():
                if rec_name not in traces[num_syns]:
                    traces[num_syns][rec_name] = np.zeros(trace_len)
                traces[num_syns][rec_name][start:end] += this_trace[baseline_len:]
    return traces


def get_args_static_EPSP_train():
    """
    A nested map operation is required to compute EPSP train features. The arguments to be mapped are the
    same (static) for each set of parameters.
    :return: list of list
    """
    syn_group_list = []
    syn_id_lists = []
    syn_condition_list = []
    
    for syn_group in ['proximal', 'distal']:
        this_syn_id_group = context.syn_id_dict[syn_group]
        for syn_condition in context.syn_conditions:
            syn_id_lists.append(this_syn_id_group)
            syn_group_list.append(syn_group)
            syn_condition_list.append(syn_condition)

    return [syn_id_lists, syn_condition_list, syn_group_list]


def compute_features_EPSP_train(x, syn_ids, syn_condition, syn_group, model_key, export=False, plot=False):
    """

    :param x: array
    :param syn_ids: list of int
    :param syn_condition: str
    :param syn_group: str
    :param model_key: int or str
    :param export: bool
    :param plot: bool
    :return: dict
    """
    start_time = time.time()
    config_sim_env(context)
    update_source_contexts(x, context)
    # zero_na(context.cell)
    
    dt = context.dt
    duration = context.sim_duration['train']
    ISI = context.ISI['train']
    equilibrate = context.equilibrate
    trace_baseline = context.trace_baseline
    
    rec_dict = context.sim.get_rec('soma')
    node = rec_dict['node']
    loc = rec_dict['loc']
    
    sim = context.sim
    sim.backup_state()
    sim.set_state(dt=dt, tstop=duration, cvode=False)  # cvode=True)
    
    sim.modify_stim('holding', node=node, loc=loc, amp=context.i_holding['soma'][context.v_active], dur=duration)
    
    syn_attrs = context.env.synapse_attributes
    sim.parameters = dict()
    sim.parameters['duration'] = duration
    sim.parameters['equilibrate'] = equilibrate
    sim.parameters['syn_secs'] = []
    sim.parameters['swc_types'] = []
    sim.parameters['syn_ids'] = syn_ids
    for i, syn_id in enumerate(syn_ids):
        spike_times = [context.equilibrate + i * ISI for i in range(context.num_pulses_train)]
        for syn_name in context.syn_mech_names:
            this_nc = syn_attrs.get_netcon(context.cell.gid, syn_id, syn_name)
            this_nc.delay = 0.
            this_nc.pre().play(h.Vector(spike_times))
            if syn_name == context.NMDA_type and syn_condition == 'AP5':
                config_syn(syn_name=syn_name, rules=syn_attrs.syn_param_rules, mech_names=syn_attrs.syn_mech_names,
                           nc=this_nc, syn=this_nc.syn(), weight=0.)  # g_unit=0.)
        syn = syn_attrs.syn_id_attr_dict[context.cell.gid][syn_id]
        node_index = syn.syn_section
        node_type = syn.swc_type
        sim.parameters['syn_secs'].append(node_index)
        sim.parameters['swc_types'].append(node_type)
        if i == 0:
            branch = context.cell.tree.get_node_with_index(node_index)
            context.sim.modify_rec('dend_local', node=branch)
    
    sim.run(context.v_active)
    
    traces_dict = {}
    start = int(equilibrate / dt)
    trace_start = start - int(trace_baseline / dt)
    baseline_start, baseline_end = int(start - 3. / dt), int(start - 1. / dt)
    for rec_name in context.synaptic_integration_rec_names:
        this_vm = np.array(context.sim.recs[rec_name]['vec'].to_python())
        baseline = np.mean(this_vm[baseline_start:baseline_end])
        this_vm = this_vm[trace_start:] - baseline
        traces_dict[rec_name] = np.array(this_vm)
    for syn_id in syn_ids:
        for syn_name in context.syn_mech_names:
            this_nc = syn_attrs.get_netcon(context.cell.gid, syn_id, syn_name)
            this_nc.pre().play(h.Vector())
    
    new_model_data = {model_key: {'EPSP_train_traces': {syn_group: {syn_condition: traces_dict}}}}
    
    dict_merge(context.temp_model_data, new_model_data)
    
    result = {'model_key': model_key}
    
    spike_times = np.array(context.cell.spike_detector.get_recordvec().to_python())
    if np.any(spike_times > equilibrate):
        result['soma_spikes'] = True
    
    title = 'EPSP_train'
    description = 'condition: %s, group: %s, first syn_id: %i' % (syn_condition, syn_group, syn_ids[0])
    sim.parameters['title'] = title
    sim.parameters['description'] = description
    
    if context.verbose > 0:
        print('compute_features_EPSP_train: pid: %i; model_id: %s; %s: %s took %.3f s' %
              (os.getpid(), model_key, title, description, time.time() - start_time))
        sys.stdout.flush()
    if plot:
        context.sim.plot()
    if export:
        context.sim.export_to_file(context.temp_output_path, model_label=model_key, category=title)
    sim.restore_state()
    
    return result


def get_objectives_synaptic_integration(features, model_key, export=False, plot=False):
    """

    :param features: dict
    :param model_key: int or str
    :param export: bool
    :param plot: bool
    :return: tuple of dict
    """
    start_time = time.time()
    objectives = dict()
    failed = False
    
    if 'soma_spikes' in features:
        if not context.debug:
            failed = True
        if context.verbose > 0:
            print('get_objectives_synaptic_integration: pid: %i; model_id: %s; aborting - dendritic spike propagated '
                  'to soma' % (os.getpid(), model_key))
            sys.stdout.flush()

    group_key = context.temp_model_data_legend[model_key]

    unitary_EPSP_traces_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    compound_EPSP_traces_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    EPSP_train_traces_dict = defaultdict(lambda: defaultdict(dict))

    description = 'unitary_EPSP_traces'
    with h5py.File(context.temp_model_data_file_path, 'r') as temp_model_data_file:
        for syn_group in temp_model_data_file[group_key][description]:
            syn_id_list = context.syn_id_dict[syn_group]
            for syn_condition in temp_model_data_file[group_key][description][syn_group]:
                this_group = temp_model_data_file[group_key][description][syn_group][syn_condition]
                for rec_name in context.synaptic_integration_rec_names:
                    for i, syn_id in enumerate(syn_id_list):
                        unitary_EPSP_traces_dict[syn_group][syn_condition][syn_id][rec_name] = this_group[rec_name][i,:]

        description = 'compound_EPSP_traces'
        for syn_group in temp_model_data_file[group_key][description]:
            num_syn_ids = len(context.syn_id_dict[syn_group])
            for syn_condition in temp_model_data_file[group_key][description][syn_group]:
                this_group = temp_model_data_file[group_key][description][syn_group][syn_condition]
                for rec_name in context.synaptic_integration_rec_names:
                    for i in range(num_syn_ids):
                        num_syns = i + 1
                        compound_EPSP_traces_dict[syn_group][syn_condition][num_syns][rec_name] = \
                            this_group[rec_name][i,:]

    syn_id_attr_dict = context.env.synapse_attributes.syn_id_attr_dict[context.cell.gid]
    control_EPSP_amp_list = []
    syn_soma_distance_list = []
    NMDA_contribution_list = []
    for syn_id in context.syn_id_dict['random']:
        control_soma_trace = np.array(unitary_EPSP_traces_dict['random']['control'][syn_id]['soma'][:])
        control_soma_amp = np.max(control_soma_trace)
        control_EPSP_amp_list.append(control_soma_amp)
        syn_section = syn_id_attr_dict[syn_id].syn_section
        syn_loc = syn_id_attr_dict[syn_id].syn_loc
        this_distance = get_distance_to_node(context.cell, context.cell.tree.root,
                                             context.cell.tree.get_node_with_index(syn_section),
                                             loc=syn_loc)
        syn_soma_distance_list.append(this_distance)
        AP5_soma_trace = np.array(unitary_EPSP_traces_dict['random']['AP5'][syn_id]['soma'][:])
        AP5_soma_amp = np.max(AP5_soma_trace)
        NMDA_contribution_list.append((control_soma_amp - AP5_soma_amp) / control_soma_amp)

    mean_unitary_EPSP_amp_residuals = \
        np.sum(((np.array(control_EPSP_amp_list) - context.target_val['mean_unitary_EPSP_amp']) /
                 context.target_range['mean_unitary_EPSP_amp']) ** 2.)
    mean_NMDA_contribution_residuals = \
        np.mean(((np.array(NMDA_contribution_list) - context.target_val['mean_NMDA_contribution']) /
                 context.target_range['mean_NMDA_contribution']) ** 2.)

    features['mean_unitary_EPSP_amp'] = np.mean(control_EPSP_amp_list)
    features['mean_NMDA_contribution'] = np.mean(NMDA_contribution_list)
    objectives['mean_unitary_EPSP_amp_residuals'] = mean_unitary_EPSP_amp_residuals
    objectives['mean_NMDA_contribution_residuals'] = mean_NMDA_contribution_residuals

    for syn_group in compound_EPSP_traces_dict:
        for syn_condition in list(compound_EPSP_traces_dict[syn_group].keys()):
            expected_key = 'expected_' + syn_condition
            compound_EPSP_traces_dict[syn_group][expected_key] = \
                get_expected_compound_EPSP_traces(unitary_EPSP_traces_dict[syn_group][syn_condition],
                                                  context.syn_id_dict[syn_group])

    soma_compound_EPSP_amp = defaultdict(lambda: defaultdict(list))
    initial_gain = defaultdict(list)
    initial_gain_residuals = defaultdict(list)
    integration_gain = defaultdict(list)
    integration_gain_residuals = defaultdict(list)
    for syn_group in compound_EPSP_traces_dict:
        for syn_condition in compound_EPSP_traces_dict[syn_group]:
            max_num_syns = max(compound_EPSP_traces_dict[syn_group][syn_condition].keys())
            for num_syns in range(1, max_num_syns + 1):
                soma_compound_EPSP_amp[syn_group][syn_condition].append(
                    np.max(compound_EPSP_traces_dict[syn_group][syn_condition][num_syns]['soma']))
        for syn_condition in context.syn_conditions:
            expected_key = 'expected_' + syn_condition
            this_actual = np.array(soma_compound_EPSP_amp[syn_group][syn_condition])
            this_expected = np.array(soma_compound_EPSP_amp[syn_group][expected_key])
            indexes = np.where(this_expected <= context.max_expected_compound_EPSP_amp)[0]
            if not np.any(indexes) or \
                    (not context.limited_branches and max(this_expected) < context.min_expected_compound_EPSP_amp) or \
                    (this_expected[1] - this_expected[0] <= 0.):
                failed = True
                if context.verbose > 0:
                    print('optimize_DG_GC_synaptic_integration: get_objectives: pid: %i; model_id: %s; syn_group: %s; '
                          'syn_condition: %s; aborting - expected compound EPSP amplitude below criterion' %
                          (os.getpid(), model_key, syn_group, syn_condition))
                    sys.stdout.flush()
            else:
                # Integration should be close to linear without gain for the first few synapses.
                this_initial_gain = (this_actual[1] - this_actual[0]) / (this_expected[1] - this_expected[0])
                initial_gain[syn_condition].append(this_initial_gain)
                feature_key = 'initial_gain_%s' % syn_condition
                this_initial_gain_residuals = ((this_initial_gain - context.target_val[feature_key]) /
                                               context.target_range[feature_key]) ** 2.
                initial_gain_residuals[syn_condition].append(this_initial_gain_residuals)
                slope, intercept, r_value, p_value, std_err = \
                    stats.linregress(this_expected[indexes], this_actual[indexes])
                integration_gain[syn_condition].append(slope)
                feature_key = 'integration_gain_%s' % syn_condition
                this_target = context.target_val[feature_key] * this_expected[indexes] + intercept
                this_integration_gain_residuals = 0.
                for target_val, actual_val in zip(this_target, this_actual[indexes]):
                    this_integration_gain_residuals += ((actual_val - target_val) /
                                                        context.target_range['mean_unitary_EPSP_amp']) ** 2.
                this_integration_gain_residuals /= float(len(indexes))
                integration_gain_residuals[syn_condition].append(this_integration_gain_residuals)
    
    with h5py.File(context.temp_model_data_file_path, 'r') as temp_model_data_file:
        description = 'EPSP_train_traces'
        for syn_group in temp_model_data_file[group_key][description]:
            for syn_condition in temp_model_data_file[group_key][description][syn_group]:
                this_group = temp_model_data_file[group_key][description][syn_group][syn_condition]
                for rec_name in context.synaptic_integration_rec_names:
                    EPSP_train_traces_dict[syn_group][syn_condition][rec_name] = this_group[rec_name][:]
    
    start_pulse_first = int(context.trace_baseline / context.dt)
    end_pulse_first = start_pulse_first + int(context.ISI['train'] / context.dt)
    start_pulse_last = start_pulse_first + int((context.num_pulses_train - 1) * context.ISI['train'] / context.dt)
    end_pulse_last = start_pulse_last + int(context.ISI['units'] / context.dt)
    soma_compound_EPSP_summation = defaultdict(dict)
    for syn_group in EPSP_train_traces_dict:
        for syn_condition in EPSP_train_traces_dict[syn_group]:
            this_trace = EPSP_train_traces_dict[syn_group][syn_condition]['soma']
            pulse_amp_first = np.max(this_trace[start_pulse_first:end_pulse_first])
            pulse_amp_last = np.max(this_trace[start_pulse_last:end_pulse_last])
            soma_compound_EPSP_summation[syn_group][syn_condition] = 100. * (pulse_amp_last / pulse_amp_first - 1)
    
    if export:
        with h5py.File(context.temp_output_path, 'a') as f:
            description = 'mean_unitary_EPSP_traces'
            group = get_h5py_group(f, [model_key, description], create=True)
            t = np.arange(-context.trace_baseline, context.ISI['units'], context.dt)
            group.create_dataset('time', compression='gzip', data=t)
            data_group = group.create_group('data')
            for syn_group in unitary_EPSP_traces_dict:
                data_group.create_group(syn_group)
                for syn_condition in unitary_EPSP_traces_dict[syn_group]:
                    this_group = data_group[syn_group].create_group(syn_condition)
                    for rec_name in next(iter(unitary_EPSP_traces_dict[syn_group][syn_condition].values())):
                        this_condition_array_list = []
                        for syn_id in unitary_EPSP_traces_dict[syn_group][syn_condition]:
                            this_condition_array_list.append(
                                unitary_EPSP_traces_dict[syn_group][syn_condition][syn_id][rec_name])
                        this_group.create_dataset(rec_name, compression='gzip',
                                                  data=np.mean(this_condition_array_list, axis=0))

            description = 'unitary_EPSP_summary'
            group = get_h5py_group(f, [model_key, description], create=True)
            group.create_dataset('soma_distance', compression='gzip', data=syn_soma_distance_list)
            group.create_dataset('soma_unitary_EPSP_amp', compression='gzip', data=control_EPSP_amp_list)

            description = 'compound_EPSP_summary'
            group = get_h5py_group(f, [model_key, description], create=True)
            t = np.arange(-context.trace_baseline, context.sim_duration['clustered'] - context.equilibrate, context.dt)
            group.create_dataset('time', compression='gzip', data=t)
            data_group = group.create_group('traces')
            for syn_group in compound_EPSP_traces_dict:
                data_group.create_group(syn_group)
                for syn_condition in compound_EPSP_traces_dict[syn_group]:
                    data_group[syn_group].create_group(syn_condition)
                    for num_syns in compound_EPSP_traces_dict[syn_group][syn_condition]:
                        this_group = data_group[syn_group][syn_condition].create_group(str(num_syns))
                        for rec_name in compound_EPSP_traces_dict[syn_group][syn_condition][num_syns]:
                            this_group.create_dataset(rec_name, compression='gzip',
                                        data=compound_EPSP_traces_dict[syn_group][syn_condition][num_syns][rec_name])
            data_group = group.create_group('soma_compound_EPSP_amp')
            for syn_group in soma_compound_EPSP_amp:
                this_group = data_group.create_group(syn_group)
                for syn_condition in soma_compound_EPSP_amp[syn_group]:
                    this_group.create_dataset(syn_condition, compression='gzip',
                                              data=soma_compound_EPSP_amp[syn_group][syn_condition])
            
            description = 'EPSP_train_summary'
            group = get_h5py_group(f, [model_key, description], create=True)
            t = np.arange(-context.trace_baseline, context.sim_duration['train'] - context.equilibrate, context.dt)
            group.create_dataset('time', compression='gzip', data=t)
            data_group = group.create_group('traces')
            for syn_group in EPSP_train_traces_dict:
                data_group.create_group(syn_group)
                for syn_condition in EPSP_train_traces_dict[syn_group]:
                    this_group = data_group[syn_group].create_group(syn_condition)
                    for rec_name in EPSP_train_traces_dict[syn_group][syn_condition]:
                        this_group.create_dataset(
                            rec_name, compression='gzip',
                            data=EPSP_train_traces_dict[syn_group][syn_condition][rec_name])
            data_group = group.create_group('soma_compound_EPSP_summation')
            for syn_group in soma_compound_EPSP_summation:
                this_group = data_group.create_group(syn_group)
                for syn_condition in soma_compound_EPSP_summation[syn_group]:
                    this_group.create_dataset(syn_condition, compression='gzip',
                                              data=[soma_compound_EPSP_summation[syn_group][syn_condition]])

    if context.verbose > 1:
        print('get_objectives_synaptic_integration: pid: %i; model_id: %s; took %.3f s' %
              (os.getpid(), model_key, time.time() - start_time))
        sys.stdout.flush()

    if failed:
        return dict(), dict()

    for feature_name, feature_dict in zip(['initial_gain', 'integration_gain'], [initial_gain, integration_gain]):
        for syn_condition in context.syn_conditions:
            feature_key = '%s_%s' % (feature_name, syn_condition)
            features[feature_key] = np.mean(feature_dict[syn_condition])

    for feature_name, feature_dict in zip(['initial_gain', 'integration_gain'],
                                          [initial_gain_residuals, integration_gain_residuals]):
        for syn_condition in context.syn_conditions:
            residuals_key = '%s_%s_residuals' % (feature_name, syn_condition)
            objectives[residuals_key] = np.mean(feature_dict[syn_condition])
            
    for syn_group in soma_compound_EPSP_summation:
        for syn_condition in soma_compound_EPSP_summation[syn_group]:
            feature_key = 'EPSP_summation_%s_%s' % (syn_group, syn_condition)
            features[feature_key] = soma_compound_EPSP_summation[syn_group][syn_condition]
            objective_val = ((context.target_val[feature_key] -
                              soma_compound_EPSP_summation[syn_group][syn_condition]) /
                             context.target_range['EPSP_summation']) ** 2.
            objectives[feature_key] = objective_val

    return features, objectives


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)