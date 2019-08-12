"""
Uses nested.optimize to tune spatiotemporal integration of AMPA + NMDA mixed EPSPs in dentate granule cell dendrites.

Requires a YAML file to specify required configuration parameters.
Requires use of a nested.parallel interface.
"""
__author__ = 'Aaron D. Milstein and Grace Ng'
from dentate.biophysics_utils import *
from nested.parallel import *
from nested.optimize_utils import *
from cell_utils import *
import click
import uuid

context = Context()


def config_worker():
    """

    """
    if 'plot' not in context():
        context.plot = False
    if 'debug' not in context():
        context.debug = False
    if 'limited_branches' not in context():
        context.limited_branches = False
    if not context_has_sim_env(context):
        build_sim_env(context, **context.kwargs)


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

    context.interface = get_parallel_interface(source_file=__file__, source_package=__package__, **kwargs)
    context.interface.start(disp=context.disp)
    context.interface.ensure_controller()
    config_optimize_interactive(__file__, config_file_path=config_file_path, output_dir=output_dir,
                                export=export, export_file_path=export_file_path, label=label,
                                disp=context.disp, interface=context.interface, verbose=verbose, plot=plot,
                                debug=debug, **kwargs)

    if plot:
        from dentate.plot import plot_synaptic_attribute_distribution
        plot_synaptic_attribute_distribution(context.cell, context.env, context.NMDA_type, 'g_unit',
                                             from_mech_attrs=True, from_target_attrs=True, show=True)
        plot_synaptic_attribute_distribution(context.cell, context.env, context.AMPA_type, 'g_unit',
                                             from_mech_attrs=True, from_target_attrs=True, show=True)

    if not debug:
        features = dict()

        # Stage 0:
        args = context.interface.execute(get_args_dynamic_unitary_EPSP_amp, context.x0_array, features)
        group_size = len(args[0])
        sequences = [[context.x0_array] * group_size] + args + [[context.export] * group_size] + \
                    [[context.plot] * group_size]
        primitives = context.interface.map(compute_features_unitary_EPSP_amp, *sequences)
        this_features = {key: value for feature_dict in primitives for key, value in viewitems(feature_dict)}
        features.update(this_features)
        context.update(locals())
        context.interface.apply(export_unitary_EPSP_traces)

        # Stage 1:
        args = context.interface.execute(get_args_dynamic_compound_EPSP_amp, context.x0_array, features)
        group_size = len(args[0])
        sequences = [[context.x0_array] * group_size] + args + [[context.export] * group_size] + \
                    [[context.plot] * group_size]
        primitives = context.interface.map(compute_features_compound_EPSP_amp, *sequences)
        this_features = {key: value for feature_dict in primitives for key, value in viewitems(feature_dict)}
        features.update(this_features)
        context.update(locals())
        context.interface.apply(export_compound_EPSP_traces)

        features, objectives = context.interface.execute(get_objectives_synaptic_integration, features, context.export)
        if export:
            collect_and_merge_temp_output(context.interface, context.export_file_path, verbose=context.disp)
        sys.stdout.flush()
        time.sleep(1.)
        context.interface.apply(shutdown_worker)
        print('params:')
        pprint.pprint(context.x0_dict)
        print('features:')
        pprint.pprint(features)
        print('objectives:')
        pprint.pprint(objectives)
        sys.stdout.flush()
        time.sleep(1.)
        if context.plot:
            context.interface.apply(plt.show)
    context.update(locals())

    if not interactive:
        context.interface.stop()


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

    min_expected_compound_EPSP_amp, max_expected_compound_EPSP_amp = 6., 12.  # mV

    clustered_branch_names = ['clustered%i' % i for i in range(num_clustered_branches)]

    ISI = {'units': 200., 'clustered': 1.1}  # inter-stimulus interval for synaptic stim (ms)
    units_per_sim = 5
    equilibrate = 250.  # time to steady-state
    stim_dur = 150.
    sim_duration = {'units': equilibrate + units_per_sim * ISI['units'],
                    'clustered': equilibrate + ISI['units'] + num_syns_per_clustered_branch * ISI['clustered'],
                    'default': equilibrate + stim_dur}
    trace_baseline = 10.
    duration = max(sim_duration.values())

    AMPA_type = 'AMPA'
    NMDA_type = 'NMDA'
    syn_mech_names = [AMPA_type, NMDA_type]

    context.update(locals())


def build_sim_env(context, verbose=2, cvode=True, daspk=True, **kwargs):
    """

    :param context: :class:'Context'
    :param verbose: int
    :param cvode: bool
    :param daspk: bool
    """
    init_context()
    context.env = Env(comm=context.comm, verbose=verbose > 1, **kwargs)
    configure_hoc_env(context.env)
    cell = get_biophys_cell(context.env, gid=context.gid, pop_name=context.cell_type, set_edge_delays=False,
                            mech_file_path=context.mech_file_path)
    init_biophysics(cell, reset_cable=True, correct_cm=context.correct_for_spines,
                    correct_g_pas=context.correct_for_spines, env=context.env, verbose=verbose > 1)
    init_syn_mech_attrs(cell, context.env)
    context.sim = QuickSim(context.duration, cvode=cvode, daspk=daspk, dt=context.dt, verbose=verbose > 1)
    context.spike_output_vec = h.Vector()
    cell.spike_detector.record(context.spike_output_vec)
    context.cell = cell
    context.temp_model_data = dict()
    context.temp_model_data_file_path = None
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
        sim.append_rec(cell, dend, name='local_branch', loc=0.5)

    equilibrate = context.equilibrate
    duration = context.duration

    if not sim.has_stim('holding'):
        sim.append_stim(cell, cell.tree.root, name='holding', loc=0.5, amp=0., delay=0., dur=duration)
        offset_vm('soma', context, vm_target=context.v_active, i_history=context.i_holding)

    if 'syn_id_dict' not in context():
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
        # NMDAR-R properties to match target features for spatiotemporal integration
        candidate_branches = [branch for branch in cell.apical if
                              50. < get_distance_to_node(cell, cell.tree.root, branch) < 150. and
                              90. < branch.sec.L < 120.]
        context.local_random.shuffle(candidate_branches)

        parents = []
        branch_count = 0
        for branch in (branch for branch in candidate_branches if branch.parent not in parents):
            this_syn_ids = syn_attrs.get_filtered_syn_ids(cell.gid, syn_sections=[branch.index],
                                                          syn_types=[env.Synapse_Types['excitatory']])
            candidate_syn_ids = []
            for syn_id in this_syn_ids:
                syn_loc = syn_attrs.syn_id_attr_dict[cell.gid][syn_id].syn_loc
                if 30. <= syn_loc * branch.sec.L <= 60.:
                    candidate_syn_ids.append(syn_id)
            if len(candidate_syn_ids) >= context.num_syns_per_clustered_branch:
                branch_key = context.clustered_branch_names[branch_count]
                syn_id_dict[branch_key].extend(context.local_random.sample(candidate_syn_ids,
                                                                           context.num_syns_per_clustered_branch))
                branch_count += 1
                parents.append(branch.parent)
            if branch_count >= context.num_clustered_branches:
                break
        if branch_count < context.num_clustered_branches:
            raise RuntimeError('optimize_DG_GC_synaptic_integration: problem finding required number of branches that'
                               'satisfy the requirement for clustered synapses: %i/%i' %
                               (branch_count, context.num_clustered_branches))

        context.syn_id_dict = syn_id_dict
        syn_id_set = set()
        for group_key in syn_id_dict:
            syn_id_set.update(context.syn_id_dict[group_key])
        context.syn_id_list = list(syn_id_set)

        config_biophys_cell_syns(env=context.env, gid=context.cell.gid, postsyn_name=context.cell.pop_name,
                                 syn_ids=context.syn_id_list, insert=True, insert_netcons=True, insert_vecstims=True,
                                 verbose=context.verbose > 1, throw_error=False)

    sim.parameters['duration'] = duration
    sim.parameters['equilibrate'] = equilibrate
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
    modify_syn_param(cell, env, 'apical', context.NMDA_type, param_name='Kd', value=x_dict['NMDA.Kd'],
                     update_targets=False)
    modify_syn_param(cell, env, 'apical', context.NMDA_type, param_name='gamma', value=x_dict['NMDA.gamma'],
                     update_targets=False)
    modify_syn_param(cell, env, 'apical', context.NMDA_type, param_name='g_unit', value=x_dict['NMDA.g_unit'],
                     update_targets=False)
    modify_syn_param(cell, env, 'apical', context.NMDA_type, param_name='vshift', value=x_dict['NMDA.vshift'],
                     update_targets=False)
    config_biophys_cell_syns(env=env, gid=cell.gid, postsyn_name=cell.pop_name, syn_ids=context.syn_id_list,
                             verbose=context.verbose > 1, throw_error=True)


def shutdown_worker():
    """

    """
    # context.temp_model_data_file.close()
    if context.temp_model_data_file is not None:
        context.temp_model_data_file.close()
    time.sleep(2.)
    if context.interface.global_comm.rank == 0:
        os.remove(context.temp_model_data_file_path)


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
            for rec_name in context.sim.recs:
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
            for rec_name in context.sim.recs:
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
    Global MPI rank 0 cannot participate in global collective operations while monitoring the NEURON ParallelContext
    bulletin board, so global rank 1 serves as root for this collective write procedure.
    """
    start_time = time.time()
    description = 'unitary_EPSP_traces'
    trace_len = int((context.ISI['units'] + context.trace_baseline) / context.dt)
    context.temp_model_data_legend = dict()

    if context.interface.global_comm.rank == 0:
        context.interface.pc.post('merge1', context.temp_model_data)
    elif context.interface.global_comm.rank == 1:
        context.interface.pc.take('merge1')
        temp_model_data_from_master = context.interface.pc.upkpyobj()
        dict_merge(context.temp_model_data, temp_model_data_from_master)

    if context.interface.global_comm.rank > 0:
        model_keys = list(context.temp_model_data.keys())
        model_keys = context.interface.worker_comm.gather(model_keys, root=0)
        if context.interface.worker_comm.rank == 0:
            model_keys = list(set([key for key_list in model_keys for key in key_list]))
        else:
            model_keys = None
        model_keys = context.interface.worker_comm.bcast(model_keys, root=0)

        if context.temp_model_data_file_path is None:
            if context.interface.worker_comm.rank == 0:
                context.temp_model_data_file_path = '%s/%s_%s_temp_model_data.hdf5' % \
                                                    (context.output_dir,
                                                     datetime.datetime.today().strftime('%Y%m%d_%H%M'),
                                                     context.optimization_title)
            context.temp_model_data_file_path = \
                context.interface.worker_comm.bcast(context.temp_model_data_file_path, root=0)
            context.temp_model_data_file = h5py.File(context.temp_model_data_file_path, 'a', driver='mpio',
                                                     comm=context.interface.worker_comm)

        for i, model_key in enumerate(model_keys):
            group_key = str(i)
            context.temp_model_data_legend[model_key] = group_key
            if group_key not in context.temp_model_data_file:
                context.temp_model_data_file.create_group(group_key)
            if description not in context.temp_model_data_file[group_key]:
                context.temp_model_data_file[group_key].create_group(description)
                for syn_group in context.syn_id_dict:
                    context.temp_model_data_file[group_key][description].create_group(syn_group)
                    num_syn_ids = len(context.syn_id_dict[syn_group])
                    for syn_condition in context.syn_conditions:
                        context.temp_model_data_file[group_key][description][syn_group].create_group(syn_condition)
                        for rec_name in context.sim.recs:
                            context.temp_model_data_file[group_key][description][syn_group][
                                syn_condition].create_dataset(rec_name, (num_syn_ids, trace_len), dtype='f8')

            target_rank = i % context.interface.worker_comm.size
            if model_key in context.temp_model_data:
                this_temp_model_data = context.temp_model_data.pop(model_key)
            else:
                this_temp_model_data = {}
            this_temp_model_data = context.interface.worker_comm.gather(this_temp_model_data, root=target_rank)
            if context.interface.worker_comm.rank == target_rank:
                context.temp_model_data[model_key] = {description: {}}
                for element in this_temp_model_data:
                    if element:
                        dict_merge(context.temp_model_data[model_key], element)
            context.interface.worker_comm.barrier()

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

        context.interface.worker_comm.barrier()
        context.temp_model_data_file.flush()

    del context.temp_model_data
    context.temp_model_data = dict()

    if context.interface.global_comm.rank == 0:
        context.interface.pc.take('merge2')
        context.temp_model_data_file_path = context.interface.pc.upkpyobj()[0]
        context.temp_model_data_file = None
        context.interface.pc.take('merge3')
        context.temp_model_data_legend = context.interface.pc.upkpyobj()
    elif context.interface.global_comm.rank == 1:
        context.interface.pc.post('merge2', [context.temp_model_data_file_path])
        context.interface.pc.post('merge3', context.temp_model_data_legend)

    sys.stdout.flush()
    time.sleep(1.)
    if context.interface.global_comm.rank == 1 and context.disp:
        print('optimize_DG_GC_synaptic_integration: export_unitary_EPSP_traces took %.2f s' %
              (time.time() - start_time))
        sys.stdout.flush()
        time.sleep(1.)


def export_compound_EPSP_traces():
    """
    Data from model simulations is temporarily stored locally on each worker. This method uses collective operations to
    export the data to disk, with one hdf5 group per model.
    """
    start_time = time.time()
    description = 'compound_EPSP_traces'
    trace_len = int((context.sim_duration['clustered'] - context.equilibrate + context.trace_baseline) / context.dt)

    if context.interface.global_comm.rank == 0:
        context.interface.pc.post('merge4', context.temp_model_data)
    elif context.interface.global_comm.rank == 1:
        context.interface.pc.take('merge4')
        temp_model_data_from_master = context.interface.pc.upkpyobj()
        dict_merge(context.temp_model_data, temp_model_data_from_master)

    if context.interface.global_comm.rank > 0:
        model_keys = list(context.temp_model_data.keys())
        model_keys = context.interface.worker_comm.gather(model_keys, root=0)
        if context.interface.worker_comm.rank == 0:
            model_keys = list(set([key for key_list in model_keys for key in key_list]))
        else:
            model_keys = None
        model_keys = context.interface.worker_comm.bcast(model_keys, root=0)

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
                        for rec_name in context.sim.recs:
                            context.temp_model_data_file[group_key][description][syn_group][
                                syn_condition].create_dataset(rec_name, (num_syn_ids, trace_len), dtype='f8')

            target_rank = i % context.interface.worker_comm.size
            if model_key in context.temp_model_data:
                this_temp_model_data = context.temp_model_data.pop(model_key)
            else:
                this_temp_model_data = {}
            this_temp_model_data = context.interface.worker_comm.gather(this_temp_model_data, root=target_rank)
            if context.interface.worker_comm.rank == target_rank:
                context.temp_model_data[model_key] = {description: {}}
                for element in this_temp_model_data:
                    if element:
                        dict_merge(context.temp_model_data[model_key], element)
            context.interface.worker_comm.barrier()

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

        context.interface.worker_comm.barrier()
        context.temp_model_data_file.flush()

    del context.temp_model_data
    context.temp_model_data = dict()

    if context.interface.global_comm.rank == 0:
        context.interface.pc.take('merge5')
        handshake = context.interface.pc.upkscalar()
    elif context.interface.global_comm.rank == 1:
        context.interface.pc.post('merge5', 0)

    sys.stdout.flush()
    time.sleep(1.)
    if context.interface.global_comm.rank == 1 and context.disp:
        print('optimize_DG_GC_synaptic_integration: export_compound_EPSP_traces took %.2f s' %
              (time.time() - start_time))
        sys.stdout.flush()
        time.sleep(1.)


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
    config_sim_env(context)
    update_source_contexts(x, context)
    zero_na(context.cell)

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
    context.sim.parameters['syn_secs'] = []
    context.sim.parameters['swc_types'] = []
    context.sim.parameters['syn_ids'] = syn_ids
    for i, syn_id in enumerate(syn_ids):
        syn_id = int(syn_id)
        spike_time = context.equilibrate + i * ISI
        for syn_name in context.syn_mech_names:
            this_nc = syn_attrs.get_netcon(context.cell.gid, syn_id, syn_name)
            this_nc.delay = 0.
            this_nc.pre().play(h.Vector([spike_time]))
            if syn_name == context.NMDA_type and syn_condition == 'AP5':
                config_syn(syn_name=syn_name, rules=syn_attrs.syn_param_rules, mech_names=syn_attrs.syn_mech_names,
                           nc=this_nc, syn=this_nc.syn(), g_unit=0.)
        syn = syn_attrs.syn_id_attr_dict[context.cell.gid][syn_id]
        node_index = syn.syn_section
        node_type = syn.swc_type
        context.sim.parameters['syn_secs'].append(node_index)
        context.sim.parameters['swc_types'].append(node_type)
        if i == 0:
            branch = context.cell.tree.get_node_with_index(node_index)
            context.sim.modify_rec('local_branch', node=branch)

    sim.run(context.v_active)

    traces_dict = defaultdict(dict)
    for i, syn_id in enumerate(syn_ids):
        start = int((equilibrate + i * ISI) / dt)
        end = start + int(ISI / dt)
        trace_start = start - int(trace_baseline / dt)
        baseline_start, baseline_end = int(start - 3. / dt), int(start - 1. / dt)
        syn_id = int(syn_id)
        for rec_name in context.sim.recs:
            this_vm = np.array(context.sim.recs[rec_name]['vec'])
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
    sim.parameters['duration'] = duration
    sim.parameters['title'] = title
    sim.parameters['description'] = description

    if context.verbose > 0:
        print('compute_features_unitary_EPSP_amp: pid: %i; %s: %s took %.3f s' %
              (os.getpid(), title, description, time.time() - start_time))
        sys.stdout.flush()
    if plot:
        context.sim.plot()

    if export:
        print('debug: pid: %i; temp_output_path: %s' % (os.getpid(), context.temp_output_path))
        sys.stdout.flush()
        context.sim.export_to_file(context.temp_output_path)

    sim.restore_state()

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
    config_sim_env(context)
    update_source_contexts(x, context)
    zero_na(context.cell)

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
    context.sim.parameters['syn_secs'] = []
    context.sim.parameters['swc_types'] = []
    context.sim.parameters['syn_ids'] = syn_ids
    for i, syn_id in enumerate(syn_ids):
        spike_time = context.equilibrate + i * ISI
        for syn_name in context.syn_mech_names:
            this_nc = syn_attrs.get_netcon(context.cell.gid, syn_id, syn_name)
            this_nc.delay = 0.
            this_nc.pre().play(h.Vector([spike_time]))
            if syn_name == context.NMDA_type and syn_condition == 'AP5':
                config_syn(syn_name=syn_name, rules=syn_attrs.syn_param_rules, mech_names=syn_attrs.syn_mech_names,
                           nc=this_nc, syn=this_nc.syn(), g_unit=0.)
        syn = syn_attrs.syn_id_attr_dict[context.cell.gid][syn_id]
        node_index = syn.syn_section
        node_type = syn.swc_type
        context.sim.parameters['syn_secs'].append(node_index)
        context.sim.parameters['swc_types'].append(node_type)
        if i == 0:
            branch = context.cell.tree.get_node_with_index(node_index)
            context.sim.modify_rec('local_branch', node=branch)

    sim.run(context.v_active)

    traces_dict = {}
    start = int(equilibrate / dt)
    trace_start = start - int(trace_baseline / dt)
    baseline_start, baseline_end = int(start - 3. / dt), int(start - 1. / dt)
    for rec_name in context.sim.recs:
        this_vm = np.array(context.sim.recs[rec_name]['vec'])
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

    title = 'compound_EPSP_amp'
    description = 'condition: %s, group: %s, num_syns: %i, first syn_id: %i' % \
                  (syn_condition, syn_group, len(syn_ids), syn_ids[0])
    sim.parameters['duration'] = duration
    sim.parameters['title'] = title
    sim.parameters['description'] = description

    if context.verbose > 0:
        print('compute_features_compound_EPSP_amp: pid: %i; %s: %s took %.3f s' %
              (os.getpid(), title, description, time.time() - start_time))
        sys.stdout.flush()
    if plot:
        context.sim.plot()
    if export:
        context.sim.export_to_file(context.temp_output_path)
    sim.restore_state()

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
    for i in range(len(syn_id_dict)):
        num_syns = i + 1
        traces[num_syns] = {}
        for count, syn_id in enumerate(syn_id_dict[:num_syns]):
            start = baseline_len + int(count * context.ISI['clustered'] / context.dt)
            end = start + unitary_len
            for rec_name, this_trace in viewitems(unitary_traces_dict[syn_id]):
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
    start_time = time.time()
    objectives = dict()
    model_key = features['model_key']
    group_key = context.temp_model_data_legend[model_key]

    unitary_EPSP_traces_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    compound_EPSP_traces_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    description = 'unitary_EPSP_traces'
    with h5py.File(context.temp_model_data_file_path, 'r') as temp_model_data_file:
        for syn_group in temp_model_data_file[group_key][description]:
            syn_id_list = context.syn_id_dict[syn_group]
            for syn_condition in temp_model_data_file[group_key][description][syn_group]:
                this_group = temp_model_data_file[group_key][description][syn_group][syn_condition]
                for rec_name in context.sim.recs:
                    for i, syn_id in enumerate(syn_id_list):
                        unitary_EPSP_traces_dict[syn_group][syn_condition][syn_id][rec_name] = this_group[rec_name][i,:]

        description = 'compound_EPSP_traces'
        for syn_group in temp_model_data_file[group_key][description]:
            num_syn_ids = len(context.syn_id_dict[syn_group])
            for syn_condition in temp_model_data_file[group_key][description][syn_group]:
                this_group = temp_model_data_file[group_key][description][syn_group][syn_condition]
                for rec_name in context.sim.recs:
                    for i in range(num_syn_ids):
                        num_syns = i + 1
                        compound_EPSP_traces_dict[syn_group][syn_condition][num_syns][rec_name] = \
                            this_group[rec_name][i,:]

    control_EPSP_amp_list = []
    NMDA_contribution_list = []
    for syn_id in context.syn_id_dict['random']:
        control_soma_trace = np.array(unitary_EPSP_traces_dict['random']['control'][syn_id]['soma'][:])
        control_soma_amp = np.max(control_soma_trace)
        control_EPSP_amp_list.append(control_soma_amp)
        AP5_soma_trace = np.array(unitary_EPSP_traces_dict['random']['AP5'][syn_id]['soma'][:])
        AP5_soma_amp = np.max(AP5_soma_trace)
        NMDA_contribution_list.append((control_soma_amp - AP5_soma_amp) / control_soma_amp)

    mean_unitary_EPSP_amp_residuals = \
        np.mean(((np.array(control_EPSP_amp_list) - context.target_val['mean_unitary_EPSP_amp']) /
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
            this_initial_gain = (this_actual[1] - this_actual[0]) / (this_expected[1] - this_expected[0])
            # Integration should be close to linear without gain for the first few synapses.
            initial_gain[syn_condition].append(this_initial_gain)
            feature_key = 'initial_gain_%s' % syn_condition
            this_initial_gain_residuals = ((this_initial_gain - context.target_val[feature_key]) /
                                           context.target_range[feature_key]) ** 2.
            initial_gain_residuals[syn_condition].append(this_initial_gain_residuals)
            indexes = np.where(this_expected <= context.max_expected_compound_EPSP_amp)[0]
            if not np.any(indexes) or (not context.limited_branches and syn_condition == 'control' and
                                       max(this_expected) < context.min_expected_compound_EPSP_amp):
                if context.verbose > 0:
                    print('optimize_DG_GC_synaptic_integration: get_objectives: pid: %i; aborting - expected ' \
                          'compound EPSP amplitude below criterion' % os.getpid())
                    sys.stdout.flush()
                return dict(), dict()
            slope, intercept, r_value, p_value, std_err = stats.linregress(this_expected[indexes], this_actual[indexes])
            integration_gain[syn_condition].append(slope)
            feature_key = 'integration_gain_%s' % syn_condition
            this_target = context.target_val[feature_key] * this_expected[indexes] + intercept
            this_integration_gain_residuals = 0.
            for target_val, actual_val in zip(this_target, this_actual[indexes]):
                this_integration_gain_residuals += ((actual_val - target_val) /
                                                    context.target_range['mean_unitary_EPSP_amp']) ** 2.
            this_integration_gain_residuals /= float(len(indexes))
            integration_gain_residuals[syn_condition].append(this_integration_gain_residuals)

    for feature_name, feature_dict in zip(['initial_gain', 'integration_gain'], [initial_gain, integration_gain]):
        for syn_condition in context.syn_conditions:
            feature_key = '%s_%s' % (feature_name, syn_condition)
            features[feature_key] = np.mean(feature_dict[syn_condition])

    for feature_name, feature_dict in zip(['initial_gain', 'integration_gain'],
                                          [initial_gain_residuals, integration_gain_residuals]):
        for syn_condition in context.syn_conditions:
            residuals_key = '%s_%s_residuals' % (feature_name, syn_condition)
            objectives[residuals_key] = np.mean(feature_dict[syn_condition])

    if export:
        with h5py.File(context.export_file_path, 'a') as f:
            description = 'mean_unitary_EPSP_traces'
            if description not in f:
                f.create_group(description)
                f[description].attrs['enumerated'] = False
            group = f[description]
            t = np.arange(-context.trace_baseline, context.ISI['units'], context.dt)
            group.create_dataset('time', compression='gzip', data=t)
            data_group = group.create_group('data')
            for syn_group in unitary_EPSP_traces_dict:
                data_group.create_group(syn_group)
                for syn_condition in unitary_EPSP_traces_dict[syn_group]:
                    this_group = data_group[syn_group].create_group(syn_condition)
                    for rec_name in next(iter(viewvalues(unitary_EPSP_traces_dict[syn_group][syn_condition]))):
                        this_condition_array_list = []
                        for syn_id in unitary_EPSP_traces_dict[syn_group][syn_condition]:
                            this_condition_array_list.append(
                                unitary_EPSP_traces_dict[syn_group][syn_condition][syn_id][rec_name])
                        this_group.create_dataset(rec_name, compression='gzip',
                                                  data=np.mean(this_condition_array_list, axis=0))

            description = 'compound_EPSP_summary'
            if description not in f:
                f.create_group(description)
                f[description].attrs['enumerated'] = False
            group = f[description]
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

    if context.verbose > 1:
        print('get_objectives_synaptic_integration: pid: %i; took %.3f s' % (os.getpid(), time.time() - start_time))
        sys.stdout.flush()

    return features, objectives


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)