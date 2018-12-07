"""
Uses nested.optimize to tune spatiotemporal integration of AMPA + NMDA mixed EPSPs in dentate granule cell dendrites.

Requires a YAML file to specify required configuration parameters.
Requires use of a nested.parallel interface.
"""
__author__ = 'Aaron D. Milstein and Grace Ng'
from dentate.biophysics_utils import *
from nested.optimize_utils import *
from optimize_cells_utils import *
import click


context = Context()


@click.command()
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/optimize_DG_GC_synaptic_integration_config.yaml')
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--export", is_flag=True)
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default=None)
@click.option("--verbose", type=int, default=2)
@click.option("--plot", is_flag=True)
@click.option("--run-tests", is_flag=True)
def main(config_file_path, output_dir, export, export_file_path, label, verbose, plot, run_tests):
    """

    :param config_file_path: str (path)
    :param output_dir: str (path)
    :param export: bool
    :param export_file_path: str
    :param label: str
    :param verbose: bool
    :param plot: bool
    :param run_tests: bool
    """
    # requires a global variable context: :class:'Context'
    context.update(locals())
    disp = verbose > 0
    config_interactive(context, __file__, config_file_path=config_file_path, output_dir=output_dir, export=export,
                       export_file_path=export_file_path, label=label, disp=disp)

    if run_tests:
        unit_tests_synaptic_integration()


def unit_tests_synaptic_integration():
    """

    """
    features = dict()
    objectives = dict()

    # Stage 0:
    args = get_args_dynamic_unitary_EPSP_amp(context.x0_array, features)
    group_size = len(args[0])
    sequences = [[context.x0_array] * group_size] + args + [[context.export] * group_size] + \
                [[context.plot] * group_size]
    primitives = map(compute_features_unitary_EPSP_amp, *sequences)
    """
    primitives = [compute_features_unitary_EPSP_amp(*zip(*sequences)[0]),
                  compute_features_unitary_EPSP_amp(*zip(*sequences)[1]),
                  compute_features_unitary_EPSP_amp(*zip(*sequences)[17])]
    """
    this_features = filter_features_unitary_EPSP_amp(primitives, features, context.export)
    features.update(this_features)

    # Stage 1:
    args = get_args_dynamic_compound_EPSP_amp(context.x0_array, features)
    group_size = len(args[0])
    sequences = [[context.x0_array] * group_size] + args + [[context.export] * group_size] + \
                [[context.plot] * group_size]
    primitives = map(compute_features_compound_EPSP_amp, *sequences)
    this_features = filter_features_compound_EPSP_amp(primitives, features, context.export)
    features.update(this_features)
    
    features, objectives = get_objectives_synaptic_integration(features)

    context.update(locals())
    print 'params:'
    pprint.pprint(context.x0_dict)
    print 'features:'
    pprint.pprint(features)
    print 'objectives:'
    pprint.pprint(objectives)


def config_worker():
    """

    """
    if 'plot' not in context():
        context.plot = False
    if not context_has_sim_env(context):
        build_sim_env(context, **context.kwargs)


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

    # for clustered inputs, num_syns corresponds to number of clustered inputs per branch
    max_syns_per_random_branch = 5
    min_random_inter_syn_distance = 30.  # um
    num_syns_per_clustered_branch = 5  # 20
    syn_conditions = ['control', 'AP5']

    # number of branches to test temporal integration of clustered inputs
    num_clustered_branches = 1  # 2
    clustered_branch_names = ['clustered%i' % i for i in xrange(num_clustered_branches)]

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
    cell = get_biophys_cell(context.env, gid=context.gid, pop_name=context.cell_type)
    init_biophysics(cell, reset_cable=True, from_file=True, mech_file_path=context.mech_file_path,
                    correct_cm=context.correct_for_spines, correct_g_pas=context.correct_for_spines, env=context.env)
    init_syn_mech_attrs(cell, context.env, from_file=True)
    context.sim = QuickSim(context.duration, cvode=cvode, daspk=daspk, dt=context.dt, verbose=verbose>1)
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
        dend, dend_loc = get_DG_GC_thickest_dend_branch(context.cell, 100., terminal=False)
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
        syn_indexes = defaultdict(list)
        # choose a random subset of synapses across all apical branches for tuning a distance-dependent AMPA-R gradient
        for branch in cell.apical:
            if branch.index in syn_attrs.sec_index_map[cell.gid]:
                this_syn_indexes = syn_attrs.sec_index_map[cell.gid][branch.index]
                this_syn_indexes = syn_attrs.get_filtered_syn_indexes(cell.gid, syn_indexes=this_syn_indexes,
                                                                      syn_types=[env.syntypes_dict['excitatory']])
                if len(this_syn_indexes) > 1:
                    if branch.sec.L <= context.min_random_inter_syn_distance:
                        syn_indexes['random'].extend(context.local_random.sample(this_syn_indexes, 1))
                    else:
                        this_num_syns = min(len(this_syn_indexes),
                                            int(branch.sec.L / context.min_random_inter_syn_distance),
                                            context.max_syns_per_random_branch)
                        syn_indexes['random'].extend(context.local_random.sample(this_syn_indexes, this_num_syns))
                elif len(this_syn_indexes) > 0:
                    syn_indexes['random'].append(this_syn_indexes[0])

        # choose a random subset of apical branches that contain the required number of clustered synapses to tune
        # NMDAR-R properties to match target features for spatiotemporal integration
        candidate_branches = [branch for branch in cell.apical if
                              50. < get_distance_to_node(cell, cell.tree.root, branch) < 150. and
                              branch.sec.L > 80.]
        context.local_random.shuffle(candidate_branches)

        parents = []
        branch_count = 0
        for branch in (branch for branch in candidate_branches if branch.parent not in parents):
            if branch.index in syn_attrs.sec_index_map[cell.gid]:
                this_syn_indexes = syn_attrs.sec_index_map[cell.gid][branch.index]
                this_syn_indexes = syn_attrs.get_filtered_syn_indexes(cell.gid, syn_indexes=this_syn_indexes,
                                                                      syn_types=[env.syntypes_dict['excitatory']])
                candidate_syn_indexes = []
                for syn_index in this_syn_indexes:
                    syn_loc = syn_attrs.syn_id_attr_dict[cell.gid]['syn_locs'][syn_index]
                    if 30. <= syn_loc * branch.sec.L <= 60.:
                        candidate_syn_indexes.append(syn_index)
                if len(candidate_syn_indexes) >= context.num_syns_per_clustered_branch:
                    branch_key = context.clustered_branch_names[branch_count]
                    syn_indexes[branch_key].extend(context.local_random.sample(candidate_syn_indexes,
                                                                               context.num_syns_per_clustered_branch))
                    branch_count += 1
                    parents.append(branch.parent)
            if branch_count >= context.num_clustered_branches:
                break
        if branch_count < context.num_clustered_branches:
            raise RuntimeError('optimize_DG_GC_synaptic_integration: problem finding required number of branches that'
                               'satisfy the requirement for clustered synapses: %i/%i' %
                               (branch_count, context.num_clustered_branches))

        context.syn_id_dict = defaultdict(list)
        syn_id_set = set()
        context.syn_id_list = []
        for group_key in syn_indexes:
            context.syn_id_dict[group_key] = syn_attrs.syn_id_attr_dict[cell.gid]['syn_ids'][syn_indexes[group_key]]
            syn_id_set.update(context.syn_id_dict[group_key])
        context.syn_id_list = list(syn_id_set)

        config_syns_from_mech_attrs(context.cell.gid, context.env, context.cell.pop_name, syn_ids=context.syn_id_list,
                                    insert=True, verbose=True)

    sim.parameters['duration'] = duration
    sim.parameters['equilibrate'] = equilibrate
    context.previous_module = __file__
    if context.plot:
        from dentate.plot import plot_synaptic_attribute_distribution
        plot_synaptic_attribute_distribution(cell, env, context.NMDA_type, 'g_unit', from_mech_attrs=True,
                                                          from_target_attrs=True, show=True)
        plot_synaptic_attribute_distribution(cell, env, context.AMPA_type, 'g_unit', from_mech_attrs=True,
                                                          from_target_attrs=True, show=True)


def update_syn_mechanisms(x, context=None):
    """

    :param x:
    :param context:
    """
    if context is None:
        raise RuntimeError('update_syn_mechanisms: missing required Context object')
    x_dict = param_array_to_dict(x, context.param_names)
    cell = context.cell
    env = context.env
    modify_syn_mech_param(cell, env, 'apical', context.AMPA_type, param_name='g_unit', value=x_dict['AMPA.g0'],
                          filters={'syn_types': ['excitatory']}, origin='soma', slope=x_dict['AMPA.slope'],
                          tau=x_dict['AMPA.tau'], update_targets=True)
    modify_syn_mech_param(cell, env, 'apical', context.AMPA_type, param_name='g_unit',
                          filters={'syn_types': ['excitatory']}, origin='parent',
                          origin_filters={'syn_types': ['excitatory']},
                          custom={'func': 'custom_filter_if_terminal'}, update_targets=True, append=True)
    modify_syn_mech_param(cell, env, 'apical', context.AMPA_type, param_name='g_unit',
                          filters={'syn_types': ['excitatory'], 'layers': ['OML']}, origin='apical',
                          origin_filters={'syn_types': ['excitatory'], 'layers': ['MML']}, update_targets=True,
                          append=True)
    modify_syn_mech_param(cell, env, 'apical', context.NMDA_type, param_name='Kd', value=x_dict['NMDA.Kd'],
                          update_targets=True)
    modify_syn_mech_param(cell, env, 'apical', context.NMDA_type, param_name='gamma', value=x_dict['NMDA.gamma'],
                          update_targets=True)
    modify_syn_mech_param(cell, env, 'apical', context.NMDA_type, param_name='g_unit', value=x_dict['NMDA.g_unit'],
                          update_targets=True)


def get_args_dynamic_unitary_EPSP_amp(x, features):
    """
    A nested map operation is required to compute unitary EPSP amplitude features. The arguments to be mapped include
    a unique file_path for each set of parameters that will be used to temporarily store simulation output.
    :param x: array
    :param features: dict
    :return: list of list
    """
    syn_group_list = []
    syn_id_lists = []
    syn_condition_list = []
    import uuid
    temp_traces_path = '%s/%s_temp_traces_%s.hdf5' % (context.output_dir,
                                                      datetime.datetime.today().strftime('%Y%m%d%H%M'),
                                                      str(uuid.uuid1()))
    if os.path.isfile(temp_traces_path):
        os.remove(temp_traces_path)
    f = h5py.File(temp_traces_path, 'w')
    f.close()

    temp_traces_path_list = []
    for syn_group in context.syn_id_dict:
        this_syn_id_group = context.syn_id_dict[syn_group]
        this_syn_id_lists = []
        start = 0
        while start < len(this_syn_id_group):
            this_syn_id_lists.append(this_syn_id_group[start:start + context.units_per_sim])
            start += context.units_per_sim
        num_sims = len(this_syn_id_lists)
        syn_id_lists.extend(this_syn_id_lists)
        syn_group_list.extend([syn_group] * num_sims)
        syn_condition_list.extend(['control'] * num_sims)
        temp_traces_path_list.extend([temp_traces_path] * num_sims)
        if syn_group == 'random':
            syn_id_lists.extend(this_syn_id_lists)
            syn_group_list.extend([syn_group] * num_sims)
            syn_condition_list.extend(['AP5'] * num_sims)
            temp_traces_path_list.extend([temp_traces_path] * num_sims)

    return [syn_id_lists, syn_condition_list, syn_group_list, temp_traces_path_list]


def compute_features_unitary_EPSP_amp(x, syn_ids, syn_condition, syn_group, temp_traces_path, export=False, plot=False):
    """

    :param x: array
    :param syn_ids: list of int
    :param syn_condition: str
    :param syn_group: str
    :param temp_traces_path: str
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

    sim.modify_stim('holding', node=node, loc=loc, amp=context.i_holding['soma'][context.v_active])

    syn_attrs = context.env.synapse_attributes
    context.sim.parameters['syn_secs'] = []
    context.sim.parameters['swc_types'] = []
    context.sim.parameters['syn_ids'] = syn_ids
    for i, syn_id in enumerate(syn_ids):
        spike_time = context.equilibrate + i * ISI
        for syn_name in context.syn_mech_names:
            this_nc = syn_attrs.get_netcon(context.cell.gid, syn_id, syn_name)
            this_nc.pre().play(h.Vector([spike_time]))
            if syn_name == context.NMDA_type and syn_condition == 'AP5':
                config_syn(syn_name=syn_name, rules=syn_attrs.syn_param_rules, mech_names=syn_attrs.syn_mech_names,
                           nc=this_nc, syn=this_nc.syn(), g_unit=0.)
        syn_index = syn_attrs.syn_id_attr_index_map[context.cell.gid][syn_id]
        node_index = syn_attrs.syn_id_attr_dict[context.cell.gid]['syn_secs'][syn_index]
        node_type = syn_attrs.syn_id_attr_dict[context.cell.gid]['swc_types'][syn_index]
        context.sim.parameters['syn_secs'].append(node_index)
        context.sim.parameters['swc_types'].append(node_type)
        if i == 0:
            branch = context.cell.tree.get_node_with_index(node_index)
            context.sim.modify_rec('local_branch', node=branch)

    sim.run(context.v_active)

    soma_EPSP_amp_dict = {}
    traces_dict = defaultdict(dict)
    for i, syn_id in enumerate(syn_ids):
        start = int((equilibrate + i * ISI) / dt)
        end = start + int(ISI / dt)
        trace_start = start - int(trace_baseline / dt)
        baseline_start, baseline_end = int(start - 3. / dt), int(start - 1. / dt)
        for rec_name in context.sim.recs:
            this_vm = np.array(context.sim.recs[rec_name]['vec'])
            baseline = np.mean(this_vm[baseline_start:baseline_end])
            this_vm = this_vm[trace_start:end] - baseline
            peak = np.max(this_vm)
            peak_index = np.argmax(this_vm)
            zero_index = np.where(this_vm[peak_index:] <= 0.)[0]
            if np.any(zero_index):
                this_vm[peak_index+zero_index[0]:] = 0.
            if rec_name == 'soma':
                soma_EPSP_amp_dict[syn_id] = peak
            traces_dict[syn_id][rec_name] = np.array(this_vm)
        for syn_name in context.syn_mech_names:
            this_nc = syn_attrs.get_netcon(context.cell.gid, syn_id, syn_name)
            this_nc.pre().play(h.Vector())

    description = 'unitary_EPSP_traces'
    with h5py.File(temp_traces_path, 'a', driver='mpio', comm=context.comm) as f:
        if description not in f:
            f.create_group(description)
            f[description].attrs['enumerated'] = False
        group = f[description]
        if 'time' not in group:
            t = np.arange(-trace_baseline, ISI, context.dt)
            group.create_dataset('time', data=t)
        if 'data' not in group:
            group.create_group('data')
        data_group = group['data']
        if syn_group not in data_group:
            data_group.create_group(syn_group)
        if syn_condition not in data_group[syn_group]:
            data_group[syn_group].create_group(syn_condition)
        for syn_id in traces_dict:
            key = str(syn_id)
            this_group = data_group[syn_group][syn_condition].create_group(key)
            for rec_name in traces_dict[syn_id]:
                this_group.create_dataset(rec_name, data=traces_dict[syn_id][rec_name])

    result = {'syn_group': syn_group, 'syn_condition': syn_condition, 'soma_unitary_EPSP_amp': soma_EPSP_amp_dict,
              'temp_traces_path': temp_traces_path}

    title = 'unitary_EPSP_amp'
    description = 'condition: %s, group: %s, num_syns: %i, first syn_id: %i' % \
                  (syn_condition, syn_group, len(syn_ids), syn_ids[0])
    sim.parameters['duration'] = duration
    sim.parameters['title'] = title
    sim.parameters['description'] = description

    if context.verbose > 0:
        print 'compute_features_unitary_EPSP_amp: pid: %i; %s: %s took %.3f s' % \
              (os.getpid(), title, description, time.time() - start_time)
    if plot:
        context.sim.plot()
    if export:
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
    soma_unitary_EPSP_amp_dict = defaultdict(lambda: defaultdict(dict))
    temp_traces_path = None
    for this_feature_dict in primitives:
        this_temp_traces_path = this_feature_dict['temp_traces_path']
        if temp_traces_path is None:
            temp_traces_path = this_temp_traces_path
        elif this_temp_traces_path != temp_traces_path:
            raise Exception('filter_features_unitary_EPSP_amp: temp_traces_paths do not match')
        syn_group = this_feature_dict['syn_group']
        syn_condition = this_feature_dict['syn_condition']
        soma_unitary_EPSP_amp_dict[syn_group][syn_condition].update(this_feature_dict['soma_unitary_EPSP_amp'])
    control_EPSP_amp_list = []
    NMDA_contribution_list = []
    for syn_id in soma_unitary_EPSP_amp_dict['random']['AP5']:
        control_amp = soma_unitary_EPSP_amp_dict['random']['control'][syn_id]
        control_EPSP_amp_list.append(control_amp)
        AP5_amp = soma_unitary_EPSP_amp_dict['random']['AP5'][syn_id]
        NMDA_contribution_list.append((control_amp - AP5_amp) / control_amp)
    features['mean_unitary_EPSP_amp'] = np.mean(control_EPSP_amp_list)
    features['mean_NMDA_contribution'] = np.mean(NMDA_contribution_list)
    features['temp_traces_path'] = temp_traces_path

    if export:
        with h5py.File(temp_traces_path, 'r') as f:
            description = 'unitary_EPSP_traces'
            t = f[description]['time'][:]
            mean_unitary_EPSP_traces_dict = defaultdict(lambda: defaultdict(dict))
            for syn_condition in f[description]['data']['random']:
                this_group = f[description]['data']['random'][syn_condition]
                first_key = this_group.iterkeys().next()
                for rec_name in this_group[first_key]:
                    this_mean_trace = np.mean([this_group[key][rec_name][:] for key in this_group], axis=0)
                    mean_unitary_EPSP_traces_dict[syn_condition][rec_name] = this_mean_trace
        with h5py.File(context.export_file_path, 'a') as f:
            description = 'mean_unitary_EPSP_traces'
            if description not in f:
                f.create_group(description)
                f[description].attrs['enumerated'] = False
            group = f[description]
            group.create_dataset('time', compression='gzip', data=t)
            data_group = group.create_group('data')
            for syn_condition in mean_unitary_EPSP_traces_dict:
                this_group = data_group.create_group(syn_condition)
                for rec_name in mean_unitary_EPSP_traces_dict[syn_condition]:
                    this_group.create_dataset(rec_name, compression='gzip',
                                              data=mean_unitary_EPSP_traces_dict[syn_condition][rec_name])

    return features


def get_args_dynamic_compound_EPSP_amp(x, features):
    """
    A nested map operation is required to compute compound EPSP amplitude features. The arguments to be mapped include
    a unique file_path for each set of parameters that will be used to temporarily store simulation output.
    :param x: array
    :param features: dict
    :return: list of list
    """
    syn_group_list = []
    syn_id_lists = []
    syn_condition_list = []
    temp_traces_path = features['temp_traces_path']
    temp_traces_path_list = []
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
            temp_traces_path_list.extend([temp_traces_path] * num_sims)

    return [syn_id_lists, syn_condition_list, syn_group_list, temp_traces_path_list]


def compute_features_compound_EPSP_amp(x, syn_ids, syn_condition, syn_group, temp_traces_path, export=False,
                                       plot=False):
    """

    :param x: array
    :param syn_ids: list of int
    :param syn_condition: str
    :param syn_group: str
    :param temp_traces_path: str
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

    sim.modify_stim('holding', node=node, loc=loc, amp=context.i_holding['soma'][context.v_active])

    syn_attrs = context.env.synapse_attributes
    context.sim.parameters['syn_secs'] = []
    context.sim.parameters['swc_types'] = []
    context.sim.parameters['syn_ids'] = syn_ids
    for i, syn_id in enumerate(syn_ids):
        spike_time = context.equilibrate + i * ISI
        for syn_name in context.syn_mech_names:
            this_nc = syn_attrs.get_netcon(context.cell.gid, syn_id, syn_name)
            this_nc.pre().play(h.Vector([spike_time]))
            if syn_name == context.NMDA_type and syn_condition == 'AP5':
                config_syn(syn_name=syn_name, rules=syn_attrs.syn_param_rules, mech_names=syn_attrs.syn_mech_names,
                           nc=this_nc, syn=this_nc.syn(), g_unit=0.)
        syn_index = syn_attrs.syn_id_attr_index_map[context.cell.gid][syn_id]
        node_index = syn_attrs.syn_id_attr_dict[context.cell.gid]['syn_secs'][syn_index]
        node_type = syn_attrs.syn_id_attr_dict[context.cell.gid]['swc_types'][syn_index]
        context.sim.parameters['syn_secs'].append(node_index)
        context.sim.parameters['swc_types'].append(node_type)
        if i == 0:
            branch = context.cell.tree.get_node_with_index(node_index)
            context.sim.modify_rec('local_branch', node=branch)

    sim.run(context.v_active)

    traces_dict = {}
    for i, syn_id in enumerate(syn_ids):
        start = int(equilibrate / dt)
        trace_start = start - int(trace_baseline / dt)
        baseline_start, baseline_end = int(start - 3. / dt), int(start - 1. / dt)
        for rec_name in context.sim.recs:
            this_vm = np.array(context.sim.recs[rec_name]['vec'])
            baseline = np.mean(this_vm[baseline_start:baseline_end])
            this_vm = this_vm[trace_start:] - baseline
            traces_dict[rec_name] = np.array(this_vm)
        for syn_name in context.syn_mech_names:
            this_nc = syn_attrs.get_netcon(context.cell.gid, syn_id, syn_name)
            this_nc.pre().play(h.Vector())

    description = 'compound_EPSP_traces'
    with h5py.File(temp_traces_path, 'a') as f:
        if description not in f:
            f.create_group(description)
            f[description].attrs['enumerated'] = False
        group = f[description]
        if 'time' not in group:
            t = np.arange(-trace_baseline, duration - equilibrate, context.dt)
            group.create_dataset('time', data=t)
        if 'data' not in group:
            group.create_group('data')
        data_group = group['data']
        if syn_group not in data_group:
            data_group.create_group(syn_group)
        if syn_condition not in data_group[syn_group]:
            data_group[syn_group].create_group(syn_condition)
        key = str(len(syn_ids))
        this_group = data_group[syn_group][syn_condition].create_group(key)
        for rec_name in traces_dict:
            this_group.create_dataset(rec_name, data=traces_dict[rec_name])

    result = {'syn_group': syn_group, 'syn_condition': syn_condition, 'syn_ids': syn_ids}

    title = 'compound_EPSP_amp'
    description = 'condition: %s, group: %s, num_syns: %i, first syn_id: %i' % \
                  (syn_condition, syn_group, len(syn_ids), syn_ids[0])
    sim.parameters['duration'] = duration
    sim.parameters['title'] = title
    sim.parameters['description'] = description

    if context.verbose > 0:
        print 'compute_features_compound_EPSP_amp: pid: %i; %s: %s took %.3f s' % \
              (os.getpid(), title, description, time.time() - start_time)
    if plot:
        context.sim.plot()
    if export:
        context.sim.export_to_file(context.temp_output_path)
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


def filter_features_compound_EPSP_amp(primitives, current_features, export=False):
    """
    :param primitives: list of dict (each dict contains results from a single simulation)
    :param current_features: dict
    :param export: bool
    :return: dict
    """
    features = {}
    temp_traces_path = current_features['temp_traces_path']
    syn_ids_dict = defaultdict(lambda: defaultdict(dict))
    for this_feature_dict in primitives:
        syn_group = this_feature_dict['syn_group']
        syn_condition = this_feature_dict['syn_condition']
        syn_ids = this_feature_dict['syn_ids']
        num_syns = len(syn_ids)
        syn_ids_dict[syn_group][syn_condition][num_syns] = syn_ids
    with h5py.File(temp_traces_path, 'r') as f:
        description = 'compound_EPSP_traces'
        t = f[description]['time'][:]
        compound_EPSP_traces_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for syn_group in syn_ids_dict:
            for syn_condition in syn_ids_dict[syn_group]:
                for num_syns in syn_ids_dict[syn_group][syn_condition]:
                    this_group = f[description]['data'][syn_group][syn_condition][str(num_syns)]
                    for rec_name in this_group:
                        compound_EPSP_traces_dict[syn_group][syn_condition][num_syns][rec_name] = \
                            this_group[rec_name][:]
        description = 'unitary_EPSP_traces'
        unitary_EPSP_traces_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for syn_group in syn_ids_dict:
            this_group = f[description]['data'][syn_group]['control']
            for syn_id_key in this_group:
                syn_id = int(syn_id_key)
                for rec_name in this_group[syn_id_key]:
                    unitary_EPSP_traces_dict[syn_group]['control'][syn_id][rec_name] = \
                        this_group[syn_id_key][rec_name][:]
    os.remove(temp_traces_path)

    for syn_group in compound_EPSP_traces_dict:
        compound_EPSP_traces_dict[syn_group]['expected'] = \
            get_expected_compound_EPSP_traces(unitary_EPSP_traces_dict[syn_group]['control'],
                                              syn_ids_dict[syn_group]['control'])

    soma_compound_EPSP_amp = defaultdict(lambda: defaultdict(list))
    initial_gain = defaultdict(list)
    integration_gain = defaultdict(list)
    for syn_group in compound_EPSP_traces_dict:
        for syn_condition in compound_EPSP_traces_dict[syn_group]:
            max_num_syns = max(compound_EPSP_traces_dict[syn_group][syn_condition].keys())
            for num_syns in range(1, max_num_syns + 1):
                soma_compound_EPSP_amp[syn_group][syn_condition].append(
                    np.max(compound_EPSP_traces_dict[syn_group][syn_condition][num_syns]['soma']))
        for syn_condition in context.syn_conditions:
            this_actual = np.array(soma_compound_EPSP_amp[syn_group][syn_condition])
            this_expected = np.array(soma_compound_EPSP_amp[syn_group]['expected'])
            this_ratio = np.divide(this_actual, this_expected)
            # Integration should be close to linear without gain for the first few synapses.
            initial_gain[syn_condition].append(np.mean(this_ratio[:2]))
            slope, intercept, r_value, p_value, std_err = stats.linregress(this_expected, this_actual)
            integration_gain[syn_condition].append(slope)

    features['initial_gain_control'] = np.mean(initial_gain['control'])
    features['initial_gain_AP5'] = np.mean(initial_gain['AP5'])
    features['integration_gain_control'] = np.mean(integration_gain['control'])
    features['integration_gain_AP5'] = np.mean(integration_gain['AP5'])

    if context.plot:
        fig, axes = plt.subplots(1, len(soma_compound_EPSP_amp))
        for i, syn_group in enumerate(soma_compound_EPSP_amp):
            if len(soma_compound_EPSP_amp) > 1:
                this_axes = axes[i]
            else:
                this_axes = axes
            for syn_condition in soma_compound_EPSP_amp[syn_group]:
                this_axes.plot(range(1, num_syns + 1), soma_compound_EPSP_amp[syn_group][syn_condition],
                               label=syn_condition)
            this_axes.set_title(syn_group)
            this_axes.legend(loc='best', frameon=False, framealpha=0.5)
        fig.show()

    if export:
        description = 'compound_EPSP_summary'
        with h5py.File(context.export_file_path, 'a') as f:
            if description not in f:
                f.create_group(description)
                f[description].attrs['enumerated'] = False
            group = f[description]
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

    return features


def get_objectives_synaptic_integration(features):
    """

    :param features: dict
    :return: tuple of dict
    """
    objectives = dict()
    for objective_name in context.objective_names:
        if objective_name not in features:
            return dict(), dict()
        objectives[objective_name] = \
            ((features[objective_name] - context.target_val[objective_name]) /
             context.target_range[objective_name]) ** 2.
    return features, objectives


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)