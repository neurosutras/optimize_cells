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
        pass
        # unit_tests_iEPSP()


def unit_tests_iEPSP():
    """

    """
    features = dict()
    # Stage 0:
    args = get_args_dynamic_i_holding(context.x0_array, features)
    group_size = len(args[0])
    sequences = [[context.x0_array] * group_size] + args + [[context.export] * group_size] + \
                [[context.plot] * group_size]
    primitives = map(compute_features_iEPSP_i_unit, *sequences)
    features = {key: value for feature_dict in primitives for key, value in feature_dict.iteritems()}

    # Stage 1:
    args = get_args_dynamic_iEPSP_attenuation(context.x0_array, features)
    group_size = len(args[0])
    sequences = [[context.x0_array] * group_size] + args + [[context.export] * group_size] + \
                [[context.plot] * group_size]
    primitives = map(compute_features_iEPSP_attenuation, *sequences)
    this_features = {key: value for feature_dict in primitives for key, value in feature_dict.iteritems()}
    features.update(this_features)

    features, objectives = get_objectives_iEPSP_propagation(features)
    print 'params:'
    pprint.pprint(context.x0_dict)
    print 'features:'
    pprint.pprint(features)
    print 'objectives:'
    pprint.pprint(objectives)


def config_controller(export_file_path, output_dir, **kwargs):
    """

    :param export_file_path: str (path)
    :param output_dir: str (dir)
    """
    context.update(locals())
    context.update(kwargs)
    init_context()


def config_worker(update_context_funcs, param_names, default_params, feature_names, objective_names, target_val,
                  target_range, temp_output_path, export_file_path, output_dir, disp, mech_file_path, gid,
                  cell_type, correct_for_spines, **kwargs):
    """
    :param update_context_funcs: list of function references
    :param param_names: list of str
    :param default_params: dict
    :param feature_names: list of str
    :param objective_names: list of str
    :param target_val: dict
    :param target_range: dict
    :param temp_output_path: str
    :param export_file_path: str
    :param output_dir: str (dir path)
    :param disp: bool
    :param mech_file_path: str
    :param gid: int
    :param cell_type: str
    :param correct_for_spines: bool
    """
    context.update(locals())
    context.update(kwargs)
    if not context_has_sim_env(context):
        build_sim_env(context, **kwargs)


def context_has_sim_env(context):
    """

    :param context: :class:'Context
    :return: bool
    """
    return 'env' in context() and 'sim' in context() and 'cell' in context()


def init_context():
    """

    """
    ISI = {'long': 10., 'short': 1.}  # inter-stimulus interval for synaptic stim (ms)
    equilibrate = 250.  # time to steady-state
    stim_dur = 50.
    num_pulses = 5
    sim_duration = {'long': equilibrate + (num_pulses - 1.) * ISI['long'] + stim_dur,
                    'short': equilibrate + (num_pulses - 1.) * ISI['short'] + stim_dur,
                    'unit': equilibrate + stim_dur}
    trace_baseline = 10.
    duration = max(sim_duration.values())
    dt = 0.025
    v_init = -77.
    v_active = -77.

    seed_offset = 7. * 2e6
    local_random = random.Random()

    # for clustered inputs, num_syns corresponds to number of clustered inputs per branch
    num_syns = {'random': 30, 'clustered': 20}  # {'random': 30, 'clustered': 20}

    # number of branches to test temporal integration of clustered inputs
    num_clustered_branches = 2
    clustered_branch_names = ['clustered%i' % i for i in xrange(num_clustered_branches)]
    synapse_indexes = {'random': range(num_syns['random'])}
    for i, branch in enumerate(clustered_branch_names):
        synapse_indexes[branch] = range(num_syns['random'] + i * num_syns['clustered'],
                                        num_syns['random'] + (i + 1) * num_syns['clustered'])
    syn_conditions = ['control', 'AP5']
    ISI = {'units': 150., 'clustered': 1.1}  # inter-stimulus interval for synaptic stim (ms)
    units_per_sim = 5
    equilibrate = 250.  # time to steady-state
    stim_dur = 150.
    sim_duration = {'units': equilibrate + units_per_sim * ISI['units'],
                    'clustered': equilibrate + 200.,
                    'default': equilibrate + stim_dur}
    trace_baseline = 10.
    duration = max(sim_duration.values())

    NMDA_type = 'NMDA'
    syn_types = ['SatAMPA', NMDA_type]

    context.update(locals())


def build_sim_env(context, verbose=2, cvode=True, daspk=True, **kwargs):
    """

    :param context: :class:'Context'
    :param verbose: int
    :param cvode: bool
    :param daspk: bool
    """
    init_context()
    context.env = Env(comm=context.comm, **kwargs)
    configure_env(context.env)
    cell = get_biophys_cell(context.env, context.gid, context.cell_type)
    init_biophysics(cell, reset_cable=True, from_file=True, mech_file_path=context.mech_file_path,
                    correct_cm=context.correct_for_spines, correct_g_pas=context.correct_for_spines, env=context.env)
    init_syn_mech_attrs(context.cell, context.env, from_file=True)
    context.sim = QuickSim(context.duration, cvode=cvode, daspk=daspk, dt=context.dt, verbose=verbose>1)
    context.spike_output_vec = h.Vector()
    cell.spike_detector.record(context.spike_output_vec)
    context.cell = cell
    config_sim_env(context)


def config_sim_env(context):
    """

    :param context: :class:'Context'
    """
    context.local_random.seed(int(context.seed_offset + context.gid))
    if 'previous_module' in context() and context.previous_module == __file__:
        return
    init_context()
    if 'i_holding' not in context():
        context.i_holding = defaultdict(dict)

    cell = context.cell
    sim = context.sim
    if not sim.has_rec('soma'):
        sim.append_rec(cell, cell.tree.root, name='soma', loc=0.5)
    if context.v_active not in context.i_holding['soma']:
        context.i_holding['soma'][context.v_active] = 0.
    if not sim.has_rec('dend'):
        dend, dend_loc = get_DG_GC_thickest_dend_branch(context.cell, 200., terminal=False)
        sim.append_rec(cell, dend, name='dend', loc=dend_loc)

    equilibrate = context.equilibrate
    duration = context.duration

    if not sim.has_stim('holding'):
        sim.append_stim(cell, cell.tree.root, name='holding', loc=0.5, amp=0., delay=0., dur=duration)

    if 'local_syn_ids' not in context():
        syn_attrs = context.env.synapse_attributes
        syn_indexes = []
        for branch in cell.apical:
            this_syn_indexes = syn_attrs.sec_index_map[cell.gid][branch.index]
            this_syn_indexes = syn_attrs.get_filtered_syn_indexes(cell.gid, this_syn_indexes,
                                                                  syn_category='excitatory')
            if len(this_syn_indexes) > 1:
                candidates = [(branch.index, syn_id, exc_syn_index)
                              for (exc_syn_index, syn_id) in enumerate(this_synapse_attributes['syn_id'])]
                if branch.sec.L <= 10:
                    this_syn_pointer_list.extend(context.local_random.sample(candidates, 1))
                else:
                    this_num_syns = min(len(this_synapse_attributes['syn_locs']), int(branch.sec.L / 10.))
                    this_syn_pointer_list.extend(context.local_random.sample(candidates, this_num_syns))
            elif this_synapse_attributes['syn_locs']:
                this_syn_pointer_list.append(this_synapse_attributes['syn_id'][0])
        syn_pointer_list.extend(context.local_random.sample(this_syn_pointer_list, context.num_syns['random']))

        syn_indexes = syn_attrs.get_filtered_syn_indexes(context.cell.gid,
                                                         syn_types=[context.env.syntypes_dict['excitatory']],
                                                         swc_types=[context.env.swctypes_dict['apical']])

        context.local_syn_ids = syn_attrs.syn_id_attr_dict[context.gid]['syn_ids'][syn_indexes]
    """
    if 'i_syn' not in context():
        rec_dict = sim.get_rec('dend')
        node = rec_dict['node']
        loc = rec_dict['loc']
        seg = node.sec(loc)
        context.i_syn = add_unique_synapse(context.syn_mech_name, seg)
        config_syn(context.syn_mech_name, context.env.synapse_attributes.syn_param_rules, syn=context.i_syn,
                   i_unit=context.i_EPSC['dend'], tau_rise=0.1, tau_decay=5.)
    if 'i_nc' not in context():
        context.i_nc, context.i_vs = mknetcon_vecstim(context.i_syn)
    """

    sim.parameters['duration'] = duration
    sim.parameters['equilibrate'] = equilibrate
    context.previous_module = __file__


def iEPSP_amp_error(x):
    """

    :param x: array
    :return: float
    """
    start_time = time.time()
    config_syn(context.syn_mech_name, context.env.synapse_attributes.syn_param_rules, syn=context.i_syn, i_unit=x[0])

    dt = context.dt
    equilibrate = context.equilibrate
    sim = context.sim
    sim.run(context.v_active)

    vm = np.array(context.sim.get_rec('soma')['vec'])
    baseline = np.mean(vm[int((equilibrate - 3.) / dt):int((equilibrate - 1.) / dt)])
    vm -= baseline
    iEPSP_amp = np.max(vm[int(equilibrate / dt):])
    Err = ((iEPSP_amp - context.target_val['iEPSP_unit_amp']) / (0.01 * context.target_val['iEPSP_unit_amp'])) ** 2.
    if context.verbose > 1:
        print 'iEPSP_amp_error: %s.i_unit: %.3f, soma iEPSP amp: %.2f; took %.1f s' % \
              (context.syn_mech_name, x[0], iEPSP_amp, time.time() - start_time)
    return Err


def get_args_dynamic_i_holding(x, features):
    """
    A nested map operation is required to compute iEPSP features. The arguments to be mapped depend on prior features
    (dynamic).
    :param x: array
    :param features: dict
    :return: list of list
    """
    if 'i_holding' not in features:
        i_holding = context.i_holding
    else:
        i_holding = features['i_holding']
    return [[i_holding]]


def compute_features_iEPSP_i_unit(x, i_holding, export=False, plot=False):
    """

    :param x: array
    :param i_holding: defaultdict(dict: float)
    :param export: bool
    :param plot: bool
    :return: dict
    """
    start_time = time.time()
    config_sim_env(context)
    update_source_contexts(x, context)
    zero_na(context.cell)

    dt = context.dt
    duration = context.sim_duration['unit']

    v_active = context.v_active
    context.i_holding = i_holding
    offset_vm('soma', context, v_active, i_history=context.i_holding)

    sim = context.sim
    sim.backup_state()
    sim.set_state(dt=dt, tstop=duration, cvode=False)

    context.i_vs.play(h.Vector([context.equilibrate]))
    i_EPSC = context.i_EPSC['dend']
    bounds = [(-0.3, -0.01)]
    i_EPSC_result = scipy.optimize.minimize(iEPSP_amp_error, [i_EPSC], method='L-BFGS-B', bounds=bounds,
                                   options={'ftol': 1e-3, 'disp': context.verbose > 1, 'maxiter': 5})
    i_EPSC = i_EPSC_result.x[0]
    context.i_EPSC['dend'] = i_EPSC

    result = dict()
    result['iEPSP_i_unit'] = i_EPSC
    result['i_holding'] = context.i_holding

    title = 'iEPSP'
    description = 'i_unit: %.3f (nA)' % i_EPSC
    sim.parameters['duration'] = duration
    sim.parameters['title'] = title
    sim.parameters['description'] = description

    if context.verbose > 0:
        print 'compute_features_iEPSP_i_unit: pid: %i; %s: %s took %.3f s' % \
              (os.getpid(), title, description, time.time() - start_time)
    if plot:
        context.sim.plot()
    if export:
        context.sim.export_to_file(context.temp_output_path)
    sim.restore_state()
    context.i_vs.play(h.Vector())
    return result


def get_args_dynamic_iEPSP_attenuation(x, features):
    """
    A nested map operation is required to compute iEPSP_propagation features. The arguments to be mapped depend on
    prior features (dynamic).
    :param x: array
    :param features: dict
    :return: list of list
    """
    if 'i_holding' not in features:
        i_holding = context.i_holding
    else:
        i_holding = features['i_holding']
    i_EPSC = features['iEPSP_i_unit']
    return [[i_holding] * 2, ['long', 'short'], [i_EPSC, i_EPSC]]


def compute_features_iEPSP_attenuation(x, i_holding, ISI_key, i_EPSC, export=False, plot=False):
    """

    :param x:
    :param i_holding: defaultdict(dict: float)
    :param ISI_key: str
    :param i_EPSC: float
    :param export:
    :param plot:
    :return: dict
    """
    start_time = time.time()
    config_sim_env(context)
    update_source_contexts(x, context)
    zero_na(context.cell)

    result = dict()

    dt = context.dt
    duration = context.sim_duration[ISI_key]
    ISI = context.ISI[ISI_key]
    equilibrate = context.equilibrate
    context.i_EPSC['dend'] = i_EPSC

    v_active = context.v_active
    context.i_holding = i_holding
    offset_vm('soma', context, v_active, i_history=context.i_holding)

    sim = context.sim
    sim.backup_state()
    sim.set_state(dt=dt, tstop=duration, cvode=False)

    context.i_vs.play(h.Vector([equilibrate + i * ISI for i in xrange(context.num_pulses)]))
    config_syn(context.syn_mech_name, context.env.synapse_attributes.syn_param_rules, syn=context.i_syn, i_unit=i_EPSC)
    sim.run(v_active)

    soma_vm = np.array(sim.get_rec('soma')['vec'])
    soma_baseline = np.mean(soma_vm[int((equilibrate - 3.) / dt):int((equilibrate - 1.) / dt)])
    soma_vm -= soma_baseline
    soma_iEPSP_amp = np.max(soma_vm[int(equilibrate / dt):])

    dend_vm = np.array(sim.get_rec('dend')['vec'])
    dend_baseline = np.mean(dend_vm[int((equilibrate - 3.) / dt):int((equilibrate - 1.) / dt)])
    dend_vm -= dend_baseline
    dend_iEPSP_amp = np.max(dend_vm[int(equilibrate / dt):])

    result['iEPSP_attenuation_%s' % ISI_key] = soma_iEPSP_amp / dend_iEPSP_amp

    title = 'iEPSP_attenuation'
    description = 'ISI: %.1f' % ISI
    sim.parameters['duration'] = duration
    sim.parameters['title'] = title
    sim.parameters['description'] = description

    if context.verbose > 0:
        print 'compute_features_iEPSP_attenuation: pid: %i; %s: %s took %.3f s' % \
              (os.getpid(), title, description, time.time() - start_time)
    if plot:
        context.sim.plot()
    if export:
        context.sim.export_to_file(context.temp_output_path)
    sim.restore_state()
    context.i_vs.play(h.Vector())
    return result


def get_objectives_iEPSP_propagation(features):
    """

    :param features: dict
    :return: tuple of dict
    """
    objectives = dict()
    for ISI_key in context.ISI:
        target = 'iEPSP_attenuation_%s' % ISI_key
        objectives[target] = \
            ((features[target] - context.target_val[target]) / (0.01 * context.target_val[target])) ** 2.
    return features, objectives


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)