"""
Uses nested.optimize to tune frequency-dependent attenuation of EPSPs from dendrite to soma in dentate granule cells.

Requires a YAML file to specify required configuration parameters.
Requires use of a nested.parallel interface.
"""
__author__ = 'Aaron D. Milstein and Grace Ng'
from dentate.biophysics_utils import *
from nested.optimize_utils import *
from cell_utils import *
import click


context = Context()


@click.command()
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/optimize_DG_GC_iEPSP_propagation_config.yaml')
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
    config_optimize_interactive(__file__, config_file_path=config_file_path, output_dir=output_dir, export=export,
                       export_file_path=export_file_path, label=label, disp=disp)

    if run_tests:
        unit_tests_iEPSP()


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

    features, objectives = get_objectives_iEPSP_propagation(features, context.export)
    print 'params:'
    pprint.pprint(context.x0_dict)
    print 'features:'
    pprint.pprint(features)
    print 'objectives:'
    pprint.pprint(objectives)


def config_worker():
    """

    """
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
    syn_mech_name = 'EPSC'

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
    cell = get_biophys_cell(context.env, gid=context.gid, pop_name=context.cell_type, load_edges=False)
    init_biophysics(cell, reset_cable=True, from_file=True, mech_file_path=context.mech_file_path,
                    correct_cm=context.correct_for_spines, correct_g_pas=context.correct_for_spines, env=context.env,
                    verbose=verbose>1)
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
    if 'i_EPSC' not in context():
        context.i_EPSC = defaultdict()
    cell = context.cell
    sim = context.sim
    if not sim.has_rec('soma'):
        sim.append_rec(cell, cell.tree.root, name='soma', loc=0.5)
    if context.v_active not in context.i_holding['soma']:
        context.i_holding['soma'][context.v_active] = 0.
    if not sim.has_rec('dend'):
        dend, dend_loc = get_thickest_dend_branch(context.cell, 100., terminal=False)
        sim.append_rec(cell, dend, name='dend', loc=dend_loc)
    if 'dend' not in context.i_EPSC:
        context.i_EPSC['dend'] = -0.05

    equilibrate = context.equilibrate
    duration = context.duration

    if not sim.has_stim('holding'):
        sim.append_stim(cell, cell.tree.root, name='holding', loc=0.5, amp=0., delay=0., dur=duration)

    if 'i_syn' not in context():
        rec_dict = sim.get_rec('dend')
        node = rec_dict['node']
        loc = rec_dict['loc']
        seg = node.sec(loc)
        context.i_syn = make_unique_synapse_mech(context.syn_mech_name, seg)
        config_syn(context.syn_mech_name, context.env.synapse_attributes.syn_param_rules, syn=context.i_syn,
                   i_unit=context.i_EPSC['dend'], tau_rise=1., tau_decay=10.)
    if 'i_nc' not in context():
        context.i_nc, context.i_vs = mknetcon_vecstim(context.i_syn)

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
    offset_vm('soma', context, v_active, i_history=context.i_holding, dynamic=True)

    sim = context.sim
    sim.modify_stim('holding', dur=duration)
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
    offset_vm('soma', context, v_active, i_history=context.i_holding, dynamic=True)

    sim = context.sim
    sim.modify_stim('holding', dur=duration)
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


def get_objectives_iEPSP_propagation(features, export=False):
    """

    :param features: dict
    :param export: bool
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