"""
Uses nested.optimize to tune frequency-dependent attenuation of EPSPs from dendrite to soma in dentate granule cells.

Requires a YAML file to specify required configuration parameters.
Requires use of a nested.parallel interface.
"""
__author__ = 'Aaron D. Milstein and Grace Ng'
from biophysics_utils import *
from nested.optimize_utils import *
from optimize_cells_utils import *
import click


context = Context()


@click.command()
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/optimize_DG_GC_spiking_config.yaml')
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--export", is_flag=True)
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default=None)
@click.option("--verbose", type=int, default=2)
def main(config_file_path, output_dir, export, export_file_path, label, verbose):
    """

    :param config_file_path: str (path)
    :param output_dir: str (path)
    :param export: bool
    :param export_file_path: str
    :param label: str
    :param verbose: bool
    """
    # requires a global variable context: :class:'Context'
    context.update(locals())
    disp = verbose > 0
    config_interactive(context, __file__, config_file_path=config_file_path, output_dir=output_dir, export=export,
                       export_file_path=export_file_path, label=label, disp=disp)
    # Stage 0:
    args = []
    group_size = 1
    sequences = [[context.x0_array] * group_size] + args + [[context.export] * group_size]
    primitives = map(compute_features_spike_shape, *sequences)
    features = {key: value for feature_dict in primitives for key, value in feature_dict.iteritems()}

    # Stage 1:
    args = get_args_dynamic_fI(context.x0_array, features)
    group_size = len(args[0])
    sequences = [[context.x0_array] * group_size] + args + [[context.export] * group_size]
    primitives = map(compute_features_fI, *sequences)
    this_features = filter_features_fI(primitives, features, context.export)
    features.update(this_features)

    features, objectives = get_objectives_spiking(features)
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
    num_pulses = 5.
    sim_duration = {'long': equilibrate + (num_pulses - 1.) * ISI['long'] + stim_dur,
                    'short': equilibrate + (num_pulses - 1.) * ISI['short'] + stim_dur,
                    'unit': equilibrate + stim_dur}
    trace_baseline = 10.
    duration = max(sim_duration.values())
    dt = 0.025
    v_init = -77.
    v_active = -77.
    syn_mech_name = 'EPSC'

    target_iEPSP_amp = 1.  # mV
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
    context.sim = QuickSim(context.duration, cvode=cvode, daspk=daspk, dt=context.dt, verbose=verbose>1)
    context.spike_output_vec = h.Vector()
    cell.spike_detector.record(context.spike_output_vec)
    context.cell = cell


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
        dend, dend_loc = get_DG_GC_thickest_dend_branch(context.cell, 200., terminal=False)
        sim.append_rec(cell, dend, name='dend', loc=dend_loc)
    if context.v_active not in context.i_holding['dend']:
        context.i_holding['dend'][context.v_active] = 0.1
    if context.v_active not in context.i_EPSC['dend']:
        context.i_EPSC['dend'] = -0.15

    equilibrate = context.equilibrate
    duration = context.duration

    if not sim.has_stim('holding'):
        sim.append_stim(cell, cell.tree.root, name='holding', loc=0.5, amp=0., delay=0., dur=duration)

    if 'i_syn' not in context():
        rec = sim.get_rec('dend')
        node = rec.node
        loc = rec.loc
        seg = node.sec(loc)
        context.i_syn = add_unique_synapse(context.syn_mech_name, seg)
        config_syn(context.syn_mech_name, context.env.synapse_attributes.syn_param_rules, syn=context.i_syn,
                   i_unit=context.i_EPSC['dend'])
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

    rec = context.sim.get_rec('soma')['vec']
    t = np.arange(0., sim.tstop, dt)
    vm = np.interp(t, sim.tvec, rec)
    baseline = np.mean(vm[int((equilibrate - 3.) / dt):int((equilibrate - 1.) / dt)])
    vm -= baseline
    iEPSP_amp = np.max(vm[int(equilibrate / dt):])
    Err = ((iEPSP_amp - context.target_iEPSP_amp) / 0.01) ** 2.
    if context.verbose > 1:
        print 'iEPSP_amp_error: %s.i_unit: %.3f, soma iEPSP amp: %.2f; took %.1f s' % \
              (context.syn_mech_name, x[0], iEPSP_amp, time.time() - start_time)
    return Err


def compute_features_iEPSP_unit(x, export=False, plot=False):
    """

    :param x: array
    :param export: bool
    :param plot: bool
    :return: dict
    """
    start_time = time.time()
    config_sim_env(context)
    update_source_contexts(x, context)
    zero_na(context.cell)

    result = dict()
    sim = context.sim
    orig_cvode = sim.cvode
    sim.cvode = False
    dt = context.dt
    orig_dt = sim.dt
    sim.dt = dt
    orig_duration = sim.tstop
    duration = context.sim_duration['unit']
    sim.tstop = duration

    v_active = context.v_active
    offset_vm('soma', context, v_active)
    context.i_vs.play(h.Vector([context.equilibrate]))
    i_EPSC = context.i_EPSC['dend']
    bounds = [-0.3, -0.01]
    i_EPSC_result = scipy.optimize.minimize(iEPSP_amp_error, [i_EPSC], method='L-BFGS-B', bounds=bounds,
                                   options={'ftol': 1e-3, 'disp': context.verbose > 1, 'maxiter': 5})
    i_EPSC = i_EPSC_result.x[0]
    context.i_EPSC['dend'] = i_EPSC
    result['i_EPSC'] = i_EPSC

    title = 'iEPSP_unit_amp'
    description = 'i_EPSC: %.3f (nA)' % i_EPSC
    sim.parameters['duration'] = duration
    sim.parameters['title'] = title
    sim.parameters['description'] = description

    if context.verbose > 0:
        print 'compute_features_iEPSP_unit: pid: %i; %s: %s took %.3f s' % \
              (os.getpid(), title, description, time.time() - start_time)
    if plot:
        context.sim.plot()
    if export:
        context.sim.export_to_file(context.temp_output_path)
    sim.cvode = orig_cvode
    sim.dt = orig_dt
    sim.tstop = orig_duration
    context.i_vs.play(h.Vector())
    return result


def get_args_dynamic_iEPSP_propagation(x, features):
    """
    A nested map operation is required to compute iEPSP_propagation features. The arguments to be mapped depend on each
    set of parameters and prior features (dynamic).
    :param x: array
    :param features: dict
    :return: list of list
    """
    i_EPSC = features['i_EPSC']
    return [['long', 'short'], [i_EPSC, i_EPSC]]


def compute_features_iEPSP_propagation(x, ISI_key, i_EPSC, export=False, plot=False):
    """

    :param x:
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
    sim = context.sim
    orig_cvode = sim.cvode
    sim.cvode = False
    dt = context.dt
    orig_dt = sim.dt
    sim.dt = dt
    orig_duration = sim.tstop
    duration = context.sim_duration[ISI_key]
    ISI = context.ISI[ISI_key]
    sim.tstop = duration
    equilibrate = context.equilibrate
    context.i_EPSC['dend'] = i_EPSC

    v_active = context.v_active
    offset_vm('soma', context, v_active)

    context.i_vs.play(h.Vector([equilibrate + i * ISI for i in xrange(context.num_pulses)]))
    config_syn(context.syn_mech_name, context.env.synapse_attributes.syn_param_rules, syn=context.i_syn, i_unit=i_EPSC)
    sim.run(context.v_active)

    t = np.arange(0., duration, dt)
    soma_rec = sim.get_rec('soma')['vec']
    soma_vm = np.interp(t, sim.tvec, soma_rec)
    soma_baseline = np.mean(soma_vm[int((equilibrate - 3.) / dt):int((equilibrate - 1.) / dt)])
    soma_vm -= soma_baseline
    soma_iEPSP_amp = np.max(soma_vm[int(equilibrate / dt):])

    dend_rec = sim.get_rec('dend')['vec']
    dend_vm = np.interp(t, sim.tvec, dend_rec)
    dend_baseline = np.mean(dend_vm[int((equilibrate - 3.) / dt):int((equilibrate - 1.) / dt)])
    dend_vm -= dend_baseline
    dend_iEPSP_amp = np.max(dend_vm[int(equilibrate / dt):])

    result['iEPSP_amp_%s_ISI_soma' % ISI_key] = soma_iEPSP_amp
    result['iEPSP_amp_%s_ISI_dend' % ISI_key] = dend_iEPSP_amp
    result['iEPSP_ratio_%s_ISI' % ISI_key] = soma_iEPSP_amp / dend_iEPSP_amp

    description = 'ISI: %.1f' % ISI
    title = 'iEPSP_propagation'

    sim.parameters['duration'] = duration
    sim.parameters['title'] = title
    sim.parameters['description'] = description

    if context.verbose > 0:
        print 'compute_features_iEPSP_propagation: pid: %i; %s: %s took %.3f s' % \
              (os.getpid(), title, description, time.time() - start_time)
    if plot:
        context.sim.plot()
    if export:
        context.sim.export_to_file(context.temp_output_path)
    sim.cvode = orig_cvode
    sim.dt = orig_dt
    sim.tstop = orig_duration
    context.i_vs.play(h.Vector())
    return result


def get_objectives_spiking(features):
    """

    :param features: dict
    :return: tuple of dict
    """
    if features is None:  # No rheobase value found
        objectives = None
    else:
        objectives = {}
        rheobase = features['rheobase']
        for target in ['vm_th', 'ADP', 'AHP', 'rebound_firing', 'vm_stability', 'ais_delay',
                       'slow_depo', 'dend_bAP_ratio', 'soma_spike_amp', 'th_count']:
            # don't penalize AHP or slow_depo less than target
            if not ((target == 'AHP' and features[target] < context.target_val[target]) or
                        (target == 'slow_depo' and features[target] < context.target_val[target])):
                objectives[target] = ((context.target_val[target] - features[target]) /
                                      context.target_range[target]) ** 2.
            else:
                objectives[target] = 0.
        objectives['adi'] = 0.
        all_adi = []
        all_exp_adi = []
        for i, this_adi in enumerate(features['adi']):
            if this_adi is not None and features['exp_adi'] is not None:
                objectives['adi'] += ((this_adi - features['exp_adi'][i]) / (0.01 * features['exp_adi'][i])) ** 2.
                all_adi.append(this_adi)
                all_exp_adi.append(features['exp_adi'][i])
        features['adi'] = np.mean(all_adi)
        features['exp_adi'] = np.mean(all_exp_adi)
        num_increments = context.num_increments
        i_inj_increment = context.i_inj_increment
        target_f_I = [context.target_val['f_I_slope'] * np.log((rheobase + i_inj_increment * (i + 1)) / rheobase)
                      for i in xrange(num_increments)]
        f_I_residuals = [(features['f_I'][i] - target_f_I[i]) for i in xrange(num_increments)]
        features['f_I_residuals'] = np.mean(np.abs(f_I_residuals))
        objectives['f_I_slope'] = 0.
        for i in xrange(num_increments):
            objectives['f_I_slope'] += (f_I_residuals[i] / (0.01 * target_f_I[i])) ** 2.
        I_inj = [np.log((rheobase + i_inj_increment * (i + 1)) / rheobase) for i in xrange(num_increments)]
        slope, intercept, r_value, p_value, std_err = stats.linregress(I_inj, features['f_I'])
        features['f_I_slope'] = slope
        features.pop('f_I')
    return features, objectives


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)