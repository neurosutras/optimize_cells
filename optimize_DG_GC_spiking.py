"""
Uses nested.optimize to tune somatodendritic spike shape, f-I curve, and spike adaptation in dentate granule cells.

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
              default='config/optimize_DG_GC_spiking_config.yaml')
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--export", is_flag=True)
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default=None)
@click.option("--verbose", type=int, default=2)
@click.option("--plot", is_flag=True)
@click.option("--debug", is_flag=True)
@click.option("--run-tests", is_flag=True)
def main(config_file_path, output_dir, export, export_file_path, label, verbose, plot, debug, run_tests):
    """

    :param config_file_path: str (path)
    :param output_dir: str (path)
    :param export: bool
    :param export_file_path: str
    :param label: str
    :param verbose: bool
    :param plot: bool
    :param debug: bool
    :param run_tests: bool
    """
    # requires a global variable context: :class:'Context'
    context.update(locals())
    disp = verbose > 0
    #load context with relative bounds, parameters, paths, etc. for simulation
    config_interactive(context, __file__, config_file_path=config_file_path, output_dir=output_dir, export=export,
                       export_file_path=export_file_path, label=label, disp=disp)

    if debug:
        # add_diagnostic_recordings(context)
        add_complete_axon_recordings(context)

    if run_tests:
        unit_tests_spiking(context)


def unit_tests_spiking(context):
    """

    :param context: :class:'Context'
    """
    features = dict()
    # Stage 0: Get shape of single spike at rheobase in soma and axon
    args = get_args_dynamic_i_holding(context.x0_array, features)
    group_size = len(args[0])
    sequences = [[context.x0_array] * group_size] + args + [[context.export] * group_size] + \
                [[context.plot] * group_size]
    primitives = map(compute_features_spike_shape, *sequences)
    features = {key: value for feature_dict in primitives for key, value in feature_dict.iteritems()}

    # Stage 1: Run simulations with a range of amplitudes of step current injections to the soma
    args = get_args_dynamic_fI(context.x0_array, features)
    group_size = len(args[0])
    sequences = [[context.x0_array] * group_size] + args + [[context.export] * group_size] + \
                [[context.plot] * group_size]
    primitives = map(compute_features_fI, *sequences) #compute features for each sequence
    this_features = filter_features_fI(primitives, features, context.export)
    features.update(this_features)

    # Stage 2: Run simulations with a range of amplitudes of step current injections to the soma
    args = get_args_dynamic_spike_adaptation(context.x0_array, features)
    group_size = len(args[0])
    sequences = [[context.x0_array] * group_size] + args + [[context.export] * group_size] + \
                [[context.plot] * group_size]
    primitives = map(compute_features_spike_adaptation, *sequences)  # compute features for each sequence
    this_features = filter_features_spike_adaptation(primitives, features, context.export)
    features.update(this_features)

    # Stage 3: Run simulations with a range of amplitudes of step current injections to the dendrite
    args = get_args_dynamic_dend_spike(context.x0_array, features)
    group_size = len(args[0])
    sequences = [[context.x0_array] * group_size] + args + [[context.export] * group_size] + \
                [[context.plot] * group_size]
    primitives = map(compute_features_dend_spike, *sequences)
    this_features = filter_features_dend_spike(primitives, features, context.export)
    features.update(this_features)

    features, objectives = get_objectives_spiking(features)
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
    equilibrate = 250.  # time to steady-state
    default_stim_dur = 200.  # ms
    stim_dur_f_I = 1000.
    stim_dur_spike_adaptation = 100.
    duration = equilibrate + default_stim_dur
    dend_spike_stim_dur = 5.
    dend_spike_duration = equilibrate + dend_spike_stim_dur + 10.
    dt = 0.025
    th_dvdt = 10.
    dend_th_dvdt = 30.
    v_init = -77.
    v_active = -77.
    i_th_start = 0.2
    i_th_max = 0.4

    # GC experimental spike adaptation data from:
    # Gao, T. M., Howard, E. M., & Xu, Z. C. (1998). Transient neurophysiological changes in CA3 neurons and dentate
    # granule cells after severe forebrain ischemia in vivo. Journal of Neurophysiology, 80(6), 2860-2869
    # http://doi.org/10.1152/jn.1998.80.6.2860

    exp_i_inj_amp_spike_adaptation_0 = [0.5 + 0.1 * i for i in xrange(12)]  # nA
    exp_ISI1_array_0 = [7.978142077, 5.847797063, 4.818481848, 3.828671329, 4.235976789, 3.914209115, 3.465189873,
                      3.549432739, 3.097595474, 3.043780403, 2.95148248, 2.95148248]
    exp_ISI2_array_0 = [8.941684665, 9.179600887, 9.430523918, 6.169895678, 6.561014263, 5.369649805, 4.842105263,
                      4.228804903, 4.316996872, 3.924170616, 3.822714681, 3.584415584]
    fit_params_ISI_0 = [-4., 4.]
    exp_fit_params_ISI1, pcov = scipy.optimize.curve_fit(log10_fit, exp_i_inj_amp_spike_adaptation_0, exp_ISI1_array_0,
                                                        fit_params_ISI_0)
    exp_fit_params_ISI2, pcov = scipy.optimize.curve_fit(log10_fit, exp_i_inj_amp_spike_adaptation_0, exp_ISI2_array_0,
                                                         fit_params_ISI_0)
    i_inj_increment_spike_adaptation = 0.15
    num_increments_spike_adaptation = 4

    exp_rheobase_spike_adaptation = 0.28  # nA
    i_inj_relative_amp_array_spike_adaptation = np.array([0.25 + i_inj_increment_spike_adaptation * i
                                                          for i in xrange(num_increments_spike_adaptation)])
    exp_i_inj_amp_array_spike_adaptation = np.add(exp_rheobase_spike_adaptation,
                                                  i_inj_relative_amp_array_spike_adaptation)
    exp_ISI1_array = log10_fit(exp_i_inj_amp_array_spike_adaptation, *exp_fit_params_ISI1)
    exp_ISI2_array = log10_fit(exp_i_inj_amp_array_spike_adaptation, *exp_fit_params_ISI2)

    # GC experimental f-I data from:
    # Yun, S. H., Gamkrelidze, G., Stine, W. B., Sullivan, P. M., Pasternak, J. F., LaDu, M. J., & Trommer, B. L.
    # (2006). Amyloid-beta1-42 reduces neuronal excitability in mouse dentate gyrus. Neuroscience Letters, 403(1-2),
    # 162-165. http://doi.org/10.1016/j.neulet.2006.04.065
    i_inj_increment_f_I = 0.05
    num_increments_f_I = 6
    rate_at_rheobase = 5.  # Hz, corresponds to 1 spike in a 200 ms current injection

    exp_i_inj_amp_f_I_0 = [0.1 + 0.05 * i for i in xrange(4)]  # nA
    exp_rate_f_I_0 = [5.547235467, 15.19694751, 20.28774047, 24.0119589]  # inferred from 100 ms and 1 s i_inj
    fit_params_f_I_0 = [50., 10.]

    exp_fit_params_f_I, pcov = scipy.optimize.curve_fit(log10_fit, exp_i_inj_amp_f_I_0, exp_rate_f_I_0,
                                                        fit_params_f_I_0)
    exp_rheobase_f_I = inverse_log10_fit(rate_at_rheobase, *exp_fit_params_f_I)
    i_inj_relative_amp_array_f_I = np.array([i_inj_increment_f_I * i for i in xrange(1, num_increments_f_I + 1)])
    exp_i_inj_amp_array_f_I = np.add(exp_rheobase_f_I, i_inj_relative_amp_array_f_I)
    exp_rate_f_I_array = log10_fit(exp_i_inj_amp_array_f_I, *exp_fit_params_f_I)
    
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
    configure_hoc_env(context.env)
    cell = get_biophys_cell(context.env, gid=context.gid, pop_name=context.cell_type)
    init_biophysics(cell, reset_cable=True, from_file=True, mech_file_path=context.mech_file_path,
                    correct_cm=context.correct_for_spines, correct_g_pas=context.correct_for_spines, env=context.env)
    context.sim = QuickSim(context.duration, cvode=cvode, daspk=daspk, dt=context.dt, verbose=verbose>1)
    context.spike_output_vec = h.Vector()
    cell.spike_detector.record(context.spike_output_vec)
    context.cell = cell
    config_sim_env(context)


def get_spike_adaptation_indexes(spike_times):
    """
    Spike rate adaptation refers to changes in inter-spike intervals during a spike train. Larger values indicate
    larger increases in inter-spike intervals.
    :param spike_times: list of float
    :return: array
    """
    if len(spike_times) < 3:
        return None
    isi = np.diff(spike_times)
    adi = []
    for i in xrange(len(isi) - 1):
        adi.append((isi[i + 1] - isi[i]) / (isi[i + 1] + isi[i]))
    return np.array(adi)


def config_sim_env(context):
    """

    :param context: :class:'Context'
    """
    if 'previous_module' in context() and context.previous_module == __file__:
        return
    init_context()
    if 'i_holding' not in context():
        context.i_holding = defaultdict(dict)
    # if 'i_th_history' not in context():
    #    context.i_th_history = defaultdict(dict)
    cell = context.cell
    sim = context.sim
    if not sim.has_rec('soma'):
        sim.append_rec(cell, cell.tree.root, name='soma', loc=0.5)
    if context.v_active not in context.i_holding['soma']:
        context.i_holding['soma'][context.v_active] = 0.
    # if context.v_active not in context.i_th_history['soma']:
    #    context.i_th_history['soma'][context.v_active] = context.i_th_start
    if not sim.has_rec('dend'):
        dend, dend_loc = get_DG_GC_thickest_dend_branch(context.cell, 200., terminal=False)
        sim.append_rec(cell, dend, name='dend', loc=dend_loc)
    if not sim.has_rec('ais'):
        sim.append_rec(cell, cell.ais[0], name='ais', loc=1.)
    if not sim.has_rec('axon'):
        axon_seg_locs = [seg.x for seg in cell.axon[0].sec]
        sim.append_rec(cell, cell.axon[0], name='axon', loc=axon_seg_locs[0])

    equilibrate = context.equilibrate
    stim_dur = context.default_stim_dur
    duration = context.duration

    if not sim.has_stim('step'):
        sim.append_stim(cell, cell.tree.root, name='step', loc=0.5, amp=0., delay=equilibrate, dur=stim_dur)
    if not sim.has_stim('holding'):
        sim.append_stim(cell, cell.tree.root, name='holding', loc=0.5, amp=0., delay=0., dur=duration)

    sim.parameters['duration'] = duration
    sim.parameters['equilibrate'] = equilibrate
    context.previous_module = __file__


def get_args_dynamic_i_holding(x, features):
    """
    A nested map operation is required to compute spike_shape features. The arguments to be mapped depend on prior
    features (dynamic).
    :param x: array
    :param features: dict
    :return: list of list
    """
    if 'i_holding' not in features:
        i_holding = context.i_holding
    else:
        i_holding = features['i_holding']
    return [[i_holding]]


def compute_features_spike_shape(x, i_holding, export=False, plot=False):
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

    equilibrate = context.equilibrate
    dt = context.dt
    stim_dur = context.default_stim_dur
    duration = equilibrate + stim_dur
    v_active = context.v_active
    sim = context.sim
    context.i_holding = i_holding
    offset_vm('soma', context, v_active, i_history=context.i_holding)
    spike_times = np.array(context.cell.spike_detector.get_recordvec())
    if np.any(spike_times < equilibrate):
        if context.verbose > 0:
            print 'compute_features_spike_shape: pid: %i; aborting - spontaneous firing' % (os.getpid())
        return dict()

    result = dict()
    rec_dict = sim.get_rec('soma')
    loc = rec_dict['loc']
    node = rec_dict['node']
    soma_rec = rec_dict['vec']

    """
    if v_active not in context.i_th_history['soma']:
        i_th = context.i_th_start
        context.i_th_history['soma'][v_active] = i_th
    else:
        i_th = context.i_th_history['soma'][v_active]
    """
    i_th = context.i_th_start

    sim.modify_stim('step', node=node, loc=loc, dur=stim_dur, amp=i_th)
    sim.backup_state()
    sim.set_state(dt=dt, tstop=duration, cvode=False)
    sim.run(v_active)
    spike_times = np.array(context.cell.spike_detector.get_recordvec())

    spike = np.any(spike_times > equilibrate)
    if spike:
        i_inc = -0.01
        delta_str = 'decreased'
        while spike and i_th > 0.:
            i_th += i_inc
            sim.modify_stim('step', amp=i_th)
            sim.run(v_active)
            spike_times = np.array(context.cell.spike_detector.get_recordvec())
            if sim.verbose:
                print 'compute_features_spike_shape: pid: %i; %s; %s i_th to %.3f nA; num_spikes: %i' % \
                      (os.getpid(), 'soma', delta_str, i_th, len(spike_times))
            spike = np.any(spike_times > equilibrate)
    if i_th <= 0.:
        if context.verbose > 0:
            print 'compute_features_spike_shape: pid: %i; aborting - spontaneous firing' % (os.getpid())
        return dict()

    i_inc = 0.01
    delta_str = 'increased'
    while not spike:  # Increase step current until spike. Resting Vm is v_active.
        i_th += i_inc
        if i_th > context.i_th_max:
            if context.verbose > 0:
                print 'compute_features_spike_shape: pid: %i; aborting - rheobase outside target range' % (os.getpid())
            return dict()
        sim.modify_stim('step', amp=i_th)
        sim.run(v_active)
        spike_times = np.array(context.cell.spike_detector.get_recordvec())
        if sim.verbose:
            print 'compute_features_spike_shape: pid: %i; %s; %s i_th to %.3f nA; num_spikes: %i' % \
                  (os.getpid(), 'soma', delta_str, i_th, len(spike_times))
        spike = np.any(spike_times > equilibrate)

    # context.i_th_history['soma'][v_active] = i_th

    soma_vm = np.array(soma_rec)  # Get voltage waveforms of spike from various subcellular compartments
    ais_vm = np.array(sim.get_rec('ais')['vec'])
    axon_vm = np.array(sim.get_rec('axon')['vec'])
    dend_vm = np.array(sim.get_rec('dend')['vec'])

    title = 'spike_shape'  # simulation metadata to be exported to file along with the data
    description = 'rheobase: %.3f' % i_th
    sim.parameters['amp'] = i_th
    sim.parameters['title'] = title
    sim.parameters['description'] = description
    sim.parameters['duration'] = duration

    spike_shape_dict = get_spike_shape(soma_vm, spike_times, context)
    if spike_shape_dict is None:
        if context.verbose > 0:
            print 'compute_features_spike_shape: pid: %i; aborting - problem analyzing spike shape' % (os.getpid())
        return dict()
    peak = spike_shape_dict['v_peak']
    threshold = spike_shape_dict['th_v']
    fAHP = spike_shape_dict['fAHP']
    mAHP = spike_shape_dict['mAHP']
    ADP = spike_shape_dict['ADP']
    spike_detector_delay = spike_shape_dict['spike_detector_delay']

    result['soma_spike_amp'] = peak - threshold
    result['vm_th'] = threshold
    result['fAHP'] = fAHP
    result['mAHP'] = mAHP
    result['ADP'] = ADP
    result['rheobase'] = i_th
    result['i_holding'] = context.i_holding
    result['spike_detector_delay'] = spike_detector_delay

    if context.verbose > 1:
        print 'compute_features_spike_shape: pid: %i; spike detector delay: %.3f (ms)' % \
              (os.getpid(), spike_detector_delay)

    start = int((equilibrate + 1.) / dt)
    th_x = np.where(soma_vm[start:] >= threshold)[0][0] + start
    if len(spike_times) > 1:
        end = min(th_x + int(10. / dt), int((spike_times[1] - spike_detector_delay) / dt))
    else:
        end = th_x + int(10. / dt)
    dend_peak = np.max(dend_vm[th_x:end])
    dend_pre = np.mean(dend_vm[th_x - int(3. / dt):th_x - int(1. / dt)])
    result['dend_bAP_ratio'] = (dend_peak - dend_pre) / result['soma_spike_amp'] #back propagation ratio

    # calculate AIS delay
    soma_dvdt = np.gradient(soma_vm, dt)
    ais_dvdt = np.gradient(ais_vm, dt)
    axon_dvdt = np.gradient(axon_vm, dt)
    left = th_x - int(2. / dt)
    right = th_x + int(5. / dt)
    soma_peak = np.max(soma_dvdt[left:right])
    soma_peak_t = np.where(soma_dvdt[left:right] == soma_peak)[0][0] * dt
    ais_peak = np.max(ais_dvdt[left:right])
    ais_peak_t = np.where(ais_dvdt[left:right] == ais_peak)[0][0] * dt
    axon_peak = np.max(axon_dvdt[left:right])
    axon_peak_t = np.where(axon_dvdt[left:right] == axon_peak)[0][0] * dt
    result['ais_delay'] = max(0., ais_peak_t + dt - soma_peak_t) + max(0., ais_peak_t + dt - axon_peak_t)

    if context.verbose > 0:
        print 'compute_features_spike_shape: pid: %i; %s: %s took %.1f s; vm_th: %.1f' % \
              (os.getpid(), title, description, time.time() - start_time, threshold)
    if plot:
        sim.plot()
    if export:
        context.sim.export_to_file(context.temp_output_path)
    sim.restore_state()
    sim.modify_stim('step', amp=0.)
    return result


def get_args_dynamic_fI(x, features):
    """
    A nested map operation is required to compute fI features. The arguments to be mapped depend on prior features
    (dynamic).
    :param x: array
    :param features: dict
    :return: list of list
    """
    if 'i_holding' not in features:
        i_holding = context.i_holding
    else:
        i_holding = features['i_holding']
    rheobase = features['rheobase']
    spike_detector_delay = features['spike_detector_delay']
    
    # Calculate firing rates for a range of i_inj amplitudes using a stim duration of 1 s
    group_size = context.num_increments_f_I
    return [[i_holding] * group_size, [spike_detector_delay] * group_size, [rheobase] * group_size,
            context.i_inj_relative_amp_array_f_I, [False] * (group_size - 1) + [True]]


def compute_features_fI(x, i_holding, spike_detector_delay, rheobase, relative_amp, extend_dur=False, export=False,
                        plot=False):
    """

    :param x: array
    :param i_holding: defaultdict(dict: float)
    :param spike_detector_delay: float (ms)
    :param rheobase: float
    :param relative_amp: float
    :param extend_dur: bool
    :param export: bool
    :param plot: bool
    :return: dict
    """
    start_time = time.time()
    config_sim_env(context)
    update_source_contexts(x, context)

    v_active = context.v_active
    context.i_holding = i_holding
    offset_vm('soma', context, v_active, i_history=context.i_holding)
    sim = context.sim
    dt = context.dt
    stim_dur = context.stim_dur_f_I
    equilibrate = context.equilibrate

    if extend_dur:
        # extend duration of simulation to examine rebound
        duration = equilibrate + stim_dur + 100.
    else:
        duration = equilibrate + stim_dur

    rec_dict = sim.get_rec('soma')
    loc = rec_dict['loc']
    node = rec_dict['node']
    soma_rec = rec_dict['vec']

    amp = rheobase + relative_amp

    title = 'f_I'
    description = 'step current amp: %.3f' % amp
    sim.parameters['duration'] = duration
    sim.parameters['title'] = title
    sim.parameters['description'] = description
    sim.parameters['i_amp'] = amp
    sim.backup_state()
    sim.set_state(dt=dt, tstop=duration, cvode=False)
    sim.modify_stim('step', node=node, loc=loc, dur=stim_dur, amp=amp)
    sim.run(v_active)

    spike_times = np.subtract(np.array(context.cell.spike_detector.get_recordvec()), equilibrate + spike_detector_delay)

    result = dict()
    result['i_amp'] = amp
    vm = np.array(soma_rec)

    """
    # Make sure spike_detector_delay is accurately transposing threshold crossing at the spike detector location to
    # spike onset time at the soma
    if plot:
        fig = plt.figure()
        plt.plot(sim.tvec, vm)
        axon_indexes = [int(spike_time / dt) for spike_time in np.array(context.cell.spike_detector.get_recordvec())]
        plt.scatter(np.array(sim.tvec)[axon_indexes], vm[axon_indexes], c='r')
        soma_indexes = [int((spike_time + equilibrate) / dt) for spike_time in spike_times]
        plt.scatter(np.array(sim.tvec)[soma_indexes], vm[soma_indexes], c='k')
        fig.show()
    """

    if extend_dur:
        vm_rest = np.mean(vm[int((equilibrate - 3.) / dt):int((equilibrate - 1.) / dt)])
        v_after = np.max(vm[-int(50. / dt):-1])
        vm_stability = abs(v_after - vm_rest)
        result['vm_stability'] = vm_stability
        result['rebound_firing'] = len(np.where(spike_times > stim_dur)[0])
        last_spike_time = spike_times[np.where(spike_times < stim_dur)[0][-1]]
        last_spike_index = int((last_spike_time + equilibrate) / dt)
        start = last_spike_index
        dvdt = np.gradient(vm, dt)
        th_x_indexes = np.where(dvdt[start:] > context.th_dvdt)[0]
        if th_x_indexes.any():
            end = start + th_x_indexes[0] - int(1.6 / dt)
            vm_th_late = np.mean(vm[end - int(0.1 / dt):end])
            result['vm_th_late'] = vm_th_late

    spike_times = spike_times[np.where(spike_times < stim_dur)[0]]
    result['spike_times'] = spike_times

    if context.verbose > 0:
        print 'compute_features_fI: pid: %i; %s: %s took %.1f s; num_spikes: %i' % \
              (os.getpid(), title, description, time.time() - start_time, len(spike_times))
    if plot:
        sim.plot()
    if export:
        context.sim.export_to_file(context.temp_output_path)
    sim.restore_state()
    sim.modify_stim('step', amp=0.)
    return result


def filter_features_fI(primitives, current_features, export=False):
    """

    :param primitives: list of dict (each dict contains results from a single simulation)
    :param current_features: dict
    :param export: bool
    :return: dict
    """
    stim_dur = context.stim_dur_f_I
    rheobase = current_features['rheobase']

    new_features = dict()
    i_relative_amp = [this_dict['i_amp'] - rheobase for this_dict in primitives]
    rate = []

    indexes = range(len(i_relative_amp))
    indexes.sort(key=i_relative_amp.__getitem__)
    i_relative_amp = map(i_relative_amp.__getitem__, indexes)
    for i in indexes:
        this_dict = primitives[i]
        if 'vm_stability' in this_dict:
            new_features['vm_stability'] = this_dict['vm_stability']
        if 'rebound_firing' in this_dict:
            new_features['rebound_firing'] = this_dict['rebound_firing']
        if 'vm_th_late' in this_dict:
            new_features['slow_depo'] = abs(this_dict['vm_th_late'] - current_features['vm_th'])
        spike_times = this_dict['spike_times']
        this_rate = len(spike_times) / stim_dur * 1000.
        rate.append(this_rate)
    new_features['f_I_rate'] = rate
    if 'slow_depo' not in new_features:
        feature_name = 'slow_depo'
        if context.verbose > 0:
            print 'filter_features_fI: pid: %i; aborting - failed to compute required feature: %s' % \
                  (os.getpid(), feature_name)
        return dict()

    if export:
        description = 'f_I'
        with h5py.File(context.export_file_path, 'a') as f:
            if description not in f:
                f.create_group(description)
                f[description].attrs['enumerated'] = False
            group = f[description]
            group.attrs['rheobase'] = rheobase
            group.attrs['exp_rheobase'] = context.exp_rheobase_f_I
            group.create_dataset('i_relative_amp', compression='gzip', data=i_relative_amp)
            group.create_dataset('rate', compression='gzip', data=rate)
            group.create_dataset('exp_rate', compression='gzip', data=context.exp_rate_f_I_array)
    return new_features


def get_args_dynamic_spike_adaptation(x, features):
    """
    A nested map operation is required to compute spike adaptation features. The arguments to be mapped depend on prior
    features (dynamic).
    :param x: array
    :param features: dict
    :return: list of list
    """
    if 'i_holding' not in features:
        i_holding = context.i_holding
    else:
        i_holding = features['i_holding']
    rheobase = features['rheobase']
    spike_detector_delay = features['spike_detector_delay']

    # Calculate first and second inter-spike intervals (ISIs) for a range of of i_inj amplitudes using a stim duration
    # of 100 ms
    group_size = context.num_increments_spike_adaptation
    return [[i_holding] * group_size, [spike_detector_delay] * group_size, [rheobase] * group_size,
            context.i_inj_relative_amp_array_spike_adaptation]


def compute_features_spike_adaptation(x, i_holding, spike_detector_delay, rheobase, relative_amp, export=False,
                                      plot=False):
    """

    :param x: array
    :param i_holding: defaultdict(dict: float)
    :param spike_detector_delay: float (ms)
    :param rheobase: float
    :param relative_amp: float
    :param export: bool
    :param plot: bool
    :return: dict
    """
    start_time = time.time()
    config_sim_env(context)
    update_source_contexts(x, context)

    v_active = context.v_active
    context.i_holding = i_holding
    offset_vm('soma', context, v_active, i_history=context.i_holding)
    sim = context.sim
    dt = context.dt
    stim_dur = context.stim_dur_spike_adaptation
    equilibrate = context.equilibrate
    duration = equilibrate + stim_dur

    rec_dict = sim.get_rec('soma')
    loc = rec_dict['loc']
    node = rec_dict['node']
    soma_rec = rec_dict['vec']

    amp = rheobase + relative_amp

    title = 'spike_adaptation'
    description = 'step current amp: %.3f' % amp
    sim.parameters['duration'] = duration
    sim.parameters['title'] = title
    sim.parameters['description'] = description
    sim.parameters['i_amp'] = amp
    sim.backup_state()
    sim.set_state(dt=dt, tstop=duration, cvode=False)
    sim.modify_stim('step', node=node, loc=loc, dur=stim_dur, amp=amp)
    sim.run(v_active)

    spike_times = np.subtract(np.array(context.cell.spike_detector.get_recordvec()), equilibrate + spike_detector_delay)

    result = dict()
    result['i_amp'] = amp
    if len(spike_times) >= 3:
        result['ISI1'] = spike_times[1] - spike_times[0]
        result['ISI2'] = spike_times[2] - spike_times[1]
        if context.verbose > 0:
            print 'compute_features_spike_adaptation: pid: %i; %s: %s took %.1f s; ISI1: %.1f; ISI2: %.1f' % \
                  (os.getpid(), title, description, time.time() - start_time, result['ISI1'], result['ISI2'])
    else:
        if context.verbose > 0:
            print 'compute_features_spike_adaptation: pid: %i; %s: %s took %.1f s; not enough spikes to compute ISI1 ' \
                  'and ISI2' % (os.getpid(), title, description, time.time() - start_time)

    if plot:
        sim.plot()
    if export:
        context.sim.export_to_file(context.temp_output_path)
    sim.restore_state()
    sim.modify_stim('step', amp=0.)
    return result


def filter_features_spike_adaptation(primitives, current_features, export=False):
    """

    :param primitives: list of dict (each dict contains results from a single simulation)
    :param current_features: dict
    :param export: bool
    :return: dict
    """
    rheobase = current_features['rheobase']

    new_features = dict()
    i_relative_amp = [this_dict['i_amp'] - rheobase for this_dict in primitives]
    model_ISI1 = []
    model_ISI2 = []

    indexes = range(len(i_relative_amp))
    indexes.sort(key=i_relative_amp.__getitem__)
    i_relative_amp = map(i_relative_amp.__getitem__, indexes)
    for i in indexes:
        this_dict = primitives[i]
        if 'ISI1' not in this_dict or 'ISI2' not in this_dict:
            if context.verbose > 0:
                print 'filter_features_spike_adaptation: pid: %i; aborting - failed to compute required features: ' \
                      'ISI1 and ISI2' % os.getpid()
            return dict()
        model_ISI1.append(this_dict['ISI1'])
        model_ISI2.append(this_dict['ISI2'])
    new_features['ISI1'] = model_ISI1
    new_features['ISI2'] = model_ISI2

    if export:
        description = 'spike_adaptation'
        with h5py.File(context.export_file_path, 'a') as f:
            if description not in f:
                f.create_group(description)
                f[description].attrs['enumerated'] = False
            group = f[description]
            group.attrs['rheobase'] = rheobase
            group.attrs['exp_rheobase'] = context.exp_rheobase_spike_adaptation
            group.create_dataset('i_relative_amp', compression='gzip', data=i_relative_amp)
            group.create_dataset('model_ISI1', compression='gzip', data=model_ISI1)
            group.create_dataset('model_ISI2', compression='gzip', data=model_ISI1)
            group.create_dataset('exp_ISI1', compression='gzip', data=context.exp_ISI1_array)
            group.create_dataset('exp_ISI2', compression='gzip', data=context.exp_ISI2_array)
    return new_features


def get_args_dynamic_dend_spike(x, features):
    """
    A nested map operation is required to compute dend_spike features. The arguments to be mapped depend on prior
    features (dynamic).
    :param x: array
    :param features: dict
    :return: list of list
    """
    if 'i_holding' not in features:
        i_holding = context.i_holding
    else:
        i_holding = features['i_holding']
    return [[i_holding] * 2, [0.4, 0.7]]


def compute_features_dend_spike(x, i_holding, amp, export=False, plot=False):
    """

    :param x: array
    :param i_holding: defaultdict(dict: float)
    :param amp: float
    :param export: bool
    :param plot: bool
    :return: dict
    """
    start_time = time.time()
    config_sim_env(context)
    update_source_contexts(x, context)

    v_active = context.v_active
    context.i_holding = i_holding
    offset_vm('soma', context, v_active, i_history=context.i_holding)
    sim = context.sim
    dt = context.dt
    stim_dur = context.dend_spike_stim_dur
    equilibrate = context.equilibrate
    duration = context.dend_spike_duration

    rec_dict = sim.get_rec('dend')
    loc = rec_dict['loc']
    node = rec_dict['node']
    dend_rec = rec_dict['vec']

    title = 'dendritic spike'
    description = 'step current amp: %.3f' % amp
    sim.parameters['duration'] = duration
    sim.parameters['title'] = title
    sim.parameters['description'] = description
    sim.parameters['i_amp'] = amp
    sim.backup_state()
    sim.set_state(dt=dt, tstop=duration, cvode=False)
    sim.modify_stim('step', node=node, loc=loc, dur=stim_dur, amp=amp)
    sim.run(v_active)

    result = dict()
    result['i_amp'] = amp
    vm = np.array(dend_rec)
    # dvdt = np.gradient(vm, dt)
    # dvdt2 = np.gradient(dvdt, dt)
    start = int((equilibrate + 0.2) / dt)
    end = int((equilibrate + stim_dur) / dt)
    peak_index = np.argmax(vm[start:end]) + start
    peak_vm = vm[peak_index]
    dend_spike_amp_by_vm_late = peak_vm - np.mean(vm[end - int(0.1 / dt):end])

    if peak_index < end:
        min_vm = np.min(vm[peak_index:end])
        dend_spike_amp_from_min = peak_vm - min_vm
        dend_spike_amp = max(dend_spike_amp_from_min, dend_spike_amp_by_vm_late)
    else:
        dend_spike_amp = 0.
    result['dend_spike_amp'] = dend_spike_amp

    if context.verbose > 0:
        print 'compute_features_dend_spike: pid: %i; %s: %s took %.1f s; dend_spike_amp: %.1f' % \
              (os.getpid(), title, description, time.time() - start_time, dend_spike_amp)
    if plot:
        sim.plot()
    if export:
        context.sim.export_to_file(context.temp_output_path)
    sim.restore_state()
    sim.modify_stim('step', amp=0.)
    return result


def filter_features_dend_spike(primitives, current_features, export=False):
    """

    :param primitives: list of dict (each dict contains results from a single simulation)
    :param current_features: dict
    :param export: bool
    :return: dict
    """
    new_features = dict()
    dend_spike_score = 0.
    for i, this_dict in enumerate(primitives):
        i_amp = this_dict['i_amp']
        spike_amp = this_dict['dend_spike_amp']
        if i_amp == 0.4:
            target_amp = 0.
        elif i_amp == 0.7:
            target_amp = context.target_val['dend_spike_amp']
            new_features['dend_spike_amp'] = spike_amp
        dend_spike_score += ((spike_amp - target_amp) / context.target_range['dend_spike_amp']) ** 2.
    new_features['dend_spike_score'] = dend_spike_score
    return new_features


def get_objectives_spiking(features):
    """

    :param features: dict
    :return: tuple of dict
    """
    objectives = dict()
    for target in ['vm_th', 'fAHP', 'mAHP', 'ADP', 'rebound_firing', 'vm_stability', 'ais_delay', 'dend_bAP_ratio',
                   'soma_spike_amp']:
        objectives[target] = ((context.target_val[target] - features[target]) / context.target_range[target]) ** 2.

    # don't penalize slow_depo outside target range:
    target = 'slow_depo'
    if features[target] > context.target_val[target]:
        objectives[target] = ((features[target] - context.target_val[target]) /
                              (0.01 * context.target_val[target])) ** 2.
    else:
        objectives[target] = 0.

    exp_ISI1 = context.exp_ISI1_array
    exp_ISI2 = context.exp_ISI2_array
    model_ISI1 = features['ISI1']
    model_ISI2 = features['ISI2']
    spike_adaptation_residuals = 0.
    for i in xrange(len(exp_ISI1)):
        spike_adaptation_residuals += ((model_ISI1[i] - exp_ISI1[i]) / (0.01 * exp_ISI1[i])) ** 2.
        spike_adaptation_residuals += ((model_ISI2[i] - exp_ISI2[i]) / (0.01 * exp_ISI2[i])) ** 2.
    objectives['spike_adaptation_residuals'] = spike_adaptation_residuals / len(exp_ISI1)

    exp_rheobase_f_I = context.exp_rheobase_f_I
    model_rate_f_I = features['f_I_rate']
    exp_rate_f_I = context.exp_rate_f_I_array
    model_i_inj_amp_array = np.add(exp_rheobase_f_I, context.i_inj_relative_amp_array_f_I)
    model_f_I_fit_params, pcov = scipy.optimize.curve_fit(log10_fit, model_i_inj_amp_array, model_rate_f_I,
                                                          context.fit_params_f_I_0)
    slope = model_f_I_fit_params[0]
    features['f_I_log10_slope'] = slope

    f_I_residuals = 0.
    for i, this_rate in enumerate(model_rate_f_I):
        f_I_residuals += ((this_rate - exp_rate_f_I[i]) / context.target_range['spike_rate']) ** 2.
    objectives['f_I_residuals'] = f_I_residuals
    del features['f_I_rate']

    objectives['dend_spike_score'] = features['dend_spike_score']

    return features, objectives


def update_mechanisms_spiking(x, context=None):
    """
    :param x: array
    :param context: :class:'Context'
    """
    if context is None:
        raise RuntimeError('update_mechanisms_spiking: missing required Context object')
    cell = context.cell
    x_dict = param_array_to_dict(x, context.param_names)
    modify_mech_param(cell, 'soma', 'nas', 'gbar', x_dict['soma.gbar_nas'])
    modify_mech_param(cell, 'soma', 'kdr', 'gkdrbar', x_dict['soma.gkdrbar'])
    modify_mech_param(cell, 'soma', 'kap', 'gkabar', x_dict['soma.gkabar'])
    slope = (x_dict['dend.gkabar'] - x_dict['soma.gkabar']) / 300.
    modify_mech_param(cell, 'soma', 'nas', 'sh', x_dict['soma.sh_nas/x'])
    for sec_type in ['apical']:
        modify_mech_param(cell, sec_type, 'kap', 'gkabar', origin='soma', max_loc=75., slope=slope, outside=0.)
        modify_mech_param(cell, sec_type, 'kad', 'gkabar', origin='soma', min_loc=75., max_loc=300., slope=slope,
                          value=(x_dict['soma.gkabar'] + slope * 75.), outside=0.)
        modify_mech_param(cell, sec_type, 'kad', 'gkabar', origin='soma', min_loc=300.,
                          value=(x_dict['soma.gkabar'] + slope * 300.), append=True)
        modify_mech_param(cell, sec_type, 'kdr', 'gkdrbar', origin='soma')
        modify_mech_param(cell, sec_type, 'nas', 'sha', 0.)
        modify_mech_param(cell, sec_type, 'nas', 'sh', origin='soma')
        modify_mech_param(cell, sec_type, 'nas', 'gbar', x_dict['dend.gbar_nas'])
        """
        modify_mech_param(cell, sec_type, 'nas', 'gbar', origin='parent', slope=x_dict['dend.gbar_nas slope'],
                          min=x_dict['dend.gbar_nas min'],
                          custom={'func': 'custom_filter_by_branch_order',
                                  'branch_order': x_dict['dend.gbar_nas bo']}, append=True)
        """
        modify_mech_param(cell, sec_type, 'nas', 'gbar', origin='parent', slope=0., min=x_dict['dend.gbar_nas min'],
                          custom={'func': 'custom_filter_by_terminal'}, append=True)
    modify_mech_param(cell, 'hillock', 'kap', 'gkabar', origin='soma')
    modify_mech_param(cell, 'hillock', 'kdr', 'gkdrbar', origin='soma')
    modify_mech_param(cell, 'ais', 'kdr', 'gkdrbar', x_dict['axon.gkdrbar'])
    modify_mech_param(cell, 'ais', 'kap', 'gkabar', x_dict['axon.gkabar'])
    modify_mech_param(cell, 'axon', 'kdr', 'gkdrbar', origin='ais')
    modify_mech_param(cell, 'axon', 'kap', 'gkabar', origin='ais')
    modify_mech_param(cell, 'hillock', 'nax', 'sh', x_dict['soma.sh_nas/x'])
    modify_mech_param(cell, 'hillock', 'nax', 'gbar', x_dict['soma.gbar_nas'])
    modify_mech_param(cell, 'axon', 'nax', 'gbar', x_dict['axon.gbar_nax'])
    for sec_type in ['ais', 'axon']:
        modify_mech_param(cell, sec_type, 'nax', 'sh', origin='hillock')
    modify_mech_param(cell, 'soma', 'Ca', 'gcamult', x_dict['soma.gCa factor'])
    modify_mech_param(cell, 'soma', 'CadepK', 'gcakmult', x_dict['soma.gCadepK factor'])
    modify_mech_param(cell, 'soma', 'Cacum', 'tau', x_dict['soma.tau_Cacum'])
    modify_mech_param(cell, 'soma', 'km3', 'gkmbar', x_dict['soma.gkmbar'])
    modify_mech_param(cell, 'ais', 'km3', 'gkmbar', x_dict['ais.gkmbar'])
    modify_mech_param(cell, 'hillock', 'km3', 'gkmbar', origin='soma')
    modify_mech_param(cell, 'axon', 'km3', 'gkmbar', origin='ais')
    modify_mech_param(cell, 'ais', 'nax', 'sha', x_dict['ais.sha_nax'])
    modify_mech_param(cell, 'ais', 'nax', 'gbar', x_dict['ais.gbar_nax'])


def add_diagnostic_recordings(context):
    """

    :param context: :class:'Context'
    """
    cell = context.cell
    sim = context.sim
    if not sim.has_rec('ica'):
        sim.append_rec(cell, cell.tree.root, name='ica', param='_ref_ica', loc=0.5)
    if not sim.has_rec('isk'):
        sim.append_rec(cell, cell.tree.root, name='isk', param='_ref_isk_CadepK', loc=0.5)
    if not sim.has_rec('ibk'):
        sim.append_rec(cell, cell.tree.root, name='ibk', param='_ref_ibk_CadepK', loc=0.5)
    if not sim.has_rec('ika'):
        sim.append_rec(cell, cell.tree.root, name='ika', param='_ref_ik_kap', loc=0.5)
    if not sim.has_rec('ikdr'):
        sim.append_rec(cell, cell.tree.root, name='ikdr', param='_ref_ik_kdr', loc=0.5)
    if not sim.has_rec('ikm'):
        sim.append_rec(cell, cell.tree.root, name='ikm', param='_ref_ik_km3', loc=0.5)
    if not sim.has_rec('cai'):
        sim.append_rec(cell, cell.tree.root, name='cai', param='_ref_cai', loc=0.5)
    if not sim.has_rec('ina'):
        sim.append_rec(cell, cell.tree.root, name='ina', param='_ref_ina', loc=0.5)
    if not sim.has_rec('axon_end'):
        axon_seg_locs = [seg.x for seg in cell.axon[0].sec]
        sim.append_rec(cell, cell.axon[0], name='axon_end', loc=axon_seg_locs[-1])


def add_complete_axon_recordings(context):
    """

    :param context: :class:'Context'
    """
    cell = context.cell
    sim = context.sim
    target_distance = 0.
    for i, seg in enumerate(cell.axon[0].sec):
        loc=seg.x
        distance = get_distance_to_node(cell, cell.tree.root, cell.axon[0], loc=loc)
        if distance >= target_distance:
            name = 'axon_seg%i' % i
            print name, distance
            if not sim.has_rec(name):
                sim.append_rec(cell, cell.axon[0], name=name, loc=loc)
            target_distance += 100.


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
