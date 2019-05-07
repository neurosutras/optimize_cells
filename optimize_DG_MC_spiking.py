"""
Uses nested.optimize to tune somatodendritic spike shape, f-I curve, and spike adaptation in dentate granule cells.

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
              default='config/optimize_DG_MC_spiking_config.yaml')
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--export", is_flag=True)
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default=None)
@click.option("--verbose", type=int, default=2)
@click.option("--plot", is_flag=True)
@click.option("--interactive", is_flag=True)
@click.option("--debug", is_flag=True)
@click.option("--diagnostic-recordings", is_flag=True)
def main(config_file_path, output_dir, export, export_file_path, label, verbose, plot, interactive, debug,
         diagnostic_recordings):
    """

    :param config_file_path: str (path)
    :param output_dir: str (path)
    :param export: bool
    :param export_file_path: str
    :param label: str
    :param verbose: bool
    :param plot: bool
    :param interactive: bool
    :param debug: bool
    :param diagnostic_recordings: bool
    """
    # requires a global variable context: :class:'Context'
    context.update(locals())
    comm = MPI.COMM_WORLD

    from nested.parallel import ParallelContextInterface
    context.interface = ParallelContextInterface()
    context.interface.start(disp=True)
    context.interface.ensure_controller()
    context.interface.apply(config_optimize_interactive, __file__, config_file_path=config_file_path,
                            output_dir=output_dir, export=export, export_file_path=export_file_path, label=label,
                            disp=verbose > 0, verbose=verbose, plot=plot)
    sys.stdout.flush()
    time.sleep(1.)

    if diagnostic_recordings:
        add_diagnostic_recordings()

    if not context.debug:
        features = dict()
        # Stage 0: Get shape of single spike at rheobase in soma and axon
        # Get value of holding current required to maintain target baseline membrane potential
        args = context.interface.execute(get_args_dynamic_i_holding, context.x0_array, features)
        group_size = len(args[0])
        sequences = [[context.x0_array] * group_size] + args + [[context.export] * group_size] + \
                    [[context.plot] * group_size]
        primitives = context.interface.map(compute_features_spike_shape, *sequences)
        features = {key: value for feature_dict in primitives for key, value in feature_dict.iteritems()}
        context.update(locals())

        # Stage 1: Run simulations with a range of amplitudes of step current injections to the soma
        args = context.interface.execute(get_args_dynamic_fI, context.x0_array, features)
        group_size = len(args[0])
        sequences = [[context.x0_array] * group_size] + args + [[context.export] * group_size] + \
                    [[context.plot] * group_size]
        primitives = context.interface.map(compute_features_fI, *sequences)
        this_features = context.interface.execute(filter_features_fI, primitives, features, context.export)
        features.update(this_features)
        context.update(locals())

        features, objectives = context.interface.execute(get_objectives_spiking, features, context.export)
        sys.stdout.flush()
        time.sleep(1.)
        print 'params:'
        pprint.pprint(context.x0_dict)
        print 'features:'
        pprint.pprint(features)
        print 'objectives:'
        pprint.pprint(objectives)
        sys.stdout.flush()
        time.sleep(1.)
        if context.plot:
            context.interface.apply(plt.show)
    context.update(locals())

    if not interactive:
        context.interface.stop()


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
    stim_dur_f_I = 500.
    duration = equilibrate + default_stim_dur

    dt = 0.025
    th_dvdt = 10.
    dend_th_dvdt = 30.
    v_init = -66.
    v_active = -66.
    i_th_start = 0.2
    i_th_max = 0.4

    # MC experimental f-I and ISI data from:
    # Howard, A.L., Neu, A., Morgan, R.J., Echegoyen, J.C., & Soltesz, I.(2007). Opposing modifications in intrinsic
    # currents and synaptic inputs in post - traumatic mossy cells: evidence for single - cell homeostasis in a
    # hyperexcitable network. Journal of Neurophysiology, 97(3), 2394-2409. http://doi.org/10.1152/jn.00509.2006
    i_inj_increment_f_I = 0.05
    num_increments_f_I = 6
    rate_at_rheobase = 2.  # Hz, corresponds to 1 spike in a 200 ms current injection

    exp_i_inj_amp_f_I_0 = [0.07986577, 0.11912752, 0.15939597, 0.19966443, 0.23892617, 0.27919463, 0.31946309,
                           0.35872483, 0.39899329]  # nA
    exp_rate_f_I_0 = [3.59683794,  8.43873518, 13.83399209, 16.87747036, 19.92094862, 21.44268775, 21.71936759,
                      24.48616601, 26.00790514]  # from 500 ms current injections
    fit_params_f_I_0 = [50., 10.]

    exp_fit_params_f_I, pcov = scipy.optimize.curve_fit(log10_fit, exp_i_inj_amp_f_I_0, exp_rate_f_I_0,
                                                        fit_params_f_I_0)
    exp_rheobase = inverse_log10_fit(rate_at_rheobase, *exp_fit_params_f_I)
    i_inj_relative_amp_array = np.array([i_inj_increment_f_I * i for i in xrange(1, num_increments_f_I + 1)])
    exp_i_inj_amp_array = np.add(exp_rheobase, i_inj_relative_amp_array)
    exp_rate_f_I_array = log10_fit(exp_i_inj_amp_array, *exp_fit_params_f_I)

    exp_i_inj_amp_spike_adaptation_0 = [0.0786927, 0.11918506, 0.15969864, 0.20034486, 0.23995649, 0.2803958,
                                        0.32106324, 0.35967742, 0.40113009]  # nA
    # last ISI / first ISI (%)
    exp_spike_adaptation_array_0 = [120.4301075, 123.655914, 129.0322581, 147.8494624, 161.827957, 159.6774194,
                                    180.6451613, 193.5483871, 194.0860215]

    exp_fit_spike_adaptation_results = stats.linregress(exp_i_inj_amp_spike_adaptation_0, exp_spike_adaptation_array_0)

    exp_fit_spike_adaptation_slope, exp_fit_spike_adaptation_intercept0 = exp_fit_spike_adaptation_results[0], \
                                                                          exp_fit_spike_adaptation_results[1]
    spike_adaptation_at_rheobase = exp_fit_spike_adaptation_slope * exp_rheobase + exp_fit_spike_adaptation_intercept0
    exp_spike_adaptation_array = np.add(np.multiply(i_inj_relative_amp_array, exp_fit_spike_adaptation_slope),
                                        spike_adaptation_at_rheobase)
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
                    verbose=verbose > 1)
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
        return 0
    isi = np.diff(spike_times)
    adi = 100. * isi[-1] / isi[0]
    return adi


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
    sim = context.sim
    if not sim.has_rec('soma'):
        sim.append_rec(cell, cell.tree.root, name='soma', loc=0.5)
    if context.v_active not in context.i_holding['soma']:
        context.i_holding['soma'][context.v_active] = 0.
    # if context.v_active not in context.i_th_history['soma']:
    #    context.i_th_history['soma'][context.v_active] = context.i_th_start
    if not sim.has_rec('dend'):
        dend, dend_loc = get_thickest_dend_branch(context.cell, 100., terminal=False)
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
    offset_vm('soma', context, v_active, i_history=context.i_holding, dynamic=True)
    sim.modify_stim('holding', dur=duration)

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

    soma_vm = np.array(soma_rec)  # Get voltage waveforms of spike from various subcellular compartments
    ais_vm = np.array(sim.get_rec('ais')['vec'])
    axon_vm = np.array(sim.get_rec('axon')['vec'])

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
            context.i_inj_relative_amp_array, [False] * (group_size - 1) + [True]]


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
    offset_vm('soma', context, v_active, i_history=context.i_holding, dynamic=True)
    sim = context.sim
    dt = context.dt
    stim_dur = context.stim_dur_f_I
    equilibrate = context.equilibrate

    if extend_dur:
        # extend duration of simulation to examine rebound
        duration = equilibrate + stim_dur + 200.
    else:
        duration = equilibrate + stim_dur

    sim.modify_stim('holding', dur=duration)

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
    result['spike_times'] = spike_times
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
        vm_th_late = np.mean(vm[last_spike_index - int(0.1 / dt):last_spike_index])
        result['vm_th_late'] = vm_th_late

    spike_rate = len(spike_times[np.where(spike_times < stim_dur)[0]]) / stim_dur * 1000.
    result['spike_rate'] = spike_rate

    if context.verbose > 0:
        print 'compute_features_fI: pid: %i; %s: %s took %.1f s; spike_rate: %.1f' % \
              (os.getpid(), title, description, time.time() - start_time, spike_rate)
    if plot:
        sim.plot()
    if export:
        context.sim.export_to_file(context.temp_output_path)
    sim.restore_state()
    sim.modify_stim('step', amp=0.)
    return result


def check_for_pause_in_spiking(spike_times, duration):
    """

    :param spike_times: array of float
    :param duration: float
    :return: bool
    """
    if len(spike_times) >= 3.:
        ISI_array = np.diff(spike_times)
        ISI_array.sort()
        max_pause_dur = max(ISI_array[-1], duration - spike_times[-1])
        if max_pause_dur > 2. * ISI_array[-2]:
            return True
    return False


def filter_features_fI(primitives, current_features, export=False):
    """

    :param primitives: list of dict (each dict contains results from a single simulation, in this case spike times and
            current amplitude)
    :param current_features: dict
    :param export: bool
    :return: dict
    """
    rheobase = current_features['rheobase']

    new_features = dict()
    i_relative_amp = [this_dict['i_amp'] - rheobase for this_dict in primitives]
    rate = []
    adi = []

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
            new_features['slow_depo'] = this_dict['vm_th_late'] - current_features['vm_th']
        rate.append(this_dict['spike_rate'])
        spike_times = this_dict['spike_times']
        this_adi = get_spike_adaptation_indexes(spike_times)
        if check_for_pause_in_spiking(spike_times, context.stim_dur_f_I):
            if context.verbose > 0:
                print 'filter_features_fI: pid: %i; aborting - model failed due to long pause in spiking' % \
                      os.getpid()
            return dict()
        adi.append(this_adi)
    new_features['f_I_rate'] = rate
    new_features['spike_adi'] = adi

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
            group.attrs['exp_rheobase'] = context.exp_rheobase
            group.create_dataset('i_relative_amp', compression='gzip', data=i_relative_amp)
            group.create_dataset('adi', compression='gzip', data=adi)
            group.create_dataset('exp_adi', compression='gzip', data=context.exp_spike_adaptation_array)
            group.create_dataset('rate', compression='gzip', data=rate)
            group.create_dataset('exp_rate', compression='gzip', data=context.exp_rate_f_I_array)
    return new_features


def get_objectives_spiking(features, export=False):
    """

    :param features: dict
    :param export: bool
    :return: tuple of dict
    """
    objectives = dict()
    for target in ['vm_th', 'rebound_firing', 'ais_delay']:
        objectives[target] = ((context.target_val[target] - features[target]) / context.target_range[target]) ** 2.

    # only penalize certain features outside target range:
    for target in ['vm_stability']:
        if features[target] > context.target_val[target]:
            objectives[target] = ((features[target] - context.target_val[target]) / context.target_range[target]) ** 2.
        else:
            objectives[target] = 0.

    # only penalize certain features outside target range:
    for target in ['fAHP', 'ADP', 'slow_depo']:
        min_val_key = 'min_' + target
        max_val_key = 'max_' + target
        if features[target] < context.target_val[min_val_key]:
            objectives[target] = ((features[target] - context.target_val[min_val_key]) /
                                  context.target_range[target]) ** 2.
        elif features[target] > context.target_val[max_val_key]:
            objectives[target] = ((features[target] - context.target_val[max_val_key]) /
                                  context.target_range[target]) ** 2.
        else:
            objectives[target] = 0.

    exp_spike_adi_array = context.exp_spike_adaptation_array
    model_spike_adi_array = features['spike_adi']
    model_i_inj_amp_array_f_I = np.add(features['rheobase'], context.i_inj_relative_amp_array)
    model_fit_params_spike_adaptation = stats.linregress(model_i_inj_amp_array_f_I, model_spike_adi_array)
    model_fit_spike_adaptation_slope = model_fit_params_spike_adaptation[0]
    features['spike_adi_slope'] = model_fit_spike_adaptation_slope

    spike_adaptation_residuals = 0.
    for i, this_adi in enumerate(model_spike_adi_array):
        spike_adaptation_residuals += ((this_adi - exp_spike_adi_array[i]) / (0.01 * exp_spike_adi_array[i])) ** 2.
    objectives['spike_adaptation_residuals'] = spike_adaptation_residuals
    del features['spike_adi']

    exp_rate_f_I = context.exp_rate_f_I_array
    model_rate_f_I = features['f_I_rate']
    model_fit_params_f_I, pcov = \
        scipy.optimize.curve_fit(log10_fit, model_i_inj_amp_array_f_I, model_rate_f_I, context.fit_params_f_I_0)
    model_fit_f_I_log10_slope = model_fit_params_f_I[0]
    features['f_I_log10_slope'] = model_fit_f_I_log10_slope

    f_I_residuals = 0.
    for i, this_rate in enumerate(model_rate_f_I):
        f_I_residuals += ((this_rate - exp_rate_f_I[i]) / (0.01 * exp_rate_f_I[i])) ** 2.
    objectives['f_I_residuals'] = f_I_residuals
    del features['f_I_rate']

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
    modify_mech_param(cell, 'soma', 'ions', 'ek', -80.)
    modify_mech_param(cell, 'soma', 'nas', 'sh', x_dict['soma.sh_nas/x'])
    for sec_type in ['apical']:
        modify_mech_param(cell, sec_type, 'kap', 'gkabar', origin='soma')
        modify_mech_param(cell, sec_type, 'kdr', 'gkdrbar', origin='soma')
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
    modify_mech_param(cell, 'ais', 'DGC_KM', 'gbar', x_dict['ais.gkmbar'])
    modify_mech_param(cell, 'hillock', 'DGC_KM', 'gbar', x_dict['ais.gkmbar'])
    modify_mech_param(cell, 'axon', 'DGC_KM', 'gbar', origin='ais')
    modify_mech_param(cell, 'ais', 'nax', 'sha', x_dict['ais.sha_nax'])
    modify_mech_param(cell, 'ais', 'nax', 'gbar', x_dict['ais.gbar_nax'])
    for sec_type in ['apical', 'hillock', 'ais', 'axon']:
        modify_mech_param(cell, sec_type, 'ions', 'ek', origin='soma')


def add_diagnostic_recordings():
    """

    """
    cell = context.cell
    sim = context.sim
    #if not sim.has_rec('ica'):
    #    sim.append_rec(cell, cell.tree.root, name='ica', param='_ref_ica', loc=0.5)
    #if not sim.has_rec('isk'):
    #    sim.append_rec(cell, cell.tree.root, name='isk', param='_ref_isk_CadepK', loc=0.5)
    #if not sim.has_rec('ibk'):
    #    sim.append_rec(cell, cell.tree.root, name='ibk', param='_ref_ibk_CadepK', loc=0.5)
    #if not sim.has_rec('ika'):
    #    sim.append_rec(cell, cell.tree.root, name='ika', param='_ref_ik_kap', loc=0.5)
    #if not sim.has_rec('ikdr'):
    #    sim.append_rec(cell, cell.tree.root, name='ikdr', param='_ref_ik_kdr', loc=0.5)
    #if not sim.has_rec('ikm'):
    #    sim.append_rec(cell, cell.tree.root, name='ikm', param='_ref_ik_km3', loc=0.5)
    #if not sim.has_rec('cai'):
    #    sim.append_rec(cell, cell.tree.root, name='cai', param='_ref_cai', loc=0.5)
    #if not sim.has_rec('ina'):
    #    sim.append_rec(cell, cell.tree.root, name='ina', param='_ref_ina', loc=0.5)
    if not sim.has_rec('axon_end'):
        axon_seg_locs = [seg.x for seg in cell.axon[0].sec]
        sim.append_rec(cell, cell.axon[0], name='axon_end', loc=axon_seg_locs[-1])


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
