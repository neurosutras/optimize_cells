"""
Uses nested.optimize to tune somatodendritic spike shape, f-I curve, and spike adaptation in dentate granule cells.

Requires a YAML file to specify required configuration parameters.
Requires use of a nested.parallel interface.
"""
__author__ = 'Aaron D. Milstein and Grace Ng'
from biophysics_utils import *
from nested.optimize_utils import *
from optimize_cells_utils import *
import collections
import click
from plot_results import *

context = Context()


@click.command()
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/optimize_DG_GC_spiking_config.yaml')
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--export", is_flag=True)
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default=None)
@click.option("--disp", is_flag=True)
@click.option("--verbose", type=int, default=2)
def main(config_file_path, output_dir, export, export_file_path, label, disp, verbose):
    """

    :param config_file_path: str (path)
    :param output_dir: str (path)
    :param export: bool
    :param export_file_path: str
    :param label: str
    :param disp: bool
    :param verbose: bool
    """
    # requires a global variable context: :class:'Context'
    context.update(locals())
    config_interactive(context, __file__, config_file_path=config_file_path, output_dir=output_dir, export=export,
                       export_file_path=export_file_path, label=label, disp=disp, verbose=verbose)
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
    equilibrate = 250.  # time to steady-state
    stim_dur = 500.
    duration = equilibrate + stim_dur
    dt = 0.025
    th_dvdt = 10.
    v_init = -77.
    v_active = -77.
    i_th_max = 0.4

    # GC experimental spike adaptation data from Brenner...Aldrich, Nat. Neurosci., 2005
    experimental_spike_times = [0., 8.57331572, 21.79656539, 39.24702774, 60.92470277, 83.34214003, 109.5640687,
                                137.1598415, 165.7067371, 199.8546896, 236.2219287, 274.3857332, 314.2404227,
                                355.2575958,
                                395.8520476, 436.7635403]
    experimental_adaptation_indexes = []
    for i in xrange(3, len(experimental_spike_times) + 1):
        experimental_adaptation_indexes.append(get_adaptation_index(experimental_spike_times[:i]))

    # GC experimental f-I data from Kowalski J...Pernia-Andrade AJ, Hippocampus, 2016
    i_inj_increment = 0.05
    num_increments = 10
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
    duration = context.duration
    dt = context.dt
    context.sim = QuickSim(context.duration, cvode=cvode, daspk=daspk, dt=context.dt, verbose=verbose>1)
    context.spike_output_vec = h.Vector()
    cell.spike_detector.record(context.spike_output_vec)
    context.cell = cell


def get_adaptation_index(spike_times):
    """
    A large value indicates large degree of spike adaptation (large increases in interspike intervals during a train)
    :param spike_times: list of float
    :return: float
    """
    if len(spike_times) < 3:
        return None
    isi = []
    adi = []
    for i in xrange(len(spike_times) - 1):
        isi.append(spike_times[i + 1] - spike_times[i])
    for i in xrange(len(isi) - 1):
        adi.append((isi[i + 1] - isi[i]) / (isi[i + 1] + isi[i]))
    return np.mean(adi)


def config_sim_env(context):
    """

    :param context: :class:'Context'
    """
    if 'previous_module' in context() and context.previous_module == __file__:
        return
    init_context()
    if 'i_holding' not in context():
        context.i_holding = defaultdict(dict)
    if 'i_th' not in context():
        context.i_th = defaultdict(dict)
    cell = context.cell
    sim = context.sim
    if not sim.has_rec('soma'):
        sim.append_rec(cell, cell.tree.root, name='soma', loc=0.5)
    if context.v_active not in context.i_holding['soma']:
        context.i_holding['soma'][context.v_active] = 0.
    if context.v_active not in context.i_th['soma']:
        context.i_th['soma'][context.v_active] = 0.1
    if not sim.has_rec('dend'):
        dend, dend_loc = get_DG_GC_thickest_dend_branch(context.cell, 200., terminal=False)
        sim.append_rec(cell, dend, name='dend', loc=dend_loc)
    if context.v_active not in context.i_holding['dend']:
        context.i_holding['dend'][context.v_active] = 0.1
    if context.v_active not in context.i_th['dend']:
        context.i_th['dend'][context.v_active] = 0.1
    if not sim.has_rec('ais'):
        sim.append_rec(cell, cell.ais[0], name='ais', loc=1.)
    if context.v_active not in context.i_holding['ais']:
        context.i_holding['ais'][context.v_active] = 0.1
    if not sim.has_rec('axon'):
        axon_seg_locs = [seg.x for seg in cell.axon[0].sec]
        sim.append_rec(cell, cell.axon[0], name='axon', loc=axon_seg_locs[0])
    if context.v_active not in context.i_holding['axon']:
        context.i_holding['axon'][context.v_active] = 0.1

    equilibrate = context.equilibrate
    stim_dur = context.stim_dur
    duration = context.duration

    if not sim.has_stim('step'):
        sim.append_stim(cell, cell.tree.root, name='step', loc=0.5, amp=0., delay=equilibrate, dur=stim_dur)
    if not sim.has_stim('holding'):
        sim.append_stim(cell, cell.tree.root, name='holding', loc=0.5, amp=0., delay=0., dur=duration)

    sim.parameters['duration'] = duration
    sim.parameters['equilibrate'] = equilibrate
    context.previous_module = __file__


def compute_features_spike_shape(x, export=False, plot=False):
    """
    
    :param x: array
    :param export: bool
    :param plot: bool
    :return: dict
    """
    start_time = time.time()
    config_sim_env(context)
    update_source_contexts(x, context)

    result = dict()
    equilibrate = context.equilibrate
    dt = context.dt
    sim = context.sim
    cvode = sim.cvode
    sim.cvode = False
    v_active = context.v_active
    vm_rest = offset_vm('soma', context, v_active)
    spike_times = context.cell.spike_detector.get_recordvec().as_numpy()
    if np.any(spike_times < equilibrate):
        if context.verbose > 0:
            print 'compute_features_spike_shape: pid: %i; aborting - spontaneous firing' % (os.getpid())
        return None
    result['vm_rest'] = vm_rest
    rec_dict = sim.get_rec('soma')
    loc = rec_dict['loc']
    node = rec_dict['node']
    soma_rec = rec_dict['vec']
    i_th = context.i_th['soma'][v_active]
    stim_dur = 150.
    duration = equilibrate + stim_dur
    sim.tstop = duration
    t = np.arange(0., duration, dt)

    sim.modify_stim('step', node=node, loc=loc, dur=stim_dur, amp=i_th)
    sim.run(v_active)
    spike_times = context.cell.spike_detector.get_recordvec().as_numpy()
    if np.any((spike_times > equilibrate) & (spike_times < equilibrate + 50)):
        spike = True
        target = False
        soma_vm = soma_rec.as_numpy()
        ais_vm = sim.get_rec('ais')['vec'].as_numpy()
        axon_vm = sim.get_rec('axon')['vec'].as_numpy()
        dend_vm = sim.get_rec('dend')['vec'].as_numpy()
        i_inc = -0.01
        delta_str = 'decreased'
    else:
        delta_str = 'increased'
        spike = False
        target = True
        i_inc = 0.01
    while not spike == target:
        prev_spike_times = np.array(spike_times)
        if i_th > context.i_th_max:
            if context.verbose > 0:
                print 'compute_features_spike_shape: pid: %i; aborting - rheobase outside target range' % (os.getpid())
            return None
        i_th += i_inc
        sim.modify_stim('step', amp=i_th)
        sim.run(v_active)
        spike_times = context.cell.spike_detector.get_recordvec().as_numpy()
        if sim.verbose:
            print 'compute_features_spike_shape: pid: %i; %s; %s i_th to %.3f nA; num_spikes: %i' % \
                  (os.getpid(), 'soma', delta_str, i_th, len(spike_times))
        spike = np.any((spike_times > equilibrate) & (spike_times < equilibrate + 50))
    if target:
        soma_vm = soma_rec.as_numpy()
        ais_vm = sim.get_rec('ais')['vec'].as_numpy()
        axon_vm = sim.get_rec('axon')['vec'].as_numpy()
        dend_vm = sim.get_rec('dend')['vec'].as_numpy()
    else:
        i_th -= i_inc
        spike_times = np.array(prev_spike_times)
    title = 'spike_shape'
    description = 'rheobase: %.3f' % i_th
    sim.parameters['amp'] = i_th
    sim.parameters['title'] = title
    sim.parameters['description'] = description
    sim.parameters['duration'] = duration

    peak, threshold, ADP, AHP = get_spike_shape(soma_vm, spike_times, context)
    result['soma_spike_amp'] = peak - threshold
    result['vm_th'] = threshold
    result['ADP'] = ADP
    result['AHP'] = AHP
    result['rheobase'] = i_th
    result['th_count'] = len(np.where(spike_times > equilibrate)[0])
    start = int((equilibrate + 1.) / dt)
    th_x = np.where(soma_vm[start:] >= threshold)[0][0] + start
    if len(spike_times) > 1:
        end = min(th_x + int(10. / dt), int((spike_times[1] - 5.) / dt))
    else:
        end = th_x + int(10. / dt)
    dend_peak = np.max(dend_vm[th_x:end])
    dend_pre = np.mean(dend_vm[th_x - int(3. / dt):th_x - int(1. / dt)])
    result['dend_bAP_ratio'] = (dend_peak - dend_pre) / result['soma_spike_amp']

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

    sim.cvode = cvode
    context.i_th['soma'][v_active] = i_th
    sim.modify_stim('step', amp=0.)
    if context.verbose > 0:
        print 'compute_features_spike_shape: pid: %i; %s: %s took %.1f s; rheobase: %.3f; vm_th: %.1f' % \
              (os.getpid(), title, description, time.time() - start_time, i_th, threshold)
    if plot:
        sim.plot()
    if export:
        export_sim_results()
    return result


def get_args_dynamic_fI(x, features):
    """
    A nested map operation is required to compute fI features. The arguments to be mapped depend on each set of
    parameters and prior features (dynamic).
    :param x: array
    :param features: dict
    :return: list of list
    """
    rheobase = features['rheobase']
    # Calculate firing rates for a range of I_inj amplitudes using a stim duration of 500 ms
    num_incr = context.num_increments
    i_inj_increment = context.i_inj_increment
    return [[rheobase + i_inj_increment * (i + 1) for i in xrange(num_incr)], [False] * (num_incr - 1) + [True]]


def compute_features_fI(x, amp, extend_dur=False, export=False, plot=False):
    """

    :param x: array
    :param amp: float
    :param extend_dur: bool
    :param export: bool
    :param plot: bool
    :return: dict
    """
    start_time = time.time()
    config_sim_env(context)
    update_source_contexts(x, context)

    offset_vm('soma', context, context.v_active)
    sim = context.sim
    cvode = sim.cvode
    sim.parameters['amp'] = amp
    sim.parameters['description'] = 'f_I'

    stim_dur = context.stim_dur
    equilibrate = context.equilibrate
    v_active = context.v_active
    dt = context.dt

    rec_dict = sim.get_rec('soma')
    loc = rec_dict['loc']
    node = rec_dict['node']
    soma_rec = rec_dict['vec']
    sim.modify_stim('step', node=node, loc=loc, dur=stim_dur, amp=amp)
    if extend_dur:
        # extend duration of simulation to examine rebound
        duration = equilibrate + stim_dur + 100.
    else:
        duration = equilibrate + stim_dur

    title = 'f_I'
    description = 'step current amp: %.3f' % amp
    sim.tstop = duration
    sim.parameters['duration'] = duration
    sim.parameters['title'] = title
    sim.parameters['description'] = description
    sim.run(v_active)
    if plot:
        sim.plot()
    spike_times = np.subtract(context.cell.spike_detector.get_recordvec().as_numpy(), equilibrate)
    t = np.arange(0., duration, dt)
    result = {}
    result['spike_times'] = spike_times
    result['amp'] = amp
    if extend_dur:
        vm = np.interp(t, sim.tvec, soma_rec)
        v_min_late = np.min(vm[int((equilibrate + stim_dur - 20.) / dt):int((equilibrate + stim_dur - 1.) / dt)])
        result['v_min_late'] = v_min_late
        vm_rest = np.mean(vm[int((equilibrate - 3.) / dt):int((equilibrate - 1.) / dt)])
        v_after = np.max(vm[-int(50. / dt):-1])
        vm_stability = abs(v_after - vm_rest)
        result['vm_stability'] = vm_stability
        result['rebound_firing'] = len(np.where(spike_times > stim_dur)[0])
    if context.verbose > 0:
        print 'compute_features_fI: pid: %i; %s: %s took %.1f s; num_spikes: %i' % \
              (os.getpid(), title, description, time.time() - start_time, len(spike_times))
    sim.modify_stim('step', amp=0.)
    sim.cvode = cvode
    if export:
        export_sim_results()
    return result


def filter_features_fI(primitives, current_features, export=False):
    """

    :param primitives: list of dict (each dict contains results from a single simulation)
    :param current_features: dict
    :param export: bool
    :return: dict
    """
    amps = []
    new_features = {}
    new_features['adi'] = []
    new_features['exp_adi'] = []
    new_features['f_I'] = []
    for i, this_dict in enumerate(primitives):
        amps.append(this_dict['amp'])
        if 'vm_stability' in this_dict:
            new_features['vm_stability'] = this_dict['vm_stability']
        if 'rebound_firing' in this_dict:
            new_features['rebound_firing'] = this_dict['rebound_firing']
        if 'v_min_late' in this_dict:
            new_features['slow_depo'] = this_dict['v_min_late'] - current_features['vm_th']
        spike_times = this_dict['spike_times']
        experimental_spike_times = context.experimental_spike_times
        experimental_adaptation_indexes = context.experimental_adaptation_indexes
        stim_dur = context.stim_dur
        if len(spike_times) < 3:
            adi = None
            exp_adi = None
        elif len(spike_times) > len(experimental_spike_times):
            adi = get_adaptation_index(spike_times[:len(experimental_spike_times)])
            exp_adi = experimental_adaptation_indexes[len(experimental_spike_times) - 3]
        else:
            adi = get_adaptation_index(spike_times)
            exp_adi = experimental_adaptation_indexes[len(spike_times) - 3]
        new_features['adi'].append(adi)
        new_features['exp_adi'].append(exp_adi)
        this_rate = len(spike_times) / stim_dur * 1000.
        new_features['f_I'].append(this_rate)
    adapt_ind = range(len(new_features['f_I']))
    adapt_ind.sort(key=amps.__getitem__)
    new_features['adi'] = map(new_features['adi'].__getitem__, adapt_ind)
    new_features['exp_adi'] = map(new_features['exp_adi'].__getitem__, adapt_ind)
    new_features['f_I'] = map(new_features['f_I'].__getitem__, adapt_ind)
    amps = map(amps.__getitem__, adapt_ind)
    experimental_f_I_slope = context.target_val['f_I_slope']  # Hz/ln(pA); rate = slope * ln(current - rheobase)
    if export:
        print 'filter_features_fI trying to export'
        description = 'f_I_features'
        with h5py.File(context.export_file_path, 'a') as f:
            if description not in f:
                f.create_group(description)
                f[description].attrs['enumerated'] = False
            group = f[description]
            group.create_dataset('amps', compression='gzip', compression_opts=9, data=amps)
            group.create_dataset('adi', compression='gzip', compression_opts=9, data=new_features['adi'])
            group.create_dataset('exp_adi', compression='gzip', compression_opts=9, data=new_features['exp_adi'])
            group.create_dataset('f_I', compression='gzip', compression_opts=9, data=new_features['f_I'])
            num_increments = context.num_increments
            i_inj_increment = context.i_inj_increment
            rheobase = current_features['rheobase']
            exp_f_I = [experimental_f_I_slope * np.log((rheobase + i_inj_increment * (i + 1)) / rheobase)
                       for i in xrange(num_increments)]
            group.create_dataset('exp_f_I', compression='gzip', compression_opts=9, data=exp_f_I)
    return new_features


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


def update_mechanisms_spiking(x, context=None):
    """
    :param x: array ['soma.gbar_nas', 'dend.gbar_nas', 'dend.gbar_nas slope', 'dend.gbar_nas min', 'dend.gbar_nas bo',
                    'axon.gbar_nax', 'ais.gbar_nax', 'soma.gkabar', 'dend.gkabar', 'soma.gkdrbar', 'axon.gkabar',
                    'soma.sh_nas/x', 'ais.sha_nax', 'soma.gCa factor', 'soma.gCadepK factor', 'soma.gkmbar',
                    'ais.gkmbar']
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
        modify_mech_param(cell, sec_type, 'nas', 'sha', 0.)  # 5.)
        modify_mech_param(cell, sec_type, 'nas', 'gbar', x_dict['dend.gbar_nas'])
        modify_mech_param(cell, sec_type, 'nas', 'gbar', origin='parent', slope=x_dict['dend.gbar_nas slope'],
                          min=x_dict['dend.gbar_nas min'],
                          custom={'func': 'custom_filter_by_branch_order',
                                  'branch_order': x_dict['dend.gbar_nas bo']}, append=True)
        modify_mech_param(cell, sec_type, 'nas', 'gbar', origin='parent', slope=x_dict['dend.gbar_nas slope'],
                          min=x_dict['dend.gbar_nas min'],
                          custom={'func': 'custom_filter_by_terminal'}, append=True)
    update_mechanism_by_sec_type(cell, 'hillock', 'kap')
    update_mechanism_by_sec_type(cell, 'hillock', 'kdr')
    modify_mech_param(cell, 'ais', 'kdr', 'gkdrbar', origin='soma')
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
    modify_mech_param(cell, 'soma', 'km3', 'gkmbar', x_dict['soma.gkmbar'])
    modify_mech_param(cell, 'ais', 'km3', 'gkmbar', x_dict['ais.gkmbar'])
    modify_mech_param(cell, 'hillock', 'km3', 'gkmbar', origin='soma')
    modify_mech_param(cell, 'axon', 'km3', 'gkmbar', origin='ais')
    modify_mech_param(cell, 'ais', 'nax', 'sha', x_dict['ais.sha_nax'])
    modify_mech_param(cell, 'ais', 'nax', 'gbar', x_dict['ais.gbar_nax'])


def export_sim_results():
    """
    Export the most recent time and recorded waveforms from the QuickSim object.
    """
    context.sim.export_to_file(context.temp_output_path)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)