"""
Uses nested.optimize to tune somatodendritic spike shape, f-I curve, and spike adaptation in dentate granule cells.

Requires a YAML file to specify required configuration parameters.
Requires use of a nested.parallel interface.
"""
__author__ = 'Aaron D. Milstein and Grace Ng'
from specify_cells4 import *
from plot_results import *
from nested.optimize_utils import *
import collections
import click


context = Context()


@click.command()
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/optimize_DG_GC_spiking_config.yaml')
@click.option("--output-dir", type=str, default='data')
@click.option("--export", is_flag=True)
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default=None)
@click.option("--disp", is_flag=True)
@click.option("--verbose", is_flag=True)
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
    config_interactive(config_file_path=config_file_path, output_dir=output_dir, export_file_path=export_file_path,
                       label=label, verbose=verbose)
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

    features, objectives = get_objectives(features)
    print 'params:'
    pprint.pprint(context.x0_dict)
    print 'features:'
    pprint.pprint(features)
    print 'objectives:'
    pprint.pprint(objectives)


def config_interactive(config_file_path=None, output_dir=None, temp_output_path=None, export_file_path=None,
                       label=None, verbose=True, **kwargs):
    """

    :param config_file_path: str (.yaml file path)
    :param output_dir: str (dir path)
    :param temp_output_path: str (.hdf5 file path)
    :param export_file_path: str (.hdf5 file path)
    :param label: str
    :param verbose: bool
    """

    if config_file_path is not None:
        context.config_file_path = config_file_path
    if 'config_file_path' not in context() or context.config_file_path is None or \
            not os.path.isfile(context.config_file_path):
        raise Exception('config_file_path specifying required parameters is missing or invalid.')
    config_dict = read_from_yaml(context.config_file_path)
    context.param_names = config_dict['param_names']
    if 'default_params' not in config_dict or config_dict['default_params'] is None:
        context.default_params = {}
    else:
        context.default_params = config_dict['default_params']
    for param in context.default_params:
        config_dict['bounds'][param] = (context.default_params[param], context.default_params[param])
    context.bounds = [config_dict['bounds'][key] for key in context.param_names]
    if 'rel_bounds' not in config_dict or config_dict['rel_bounds'] is None:
        context.rel_bounds = None
    else:
        context.rel_bounds = config_dict['rel_bounds']
    if 'x0' not in config_dict or config_dict['x0'] is None:
        context.x0 = None
    else:
        context.x0 = config_dict['x0']
        context.x0_dict = context.x0
        context.x0_array = param_dict_to_array(context.x0_dict, context.param_names)
    context.feature_names = config_dict['feature_names']
    context.objective_names = config_dict['objective_names']
    context.target_val = config_dict['target_val']
    context.target_range = config_dict['target_range']
    context.optimization_title = config_dict['optimization_title']
    context.kwargs = config_dict['kwargs']  # Extra arguments to be passed to imported sources
    context.kwargs['verbose'] = verbose
    context.update(context.kwargs)

    missing_config = []
    if 'update_context' not in config_dict or config_dict['update_context'] is None:
        missing_config.append('update_context')
    else:
        context.update_context_list = config_dict['update_context']
    if 'get_features_stages' not in config_dict or config_dict['get_features_stages'] is None:
        missing_config.append('get_features_stages')
    else:
        context.stages = config_dict['get_features_stages']
    if 'get_objectives' not in config_dict or config_dict['get_objectives'] is None:
        missing_config.append('get_objectives')
    else:
        context.get_objectives_dict = config_dict['get_objectives']
    if missing_config:
        raise Exception('config_file at path: %s is missing the following required fields: %s' %
                        (context.config_file_path, ', '.join(str(field) for field in missing_config)))

    if label is not None:
        context.label = label
    if 'label' not in context() or context.label is None:
        label = ''
    else:
        label = '_' + context.label

    if output_dir is not None:
        context.output_dir = output_dir
    if 'output_dir' not in context():
        context.output_dir = None
    if context.output_dir is None:
        output_dir_str = ''
    else:
        output_dir_str = context.output_dir + '/'

    if temp_output_path is not None:
        context.temp_output_path = temp_output_path
    if 'temp_output_path' not in context() or context.temp_output_path is None:
        context.temp_output_path = '%s%s_pid%i_%s%s_temp_output.hdf5' % \
                               (output_dir_str, datetime.datetime.today().strftime('%Y%m%d%H%M'), os.getpid(),
                                context.optimization_title, label)
    
    if export_file_path is not None:
        context.export_file_path = export_file_path
    if 'export_file_path' not in context() or context.export_file_path is None:
        context.export_file_path = '%s%s_%s%s_interactive_exported_output.hdf5' % \
                                   (output_dir_str, datetime.datetime.today().strftime('%Y%m%d%H%M'),
                                    context.optimization_title, label)

    context.update_context_funcs = []
    for source, func_name in context.update_context_list:
        if source == os.path.basename(__file__).split('.')[0]:
            try:
                func = globals()[func_name]
                if not isinstance(func, collections.Callable):
                    raise Exception('update_context function: %s not callable' % func_name)
                context.update_context_funcs.append(func)
            except:
                raise Exception('update_context function: %s not found' % func_name)
    if not context.update_context_funcs:
        raise Exception('update_context function not found')

    config_worker(context.update_context_funcs, context.param_names, context.default_params, context.target_val,
                  context.target_range, context.temp_output_path, context.export_file_path, context.output_dir,
                  context.disp, **context.kwargs)
    update_source_contexts(context.x0_array)


def config_controller(export_file_path, output_dir, **kwargs):
    """

    :param export_file_path: str (path)
    :param output_dir: str (dir)
    """
    processed_export_file_path = export_file_path.replace('.hdf5', '_processed.hdf5')
    context.update(locals())
    context.update(kwargs)
    init_context()


def config_worker(update_context_funcs, param_names, default_params, target_val, target_range, temp_output_path,
                  export_file_path, output_dir, disp, mech_file_path, neuroH5_file_path, neuroH5_index, spines,
                  **kwargs):
    """
    :param update_context_funcs: list of function references
    :param param_names: list of str
    :param default_params: dict
    :param target_val: dict
    :param target_range: dict
    :param temp_output_path: str
    :param export_file_path: str
    :param output_dir: str (dir path)
    :param disp: bool
    :param mech_file_path: str
    :param neuroH5_file_path: str
    :param neuroH5_index: int
    :param spines: bool
    """
    context.update(kwargs)
    neuroH5_dict = read_from_pkl(neuroH5_file_path)[neuroH5_index]
    param_indexes = {param_name: i for i, param_name in enumerate(param_names)}
    processed_export_file_path = export_file_path.replace('.hdf5', '_processed.hdf5')
    context.update(locals())
    init_context()
    setup_cell(**kwargs)


def init_context():
    """

    """
    equilibrate = 250.  # time to steady-state
    stim_dur = 500.
    duration = equilibrate + stim_dur
    dt = 0.02
    th_dvdt = 10.
    v_init = -77.
    v_active = -77.
    i_holding = {'soma': 0., 'dend': 0., 'distal_dend': 0.}
    i_th = {'soma': 0.1}

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


def setup_cell(verbose=False, cvode=False, daspk=False, **kwargs):
    """

    :param verbose: bool
    :param cvode: bool
    :param daspk: bool
    """
    cell = DG_GC(neuroH5_dict=context.neuroH5_dict, mech_file_path=context.mech_file_path,
                 full_spines=context.spines)
    context.cell = cell

    # get the thickest apical dendrite ~200 um from the soma
    candidate_branches = []
    candidate_diams = []
    candidate_locs = []
    for branch in cell.apical:
        if ((cell.get_distance_to_node(cell.tree.root, branch, 0.) >= 200.) &
                (cell.get_distance_to_node(cell.tree.root, branch, 1.) > 300.) & (not cell.is_terminal(branch))):
            candidate_branches.append(branch)
            for seg in branch.sec:
                loc = seg.x
                if cell.get_distance_to_node(cell.tree.root, branch, loc) > 250.:
                    candidate_diams.append(branch.sec(loc).diam)
                    candidate_locs.append(loc)
                    break
    index = candidate_diams.index(max(candidate_diams))
    dend = candidate_branches[index]
    dend_loc = candidate_locs[index]
    axon_seg_locs = [seg.x for seg in cell.axon[2].sec]

    rec_locs = {'soma': 0., 'dend': dend_loc, 'ais': 1., 'axon': axon_seg_locs[0]}
    context.rec_locs = rec_locs
    rec_nodes = {'soma': cell.tree.root, 'dend': dend, 'ais': cell.axon[1], 'axon': cell.axon[2]}
    context.rec_nodes = rec_nodes

    equilibrate = context.equilibrate
    stim_dur = context.stim_dur
    duration = context.duration
    dt = context.dt

    sim = QuickSim(duration, cvode=cvode, daspk=daspk, dt=dt, verbose=verbose)
    sim.append_stim(cell, cell.tree.root, loc=0., amp=0., delay=equilibrate, dur=stim_dur, description='step')
    sim.append_stim(cell, cell.tree.root, loc=0., amp=0., delay=0., dur=duration, description='offset')
    for description, node in rec_nodes.iteritems():
        sim.append_rec(cell, node, loc=rec_locs[description], description=description)
    sim.parameters['duration'] = duration
    sim.parameters['equilibrate'] = equilibrate
    sim.parameters['spines'] = context.spines
    context.sim = sim

    context.spike_output_vec = h.Vector()
    cell.spike_detector.record(context.spike_output_vec)


def update_source_contexts(x, local_context=None):
    """

    :param x: array
    :param local_context: :class:'Context'
    """
    if local_context is None:
        local_context = context
    local_context.cell.reinit_mechanisms(from_file=True)
    if not local_context.spines:
        local_context.cell.correct_g_pas_for_spines()
    for update_func in local_context.update_context_funcs:
        update_func(x, local_context)


def compute_features_spike_shape(x, export=False, plot=False):
    """
    :param x: array
    :param export: bool
    :param plot: bool
    :return: float
    """
    start_time = time.time()
    update_source_contexts(x, context)
    result = {}
    v_active = context.v_active
    equilibrate = context.equilibrate
    dt = context.dt
    i_th = context.i_th

    soma_vm = offset_vm('soma', v_active)
    result['v_rest'] = soma_vm
    stim_dur = 150.
    step_stim_index = context.sim.get_stim_index('step')
    context.sim.modify_stim(step_stim_index, node=context.cell.tree.root, loc=0., dur=stim_dur)
    duration = equilibrate + stim_dur
    context.sim.tstop = duration
    t = np.arange(0., duration, dt)
    spike = False
    d_amp = 0.01
    amp = max(0., i_th['soma'] - 0.02)
    while not spike:
        context.sim.modify_stim(step_stim_index, amp=amp)
        context.sim.run(v_active)
        vm = np.interp(t, context.sim.tvec.to_python(), context.sim.get_rec('soma')['vec'].to_python())
        if np.any(vm[:int(equilibrate/dt)] > -30.):
            if context.disp:
                print 'Process %i: Aborting - spontaneous firing' % (os.getpid())
            return None
        if np.any(vm[int(equilibrate/dt):int((equilibrate+50.)/dt)] > -30.):
            spike = True
        elif amp >= 0.4:
            if context.disp:
                print 'Process %i: Aborting - rheobase outside target range' % (os.getpid())
            return None
        else:
            amp += d_amp
            if context.sim.verbose:
                print 'increasing amp to %.3f' % amp
    title = 'spike_shape_features'
    description = 'rheobase: %.3f' % amp
    context.sim.parameters['amp'] = amp
    context.sim.parameters['title'] = title
    context.sim.parameters['description'] = description
    context.sim.parameters['duration'] = duration
    i_th['soma'] = amp
    spike_times = context.cell.spike_detector.get_recordvec().to_python()
    peak, threshold, ADP, AHP = get_spike_shape(vm, spike_times)
    result['v_th'] = threshold
    result['ADP'] = ADP
    result['AHP'] = AHP
    result['rheobase'] = amp
    result['spont_firing'] = len(np.where(spike_times < equilibrate)[0])
    result['th_count'] = len(spike_times)
    dend_vm = np.interp(t, context.sim.tvec.to_python(), context.sim.get_rec('dend')['vec'].to_python())
    th_x = np.where(vm[int(equilibrate / dt):] >= threshold)[0][0] + int(equilibrate / dt)
    if len(spike_times) > 1:
        end = min(th_x + int(10. / dt), int((spike_times[1] - 5.)/dt))
    else:
        end = th_x + int(10. / dt)
    result['soma_peak'] = peak
    dend_peak = np.max(dend_vm[th_x:end])
    dend_pre = np.mean(dend_vm[int((equilibrate - 3.) / dt):int((equilibrate - 1.) / dt)])
    result['dend_amp'] = (dend_peak - dend_pre) / (peak - soma_vm)

    # calculate AIS delay
    soma_dvdt = np.gradient(vm, dt)
    ais_vm = np.interp(t, context.sim.tvec.to_python(), context.sim.get_rec('ais')['vec'].to_python())
    ais_dvdt = np.gradient(ais_vm, dt)
    axon_vm = np.interp(t, context.sim.tvec.to_python(), context.sim.get_rec('axon')['vec'].to_python())
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
    if context.disp:
        print 'Process: %i: %s: %s took %.1f s' % (os.getpid(), title, description, time.time() - start_time)
    if plot:
        context.sim.plot()
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
    return [[rheobase + i_inj_increment * (i + 1) for i in xrange(num_incr)], [False] * (num_incr-1) + [True]]


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
    update_source_contexts(x, context)

    soma_vm = offset_vm('soma', context.v_active)
    context.sim.parameters['amp'] = amp
    context.sim.parameters['description'] = 'f_I'

    stim_dur = context.stim_dur
    equilibrate = context.equilibrate
    v_active = context.v_active
    dt = context.dt

    step_stim_index = context.sim.get_stim_index('step')
    context.sim.modify_stim(step_stim_index, node=context.cell.tree.root, loc=0., dur=stim_dur, amp=amp)
    if extend_dur:
        # extend duration of simulation to examine rebound
        duration = equilibrate + stim_dur + 100.
    else:
        duration = equilibrate + stim_dur

    title = 'f_I_features'
    description = 'step current amp: %.3f' % amp
    context.sim.tstop = duration
    context.sim.parameters['duration'] = duration
    context.sim.parameters['title'] = title
    context.sim.parameters['description'] = description
    context.sim.run(v_active)
    if plot:
        context.sim.plot()
    spike_times = np.subtract(context.cell.spike_detector.get_recordvec().to_python(), equilibrate)
    t = np.arange(0., duration, dt)
    result = {}
    result['spike_times'] = spike_times
    result['amp'] = amp
    if extend_dur:
        vm = np.interp(t, context.sim.tvec.to_python(), context.sim.get_rec('soma')['vec'].to_python())
        v_min_late = np.min(vm[int((equilibrate + stim_dur - 20.) / dt):int((equilibrate + stim_dur - 1.) / dt)])
        result['v_min_late'] = v_min_late
        v_rest = np.mean(vm[int((equilibrate - 3.) / dt):int((equilibrate - 1.) / dt)])
        v_after = np.max(vm[-int(50. / dt):-1])
        vm_stability = abs(v_after - v_rest)
        result['vm_stability'] = vm_stability
        result['rebound_firing'] = len(np.where(spike_times > stim_dur)[0])
    if context.disp:
        print 'Process: %i: %s: %s took %.1f s' % (os.getpid(), title, description, time.time() - start_time)
        sys.stdout.flush()
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
            new_features['slow_depo'] = this_dict['v_min_late'] - current_features['v_th']
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
        description = 'f_I_features'
        with h5py.File(context.processed_export_file_path, 'a') as f:
            if description not in f:
                f.create_group(description)
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


def get_objectives(features):
    """

    :param features: dict
    :return: tuple of dict
    """
    if features is None:  # No rheobase value found
        objectives = None
    else:
        objectives = {}
        rheobase = features['rheobase']
        for target in ['v_th', 'ADP', 'AHP', 'spont_firing', 'rebound_firing', 'vm_stability', 'ais_delay',
                       'slow_depo', 'dend_amp', 'soma_peak', 'th_count']:
            # don't penalize AHP or slow_depo less than target
            if not ((target == 'AHP' and features[target] < context.target_val[target]) or
                        (target == 'slow_depo' and features[target] < context.target_val[target])):
                objectives[target] = ((context.target_val[target] - features[target]) / context.target_range[target]) ** 2.
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


def offset_vm(description, vm_target=None):
    """

    :param description: str
    :param vm_target: float
    """
    if vm_target is None:
        vm_target = context.v_init
    step_stim_index = context.sim.get_stim_index('step')
    offset_stim_index = context.sim.get_stim_index('offset')
    context.sim.modify_stim(step_stim_index, amp=0.)
    node = context.rec_nodes[description]
    loc = context.rec_locs[description]
    rec_dict = context.sim.get_rec(description)
    context.sim.modify_stim(offset_stim_index, node=node, loc=loc, amp=0.)
    rec = rec_dict['vec'].to_python()
    offset = True

    equilibrate = context.equilibrate
    dt = context.dt
    duration = context.duration

    context.sim.tstop = equilibrate
    t = np.arange(0., equilibrate, dt)
    context.sim.modify_stim(offset_stim_index, amp=context.i_holding[description])
    context.sim.run(vm_target)
    vm = np.interp(t, context.sim.tvec, rec)
    v_rest = np.mean(vm[int((equilibrate - 3.)/dt):int((equilibrate - 1.)/dt)])
    initial_v_rest = v_rest
    if v_rest < vm_target - 0.5:
        context.i_holding[description] += 0.01
        while offset:
            if context.sim.verbose:
                print 'increasing i_holding to %.3f (%s)' % (context.i_holding[description], description)
            context.sim.modify_stim(offset_stim_index, amp=context.i_holding[description])
            context.sim.run(vm_target)
            vm = np.interp(t, context.sim.tvec, rec)
            v_rest = np.mean(vm[int((equilibrate - 3.)/dt):int((equilibrate - 1.)/dt)])
            if v_rest < vm_target - 0.5:
                context.i_holding[description] += 0.01
            else:
                offset = False
    elif v_rest > vm_target + 0.5:
        context.i_holding[description] -= 0.01
        while offset:
            if context.sim.verbose:
                print 'decreasing i_holding to %.3f (%s)' % (context.i_holding[description], description)
            context.sim.modify_stim(offset_stim_index, amp=context.i_holding[description])
            context.sim.run(vm_target)
            vm = np.interp(t, context.sim.tvec, rec)
            v_rest = np.mean(vm[int((equilibrate - 3.)/dt):int((equilibrate - 1.)/dt)])
            if v_rest > vm_target + 0.5:
                context.i_holding[description] -= 0.01
            else:
                offset = False
    context.sim.tstop = duration
    return v_rest


def get_spike_shape(vm, spike_times):
    """

    :param vm: array
    :param spike_times: array
    :return: tuple of float: (v_peak, th_v, ADP, AHP)
    """
    equilibrate = context.equilibrate
    dt = context.dt
    th_dvdt = context.th_dvdt

    start = int((equilibrate+1.)/dt)
    vm = vm[start:]
    dvdt = np.gradient(vm, dt)
    th_x = np.where(dvdt > th_dvdt)[0]
    if th_x.any():
        th_x = th_x[0] - int(1.6/dt)
    else:
        th_x = np.where(vm > -30.)[0][0] - int(2./dt)
    th_v = vm[th_x]
    v_before = np.mean(vm[th_x-int(0.1/dt):th_x])
    v_peak = np.max(vm[th_x:th_x+int(5./dt)])
    x_peak = np.where(vm[th_x:th_x+int(5./dt)] == v_peak)[0][0]
    if len(spike_times) > 1:
        end = max(th_x + x_peak + int(2./dt), int((spike_times[1] - 4.) / dt) - start)
    else:
        end = len(vm)
    v_AHP = np.min(vm[th_x+x_peak:end])
    x_AHP = np.where(vm[th_x+x_peak:end] == v_AHP)[0][0]
    AHP = v_before - v_AHP
    # if spike waveform includes an ADP before an AHP, return the value of the ADP in order to increase error function
    ADP = 0.
    rising_x = np.where(dvdt[th_x+x_peak+1:th_x+x_peak+x_AHP-1] > 0.)[0]
    if rising_x.any():
        v_ADP = np.max(vm[th_x+x_peak+1+rising_x[0]:th_x+x_peak+x_AHP])
        pre_ADP = np.mean(vm[th_x+x_peak+1+rising_x[0] - int(0.1/dt):th_x+x_peak+1+rising_x[0]])
        ADP += v_ADP - pre_ADP
    falling_x = np.where(dvdt[th_x + x_peak + x_AHP + 1:end] < 0.)[0]
    if falling_x.any():
        v_ADP = np.max(vm[th_x + x_peak + x_AHP + 1: th_x + x_peak + x_AHP + 1 + falling_x[0]])
        ADP += v_ADP - v_AHP
    return v_peak, th_v, ADP, AHP


def update_context_spike_shape(x, local_context=None):
    """
    :param x: array ['soma.gbar_nas', 'dend.gbar_nas', 'dend.gbar_nas slope', 'dend.gbar_nas min', 'dend.gbar_nas bo',
                    'axon.gbar_nax', 'ais.gbar_nax', 'soma.gkabar', 'dend.gkabar', 'soma.gkdrbar', 'axon.gkabar',
                    'soma.sh_nas/x', 'ais.sha_nax', 'soma.gCa factor', 'soma.gCadepK factor', 'soma.gkmbar',
                    'ais.gkmbar']
    """
    if local_context is None:
        local_context = context
    cell = local_context.cell
    param_indexes = local_context.param_indexes
    cell.modify_mech_param('soma', 'nas', 'gbar', x[param_indexes['soma.gbar_nas']])
    cell.modify_mech_param('soma', 'kdr', 'gkdrbar', x[param_indexes['soma.gkdrbar']])
    cell.modify_mech_param('soma', 'kap', 'gkabar', x[param_indexes['soma.gkabar']])
    slope = (x[param_indexes['dend.gkabar']] - x[param_indexes['soma.gkabar']]) / 300.
    cell.modify_mech_param('soma', 'nas', 'sh', x[param_indexes['soma.sh_nas/x']])
    for sec_type in ['apical']:
        cell.reinitialize_subset_mechanisms(sec_type, 'nas')
        cell.modify_mech_param(sec_type, 'kap', 'gkabar', origin='soma', min_loc=75., value=0.)
        cell.modify_mech_param(sec_type, 'kap', 'gkabar', origin='soma', max_loc=75., slope=slope, replace=False)
        cell.modify_mech_param(sec_type, 'kad', 'gkabar', origin='soma', max_loc=75., value=0.)
        cell.modify_mech_param(sec_type, 'kad', 'gkabar', origin='soma', min_loc=75., max_loc=300., slope=slope,
                               value=(x[param_indexes['soma.gkabar']] + slope * 75.), replace=False)
        cell.modify_mech_param(sec_type, 'kad', 'gkabar', origin='soma', min_loc=300.,
                               value=(x[param_indexes['soma.gkabar']] + slope * 300.), replace=False)
        cell.modify_mech_param(sec_type, 'kdr', 'gkdrbar', origin='soma')
        cell.modify_mech_param(sec_type, 'nas', 'sha', 0.)  # 5.)
        cell.modify_mech_param(sec_type, 'nas', 'gbar',
                               x[param_indexes['dend.gbar_nas']])
        cell.modify_mech_param(sec_type, 'nas', 'gbar', origin='parent', slope=x[param_indexes['dend.gbar_nas slope']],
                               min=x[param_indexes['dend.gbar_nas min']],
                               custom={'method': 'custom_gradient_by_branch_order',
                                       'branch_order': x[param_indexes['dend.gbar_nas bo']]}, replace=False)
        cell.modify_mech_param(sec_type, 'nas', 'gbar', origin='parent',
                               slope=x[param_indexes['dend.gbar_nas slope']], min=x[param_indexes['dend.gbar_nas min']],
                               custom={'method': 'custom_gradient_by_terminal'}, replace=False)
    cell.reinitialize_subset_mechanisms('axon_hill', 'kap')
    cell.reinitialize_subset_mechanisms('axon_hill', 'kdr')
    cell.modify_mech_param('ais', 'kdr', 'gkdrbar', origin='soma')
    cell.modify_mech_param('ais', 'kap', 'gkabar', x[param_indexes['axon.gkabar']])
    cell.modify_mech_param('axon', 'kdr', 'gkdrbar', origin='ais')
    cell.modify_mech_param('axon', 'kap', 'gkabar', origin='ais')
    cell.modify_mech_param('axon_hill', 'nax', 'sh', x[param_indexes['soma.sh_nas/x']])
    cell.modify_mech_param('axon_hill', 'nax', 'gbar', x[param_indexes['soma.gbar_nas']])
    cell.modify_mech_param('axon', 'nax', 'gbar', x[param_indexes['axon.gbar_nax']])
    for sec_type in ['ais', 'axon']:
        cell.modify_mech_param(sec_type, 'nax', 'sh', origin='axon_hill')
    cell.modify_mech_param('soma', 'Ca', 'gcamult', x[param_indexes['soma.gCa factor']])
    cell.modify_mech_param('soma', 'CadepK', 'gcakmult', x[param_indexes['soma.gCadepK factor']])
    cell.modify_mech_param('soma', 'km3', 'gkmbar', x[param_indexes['soma.gkmbar']])
    cell.modify_mech_param('ais', 'km3', 'gkmbar', x[param_indexes['ais.gkmbar']])
    cell.modify_mech_param('axon_hill', 'km3', 'gkmbar', origin='soma')
    cell.modify_mech_param('axon', 'km3', 'gkmbar', origin='ais')
    cell.modify_mech_param('ais', 'nax', 'sha', x[param_indexes['ais.sha_nax']])
    cell.modify_mech_param('ais', 'nax', 'gbar', x[param_indexes['ais.gbar_nax']])


def export_sim_results():
    """
    Export the most recent time and recorded waveforms from the QuickSim object.
    """
    with h5py.File(context.temp_output_path, 'a') as f:
        context.sim.export_to_file(f)
        

if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)