"""
Uses nested.optimize to tune somatodendritic input resistance in dentate granule cells.

Requires a YAML file to specify required configuration parameters.
Requires use of a nested.parallel interface.
"""
__author__ = 'Aaron D. Milstein and Grace Ng'
from plot_results import *
from nested.optimize_utils import *
import collections
import click
from neuron_wrapper_utils import *

context = Context()


@click.command()
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/optimize_DG_GC_hoc_leak_config.yaml')
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--export", is_flag=True)
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default=None)
@click.option("--disp", is_flag=True)
@click.option("--verbose", type=int, default=1)
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
    config_interactive(config_file_path=config_file_path, output_dir=output_dir, export=export,
                       export_file_path=export_file_path, label=label, disp=disp, verbose=verbose)
    args = get_args_static_leak()
    group_size = len(args[0])
    sequences = [[context.x0_array] * group_size] + args + [[context.export] * group_size]
    primitives = map(compute_features_leak, *sequences)
    features = {key: value for feature_dict in primitives for key, value in feature_dict.iteritems()}
    features, objectives = get_objectives_leak(features)
    print 'params:'
    pprint.pprint(context.x0_dict)
    print 'features:'
    pprint.pprint(features)
    print 'objectives:'
    pprint.pprint(objectives)


def config_interactive(config_file_path=None, output_dir=None, temp_output_path=None, export=False,
                       export_file_path=None, label=None, disp=True, verbose=2, **kwargs):
    """

    :param config_file_path: str (.yaml file path)
    :param output_dir: str (dir path)
    :param temp_output_path: str (.hdf5 file path)
    :param export: bool
    :param export_file_path: str (.hdf5 file path)
    :param label: str
    :param disp: bool
    :param verbose: int
    """

    if config_file_path is not None:
        context.config_file_path = config_file_path
    if 'config_file_path' not in context() or context.config_file_path is None or \
            not os.path.isfile(context.config_file_path):
        raise Exception('config_file_path specifying required parameters is missing or invalid.')
    config_dict = read_from_yaml(context.config_file_path)
    if 'param_names' not in config_dict or config_dict['param_names'] is None:
        raise Exception('config_file at path: %s is missing the following required field: %s' %
                        (context.config_file_path, 'param_names'))
    else:
        context.param_names = config_dict['param_names']
    if 'default_params' not in config_dict or config_dict['default_params'] is None:
        context.default_params = {}
    else:
        context.default_params = config_dict['default_params']
    if 'bounds' not in config_dict or config_dict['bounds'] is None:
        raise Exception('config_file at path: %s is missing the following required field: %s' %
                        (context.config_file_path, 'bounds'))
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
        for param_name in context.default_params:
            context.x0_dict[param_name] = context.default_params[param_name]
        context.x0_array = param_dict_to_array(context.x0_dict, context.param_names)

    missing_config = []
    if 'feature_names' not in config_dict or config_dict['feature_names'] is None:
        missing_config.append('feature_names')
    else:
        context.feature_names = config_dict['feature_names']
    if 'objective_names' not in config_dict or config_dict['objective_names'] is None:
        missing_config.append('objective_names')
    else:
        context.objective_names = config_dict['objective_names']
    if 'target_val' in config_dict:
        context.target_val = config_dict['target_val']
    else:
        context.target_val = None
    if 'target_range' in config_dict:
        context.target_range = config_dict['target_range']
    else:
        context.target_range = None
    if 'optimization_title' in config_dict:
        if config_dict['optimization_title'] is None:
            context.optimization_title = ''
        else:
            context.optimization_title = config_dict['optimization_title']
    if 'kwargs' in config_dict and config_dict['kwargs'] is not None:
        context.kwargs = config_dict['kwargs']  # Extra arguments to be passed to imported sources
    else:
        context.kwargs = {}
    context.kwargs.update(kwargs)
    context.update(context.kwargs)

    if 'update_context' not in config_dict or config_dict['update_context'] is None:
        context.update_context_list = []
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

    context.export = export
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

    context.disp=disp
    context.rel_bounds_handler = RelativeBoundedStep(context.x0_array, context.param_names, context.bounds,
                                                     context.rel_bounds)
    config_worker(context.update_context_funcs, context.param_names, context.default_params, context.feature_names,
                  context.objective_names, context.target_val, context.target_range, context.temp_output_path,
                  context.export_file_path, context.output_dir, context.disp, **context.kwargs)
    update_source_contexts(context.x0_array, context)


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
                  population, spines, **kwargs):
    """
    :param update_context_funcs: list of function references
    :param param_names: list of str
    :param default_params: dict
    :param target_val: dict
    :param target_range: dict
    :param feature_names: list of str
    :param objective_names: list of str
    :param temp_output_path: str
    :param export_file_path: str
    :param output_dir: str (dir path)
    :param disp: bool
    :param mech_file_path: str
    :param gid: int
    :param population: str
    :param spines: bool
    """
    context.update(locals())
    context.update(kwargs)
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
    context.update(locals())


def setup_cell(verbose=1, cvode=False, daspk=False, **kwargs):
    """

    :param verbose: int
    :param cvode: bool
    :param daspk: bool
    """
    if 'comm' not in context():
        try:
            from mpi4py import MPI
            context.comm = MPI.COMM_WORLD
        except Exception:
            raise Exception('optimize_DG_GC_hoc_leak: problem importing from mpi4py; required for config_interactive')
    context.env = Env(comm=context.comm, **kwargs)
    configure_env(context.env)
    cell = get_biophys_cell(context.env, context.gid, context.population)
    init_biophysics(cell, reset_cable=True, from_file=True, mech_file_path=context.mech_file_path, correct_cm=True,
                    correct_g_pas=True, env=context.env)

    # get the thickest apical dendrite ~200 um from the soma
    candidate_branches = []
    candidate_distances = []
    candidate_diams = []
    candidate_locs = []
    for branch in cell.apical:
        if not is_terminal(branch):
            for seg in branch.sec:
                loc = seg.x
                if get_distance_to_node(cell, cell.tree.root, branch, loc) > 130.:
                    candidate_branches.append(branch)
                    candidate_distances.append(get_distance_to_node(cell, cell.tree.root, branch, loc))
                    candidate_diams.append(branch.sec(loc).diam)
                    candidate_locs.append(loc)
                    break
    distance_diffs = np.absolute(np.array(candidate_distances) - 250.)
    #Find the segments within 50 um of the target distance from the soma (250 um)
    indexes = [ind for ind in range(len(distance_diffs)) if distance_diffs[ind] < 50.]
    if len(indexes) == 0:
        index = np.argmin(distance_diffs)
    else:
        diams = np.array([candidate_diams[ind] for ind in indexes])
        index = indexes[np.argmin(diams)]
    dend = candidate_branches[index]
    dend_loc = candidate_locs[index]

    # get the most distal terminal branch > 300 um from the soma
    candidate_branches = []
    candidate_end_distances = []
    for branch in (branch for branch in cell.apical if is_terminal(branch)):
        if get_distance_to_node(cell, cell.tree.root, branch, 0.) >= 250.:
            candidate_branches.append(branch)
            candidate_end_distances.append(get_distance_to_node(cell, cell.tree.root, branch, 1.))
    index = candidate_end_distances.index(max(candidate_end_distances))
    distal_dend = candidate_branches[index]
    distal_dend_loc = 1.

    rec_locs = {'soma': 0., 'dend': dend_loc, 'distal_dend': distal_dend_loc}
    context.rec_locs = rec_locs
    rec_nodes = {'soma': cell.tree.root, 'dend': dend, 'distal_dend': distal_dend}
    context.rec_nodes = rec_nodes

    equilibrate = context.equilibrate
    stim_dur = context.stim_dur
    duration = context.duration
    dt = context.dt

    sim = QuickSim(duration, cvode=cvode, daspk=daspk, dt=dt, verbose=verbose>1)
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
    context.cell = cell


def reset_mechanisms(x, local_context=None):
    if local_context is None:
        local_context = context
    init_biophysics(local_context.cell, reset_cable=False, from_file=True, mech_file_path=local_context.mech_file_path,
                    correct_g_pas=True, env=local_context.env)


def get_args_static_leak():
    """
    A nested map operation is required to compute leak features. The arguments to be mapped are the same (static) for
    each set of parameters.
    :return: list of list
    """
    return [['soma', 'dend', 'distal_dend']]


def compute_features_leak(x, section, export=False, plot=False):
    """
    Inject a hyperpolarizing step current into the specified section, and return the steady-state input resistance.
    :param x: array
    :param section: str
    :param export: bool
    :param plot: bool
    :return: dict: {str: float}
    """
    start_time = time.time()
    update_source_contexts(x, context)
    zero_na(context.cell)

    duration = context.duration
    stim_dur = context.stim_dur
    equilibrate = context.equilibrate
    v_init = context.v_init
    title = 'Rinp_features'
    description = 'step current: %s' % section
    context.sim.tstop = duration
    context.sim.parameters['section'] = section
    context.sim.parameters['title'] = title
    context.sim.parameters['description'] = description
    context.sim.parameters['duration'] = duration
    amp = -0.05
    context.sim.parameters['amp'] = amp
    offset_vm(section)
    loc = context.rec_locs[section]
    node = context.rec_nodes[section]
    rec = context.sim.get_rec(section)
    step_stim_index = context.sim.get_stim_index('step')
    context.sim.modify_stim(step_stim_index, node=node, loc=loc, amp=amp, dur=stim_dur)
    context.sim.run(v_init)
    Rinp = get_Rinp(np.array(context.sim.tvec), np.array(rec['vec']), equilibrate, duration, amp)[2]
    result = {}
    result[section+' R_inp'] = Rinp
    print 'Process: %i: %s: %s took %.1f s, Rinp: %.1f' % (os.getpid(), title, description, time.time() - start_time,
                                                                                    Rinp)
    if plot:
        context.sim.plot()
    if export:
        export_sim_results()
    return result


def get_objectives_leak(features):
    """

    :param features: dict
    :return: tuple of dict
    """
    objectives = {}
    for feature_name in ['soma R_inp', 'dend R_inp']:
        objective_name = feature_name
        objectives[objective_name] = ((context.target_val[objective_name] - features[feature_name]) /
                                                  context.target_range[objective_name]) ** 2.
    this_feature = features['distal_dend R_inp'] - features['dend R_inp']
    objective_name = 'distal_dend R_inp'
    if this_feature < 0.:
        objectives[objective_name] = (this_feature / context.target_range['dend R_inp']) ** 2.
    else:
        objectives[objective_name] = 0.
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
    rec = rec_dict['vec']
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


def update_mechanisms_leak(x, local_context=None):
    """

    :param x: array
    :param local_context: :class:'Context'
    """
    if local_context is None:
        local_context = context
    cell = local_context.cell
    x_dict = param_array_to_dict(x, local_context.param_names)
    modify_mech_param(cell, 'soma', 'pas', 'g', x_dict['soma.g_pas'])
    modify_mech_param(cell, 'apical', 'pas', 'g', origin='soma', slope=x_dict['dend.g_pas slope'],
                      tau=x_dict['dend.g_pas tau'])
    for sec_type in ['axon_hill', 'axon', 'ais', 'apical', 'spine_neck', 'spine_head']:
        update_mechanism_by_sec_type(cell, sec_type, 'pas')
    if not local_context.spines:
        correct_cell_for_spines_g_pas(cell, local_context.env)


def export_sim_results():
    """
    Export the most recent time and recorded waveforms from the QuickSim object.
    """
    with h5py.File(context.temp_output_path, 'a') as f:
        context.sim.export_to_file(f)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
