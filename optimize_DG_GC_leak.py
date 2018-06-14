"""
Uses nested.optimize to tune somatodendritic input resistance in dentate granule cells.

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
              default='config/optimize_DG_GC_leak_config.yaml')
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--export", is_flag=True)
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default=None)
@click.option("--verbose", type=int, default=2)
@click.option("--plot", is_flag=True)
def main(config_file_path, output_dir, export, export_file_path, label, verbose, plot):
    """

    :param config_file_path: str (path)
    :param output_dir: str (path)
    :param export: bool
    :param export_file_path: str
    :param label: str
    :param verbose: bool
    :param plot: bool
    """
    # requires a global variable context: :class:'Context'
    context.update(locals())
    disp = verbose > 0
    config_interactive(context, __file__, config_file_path=config_file_path, output_dir=output_dir, export=export,
                       export_file_path=export_file_path, label=label, disp=disp)
    args = get_args_static_leak()
    group_size = len(args[0])
    sequences = [[context.x0_array] * group_size] + args + [[context.export] * group_size] + \
                [[context.plot] * group_size]
    primitives = map(compute_features_leak, *sequences)
    features = {key: value for feature_dict in primitives for key, value in feature_dict.iteritems()}
    features, objectives = get_objectives_leak(features)
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
    cell = context.cell
    sim = context.sim
    if not sim.has_rec('soma'):
        sim.append_rec(cell, cell.tree.root, name='soma', loc=0.5)
    if context.v_init not in context.i_holding['soma']:
        context.i_holding['soma'][context.v_init] = 0.
    if not sim.has_rec('dend'):
        dend, dend_loc = get_DG_GC_thickest_dend_branch(context.cell, 200., terminal=False)
        sim.append_rec(cell, dend, name='dend', loc=dend_loc)
    if context.v_init not in context.i_holding['dend']:
        context.i_holding['dend'][context.v_init] = 0.
    if not sim.has_rec('term_dend'):
        term_dend = get_DG_GC_distal_most_terminal_branch(context.cell, 250.)
        sim.append_rec(cell, term_dend, name='term_dend', loc=1.)
    if context.v_init not in context.i_holding['term_dend']:
        context.i_holding['term_dend'][context.v_init] = 0.

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


def get_args_static_leak():
    """
    A nested map operation is required to compute leak features. The arguments to be mapped are the same (static) for
    each set of parameters.
    :return: list of list
    """
    return [['soma', 'dend', 'term_dend']]


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
    config_sim_env(context)
    update_source_contexts(x, context)
    zero_na(context.cell)

    duration = context.duration
    stim_dur = context.stim_dur
    equilibrate = context.equilibrate
    v_init = context.v_init
    dt = context.dt
    sim = context.sim

    title = 'R_inp'
    description = 'step current injection to %s' % section
    sim.parameters['section'] = section
    sim.parameters['title'] = title
    sim.parameters['description'] = description
    sim.parameters['duration'] = duration
    amp = -0.05
    context.sim.parameters['amp'] = amp
    offset_vm(section, context, v_init)
    rec_dict = sim.get_rec(section)
    loc = rec_dict['loc']
    node = rec_dict['node']
    rec = rec_dict['vec']

    sim.modify_stim('step', node=node, loc=loc, amp=amp, dur=stim_dur)
    sim.backup_state()
    sim.set_state(dt=dt, tstop=duration, cvode=True)
    sim.run(v_init)

    R_inp = get_R_inp(np.array(sim.tvec), np.array(rec), equilibrate, duration, amp, dt)[2]
    result = dict()
    result['%s R_inp' % section] = R_inp
    if context.verbose > 0:
        print 'compute_features_leak: pid: %i; %s: %s took %.1f s; R_inp: %.1f' % \
              (os.getpid(), title, description, time.time() - start_time, R_inp)
    if plot:
        sim.plot()
    if export:
        context.sim.export_to_file(context.temp_output_path)
    sim.restore_state()
    sim.modify_stim('step', amp=0.)
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
    delta_term_dend_R_inp = features['term_dend R_inp'] - features['dend R_inp']
    objective_name = 'term_dend R_inp'
    if delta_term_dend_R_inp < 0.:
        objectives[objective_name] = (delta_term_dend_R_inp / context.target_range['dend R_inp']) ** 2.
    else:
        objectives[objective_name] = 0.
    return features, objectives


def update_mechanisms_leak(x, context):
    """

    :param x: array
    :param context: :class:'Context'
    """
    if context is None:
        raise RuntimeError('update_mechanisms_leak: missing required Context object')
    cell = context.cell
    x_dict = param_array_to_dict(x, context.param_names)
    modify_mech_param(cell, 'soma', 'pas', 'g', x_dict['soma.g_pas'])
    modify_mech_param(cell, 'apical', 'pas', 'g', origin='soma', slope=x_dict['dend.g_pas slope'],
                      tau=x_dict['dend.g_pas tau'])
    for sec_type in ['axon_hill', 'ais', 'axon', 'apical', 'spine_neck', 'spine_head']:
        update_mechanism_by_sec_type(cell, sec_type, 'pas')
    if context.correct_for_spines:
        correct_cell_for_spines_g_pas(cell, context.env)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
