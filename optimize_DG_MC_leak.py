"""
Uses nested.optimize to tune somatodendritic input resistance in dentate mossy cells.

Requires a YAML file to specify required configuration parameters.
Requires use of a nested.parallel interface.
"""
__author__ = 'Aaron D. Milstein and Prannath Moolchand'
from dentate.biophysics_utils import *
from nested.parallel import *
from nested.optimize_utils import *
from cell_utils import *
import click

context = Context()


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True, ))
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/optimize_DG_MC_leak_config.yaml')
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--export", is_flag=True)
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default=None)
@click.option("--verbose", type=int, default=2)
@click.option("--plot", is_flag=True)
@click.option("--interactive", is_flag=True)
@click.option("--debug", is_flag=True)
@click.pass_context
def main(cli, config_file_path, output_dir, export, export_file_path, label, verbose, plot, interactive, debug):
    """

    :param cli: contains unrecognized args as list of str
    :param config_file_path: str (path)
    :param output_dir: str (path)
    :param export: bool
    :param export_file_path: str
    :param label: str
    :param verbose: bool
    :param plot: bool
    :param interactive: bool
    :param debug: bool
    """
    # requires a global variable context: :class:'Context'
    context.update(locals())
    kwargs = get_unknown_click_arg_dict(cli.args)
    context.disp = verbose > 0

    context.interface = get_parallel_interface(source_file=__file__, source_package=__package__, **kwargs)
    context.interface.start(disp=context.disp)
    context.interface.ensure_controller()
    config_optimize_interactive(__file__, config_file_path=config_file_path, output_dir=output_dir,
                                export=export, export_file_path=export_file_path, label=label,
                                disp=context.disp, interface=context.interface, verbose=verbose, plot=plot,
                                debug=debug, **kwargs)

    if not debug:
        run_tests()

    if not interactive:
        context.interface.stop()


def run_tests():
    model_id = 0
    if 'model_key' in context() and context.model_key is not None:
        model_label = context.model_key
    else:
        model_label = [[]]

    args = context.interface.execute(get_args_static_leak)
    group_size = len(args[0])
    sequences = [[context.x0_array] * group_size] + args + [[model_id] * group_size] + \
                [[context.export] * group_size] + [[context.plot] * group_size]
    primitives = context.interface.map(compute_features_leak, *sequences)
    features = {key: value for feature_dict in primitives for key, value in viewitems(feature_dict)}
    features, objectives = context.interface.execute(get_objectives_leak, features, model_id, context.export)

    if context.export:
        merge_exported_data(context, param_arrays=[context.x0_array],
                            model_ids=[model_id], model_labels=[model_label], features=[features],
                            objectives=[objectives], export_file_path=context.export_file_path,
                            verbose=context.verbose > 1)
    sys.stdout.flush()
    print('model_id: %i; model_labels: %s' % (model_id, model_label))
    print('params:')
    pprint.pprint(context.x0_dict)
    print('features:')
    pprint.pprint(features)
    print('objectives:')
    pprint.pprint(objectives)
    sys.stdout.flush()
    time.sleep(.1)
    if context.plot:
        context.interface.apply(plt.show)
    context.update(locals())


def config_worker():
    """

    """
    if 'verbose' in context():
        context.verbose = int(context.verbose)
    if not context_has_sim_env(context):
        build_sim_env(context, **context.kwargs)
    else:
        config_sim_env(context)


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
    v_init = -66.
    v_active = -60.
    i_holding_max = 0.5  # nA
    context.update(locals())


def build_sim_env(context, verbose=2, cvode=True, daspk=True, load_edges=False, set_edge_delays=False, **kwargs):
    """

    :param context: :class:'Context'
    :param verbose: int
    :param cvode: bool
    :param daspk: bool
    """
    verbose = int(verbose)
    init_context()
    context.env = Env(comm=context.comm, verbose=verbose > 1, **kwargs)
    configure_hoc_env(context.env)
    cell = get_biophys_cell(context.env, gid=context.gid, pop_name=context.cell_type, load_edges=load_edges,
                            set_edge_delays=set_edge_delays, mech_file_path=context.mech_file_path)
    init_biophysics(cell, reset_cable=True, correct_cm=context.correct_for_spines,
                    correct_g_pas=context.correct_for_spines, env=context.env, verbose=verbose > 1)
    context.sim = QuickSim(context.duration, cvode=cvode, daspk=daspk, dt=context.dt, verbose=verbose > 1)
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
    sim = context.sim
    if not sim.has_rec('soma'):
        sim.append_rec(cell, cell.tree.root, name='soma', loc=0.5)
    if context.v_active not in context.i_holding['soma']:
        context.i_holding['soma'][context.v_active] = 0.
    if not sim.has_rec('dend'):
        dend, dend_loc = get_thickest_dend_branch(context.cell, 100., terminal=False)
        sim.append_rec(cell, dend, name='dend', loc=dend_loc)
    if context.v_active not in context.i_holding['dend']:
        context.i_holding['dend'][context.v_active] = 0.
    if not sim.has_rec('term_dend'):
        term_dend = get_distal_most_terminal_branch(context.cell, 250.)
        sim.append_rec(cell, term_dend, name='term_dend', loc=1.)
    if context.v_active not in context.i_holding['term_dend']:
        context.i_holding['term_dend'][context.v_active] = 0.

    equilibrate = context.equilibrate
    stim_dur = context.stim_dur
    duration = context.duration

    if not sim.has_stim('step'):
        sim.append_stim(cell, cell.tree.root, name='step', loc=0.5, amp=0., delay=equilibrate, dur=stim_dur)
    if not sim.has_stim('holding'):
        sim.append_stim(cell, cell.tree.root, name='holding', loc=0.5, amp=0., delay=0., dur=duration)

    context.previous_module = __file__


def get_args_static_leak():
    """
    A nested map operation is required to compute leak features. The arguments to be mapped are the same (static) for
    each set of parameters.
    :return: list of list
    """
    sections = ['soma', 'dend', 'term_dend']
    block_h = [False] * len(sections)
    sections.append('soma')
    block_h.append(True)
    return [sections, block_h]


def compute_features_leak(x, section, block_h, model_id=None, export=False, plot=False):
    """
    Inject a hyperpolarizing step current into the specified section, and return the steady-state input resistance.
    :param x: array
    :param section: str
    :param block_h: bool; whether or not to zero the h conductance
    :param model_id: int or str
    :param export: bool
    :param plot: bool
    :return: dict: {str: float}
    """
    start_time = time.time()
    config_sim_env(context)
    update_source_contexts(x, context)
    zero_na(context.cell)
    if block_h:
        zero_h(context.cell)

    duration = context.duration
    stim_dur = context.stim_dur
    equilibrate = context.equilibrate
    v_active = context.v_active
    dt = context.dt
    sim = context.sim

    title = 'R_inp'
    if block_h:
        description = 'step current injection to %s (no h)' % section
    else:
        description = 'step current injection to %s' % section
    sim.parameters = dict()
    sim.parameters['duration'] = duration
    sim.parameters['equilibrate'] = equilibrate
    sim.parameters['section'] = section
    sim.parameters['title'] = title
    sim.parameters['description'] = description
    amp = -0.025
    context.sim.parameters['amp'] = amp
    vm_rest, vm_offset, context.i_holding[section][v_active] = offset_vm(section, context, v_active, dynamic=False,
                                                                       cvode=context.cvode)
    sim.modify_stim('holding', dur=duration)
    rec_dict = sim.get_rec(section)

    loc = rec_dict['loc']
    node = rec_dict['node']
    rec = rec_dict['vec']

    sim.modify_stim('step', node=node, loc=loc, amp=amp, dur=stim_dur)
    sim.backup_state()
    sim.set_state(dt=dt, tstop=duration, cvode=context.cvode)
    sim.run(v_active)

    R_inp = get_R_inp(np.array(sim.tvec), np.array(rec), equilibrate, duration, amp, dt)[2]
    result = dict()
    failed = False
    if section == 'soma':
        if block_h:
            result['%s R_inp (no h)' % section] = R_inp
            result['soma vm_rest (no h)'] = vm_rest
        else:
            result['%s R_inp' % section] = R_inp
            result['soma vm_rest'] = vm_rest
            if abs(context.i_holding[section][v_active]) > context.i_holding_max:
                if context.verbose > 0:
                    print('compute_features_leak: pid: %i; model_id: %s; aborting - required i_holding out of range: '
                          '%.1f' % (os.getpid(), model_id, context.i_holding[section][v_active]))
                    sys.stdout.flush()
                failed = True
            else:
                result['i_holding'] = context.i_holding
    else:
        result['%s R_inp' % section] = R_inp
    if context.verbose > 0:
        print('compute_features_leak: pid: %i; model_id: %s; %s: %s took %.1f s; R_inp: %.1f' % \
              (os.getpid(), model_id, title, description, time.time() - start_time, R_inp))
        sys.stdout.flush()
    if plot:
        sim.plot()
    if export:
        context.sim.export_to_file(context.temp_output_path, model_label=model_id, category=title)
    sim.restore_state()
    sim.modify_stim('step', amp=0.)
    if failed:
        return dict()

    return result


def get_objectives_leak(features, model_id=None, export=False):
    """

    :param features: dict
    :param model_id: int or str
    :param export: bool
    :return: tuple of dict
    """
    objectives = {}
    for feature_name in ['soma R_inp', 'dend R_inp', 'soma vm_rest']:
        objective_name = feature_name
        objectives[objective_name] = ((context.target_val[objective_name] - features[feature_name]) /
                                                  context.target_range[objective_name]) ** 2.

    for base_feature_name in ['soma R_inp', 'soma vm_rest']:
        feature_name = base_feature_name + ' (no h)'
        objective_name = feature_name
        objectives[objective_name] = ((context.target_val[objective_name] - features[feature_name]) /
                                      context.target_range[base_feature_name]) ** 2.

    delta_term_dend_R_inp = None
    if features['term_dend R_inp'] < features['dend R_inp']:
        delta_term_dend_R_inp = features['term_dend R_inp'] - features['dend R_inp']
    elif features['term_dend R_inp'] > context.target_val['term_dend R_inp']:
        delta_term_dend_R_inp = features['term_dend R_inp'] - context.target_val['term_dend R_inp']

    objective_name = 'term_dend R_inp'
    if delta_term_dend_R_inp is not None:
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
    modify_mech_param(cell, 'soma', 'h', 'ghbar', x_dict['soma.ghbar'])
    modify_mech_param(cell, 'soma', 'pas', 'e', x_dict['e_pas'])
    modify_mech_param(cell, 'apical', 'pas', 'g', origin='soma', slope=x_dict['dend.g_pas slope'],
                      tau=x_dict['dend.g_pas tau'])
    modify_mech_param(cell, 'apical', 'pas', 'g', origin='parent', custom={'func': 'custom_filter_if_terminal'},
                      append=True)
#    modify_mech_param(cell, 'apical', 'h', 'ghbar', origin='soma')
    modify_mech_param(cell, 'apical', 'h', 'ghbar', x_dict['dend.ghbar'])
    for sec_type in ['hillock', 'ais', 'axon', 'apical']:
        update_mechanism_by_sec_type(cell, sec_type, 'pas')
    if context.correct_for_spines:
        correct_cell_for_spines_g_pas(cell, context.env, context.verbose > 1)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
