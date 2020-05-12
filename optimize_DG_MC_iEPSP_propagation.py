"""
Uses nested.optimize to tune frequency-dependent attenuation of EPSPs from dendrite to soma in dentate granule cells.

Requires a YAML file to specify required configuration parameters.
Requires use of a nested.parallel interface.
"""
__author__ = 'Aaron D. Milstein, Grace Ng, and Prannath Moolchand'
from dentate.biophysics_utils import *
from nested.parallel import *
from nested.optimize_utils import *
from cell_utils import *
import click

context = Context()


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True, ))
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/optimize_DG_MC_iEPSP_propagation_config.yaml')
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
        model_label = 'test'

    features = dict()
    # Stage 0:
    args = context.interface.execute(get_args_dynamic_i_EPSC, context.x0_array, features)
    group_size = len(args[0])
    sequences = [[context.x0_array] * group_size] + args + [[model_id] * group_size] + \
                [[context.export] * group_size] + [[context.plot] * group_size]
    primitives = context.interface.map(compute_features_iEPSP_i_unit, *sequences)
    features = {key: value for feature_dict in primitives for key, value in viewitems(feature_dict)}
    context.update(locals())

    # Stage 1:
    args = context.interface.execute(get_args_dynamic_iEPSP_unit, context.x0_array, features)
    group_size = len(args[0])
    sequences = [[context.x0_array] * group_size] + args + [[model_id] * group_size] + \
                [[context.export] * group_size] + [[context.plot] * group_size]
    primitives = context.interface.map(compute_features_iEPSP_i_unit, *sequences)
    this_features = context.interface.execute(filter_features_iEPSP_attenuation, primitives, features,
                                              model_id, context.export)
    features.update(this_features)
    
    features, objectives = context.interface.execute(get_objectives_iEPSP_attenuation, features, model_id,
                                                     context.export)
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

    equilibrate = 250.  # time to steady-state
    stim_dur = 50.
    trace_baseline = 10.
    duration = equilibrate + stim_dur
    dt = 0.025
    v_init = -66.
    v_active = -60.
    syn_mech_name = 'EPSC'
    initial_i_EPSC = -0.025

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
    if context.v_init not in context.i_holding['soma']:
        context.i_holding['soma'][context.v_init] = 0.
    dend, dend_loc = get_thickest_dend_branch(context.cell, 100., terminal=False)
    if not sim.has_rec('dend'):
        sim.append_rec(cell, dend, name='dend', loc=dend_loc)
    if not sim.has_rec('dend_local'):
        sim.append_rec(cell, cell.tree.root, name='dend_local', loc=0.5)

    equilibrate = context.equilibrate
    duration = context.duration

    if not sim.has_stim('holding'):
        sim.append_stim(cell, cell.tree.root, name='holding', loc=0.5, amp=0., delay=0., dur=duration)

    if 'i_syn_attrs' not in context():
        context.i_syn_attrs = []
        this_syn_attr_dict = {}
        node = dend
        loc = dend_loc
        this_syn_attr_dict['name'] = 'dend_ref'
        this_syn_attr_dict['node'] = node
        for seg in node.sec:
            if seg.x >= loc:
                loc = seg.x
                break
        seg = node.sec(loc)
        this_syn_attr_dict['loc'] = loc
        this_syn = make_unique_synapse_mech(context.syn_mech_name, seg)
        this_nc, this_vs = mknetcon_vecstim(this_syn)
        this_syn_attr_dict['syn'] = this_syn
        this_syn_attr_dict['nc'] = this_nc
        this_syn_attr_dict['vs'] = this_vs
        this_syn_attr_dict['distance'] = get_distance_to_node(context.cell, context.cell.tree.root, node, loc)
        config_syn(context.syn_mech_name, context.env.synapse_attributes.syn_param_rules, syn=this_syn, nc=this_nc,
                   i_unit=context.initial_i_EPSC, tau_rise=1., tau_decay=10.)
        context.i_syn_attrs.append(this_syn_attr_dict)

        node_arr_list = get_dend_segments(context.cell, ref_seg=dend, all_seg=True, dist_bounds=[0, 300])

        for i, node_arr in enumerate(node_arr_list):
            this_syn_attr_dict = {}
            node = node_arr[0]
            loc = node_arr[1]
            this_syn_attr_dict['name'] = 'dend_syn_{:03d}'.format(i+1)
            this_syn_attr_dict['node'] = node
            seg = node.sec(loc)
            this_syn_attr_dict['loc'] = loc
            this_syn = make_unique_synapse_mech(context.syn_mech_name, seg)
            this_nc, this_vs = mknetcon_vecstim(this_syn)
            this_syn_attr_dict['syn'] = this_syn
            this_syn_attr_dict['nc'] = this_nc
            this_syn_attr_dict['vs'] = this_vs
            this_syn_attr_dict['distance'] = get_distance_to_node(context.cell, context.cell.tree.root, node, loc)
            config_syn(context.syn_mech_name, context.env.synapse_attributes.syn_param_rules, syn=this_syn, nc=this_nc,
                       i_unit=context.initial_i_EPSC, tau_rise=1., tau_decay=10.)
            context.i_syn_attrs.append(this_syn_attr_dict)

    context.previous_module = __file__


def get_attenuation_data():
    distance = np.array([  0.        ,  40.5904059 ,  45.29520295,  47.50922509, 77.39852399,  90.12915129,
                           90.68265683, 121.95571956, 126.93726937, 133.02583026, 138.83763838, 145.20295203,
                           150.46125461, 156.5498155 , 185.60885609, 185.88560886, 293.26568266])

    attenuation = np.array([1.        , 0.4453202 , 1.03940887, 0.16157635, 0.21871921, 0.2       ,
                            0.14384236, 0.19014778, 0.12512315, 0.17536946, 0.23251232, 0.13300493,
                            0.38128079, 0.13103448, 0.0955665 , 0.13103448, 0.08571429])
    return distance, attenuation


def get_gompertz_coeffs(optimize=False):
    if optimize:
        distance, attenuation = get_attenuation_data()
        results = optimize.curve_fit(gompertz,  distance,  attenuation, p0=[-0.9, 1.5, 0.2,40])
    else:
        results = np.array([-0.87,  1.06160453,  0.06154768, 35.55592743])
    return results


def iEPSP_amp_error(x, syn_index):
    """

    :param x: array
    :param syn_index: int
    :return: float
    """
    start_time = time.time()
    this_syn_attr_dict = context.i_syn_attrs[syn_index]
    this_syn = this_syn_attr_dict['syn']
    this_nc = this_syn_attr_dict['nc']
    config_syn(context.syn_mech_name, context.env.synapse_attributes.syn_param_rules, syn=this_syn, nc=this_nc,
               i_unit=x[0])

    dt = context.dt
    equilibrate = context.equilibrate
    sim = context.sim
    sim.run(context.v_init)

    vm = np.array(context.sim.get_rec('soma')['vec'])
    baseline = np.mean(vm[int((equilibrate - 3.) / dt):int((equilibrate - 1.) / dt)])
    vm -= baseline
    iEPSP_amp = np.max(vm[int(equilibrate / dt):])
    Err = ((iEPSP_amp - context.target_val['iEPSP_unit_amp']) / (0.01 * context.target_val['iEPSP_unit_amp'])) ** 2.
    if context.verbose > 1:
        print('iEPSP_amp_error: %s.i_unit: %.3f, soma iEPSP amp: %.2f; took %.1f s' %
              (context.syn_mech_name, x[0], iEPSP_amp, time.time() - start_time))
        sys.stdout.flush()
    return Err


def get_args_dynamic_i_EPSC(x, features):
    if 'i_holding' not in features:
        i_holding = context.i_holding
    else:
        i_holding = features['i_holding']

    return [[i_holding], [0], [None]]


def get_args_dynamic_iEPSP_unit(x, features):
    if 'i_holding' not in features:
        i_holding = context.i_holding
    else:
        i_holding = features['i_holding']

    i_EPSC = features['i_EPSC']
    syn_count = len(context.i_syn_attrs) - 1

    return [[i_holding] * syn_count, list(range(1, syn_count + 1)), [i_EPSC] * syn_count]


def compute_features_iEPSP_i_unit(x, i_holding, syn_index, i_EPSC=None, model_id=None, export=False, plot=False):
    """

    :param x: array
    :param i_holding: defaultdict(dict: float)
    :param syn_index: int
    :param i_EPSC: float
    :param model_id: int or str
    :param export: bool
    :param plot: bool
    :return: dict
    """
    start_time = time.time()
    config_sim_env(context)
    update_source_contexts(x, context)
    zero_na(context.cell)

    dt = context.dt
    duration = context.duration
    equilibrate = context.equilibrate

    v_init = context.v_init
    context.i_holding = i_holding
    offset_vm('soma', context, v_init, i_history=context.i_holding, dynamic=False, cvode=context.cvode)

    sim = context.sim
    sim.modify_stim('holding', dur=duration)
    sim.backup_state()
    sim.set_state(dt=dt, tstop=duration, cvode=False)

    this_syn_attr_dict = context.i_syn_attrs[syn_index]
    this_syn_name = this_syn_attr_dict['name']
    this_node = this_syn_attr_dict['node']
    this_loc = this_syn_attr_dict['loc']
    this_syn = this_syn_attr_dict['syn']
    this_nc = this_syn_attr_dict['nc']
    this_vs = this_syn_attr_dict['vs']
    this_vs.play(h.Vector([context.equilibrate]))

    if i_EPSC is None:
        i_EPSC = context.initial_i_EPSC
        bounds = [(-0.3, -0.005)]
        i_EPSC_result = scipy.optimize.minimize(iEPSP_amp_error, [i_EPSC], args=(syn_index,), method='L-BFGS-B',
                                                bounds=bounds, options={'ftol': 1e-3, 'disp': context.verbose > 1,
                                                                        'maxiter': 3})
        i_EPSC = i_EPSC_result.x[0]

    config_syn(context.syn_mech_name, context.env.synapse_attributes.syn_param_rules, syn=this_syn, nc=this_nc,
               i_unit=i_EPSC)
    sim.modify_rec(name='dend_local', node=this_node, loc=this_loc)
    sim.run(v_init)

    soma_vm = np.array(sim.get_rec('soma')['vec'])
    soma_baseline = np.mean(soma_vm[int((equilibrate - 3.) / dt):int((equilibrate - 1.) / dt)])
    soma_vm -= soma_baseline
    soma_iEPSP_amp = np.max(soma_vm[int(equilibrate / dt):])

    dend_local_vm = np.array(sim.get_rec('dend_local')['vec'])
    dend_local_baseline = np.mean(dend_local_vm[int((equilibrate - 3.) / dt):int((equilibrate - 1.) / dt)])
    dend_local_vm -= dend_local_baseline
    dend_local_iEPSP_amp = np.max(dend_local_vm[int(equilibrate / dt):])

    result = dict()
    result['i_EPSC'] = i_EPSC
    result['i_holding'] = context.i_holding
    result['i_EPSP_{!s}'.format(this_syn_name)] = {'soma_iEPSP_amp': soma_iEPSP_amp,
                                                   'dend_local_iEPSP_amp': dend_local_iEPSP_amp,
                                                   'syn_index': syn_index}

    title = 'iEPSP_unit'
    description = '{!s} ({:d} um from soma)'.format(this_syn_name, int(this_syn_attr_dict['distance']))
    sim.parameters = dict()
    sim.parameters['duration'] = duration
    sim.parameters['equilibrate'] = equilibrate
    sim.parameters['title'] = title
    sim.parameters['description'] = description

    if context.verbose > 0:
        print('compute_features_iEPSP_i_unit: pid: %i; model_id: %s; %s: %s took %.3f s' %
              (os.getpid(), model_id, title, description, time.time() - start_time))
        sys.stdout.flush()
    if plot:
        context.sim.plot()
    if export:
        context.sim.export_to_file(context.temp_output_path, model_label=model_id, category=title)
    sim.restore_state()
    this_vs.play(h.Vector())

    return result


def filter_features_iEPSP_attenuation(primitives, features, model_id=None, export=False):
    """

    :param primitives: list of dict
    :param features: dict
    :param model_id: int or str
    :param export: bool
    :return: dict
    """

    primitives.append({'i_EPSP_dend_ref': features['i_EPSP_dend_ref']})

    soma_amp_list = []
    dend_local_amp_list = []
    atten_list = []
    dist_list = []

    for res in primitives:
        for key in res:
            if key.startswith('i_EPSP'):
                break
        syn_index = res[key]['syn_index']
        soma_iEPSP_amp = res[key]['soma_iEPSP_amp']
        dend_local_iEPSP_amp = res[key]['dend_local_iEPSP_amp']
        iEPSP_attenuation = soma_iEPSP_amp / dend_local_iEPSP_amp
        distance = context.i_syn_attrs[syn_index]['distance']

        if distance <= context.max_i_EPSP_attenuation_distance:
            soma_amp_list.append(soma_iEPSP_amp)
            dend_local_amp_list.append(dend_local_iEPSP_amp)
            atten_list.append(iEPSP_attenuation)
            dist_list.append(distance)

    dist_arr, uniq_idx = np.unique(dist_list, return_index=True)
    atten_arr = np.array(atten_list)[uniq_idx]
    soma_amp_arr = np.array(soma_amp_list)[uniq_idx]
    dend_local_amp_arr = np.array(dend_local_amp_list)[uniq_idx]

    new_features = {'attenuation_array': atten_arr, 'distance_array': dist_arr}

    if export:
        description = 'iEPSP_attenuation'
        exp_distance, exp_attenuation = get_attenuation_data()
        with h5py.File(context.temp_output_path, 'a') as f:
            target = get_h5py_group(f, [model_id, description], create=True)
            target.attrs['i_EPSC'] = features['i_EPSC']
            target.create_dataset('distance', compression='gzip', data=dist_arr)
            target.create_dataset('attenuation', compression='gzip', data=atten_arr)
            target.create_dataset('soma_iEPSP_amp', compression='gzip', data=soma_amp_arr)
            target.create_dataset('dend_local_iEPSP_amp', compression='gzip', data=dend_local_amp_arr)
            target.create_dataset('exp_distance', compression='gzip', data=exp_distance)
            target.create_dataset('exp_attenuation', compression='gzip', data=exp_attenuation)

    return new_features


def get_objectives_iEPSP_attenuation(features, model_id=None, export=False):
    """

    :param features: dict
    :param model_id: int or str
    :param export: bool
    :return: tuple of dict
    """
    objectives = dict()

    x = features['distance_array']
    atten = features['attenuation_array']

    gompertz_coeffs = get_gompertz_coeffs()
    expected_atten = gompertz(x, *gompertz_coeffs)
    atten_resi = np.mean(((expected_atten-atten)/context.target_range['iEPSP_attenuation'])**2.)
    objectives['iEPSP_attenuation_residual'] = atten_resi

    if export:
        with h5py.File(context.temp_output_path, 'a') as f:
            description = 'iEPSP_attenuation'
            target = get_h5py_group(f, [model_id, description], create=True)
            target.create_dataset('gompertz_coeffs', compression='gzip', data=gompertz_coeffs)
            target.attrs['gompertz_fn'] = '1+a*np.exp(-b*np.exp(-c*(t-m))), coeffs=(a,b,c,m)'

    return features, objectives


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
