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
    features = dict()

    # Stage 0:
    args = context.interface.execute(get_args_dynamic_iEPSP_unit_optimize, context.x0_array, features)
    group_size = len(args[0])
    sequences = [[context.x0_array] * group_size] + args + [[context.export] * group_size] + \
                [[context.plot] * group_size]
    primitives = context.interface.map(compute_features_iEPSP_i_unit, *sequences)
    features = {key: value for feature_dict in primitives for key, value in viewitems(feature_dict)}
    context.update(locals())

    args = context.interface.execute(get_args_dynamic_iEPSP_unit, context.x0_array, features)
    group_size = len(args[0])
    sequences = [[context.x0_array] * group_size] + args + [[context.export] * group_size] + \
                [[context.plot] * group_size]
    primitives = context.interface.map(compute_features_iEPSP_i_unit, *sequences)
    this_features = context.interface.execute(filter_features_attenuation, primitives, features, context.export)
    features.update(this_features)
    
    features, objectives = context.interface.execute(get_objectives_iEPSP_attenuation, features, context.export)
    if context.export:
        collect_and_merge_temp_output(context.interface, context.export_file_path, verbose=context.disp)
    sys.stdout.flush()

    time.sleep(1.)
    print('params:')
    pprint.pprint(context.x0_dict)
    print('features:')
    pprint.pprint(features)
    print('objectives:')
    pprint.pprint(objectives)
    sys.stdout.flush()
    time.sleep(1.)
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

    context.update(locals())


def build_sim_env(context, verbose=2, cvode=True, daspk=True, **kwargs):
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
    cell = get_biophys_cell(context.env, gid=context.gid, pop_name=context.cell_type, load_edges=False,
                            mech_file_path=context.mech_file_path)
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
    if 'i_EPSC' not in context():
        context.i_EPSC = defaultdict()
        context.i_EPSC['ini_syn_amp'] = -0.05
    cell = context.cell
    sim = context.sim
    if not sim.has_rec('soma'):
        sim.append_rec(cell, cell.tree.root, name='soma', loc=0.5)
    if context.v_active not in context.i_holding['soma']:
        context.i_holding['soma'][context.v_active] = 0.
    dend, dend_loc = get_thickest_dend_branch(context.cell, 100., terminal=False)
    if not sim.has_rec('dend'):
        sim.append_rec(cell, dend, name='dend', loc=dend_loc)

    if 'dend' not in context.i_EPSC:
        context.i_EPSC['dend'] = -0.05

    if not sim.has_rec('dendlocal'):
        sim.append_rec(cell, cell.tree.root, name='dendlocal', loc=0.5)

    equilibrate = context.equilibrate
    duration = context.duration

    if not sim.has_stim('holding'):
        sim.append_stim(cell, cell.tree.root, name='holding', loc=0.5, amp=0., delay=0., dur=duration)

    if 'i_syn' not in context():
        context.syns = defaultdict()
        context.i_syn = []
        rec_dict = sim.get_rec('dend')
        node = rec_dict['node']
        loc = rec_dict['loc']
        seg = node.sec(loc)
        dist = get_distance_to_node(cell, cell.tree.root, node, loc)
        context.syns['dend'] = {'node': node, 'loc': loc, 'seg': seg, 'idx': 0, 'dist': dist}
        context.i_syn.append(make_unique_synapse_mech(context.syn_mech_name, seg))
        config_syn(context.syn_mech_name, context.env.synapse_attributes.syn_param_rules, syn=context.i_syn[-1],
                   i_unit=context.i_EPSC['ini_syn_amp'], tau_rise=1., tau_decay=10.)

    if 'i_nc' not in context():
        context.i_nc = []
        context.i_vs = []
        i_nc, i_vs = mknetcon_vecstim(context.i_syn[-1])
        context.i_nc.append(i_nc)
        context.i_vs.append(i_vs)

    if not hasattr(context.cell, 'node_loc_arr'):
        node_loc_arr = get_dend_segments(context.cell, ref_seg=dend, all_seg=True, dist_bounds=[0, 300])
        context.cell.node_loc_arr = node_loc_arr
        N_syn = node_loc_arr.shape[0]
        context.cell.N_syn = N_syn
    
        for dendi, dend in enumerate(node_loc_arr):
            node = dend[0]
            loc = dend[1]
            seg = node.sec(loc)
            idx = dendi + 1
            dist = dend[2]
            dendn = 'dend{:03d}'.format(dendi)
            context.syns[dendn] = {'node': node, 'loc': loc, 'seg': seg, 'idx': idx, 'dist': dist}
            context.i_syn.append(make_unique_synapse_mech(context.syn_mech_name, seg))
            config_syn(context.syn_mech_name, context.env.synapse_attributes.syn_param_rules, syn=context.i_syn[idx],
                       i_unit=context.i_EPSC['ini_syn_amp'], tau_rise=1., tau_decay=10.)
    
            i_nc, i_vs = mknetcon_vecstim(context.i_syn[idx])
            context.i_nc.append(i_nc)
            context.i_vs.append(i_vs)

    sim.parameters['duration'] = duration
    sim.parameters['equilibrate'] = equilibrate
    context.previous_module = __file__


def iEPSP_amp_error(x, idx):
    """

    :param x: array
    :return: float
    """
    start_time = time.time()
    config_syn(context.syn_mech_name, context.env.synapse_attributes.syn_param_rules, syn=context.i_syn[idx], i_unit=x[0])

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
        print('iEPSP_amp_error: %s.i_unit: %.3f, soma iEPSP amp: %.2f; took %.1f s' %
              (context.syn_mech_name, x[0], iEPSP_amp, time.time() - start_time))
        sys.stdout.flush()
    return Err


def get_args_dynamic_iEPSP_unit_optimize(x, features):
    if 'i_holding' not in features:
        i_holding = context.i_holding
    else:
        i_holding = features['i_holding']

    if 'i_syn_amp' not in features:
        res = [[i_holding], ['dend'], [None]]
    return res


def get_args_dynamic_iEPSP_unit(x, features):
    if 'i_holding' not in features:
        i_holding = context.i_holding
    else:
        i_holding = features['i_holding']

    N_syn = context.cell.N_syn
    res = [[i_holding] * N_syn, ['dend{:03d}'.format(i) for i in range(N_syn)], [features['i_syn_amp']] * N_syn]
    return res


def compute_features_iEPSP_i_unit(x, i_holding, dend_name, i_syn_amp=None, export=False, plot=False):
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
    duration = context.duration
    equilibrate = context.equilibrate

    v_active = context.v_active
    context.i_holding = i_holding
    offset_vm('soma', context, v_active, i_history=context.i_holding, dynamic=False)

    sim = context.sim
    sim.modify_stim('holding', dur=duration)
    sim.backup_state()
    sim.set_state(dt=dt, tstop=duration, cvode=False)
    
    dend = context.syns[dend_name]
    idx = dend['idx']
    context.i_vs[idx].play(h.Vector([context.equilibrate]))

    if i_syn_amp is None:
        i_EPSC = context.i_EPSC['ini_syn_amp']
        bounds = [(-0.3, -0.01)]
        i_EPSC_result = scipy.optimize.minimize(iEPSP_amp_error, [i_EPSC], args=(idx), method='L-BFGS-B', bounds=bounds,
                                       options={'ftol': 1e-3, 'disp': context.verbose > 1, 'maxiter': 5})
        i_EPSC = i_EPSC_result.x[0]
    else:
        i_EPSC = i_syn_amp

    config_syn(context.syn_mech_name, context.env.synapse_attributes.syn_param_rules, syn=context.i_syn[idx],
               i_unit=i_EPSC)
    sim.modify_rec(name='dendlocal', node=dend['node'], loc=dend['loc'])
    sim.run(v_active)

    soma_vm = np.array(sim.get_rec('soma')['vec'])
    soma_baseline = np.mean(soma_vm[int((equilibrate - 3.) / dt):int((equilibrate - 1.) / dt)])
    soma_vm -= soma_baseline
    soma_iEPSP_amp = np.max(soma_vm[int(equilibrate / dt):])

    dend_vm = np.array(sim.get_rec('dend')['vec'])
    dend_baseline = np.mean(dend_vm[int((equilibrate - 3.) / dt):int((equilibrate - 1.) / dt)])
    dend_vm -= dend_baseline
    dend_iEPSP_amp = np.max(dend_vm[int(equilibrate / dt):])

    dendloc_vm = np.array(sim.get_rec('dendlocal')['vec'])
    dendloc_baseline = np.mean(dendloc_vm[int((equilibrate - 3.) / dt):int((equilibrate - 1.) / dt)])
    dendloc_vm -= dendloc_baseline
    dendloc_iEPSP_amp = np.max(dendloc_vm[int(equilibrate / dt):])

    result = dict()
    if i_syn_amp is None:
        result['i_syn_amp'] = i_EPSC
    
    result['i_EPSP_rep_{!s}'.format(dend_name)] = \
        {'soma_EPSP_amp': soma_iEPSP_amp, 'syn_loc_amp': dendloc_iEPSP_amp, 'index': idx, 'dist': dend['dist'],
         'att_ratio': soma_iEPSP_amp/dendloc_iEPSP_amp, 'ref_dend_amp': dend_iEPSP_amp}

    title = 'iEPSP'
    description = 'i_unit: {!s}'.format(dend_name)
    sim.parameters['duration'] = duration
    sim.parameters['title'] = title
    sim.parameters['description'] = description

    if context.verbose > 0:
        print('compute_features_iEPSP_i_unit: pid: {:d}; {!s}: {!s} took {:.3f} s'\
              .format(os.getpid(), title, description, time.time() - start_time))
        sys.stdout.flush()
    if plot:
        context.sim.plot()
    if export:
        context.sim.export_to_file(context.temp_output_path)
    sim.restore_state()
    context.i_vs[idx].play(h.Vector())
    return result


def get_objectives_iEPSP_attenuation(features, export=False):
    """

    :param features: dict
    :param export: bool
    :return: tuple of dict
    """
    objectives = dict()

#    x = features['distance_array'][:-1]
    # Discard terminal dendrite in computing objectives

#    x = features['distance_array'][:-1]
#    atten = features['attenuation_array'][:-1]

    x = features['distance_array']
    atten = features['attenuation_array']

    gompertz_coeffs = get_gompertz_coeffs()
    expected_atten = gompertz(x, *gompertz_coeffs) 
    atten_resi = np.mean(((expected_atten-atten)/0.02)**2)
    objectives['iEPSP_attenuation_residual'] = atten_resi

    if export:
        f = h5py.File(context.export_file_path, 'a')
        description = 'iEPSP_attenuation'
        f[description].create_dataset('gompertz_coeffs', compression='gzip', data=gompertz_coeffs)
        f[description].create_dataset('expected_attenuations', compression='gzip', data=expected_atten)
        f[description].attrs['gompertz_fn'] = '1+a*np.exp(-b*np.exp(-c*(t-m))), coeffs=(a,b,c,m)'
        f.close()
    return features, objectives


def filter_features_attenuation(primitives, features, export=False):
    dendkeys = [key for key in features.keys() if key.startswith('i_EPSP_rep_dend')]
    atten_lst = []
    dist_lst = []
    soma_amp = []
    syn_loc_amp = []
    ref_dend_amp = []
    for key in dendkeys:
        atten_lst.append(features[key]['att_ratio'])    
        dist_lst.append(features[key]['dist'])    
        soma_amp.append(features[key]['soma_EPSP_amp'])
        syn_loc_amp.append(features[key]['syn_loc_amp'])
        ref_dend_amp.append(features[key]['ref_dend_amp'])

    for res in primitives:
        dendkeys = [key for key in res.keys() if key.startswith('i_EPSP_rep_dend')]
        for key in dendkeys:
            atten_lst.append(res[key]['att_ratio'])    
            dist_lst.append(res[key]['dist'])    
            soma_amp.append(res[key]['soma_EPSP_amp'])
            syn_loc_amp.append(res[key]['syn_loc_amp'])
            ref_dend_amp.append(res[key]['ref_dend_amp'])
        
    dist_arr, uniq_idx = np.unique(dist_lst, return_index=True)
    atten_arr = np.array(atten_lst)[uniq_idx]
    soma_amp_arr = np.array(soma_amp)[uniq_idx]
    syn_amp_arr = np.array(syn_loc_amp)[uniq_idx]
    ref_dend_arr = np.array(ref_dend_amp)[uniq_idx]

    new_features = {'attenuation_array': atten_arr, 'distance_array': dist_arr}

    if export:
        description = 'iEPSP_attenuation'
        distance, attenuation = get_attenuation_data()
        f = h5py.File(context.export_file_path, 'a')
        if description not in f:
            f.create_group(description)
            f[description].attrs['enumerated'] = False
        group = f[description]
        group.attrs['i_syn_amp'] = features['i_syn_amp']
        group.create_dataset('distance', compression='gzip', data=dist_arr)
        group.create_dataset('attenuation', compression='gzip', data=atten_arr)
        group.create_dataset('soma_EPSP_amp', compression='gzip', data=soma_amp_arr)
        group.create_dataset('local_dend_amp', compression='gzip', data=syn_amp_arr)
        group.create_dataset('ref_dend_amp', compression='gzip', data=ref_dend_arr)
        group.create_dataset('expmt_distance', compression='gzip', data=distance)
        group.create_dataset('expmt_attenuation', compression='gzip', data=attenuation)
        f.close()

    return new_features 


def get_attenuation_data():
    distance = np.array([  0.        ,  40.5904059 ,  45.29520295,  47.50922509, \
                          77.39852399,  90.12915129,  90.68265683, 121.95571956, \
                         126.93726937, 133.02583026, 138.83763838, 145.20295203, \
                         150.46125461, 156.5498155 , 185.60885609, 185.88560886, \
                         293.26568266])

    attenuation = np.array([1.        , 0.4453202 , 1.03940887, 0.16157635, 0.21871921, \
                            0.2       , 0.14384236, 0.19014778, 0.12512315, 0.17536946, \
                            0.23251232, 0.13300493, 0.38128079, 0.13103448, 0.0955665 , \
                            0.13103448, 0.08571429])
    return distance, attenuation


def get_gompertz_coeffs(optimize=False):
    if optimize:
        distance, attenuation = get_attenuation_data()
        results = optimize.curve_fit(gompertz,  distance,  attenuation, p0=[-0.9, 1.5, 0.2,40]) 
    else:
        results = np.array([-0.87,  1.06160453,  0.06154768, 35.55592743])
    return results


def gompertz(t, a, b, c, m):
    return 1+a*np.exp(-b*np.exp(-c*(t-m)))


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
