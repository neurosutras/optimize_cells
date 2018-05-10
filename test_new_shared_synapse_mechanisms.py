from neuron_wrapper_utils import *

context = Context()


@click.command()
@click.option("--gid", required=True, type=int, default=0)
@click.option("--pop-name", required=True, type=str, default='GC')
@click.option("--config-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='../dentate/config/Small_Scale_Control_log_normal_weights.yaml')
@click.option("--template-paths", type=str, default='../dgc/Mateos-Aparicio2014:../dentate/templates')
@click.option("--hoc-lib-path", type=str, default='../dentate')
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='../dentate/datasets')  # '/mnt/s'
@click.option("--results-path", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='data')
@click.option("--mech-file-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='mechanisms/20180509_DG_GC_test_modified_Ca_mech.yaml')
@click.option('--verbose', '-v', is_flag=True)
@click.option('--mech-name', type=str, default='SatExp2Syn')
@click.option('--num-syns', type=int, default=1)
def main(gid, pop_name, config_file, template_paths, hoc_lib_path, dataset_prefix, results_path, mech_file_path,
         verbose, mech_name, num_syns):
    """

    :param gid:
    :param pop_name:
    :param config_file:
    :param template_paths:
    :param hoc_lib_path:
    :param dataset_prefix:
    :param results_path:
    :param mech_file_path: str
    :param verbose
    :param mech_name: str
    :param num_syns: int
    """
    comm = MPI.COMM_WORLD
    env = init_env(config_file=config_file, template_paths=template_paths, hoc_lib_path=hoc_lib_path, comm=comm,
                   dataset_prefix=dataset_prefix, results_path=results_path, verbose=verbose)
    cell = get_hoc_cell_wrapper(env, gid, pop_name)
    init_mechanisms(cell, reset_cable=True, from_file=True, mech_file_path=mech_file_path, cm_correct=True,
                    g_pas_correct=True, cell_attr_dict=env.cell_attr_dict[gid],
                    sec_index_map=env.sec_index_map[gid], env=env)

    h('proc record_netcon_weight_element() { $o1.record(&$o2.weight[$3], $4) }')

    netcon_list = []
    vecstim_list = []

    recordings = {'SatExp2Syn': {'mech_params': ['g'],
                                 'netcon_params': {'g0': 4}
                                 },
                  'AMPA_S': {'mech_params': ['g'],
                             'netcon_params': {'r0': 3}},
                  'FacilExp2Syn': {'mech_params': ['g'],
                                   'netcon_params': {'g0': 4, 'f1': 6}},
                  'FacilNMDA': {'mech_params': ['g', 'B'],
                                'netcon_params': {'g0': 4, 'f1': 6}},
                  'Exp2Syn': {'mech_params': ['g']}
                  }
    syn_params = {'SatExp2Syn': {'g_unit': 0.00015, 'dur_onset': 1., 'tau_offset': 5., 'sat': 0.9, 'e': 0.},
                  'AMPA_S': {},
                  'FacilExp2Syn': {'f_tau': 25., 'f_inc': 0.15, 'f_max': 0.6,
                                   'g_unit': 0.005, 'dur_onset': 10., 'tau_offset': 35., 'sat': 0.9, 'e': 0.},
                  'FacilNMDA': {'f_tau': 25., 'f_inc': 0.15, 'f_max': 0.6,
                                'g_unit': 0.00015, 'dur_onset': 10., 'tau_offset': 35., 'sat': 0.9, 'e': 0.},
                  'Exp2Syn': {'weight': 0.0016, 'tau1': 0.5, 'tau2': 5., 'e': 0.}
                  }
    syn_node = cell.apical[-1]  # cell.tree.root
    syn = add_unique_synapse(mech_name, syn_node.sec(0.5))
    config_syn(mech_name, rules=env.synapse_param_rules, syn=syn, **syn_params[mech_name])

    # drive input with spike train
    ISI = 100.
    equilibrate = 250.
    input_spike_train = [equilibrate + i * ISI for i in xrange(3)]
    last_spike_time = input_spike_train[-1]
    ISI = 10.
    input_spike_train += [last_spike_time + 150. + i * ISI for i in xrange(10)]
    last_spike_time = input_spike_train[-1]
    input_spike_train += [last_spike_time + 100.]

    each_syn_delay = 10.
    for i in xrange(num_syns):
        this_input_spike_train = [spike_time + i * each_syn_delay for spike_time in input_spike_train]
        this_vecstim = h.VecStim()
        vecstim_list.append(this_vecstim)
        this_netcon = h.NetCon(this_vecstim, syn, -30., 0., 1.)
        this_netcon.pre().play(h.Vector(this_input_spike_train))
        netcon_list.append(this_netcon)
        config_syn(mech_name, rules=env.synapse_param_rules, nc=this_netcon, **syn_params[mech_name])

    duration = this_input_spike_train[-1] + 150.
    sim = QuickSim(duration)
    sim.append_rec(cell, cell.tree.root, 0.5, description='soma Vm')
    if syn_node != cell.tree.root:
        sim.append_rec(cell, syn_node, 0.5, description='%s Vm' % syn_node.name)
    if 'mech_params' in recordings[mech_name]:
        for param_name in recordings[mech_name]['mech_params']:
            sim.append_rec(cell, syn_node, object=syn, param='_ref_'+param_name, description=param_name)

    if 'netcon_params' in recordings[mech_name]:
        for i in xrange(num_syns):
            for param_name, j in recordings[mech_name]['netcon_params'].iteritems():
                if j > 1:
                    description = 'netcon%i_%s' % (i, param_name)
                    sim.append_rec(cell, syn_node, description=description)
                    h.record_netcon_weight_element(sim.get_rec(description)['vec'], netcon_list[i], j, sim.dt)
    context.update(locals())

    sim.run()
    if mech_name in ['SatExp2Syn', 'FacilExp2Syn']:
        for i in xrange(num_syns):
            description = 'netcon%i_g0' % i
            sim.get_rec(description)['vec'] = np.divide(sim.get_rec(description)['vec'],
                                                       getattr(syn, 'g_inf') * getattr(syn, 'sat'))
    sim.plot()


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
