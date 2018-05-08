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
@click.option('--verbose', '-v', is_flag=True)
@click.option('--mech-name', type=str, default='SatExp2Syn')
@click.option('--num-syns', type=int, default=1)
def main(gid, pop_name, config_file, template_paths, hoc_lib_path, dataset_prefix, results_path, verbose,
         mech_name, num_syns):
    """

    :param gid:
    :param pop_name:
    :param config_file:
    :param template_paths:
    :param hoc_lib_path:
    :param dataset_prefix:
    :param results_path:
    :param verbose
    :param mech_name: str
    """
    comm = MPI.COMM_WORLD
    env = init_env(config_file=config_file, template_paths=template_paths, hoc_lib_path=hoc_lib_path, comm=comm,
                   dataset_prefix=dataset_prefix, results_path=results_path, verbose=verbose)
    cell = get_hoc_cell_wrapper(env, gid, pop_name)
    cell.tree.root.sec.insert('pas')
    cell.tree.root.sec.g_pas = 0.5

    h('proc record_netcon_weight_element() { $o1.record(&$o2.weight[$3], $4) }')

    netcon_list = []
    vecstim_list = []
    syn = getattr(h, mech_name)(cell.tree.root.sec(0.5))

    # drive input with spike train
    ISI = 100.
    equilibrate = 250.
    input_spike_train = [equilibrate + i * ISI for i in xrange(3)]
    last_spike_time = input_spike_train[-1]
    ISI = 10.
    input_spike_train += [last_spike_time + 150. + i * ISI for i in xrange(5)]
    last_spike_time = input_spike_train[-1]
    input_spike_train += [last_spike_time + 100.]

    mech_config = {'SatExp2Syn': {'mech_params': ['sat', 'dur_onset', 'tau_offset', 'e'],
                              'netcon_params': {'gmax': 1, 'g0': 4}
                              },
                   'FacilExp2Syn': {'mech_params': ['tau_rise', 'tau_decay', 'e'],
                                  'netcon_params': {'gmax': 1, 'A0': 2, 'B0': 3}}
                   }
    syn_params = {'SatExp2Syn': {'gmax': 1., 'dur_onset': 5., 'tau_offset': 40., 'sat': 0.5}}

    def set_syn_mech_params(mech_config, syn_list, netcon_list, **kwargs):
        """

        :param mech_config: dict
        :param syn_list: list of synapse point_process objects
        :param netcon_list: list of synapse netcon objects
        :param kwargs: dict
        """
        for syn in syn_list:
            for param_name in mech_config['mech_params']:
                if hasattr(syn, param_name) and param_name in kwargs:
                    setattr(syn, param_name, kwargs[param_name])
        for this_netcon in netcon_list:
            for param_name, i in mech_config['netcon_params'].iteritems():
                if param_name in kwargs and this_netcon.wcnt() >= i:
                    this_netcon.weight[i] = kwargs[param_name]
                    # print mech_name, param_name, kwargs[param_name], i

    """
    syn.f_tau = 30.
    syn.f_inc = 0.3
    syn.f_max = 0.75
    syn.tau_rise = 5.
    syn.tau_decay = 40.
    """

    each_syn_delay = 10.
    for i in xrange(num_syns):
        this_input_spike_train = [spike_time + i * each_syn_delay for spike_time in input_spike_train]
        this_vecstim = h.VecStim()
        vecstim_list.append(this_vecstim)
        this_netcon = h.NetCon(this_vecstim, syn, -30., 0., 1.)
        this_netcon.pre().play(h.Vector(this_input_spike_train))
        netcon_list.append(this_netcon)
    set_syn_mech_params(mech_config[mech_name], [syn], netcon_list, **syn_params[mech_name])

    duration = this_input_spike_train[-1] + 150.
    sim = QuickSim(duration)
    # sim.append_rec(cell, cell.tree.root, 0.5, description='soma Vm')
    sim.append_rec(cell, cell.tree.root, object=syn, param='_ref_g', description='g_total')
    sim.append_rec(cell, cell.tree.root, object=syn, param='_ref_g_onset', description='g_onset')
    sim.append_rec(cell, cell.tree.root, object=syn, param='_ref_g_offset', description='g_offset')
    for i in xrange(num_syns):
        for param_name, j in mech_config[mech_name]['netcon_params'].iteritems():
            if j > 1:
                description = 'netcon%i_%s' % (i, param_name)
                sim.append_rec(cell, cell.tree.root, description=description)
                h.record_netcon_weight_element(sim.get_rec(description)['vec'], netcon_list[i], j, sim.dt)
    context.update(locals())

    sim.run()
    for i in xrange(num_syns):
        if mech_name == 'FacilExp2Syn':
            plt.plot(sim.tvec, np.subtract(sim.get_rec('netcon%i_B0' % i)['vec'],
                                           sim.get_rec('netcon%i_A0' % i)['vec']), label='g%i' % i)
        elif mech_name == 'SatExp2Syn':
            description = 'netcon%i_g0' % i
            sim.get_rec(description)['vec'] = np.divide(sim.get_rec(description)['vec'],
                                                       getattr(syn, 'g_inf') * getattr(syn, 'sat'))
    sim.plot()


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
