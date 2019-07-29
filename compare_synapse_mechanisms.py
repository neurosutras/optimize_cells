from cell_utils import *
from dentate.biophysics_utils import *

context = Context()


@click.command()
@click.option("--gid", required=True, type=int, default=0)
@click.option("--pop-name", required=True, type=str, default='GC')
@click.option("--config-file", required=True, type=str,
              default='Small_Scale_Control_LN_weights_Sat.yaml')
@click.option("--template-paths", type=str, default='../DGC/Mateos-Aparicio2014:../dentate/templates')
@click.option("--hoc-lib-path", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='../dentate')
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='../dentate/datasets')
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='../dentate/config')
@click.option("--results-path", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='data')
@click.option("--mech-file-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='mechanisms/20180529_DG_GC_mech.yaml')
@click.option('--verbose', '-v', is_flag=True)
@click.option('--mech-name', '-m', type=str, multiple=True, default=['SatExp2Syn', 'FacilNMDA'])
@click.option('--num-inputs', type=int, default=1)
@click.option('--shared', is_flag=True)
def main(gid, pop_name, config_file, template_paths, hoc_lib_path, dataset_prefix, config_prefix, results_path,
         mech_file_path, verbose, mech_name, num_inputs, shared):
    """

    :param gid:
    :param pop_name:
    :param config_file:
    :param template_paths:
    :param hoc_lib_path:
    :param dataset_prefix:
    :param config_prefix:
    :param results_path:
    :param mech_file_path: str
    :param verbose
    :param mech_name: list ofstr
    :param num_ints: int
    :param shared: bool
    """
    comm = MPI.COMM_WORLD
    np.seterr(all='raise')
    env = Env(comm, config_file, template_paths, hoc_lib_path, dataset_prefix, config_prefix, verbose=verbose)
    configure_hoc_env(env)

    cell = get_biophys_cell(env, pop_name, gid, mech_file_path=mech_file_path)
    context.update(locals())

    init_biophysics(cell, reset_cable=True, correct_cm=True, correct_g_pas=True, env=env)

    zero_na(cell)

    h('proc record_netcon_weight_element() { $o1.record(&$o2.weight[$3], $4) }')

    netcon_list = []
    vecstim_list = []

    recordings = {'SatExp2Syn': {'mech_params': ['g', 'i']},  # , 'netcon_params': {'g0': 4}},
                  'AMPA_KIN': {'mech_params': ['g', 'i']},  # , 'netcon_params': {'r0': 3}},
                  'NMDA_KIN5': {'mech_params': ['g', 'i']},  # , 'netcon_params': {'r0': 3}},
                  'FacilExp2Syn': {'mech_params': ['g', 'i']},  # , 'netcon_params': {'g0': 4, 'f1': 6}},
                  'FacilNMDA': {'mech_params': ['g', 'i']},  # , 'B'], 'netcon_params': {'g0': 4, 'f1': 6}},
                  'LinExp2Syn': {'mech_params': ['g', 'i']}
                  }
    syn_params = {'SatExp2Syn': {'g_unit': 0.000361, 'dur_onset': 0.5, 'tau_offset': 3.5, 'sat': 0.9, 'e': 0.},
                  'AMPA_KIN': {'gmax': 0.001222},
                  'NMDA_KIN5': {'gmax': 0.001},
                  'FacilExp2Syn': {'f_tau': 25., 'f_inc': 0.15, 'f_max': 0.6,
                                   'g_unit': 0.00005, 'dur_onset': 10., 'tau_offset': 35., 'sat': 0.9, 'e': 0.},
                  'FacilNMDA': {'f_tau': 25., 'f_inc': 0.15, 'f_max': 0.6,
                                'g_unit': 0.00043, 'dur_onset': 10., 'tau_offset': 35., 'sat': 0.9, 'e': 0.},
                  'LinExp2Syn': {'g_unit': 0.000336, 'tau_rise': 0.2, 'tau_decay': 3.5, 'e': 0.}
                  # 'LinExp2Syn': {'g_unit': 0.000336, 'tau_rise': 50., 'tau_decay': 200., 'e': -90.}
                  }

    syn_attrs = env.synapse_attributes
    syn_node = cell.apical[-1]  # cell.tree.root

    make_synapse_mech = make_shared_synapse_mech if shared else make_unique_synapse_mech
    shared_syns_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    syn_obj_dict = defaultdict(dict)
    nc_obj_dict = defaultdict(dict)
    syns_set = set()

    for i in range(num_inputs):
        for syn_name in mech_name:
            syn = make_synapse_mech(syn_name=syn_name, seg=syn_node.sec(0.5), syns_dict=shared_syns_dict)
            syn_obj_dict[i][syn_name] = syn
            if syn not in syns_set:
                syns_set.add(syn)
                config_syn(syn_name=syn_name, rules=syn_attrs.syn_param_rules, syn=syn, **syn_params[syn_name])

    # drive input with spike train
    ISI = 100.
    equilibrate = 250.
    input_spike_train = [equilibrate + i * ISI for i in range(3)]
    last_spike_time = input_spike_train[-1]
    ISI = 10.
    input_spike_train += [last_spike_time + 150. + i * ISI for i in range(10)]
    last_spike_time = input_spike_train[-1]
    input_spike_train += [last_spike_time + 100.]
    # input_spike_train = []

    for syn_id in syn_obj_dict:
        for syn_name in syn_obj_dict[syn_id]:
            syn = syn_obj_dict[syn_id][syn_name]
            this_vecstim = h.VecStim()
            vecstim_list.append(this_vecstim)
            this_netcon = h.NetCon(this_vecstim, syn, -30., 0., 1.)
            this_netcon.pre().play(h.Vector(input_spike_train))
            nc_obj_dict[syn_id][syn_name] = this_netcon
            config_syn(syn_name, rules=syn_attrs.syn_param_rules, nc=this_netcon, **syn_params[syn_name])

    duration = input_spike_train[-1] + 150.
    # duration = 940.
    sim = QuickSim(duration, cvode=False)
    sim.append_rec(cell, cell.tree.root, 'soma', 0.5)
    if syn_node != cell.tree.root:
        sim.append_rec(cell, syn_node, syn_node.name, 0.5)

    rec_description_dict = defaultdict(lambda: defaultdict(dict))
    syns_set = set()
    for syn_id in syn_obj_dict:
        for syn_name in syn_obj_dict[syn_id]:
            if 'mech_params' in recordings[syn_name]:
                syn = syn_obj_dict[syn_id][syn_name]
                if syn not in syns_set:
                    syns_set.add(syn)
                    for param_name in recordings[syn_name]['mech_params']:
                        description = '%s_%s_%i' % (param_name, syn_name, syn_id)
                        sim.append_rec(cell, syn_node, description, object=syn, param='_ref_'+param_name)
                        rec_description_dict[syn_id][syn_name][param_name] = description

            if 'netcon_params' in recordings[syn_name]:
                this_netcon = nc_obj_dict[syn_id][syn_name]
                for param_name, j in viewitems(recordings[syn_name]['netcon_params']):
                    if j > 1:
                        description = '%s_%s_%i' % (param_name, syn_name, syn_id)
                        sim.append_rec(cell, syn_node, description)
                        h.record_netcon_weight_element(sim.get_rec(description)['vec'], this_netcon, j, sim.dt)
                        rec_description_dict[syn_id][syn_name][param_name] = description
    context.update(locals())

    sim.run()
    print('num_inputs: %i; num_mechs: %i; num_syns: %i' % (num_inputs, len(mech_name), len(syns_set)))
    rec_sum = defaultdict(lambda: defaultdict(list))
    for syn_id in rec_description_dict:
        for syn_name in rec_description_dict[syn_id]:
            for param_name in rec_description_dict[syn_id][syn_name]:
                rec_sum[syn_name][param_name].append(
                    sim.get_rec(rec_description_dict[syn_id][syn_name][param_name])['vec'].to_python())
    t = sim.tvec.as_numpy()
    start = np.where(t >= equilibrate - 10.)[0][0]
    t -= equilibrate
    left = start + int((10. - 3.) / sim.dt)
    right = left + int(2. / sim.dt)
    window = right + int(25. / sim.dt)
    soma_vm = sim.get_rec('soma')['vec'].as_numpy()
    unit_amp = np.max(soma_vm[right:window]) - np.mean(soma_vm[left:right])
    print('unit amp: %.3f' % unit_amp)

    fig, axes = plt.subplots(3, sharex=True)
    for j, syn_name in enumerate(mech_name):
        for i, param_name in enumerate(rec_sum[syn_name]):
            this_sum = np.sum(rec_sum[syn_name][param_name], axis=0)
            if param_name == 'i':
                scale = 1000.
                ylabel = 'Current (pA)'
                ylim = [-500., 50.]
            elif param_name == 'g':
                scale = 1000.
                ylabel = 'Conductance (nS)'
                ylim = [0., 10.]
            axes[i].plot(t[start:], scale*this_sum[start:], label='%s_%s' % (param_name, syn_name), zorder=j)
            axes[i].set_ylabel(ylabel)
            axes[i].set_ylim(ylim)
    axes[2].plot(t[start:], sim.get_rec('soma')['vec'].as_numpy()[start:], label='soma Vm')
    axes[2].plot(t[start:], sim.get_rec('%s' % syn_node.name)['vec'].as_numpy()[start:], label='dend Vm')
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_ylabel('Voltage (mV)')
    axes[2].set_ylim(-80., -10.)
    for i in range(len(axes)):
        axes[i].legend(loc='best', frameon=False, framealpha=0.5)
    clean_axes(axes)
    fig.tight_layout()
    fig.show()

    # sim.plot()
    """
    sim_time_labels = ['LinExp2Syn (unique)', 'LinExp2Syn (shared)', 'KIN_AMPA + KIN_NMDA',
                  'SatExp2Syn + FacilNMDA (unique)', 'SatExp2Syn + FacilNMDA (shared)']
    sim_time_vals = [297.3601818, 59.01813507, 784.1932774, 513.504982, 94.74205971]
    fig, ax = plt.subplots()

    bar_width = 0.2
    spacing = 0.3

    rects = []
    for i, (label, val) in enumerate(zip(sim_time_labels, sim_time_vals)):
        rects.append(ax.bar(0.1 + i * spacing, val, bar_width, alpha=0.8, label=label))
    ax.set_xlabel('Synaptic mechanism')
    ax.set_ylabel('Simulation time (ms)')
    ax.set_xticks([], [])
    ax.legend(loc='best', frameon=False, framealpha=0.5)
    clean_axes(ax)
    fig.tight_layout()
    fig.show()
    """


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
