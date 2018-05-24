#Expects there to already be a hoc cell with a python wrapper (as defined in cells.py); the python cell should be called cell.
from neuron_wrapper_utils import *
from optimize_cells.plot_results import *
import click


context = Context()


def compare_single_value(key, x, seg, mech_name, param_name):
    if not hasattr(seg, mech_name):
        print 'Segment does not have the mechanism %s' %mech_name
    else:
        model_val = getattr(getattr(seg, mech_name), param_name)
        exp_val = x[key]
        if model_val == exp_val:
            print 'Test %s passed' % key
        else:
            print 'Test %s failed' % key
            print 'Model %s, Expected %s' % (format(model_val, 'e'), format(exp_val, 'e'))


def run_standard_modify_mech_param_tests(cell):
    """

    :param cell:
    :return:
    """
    x = {'dend.g_pas slope': 1.058E-08, 'dend.g_pas tau': 3.886E+01, 'soma.g_pas': 1.050E-10, 'dend.gbar_nas': 0.03,
         'dend.gbar_nas bo': 4, 'dend.gbar_nas min': 0.0, 'dend.gbar_nas slope': -0.0001, 'dend.gkabar': 0.04,
         'soma.gkabar': 0.02108, 'axon.gkabar': 0.05266, 'soma.gbar_nas': 0.0308}

    modify_mech_param(cell, 'soma', 'pas', 'g', x['soma.g_pas'])
    soma_seg = cell.tree.root.sec(0.5)
    compare_single_value('soma.g_pas', x, soma_seg, 'pas', 'g')

    plot_mech_param_distribution(cell, 'pas', 'g', export='dend_gpas0.hdf5', param_label='dend.g_pas', show=False,
                                 sec_types='all', overwrite=True)
    modify_mech_param(cell, 'apical', 'pas', 'g', origin='soma', slope=x['dend.g_pas slope'], tau=x['dend.g_pas tau'])
    plot_mech_param_distribution(cell, 'pas', 'g', export='dend_gpas1.hdf5', param_label='dend.g_pas', show=False,
                                 sec_types='all', overwrite=True)
    for sec_type in ['hillock', 'ais', 'axon', 'spine_neck', 'spine_head']:
        modify_mech_param(cell, sec_type, 'pas', 'g', origin='soma')
    for sec_type in ['hillock', 'ais', 'axon', 'apical','spine_neck', 'spine_head']:
        update_mechanism_by_sec_type(cell, sec_type, 'pas')
    plot_mech_param_distribution(cell, 'pas', 'g', export='dend_gpas2.hdf5', param_label='dend.g_pas', show=False,
                                 sec_types='all', overwrite=True)
    plot_mech_param_from_file('pas', 'g', ['dend_gpas0.hdf5', 'dend_gpas1.hdf5', 'dend_gpas2.hdf5'], ['0', '1', '2'],
                              param_label='dend.gpas')

    modify_mech_param(cell, 'soma', 'kap', 'gkabar', x['soma.gkabar'])
    plot_mech_param_distribution(cell, 'kap', 'gkabar', export='old_dend_kap.hdf5', param_label='dend.kap', show=False,
                                 sec_types='all', overwrite=True)
    plot_mech_param_distribution(cell, 'kad', 'gkabar', export='old_dend_kad.hdf5', param_label='dend.kad',
                                 show=False, sec_types='all')

    slope = (x['dend.gkabar'] - x['soma.gkabar']) / 300.
    for sec_type in ['apical']:
        # modify_mech_param(cell, sec_type, 'kap', 'gkabar', origin='soma', min_loc=75., value=0.)
        modify_mech_param(cell, sec_type, 'kap', 'gkabar', origin='soma', max_loc=75., slope=slope, outside=0.)\
            # , append=True)
        # modify_mech_param(cell, sec_type, 'kad', 'gkabar', origin='soma', max_loc=75., value=0.)
        modify_mech_param(cell, sec_type, 'kad', 'gkabar', origin='soma', min_loc=75., max_loc=300., slope=slope, 
                          outside=0., value=(x['soma.gkabar'] + slope * 75.), append=True)
        modify_mech_param(cell, sec_type, 'kad', 'gkabar', origin='soma', min_loc=300.,
                               value=(x['soma.gkabar'] + slope * 300.), append=True)

    # should do nothing
    update_mechanism_by_sec_type(cell, 'axon_hill', 'kap')
    modify_mech_param(cell, 'ais', 'kap', 'gkabar', x['axon.gkabar'])
    modify_mech_param(cell, 'axon', 'kap', 'gkabar', origin='ais')
    plot_mech_param_distribution(cell, 'kap', 'gkabar', export='new_dend_kap.hdf5', param_label='dend.kap', show=False,
                                 sec_types='all', overwrite=True)
    plot_mech_param_distribution(cell, 'kad', 'gkabar', export='new_dend_kad.hdf5', param_label='dend.kad', show=False,
                                 sec_types='all')

    plot_mech_param_from_file('kap', 'gkabar', ['old_dend_kap.hdf5', 'new_dend_kap.hdf5'], ['old', 'new'],
                              param_label='dend.kap')
    plot_mech_param_from_file('kad', 'gkabar', ['old_dend_kad.hdf5', 'new_dend_kad.hdf5'], ['old', 'new'],
                              param_label='dend.kad')

    modify_mech_param(cell, 'soma', 'nas', 'gbar', x['soma.gbar_nas'])
    plot_mech_param_distribution(cell, 'nas', 'gbar', export='old_dend_nas.hdf5', param_label='dend.nas',
                                 show=False, sec_types='all', overwrite=True)
    for sec_type in ['apical']:
        update_mechanism_by_sec_type(cell, sec_type, 'nas')
        modify_mech_param(cell, sec_type, 'nas', 'gbar', origin='parent', slope=x['dend.gbar_nas slope'],
                          min=x['dend.gbar_nas min'],
                          custom={'func': 'custom_gradient_by_branch_order',
                                  'branch_order': x['dend.gbar_nas bo']}, append=True)
        modify_mech_param(cell, sec_type, 'nas', 'gbar', origin='parent', slope=x['dend.gbar_nas slope'],
                          min=x['dend.gbar_nas min'], custom={'func': 'custom_gradient_by_terminal'}, append=True)
    plot_mech_param_distribution(cell, 'nas', 'gbar', export='new_dend_nas.hdf5', param_label='dend.nas',
                                 show=False, sec_types='all', overwrite=True)

    plot_mech_param_from_file('nas', 'gbar', ['old_dend_nas.hdf5', 'new_dend_nas.hdf5'], ['old', 'new'],
                              param_label='dend.nas')


def count_nseg(cell):
    nseg = {}
    distances = {}
    for sec_type in cell.nodes:
        nseg[sec_type] = []
        distances[sec_type] = []
        for node in cell.nodes[sec_type]:
            nseg[sec_type].append(node.sec.nseg)
            distances[sec_type].append(get_distance_to_node(cell, cell.tree.root, node))
    return nseg, distances


def compare_nseg(old_nseg, old_distances, new_nseg, new_distances, label_old, label_new):
    for sec_type in old_nseg:
        plt.scatter(old_distances[sec_type], old_nseg[sec_type], c='r', label=label_old)
        plt.scatter(new_distances[sec_type], new_nseg[sec_type], c='c', label=label_new)
    plt.legend(loc='best')
    plt.xlabel('Distance from Soma (um)')
    plt.ylabel('Number of segments per section')
    plt.title('Changing Spatial Resolution')
    plt.show()
    plt.close()
    print '%s nseg apical' %(label_old)
    print old_nseg['apical']
    print '%s nseg apical' %(label_new)
    print new_nseg['apical']


def run_cm_correction_test(cell, context):
    init_mechanisms(cell, reset_cable=True, from_file=True, mech_file_path=context.mech_file_path, cm_correct=False,
                    g_pas_correct=False)
    old_nseg, old_distances = count_nseg(cell)
    plot_mech_param_distribution(cell, 'pas', 'g', export='old_dend_gpas', overwrite=False, param_label='dend.g_pas', show=False)
    plot_cable_param_distribution(cell, 'cm', export='old_cm', param_label='cm', show=False, overwrite=True,
                                  scale_factor=1)
    init_mechanisms(cell, reset_cable=True, from_file=True, mech_file_path=context.mech_file_path, cm_correct=True, g_pas_correct=True,
                    cell_attr_dict=context.cell_attr_dict[context.gid], sec_index_map=context.sec_index_map, env=context.env)
    new_nseg, new_distances = count_nseg(cell)
    compare_nseg(old_nseg, old_distances, new_nseg, new_distances, 'old', 'new')
    plot_mech_param_distribution(cell, 'pas', 'g', export='new_dend_gpas', overwrite=False, param_label='dend.g_pas', show=False)
    plot_mech_param_from_file('pas', 'g', ['old_dend_gpas', 'new_dend_gpas'], ['old', 'new'],
                              param_label='dend.gpas')
    plot_cable_param_distribution(cell, 'cm', export='new_cm', param_label='cm', show=False, overwrite=True,
                                  scale_factor=1)
    plot_mech_param_from_file('cm', None, ['old_cm', 'new_cm'], ['old', 'new'],
                              param_label='cm')


def run_cable_test(cell):
    plot_cable_param_distribution(cell, 'cm', export='old_cm.hdf5', param_label='cm', show=False, overwrite=True, scale_factor=1)
    modify_mech_param(cell, 'soma', 'cable', 'cm', value=2.)
    init_mechanisms(cell, reset_cable=True)
    plot_cable_param_distribution(cell, 'cm', export='new_cm.hdf5', param_label='cm', show=False, overwrite=True, scale_factor=1)
    plot_mech_param_from_file('cm', None, ['old_cm.hdf5', 'new_cm.hdf5'], ['old', 'new'],
                              param_label='cm')

    plot_cable_param_distribution(cell, 'Ra', export='old_Ra.hdf5', param_label='Ra', show=False, overwrite=True, scale_factor=1)
    init_mechanisms(cell, reset_cable=True, from_file=True)
    plot_cable_param_distribution(cell, 'Ra', export='reinit_Ra.hdf5', param_label='Ra', show=False, overwrite=True, scale_factor=1)
    modify_mech_param(cell, 'soma', 'cable', 'Ra', value=200.)
    init_mechanisms(cell, reset_cable=True)
    plot_cable_param_distribution(cell, 'Ra', export='modified_Ra.hdf5', param_label='Ra', show=False, overwrite=True, scale_factor=1)
    plot_mech_param_from_file('Ra', None, ['old_Ra.hdf5', 'reinit_Ra.hdf5', 'modified_Ra.hdf5'],
                              ['old', 'reinit', 'modified'], param_label='Ra')

    init_mechanisms(cell, reset_cable=True, from_file=True)
    old_nseg, old_distances = count_nseg(cell)
    modify_mech_param(cell, 'soma', 'cable', 'spatial_res', value=2.)
    init_mechanisms(cell, reset_cable=True)
    new_nseg, new_distances = count_nseg(cell)
    compare_nseg(old_nseg, old_distances, new_nseg, new_distances, 'old', 'new')

    init_mechanisms(cell, reset_cable=True, from_file=True)
    plot_cable_param_distribution(cell, 'cm', export='cm1.hdf5', param_label='cm1', show=False, overwrite=True, scale_factor=1)
    modify_mech_param(cell, 'apical', 'cable', 'cm', value=2.)
    init_mechanisms(cell, reset_cable=True)
    plot_cable_param_distribution(cell, 'cm', export='cm2.hdf5', param_label='cm2', show=False, overwrite=True, scale_factor=1)
    nseg2, distances2 = count_nseg(cell)
    modify_mech_param(cell, 'apical', 'cable', 'spatial_res', value=3.)
    plot_cable_param_distribution(cell, 'cm', export='cm3.hdf5', param_label='cm3', show=False, overwrite=True, scale_factor=1)
    nseg3, distances3 = count_nseg(cell)
    modify_mech_param(cell, 'apical', 'cable', 'cm', value=10.)
    plot_cable_param_distribution(cell, 'cm', export='cm4.hdf5', param_label='cm4', show=False, overwrite=True, scale_factor=1)
    nseg4, distances4 = count_nseg(cell)
    plot_mech_param_from_file('cm', None, ['cm1.hdf5', 'cm2.hdf5', 'cm3.hdf5', 'cm4.hdf5'], ['orig', 'post step 1',
                                                                                             'post step 2', 'post step 3'],
                              param_label='cm')
    compare_nseg(nseg2, distances2, nseg3, distances3, 'post step 2', 'post step 3')
    compare_nseg(nseg3, distances3, nseg4, distances4, 'post step 3', 'post step 4')
    #Try changing cm by a significant amount in apical branches only, and then see if this affects nseg. Then change the spatial
    #res parameter -- this should be a multiplier on the current nseg


def count_syns(cell, context):
    for node in cell.apical:
        all_syn = filtered_synapse_attributes(context.cell_attr_dict[0], np.array(context.sec_index_map[node.index]),
                                              context.env, syn_category='excitatory', output='syn_locs')['syn_locs']
        print ('%s: length %.3f, num synapses %i, num segments %i, density %.3f' %(node.name, node.sec.L, len(all_syn),
                                                                                   node.sec.nseg, len(all_syn)/node.sec.L))

@click.command()
@click.option("--gid", required=True, type=int, default=0)
@click.option("--pop-name", required=True, type=str, default='GC')
@click.option("--config-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='../dentate/config/Small_Scale_Control_log_normal_weights.yaml')
@click.option("--template-paths", type=str, default='../dgc/Mateos-Aparicio2014:../dentate/templates')
@click.option("--hoc-lib-path", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='../dentate')
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='../dentate/datasets')  # '/mnt/s')  # '../dentate/datasets'
@click.option("--mech-file-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='mechanisms/20180209_DG_GC_hoc_leak_mech.yaml')
@click.option('--verbose', '-v', is_flag=True)
def main(gid, pop_name, config_file, template_paths, hoc_lib_path, dataset_prefix, mech_file_path, verbose):
    """

    :param gid:
    :param pop_name:
    :param config_file:
    :param template_paths:
    :param hoc_lib_path:
    :param dataset_prefix:
    :param verbose
    """
    comm = MPI.COMM_WORLD
    np.seterr(all='raise')
    env = Env(comm, config_file, template_paths, hoc_lib_path, dataset_prefix, verbose=verbose)
    configure_env(env)

    cell = get_biophys_cell(env, gid, pop_name)
    context.update(locals())

    # init_biophysics(cell, reset_cable=True, from_file=True, mech_file_path=mech_file_path, correct_cm=True,
    #                correct_g_pas=True, env=env)
    run_standard_modify_mech_param_tests(cell)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)