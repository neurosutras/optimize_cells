import click
from biophysics_utils import *
from optimize_cells.plot_results import *


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


def standard_modify_mech_param_tests(cell):
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

    plot_mech_param_distribution(cell, 'pas', 'g', export='dend_gpas.hdf5', description='stage0', show=False,
                                 sec_types='all', overwrite=True)
    modify_mech_param(cell, 'apical', 'pas', 'g', origin='soma', slope=x['dend.g_pas slope'], tau=x['dend.g_pas tau'])
    plot_mech_param_distribution(cell, 'pas', 'g', export='dend_gpas.hdf5', description='stage1', show=False,
                                 sec_types='all', overwrite=True)
    for sec_type in ['hillock', 'ais', 'axon', 'spine_neck', 'spine_head']:
        modify_mech_param(cell, sec_type, 'pas', 'g', origin='soma')
    modify_mech_param(cell, 'soma', 'pas', 'g', 2. * x['soma.g_pas'])
    for sec_type in ['hillock', 'ais', 'axon', 'apical','spine_neck', 'spine_head']:
        update_mechanism_by_sec_type(cell, sec_type, 'pas')
    plot_mech_param_distribution(cell, 'pas', 'g', export='dend_gpas.hdf5', description='stage2', show=False,
                                 sec_types='all', overwrite=True)
    plot_mech_param_from_file('pas', 'g', 'dend_gpas.hdf5', param_label='dend.gpas')

    modify_mech_param(cell, 'soma', 'kap', 'gkabar', x['soma.gkabar'])
    plot_mech_param_distribution(cell, 'kap', 'gkabar', export='dend_kap.hdf5', description='stage0',
                                 param_label='dend.kap', show=False, sec_types='all', overwrite=True)

    slope = (x['dend.gkabar'] - x['soma.gkabar']) / 300.
    for sec_type in ['apical']:
        modify_mech_param(cell, sec_type, 'kap', 'gkabar', origin='soma', max_loc=75., slope=slope, outside=0.)
        modify_mech_param(cell, sec_type, 'kad', 'gkabar', origin='soma', min_loc=75., max_loc=300., slope=slope, 
                          outside=0., value=(x['soma.gkabar'] + slope * 75.))
        modify_mech_param(cell, sec_type, 'kad', 'gkabar', origin='soma', min_loc=300.,
                          value=(x['soma.gkabar'] + slope * 300.), append=True)

    # should do nothing
    update_mechanism_by_sec_type(cell, 'axon_hill', 'kap')
    modify_mech_param(cell, 'ais', 'kap', 'gkabar', x['axon.gkabar'])
    modify_mech_param(cell, 'axon', 'kap', 'gkabar', origin='ais')
    plot_mech_param_distribution(cell, 'kap', 'gkabar', export='dend_kap.hdf5', description='stage1',
                                 param_label='dend.kap', show=False, sec_types='all', overwrite=True)
    plot_mech_param_from_file('kap', 'gkabar', 'dend_kap.hdf5', param_label='dend.kap')
    plot_mech_param_distribution(cell, 'kad', 'gkabar', param_label='dend.kad', show=True, sec_types='all')
    modify_mech_param(cell, 'soma', 'nas', 'gbar', x['soma.gbar_nas'])
    plot_mech_param_distribution(cell, 'nas', 'gbar', export='dend_nas.hdf5', description='stage0',
                                 show=False, sec_types='all', overwrite=True)
    for sec_type in ['apical']:
        modify_mech_param(cell, sec_type, 'nas', 'gbar', x['dend.gbar_nas'])
        modify_mech_param(cell, sec_type, 'nas', 'gbar', origin='parent', slope=x['dend.gbar_nas slope'],
                          min=x['dend.gbar_nas min'],
                          custom={'func': 'custom_filter_by_branch_order',
                                  'branch_order': x['dend.gbar_nas bo']}, append=True)
        modify_mech_param(cell, sec_type, 'nas', 'gbar', origin='parent', slope=x['dend.gbar_nas slope'],
                          min=x['dend.gbar_nas min'], custom={'func': 'custom_filter_by_terminal'}, append=True)
    plot_mech_param_distribution(cell, 'nas', 'gbar', export='dend_nas.hdf5', description='stage1', show=False,
                                 sec_types='all', overwrite=True)
    plot_mech_param_from_file('nas', 'gbar', 'dend_nas.hdf5', ['stage0', 'stage1'])


def count_nseg(cell):
    nseg = defaultdict(list)
    distances = defaultdict(list)
    for sec_type in cell.nodes:
        for node in cell.nodes[sec_type]:
            nseg[sec_type].append(node.sec.nseg)
            distances[sec_type].append(get_distance_to_node(cell, cell.tree.root, node))
    return nseg, distances


def compare_nseg(nseg, distances, labels):
    """

    :param nseg: list of dict: {str: int}
    :param distances: list of dict: {str: float}
    :param labels: list of str
    """
    num_colors = max([len(this_nseg) for this_nseg in nseg])
    markers = mlines.Line2D.filled_markers
    color_x = np.linspace(0., 1., num_colors)
    colors = [cm.Set1(x) for x in color_x]
    for j, this_nseg in enumerate(nseg):
        for i, sec_type in enumerate(this_nseg):
            this_distances = distances[j]
            plt.scatter(this_distances[sec_type], this_nseg[sec_type], c=colors[j], marker=markers[i],
                        label=sec_type+'_'+labels[j], alpha=0.5)
            print '%s_%s nseg: %s' % (sec_type, labels[j], str(this_nseg[sec_type]))
    plt.legend(loc='best', frameon=False, framealpha=0.5)
    plt.xlabel('Distance from Soma (um)')
    plt.ylabel('Number of segments per section')
    plt.title('Changing Spatial Resolution')
    plt.show()
    plt.close()


def cm_correction_test(cell, env, mech_file_path):
    """

    :param cell:
    :param env:
    :param mech_file_path:
    """
    init_biophysics(cell, reset_cable=True, from_file=True, mech_file_path=mech_file_path, correct_cm=False,
                    correct_g_pas=False, env=context.env)
    old_nseg, old_distances = count_nseg(cell)
    plot_mech_param_distribution(cell, 'pas', 'g', export='old_dend_gpas.hdf5', overwrite=True,
                                 param_label='dend.g_pas', show=False)
    plot_cable_param_distribution(cell, 'cm', export='old_cm.hdf5', param_label='cm', show=False, overwrite=True)
    init_biophysics(cell, reset_cable=True, from_file=True, mech_file_path=mech_file_path, correct_cm=True,
                    correct_g_pas=True, env=context.env)
    new_nseg, new_distances = count_nseg(cell)
    compare_nseg([old_nseg, new_nseg], [old_distances, new_distances], ['before', 'after'])
    plot_mech_param_distribution(cell, 'pas', 'g', export='new_dend_gpas.hdf5', overwrite=True, param_label='dend.g_pas',
                                 show=False)
    plot_mech_param_from_file('pas', 'g', ['old_dend_gpas.hdf5', 'new_dend_gpas.hdf5'], ['old', 'new'],
                              param_label='dend.gpas')
    plot_cable_param_distribution(cell, 'cm', export='new_cm.hdf5', param_label='cm', show=False, overwrite=True)
    plot_mech_param_from_file('cm', None, ['old_cm.hdf5', 'new_cm.hdf5'], ['old', 'new'], param_label='cm')


def standard_cable_tests(cell, mech_file_path):
    """

    :param cell: :class:'BiophysCell'
    :param mech_file_path: str
    """
    init_biophysics(cell, reset_cable=True, from_file=True, mech_file_path=mech_file_path)
    plot_cable_param_distribution(cell, 'cm', export='old_cm.hdf5', show=False, overwrite=True)
    modify_mech_param(cell, 'soma', 'cable', 'cm', value=2.)
    init_biophysics(cell, reset_cable=True)
    plot_cable_param_distribution(cell, 'cm', export='new_cm.hdf5', show=False, overwrite=True)
    plot_mech_param_from_file('cm', None, ['old_cm.hdf5', 'new_cm.hdf5'], ['old', 'new'], param_label='cm',
                              yunits='uF/cm2', ylabel='Specific capacitance')

    init_biophysics(cell, reset_cable=True, from_file=True)
    plot_cable_param_distribution(cell, 'Ra', export='orig_Ra.hdf5', show=False, overwrite=True)
    modify_mech_param(cell, 'soma', 'cable', 'Ra', value=200.)
    init_biophysics(cell, reset_cable=True)
    plot_cable_param_distribution(cell, 'Ra', export='modified_Ra.hdf5', show=False, overwrite=True)
    plot_mech_param_from_file('Ra', None, ['orig_Ra.hdf5', 'modified_Ra.hdf5'],
                              ['orig', 'modified'], param_label='Ra', yunits='Ohm*cm', ylabel='Axial resistivity')

    init_biophysics(cell, reset_cable=True, from_file=True)
    old_nseg, old_distances = count_nseg(cell)
    modify_mech_param(cell, 'soma', 'cable', 'spatial_res', value=2.)
    init_biophysics(cell, reset_cable=True)
    new_nseg, new_distances = count_nseg(cell)
    compare_nseg([old_nseg, new_nseg], [old_distances, new_distances], ['before', 'after'])

    init_biophysics(cell, reset_cable=True, from_file=True)
    plot_cable_param_distribution(cell, 'cm', export='cm1.hdf5', show=False, overwrite=True)
    modify_mech_param(cell, 'apical', 'cable', 'cm', value=2.)
    init_biophysics(cell, reset_cable=True)
    plot_cable_param_distribution(cell, 'cm', export='cm2.hdf5', show=False, overwrite=True)
    nseg2, distances2 = count_nseg(cell)
    modify_mech_param(cell, 'apical', 'cable', 'spatial_res', value=2.)
    plot_cable_param_distribution(cell, 'cm', export='cm3.hdf5', show=False, overwrite=True)
    nseg3, distances3 = count_nseg(cell)
    plot_mech_param_from_file('cm', None, ['cm1.hdf5', 'cm2.hdf5', 'cm3.hdf5'], ['orig', 'post step 1', 'post step 2'],
                              param_label='cm', yunits='uF/cm2', ylabel='Specific capacitance')
    compare_nseg([nseg2, nseg3], [distances2, distances3], ['post step 2', 'post step 3'])


def count_spines(cell, env):
    """

    :param cell:
    :param env:
    """
    init_biophysics(cell, env, reset_cable=True, correct_cm=True)
    gid = cell.gid
    syn_attrs = env.synapse_attributes
    sec_index_map = syn_attrs.sec_index_map[gid]
    num_spines_list = []
    distances = []
    for node in cell.apical:
        num_spines = len(syn_attrs.get_filtered_syn_indexes(gid, sec_index_map[node.index],
                                                            syn_types=[env.Synapse_Types['excitatory']]))
        stored_num_spines = sum(node.spine_count)
        if num_spines != stored_num_spines:
            raise ValueError('count_spines_test: failed for node: %s; %i != %i' %
                             (node.name, num_spines, stored_num_spines))
        num_spines_list.append(num_spines)
        distances.append(get_distance_to_node(cell, cell.tree.root, node, 0.5))
        print 'count_spines_test: passed for node: %s; nseg: %i; L: %.2f um; spine_count: %i; density: %.2f /um' % \
              (node.name, node.sec.nseg, node.sec.L, num_spines, num_spines/node.sec.L)
    fig, axes = plt.subplots()
    axes.scatter(distances, num_spines_list)
    axes.set_xlabel('Distance from soma (um)')
    axes.set_ylabel('Spine count')
    clean_axes(axes)
    fig.show()


def standard_modify_syn_mech_param_tests(cell, env):
    """

    :param cell:
    :param env:
    """
    gid = cell.gid
    pop_name = cell.pop_name
    init_biophysics(cell, env, reset_cable=True, correct_cm=True)
    config_syns_from_mech_attrs(gid, env, pop_name, insert=True)
    syn_attrs = env.synapse_attributes
    syn_id_attr_dict = syn_attrs.syn_id_attr_dict[cell.gid]
    sec_index_map = syn_attrs.sec_index_map[cell.gid]
    sec_type = 'apical'
    syn_name = 'AMPA'
    param_name = 'g_unit'

    modify_syn_mech_param(cell, env, sec_type, syn_name, param_name=param_name, value=0.0005,
                          filters={'syn_types': ['excitatory']}, origin='soma', slope=0.0001, tau=50., xhalf=200.,
                          update_targets=True)
    plot_synaptic_attribute_distribution(cell, env, gid, syn_name, param_name, filters=None, from_mech_attrs=True,
                                         from_target_attrs=True, param_label='AMPA.g_unit', export='syn_attrs.hdf5',
                                         description='stage1', show=True)
    modify_syn_mech_param(cell, env, sec_type, syn_name, param_name=param_name,
                          filters={'syn_types': ['excitatory'], 'layers': ['OML']}, origin='apical',
                          origin_filters={'syn_types': ['excitatory'], 'layers': ['MML']}, update_targets=True,
                          append=True)
    plot_synaptic_attribute_distribution(cell, env, gid, syn_name, param_name, filters=None, from_mech_attrs=True,
                                         from_target_attrs=True, param_label='AMPA.g_unit', export='syn_attrs.hdf5',
                                         description='stage2', show=True)
    plot_syn_attr_from_file(syn_name, param_name, 'syn_attrs.hdf5', param_label='AMPA.g_unit')


@click.command()
@click.option("--gid", required=True, type=int, default=0)
@click.option("--pop-name", required=True, type=str, default='GC')
@click.option("--config-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='../dentate/config/Small_Scale_Control_log_normal_weights.yaml')
@click.option("--template-paths", type=str, default='../dgc/Mateos-Aparicio2014:../dentate/templates')
@click.option("--hoc-lib-path", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='../dentate')
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='/mnt/s')  # '/mnt/s')  # '../dentate/datasets'
@click.option("--mech-file-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='mechanisms/20180529_DG_GC_mech.yaml')
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

    standard_modify_mech_param_tests(cell)
    #standard_cable_tests(cell, mech_file_path)
    #cm_correction_test(cell, env, mech_file_path)
    #count_spines(cell, env)
    #standard_modify_syn_mech_param_tests(cell, env)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)