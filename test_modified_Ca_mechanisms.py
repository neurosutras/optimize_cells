from dentate.biophysics_utils import *

context = Context()


@click.command()
@click.option("--gid", required=True, type=int, default=0)
@click.option("--pop-name", required=True, type=str, default='GC')
@click.option("--config-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='../dentate/config/Small_Scale_Control_LN_weights.yaml')
@click.option("--template-paths", type=str, default='../dgc/Mateos-Aparicio2014:../dentate/templates')
@click.option("--hoc-lib-path", type=str, default='../dentate')
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='../dentate/datasets')  # '/mnt/s'
@click.option("--results-path", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='data')
@click.option("--mech-file-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='mechanisms/20180509_DG_GC_test_modified_Ca_mech.yaml')
@click.option('--verbose', '-v', is_flag=True)
def main(gid, pop_name, config_file, template_paths, hoc_lib_path, dataset_prefix, results_path, mech_file_path,
         verbose):
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
    """
    comm = MPI.COMM_WORLD
    np.seterr(all='raise')
    env = Env(comm, config_file, template_paths, hoc_lib_path, dataset_prefix, verbose=verbose)
    configure_hoc_env(env)

    cell = get_biophys_cell(env, pop_name, gid, mech_file_path=mech_file_path)
    context.update(locals())

    init_biophysics(cell, reset_cable=True, correct_cm=True, correct_g_pas=True, env=env)

    equilibrate = 250.
    stim_dur = 100.
    stim_amp = .2
    duration = equilibrate + stim_dur + 100.
    sim = QuickSim(duration)
    sim.append_rec(cell, cell.tree.root, 0.5, description='soma Vm')
    sim.append_rec(cell, cell.tree.root, 0.5, param='_ref_cai', description='cai')
    sim.append_rec(cell, cell.tree.root, 0.5, param='_ref_ica', description='ica')
    sim.append_rec(cell, cell.tree.root, 0.5, param='_ref_ik_CadepK', description='ik_CadepK')
    sim.append_stim(cell, cell.tree.root, 0.5, dur=stim_dur, delay=equilibrate, amp=stim_amp)

    context.update(locals())
    sim.run()
    sim.plot()


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
