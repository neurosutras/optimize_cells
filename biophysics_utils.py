"""
Tools for pulling individual neurons out of the dentate network simulation environment for single-cell tuning.
"""
__author__ = 'Grace Ng, Aaron D. Milstein, Ivan Raikov'
import click
from nested.utils import *
from dentate.neuron_utils import *
from neuroh5.h5py_io_utils import *
from dentate.env import Env
from dentate.cells import *
from dentate.synapses import *


context = Context()


class QuickSim(object):
    """
    This class is used to organize and run simple single cell simulations. Stores references and metadata for current
    injection stimuli and vector recordings. Handles export of recorded simulation data to HDF5 files.
    """
    def __init__(self, tstop=400., cvode=True, daspk=False, dt=0.025, verbose=True):
        """

        :param tstop: float
        :param cvode: bool
        :param daspk: bool
        :param dt: float
        :param verbose: bool
        """
        self.recs = defaultdict(dict)  # dict: {rec_name: dict containing recording metadata }
        self.stims = defaultdict(dict) # dict: {stim_name: dict containing stimulation metadata }
        self.tstop = tstop
        h.load_file('stdrun.hoc')
        h.celsius = 35.0
        h.cao0_ca_ion = 1.3
        self.cvode_atol = 0.01  # 0.001
        self.daspk = daspk
        self._cvode = cvode
        self.cvode = cvode
        self.dt = dt
        self.verbose = verbose
        self.tvec = h.Vector()
        self.tvec.record(h._ref_t, self.dt)
        self.parameters = {}

    def run(self, v_init=-65.):
        """

        :param v_init: float
        """
        start_time = time.time()
        h.tstop = self.tstop
        if self._cvode != self.cvode:
            self.cvode = self._cvode
        if not self._cvode:
            h.dt = self.dt
            h.steps_per_ms = int(1. / self.dt)
        h.v_init = v_init
        h.run()
        if self.verbose:
            print 'Simulation runtime: %.2f s' % (time.time() - start_time)

    def append_rec(self, cell, node, name=None, loc=None, param='_ref_v', object=None, ylabel='Vm', units='mV',
                   description=''):
        """

        :param cell: :class:'BiophysCell'
        :param node: :class:'SHocNode'
        :param name: str
        :param loc: float
        :param param: str
        :param object: :class:'HocObject'
        :param ylabel: str
        :param units: str
        :param description: str
        """
        if name is None:
            name = 'rec%i' % len(self.recs)
        elif name in self.recs:
            name = '%s%i' % (name, len(self.recs))
        self.recs[name]['cell'] = cell
        self.recs[name]['node'] = node
        self.recs[name]['ylabel'] = ylabel
        self.recs[name]['units'] = units
        self.recs[name]['vec'] = h.Vector()
        if object is None:
            if loc is None:
                loc = 0.5
            self.recs[name]['vec'].record(getattr(node.sec(loc), param), self.dt)
        else:
            if loc is None:
                try:
                    loc = object.get_segment().x
                except:
                    loc = 0.5  # if the object doesn't have a .get_segment() method, default to 0.5
            if param is None:
                self.recs[name]['vec'].record(object, self.dt)
            else:
                self.recs[name]['vec'].record(getattr(object, param), self.dt)
        self.recs[name]['loc'] = loc
        self.recs[name]['description'] = description

    def has_rec(self, name):
        """
        Report whether a recording exists with the provided name.
        :param name: str
        :return: bool
        """
        return name in self.recs

    def get_rec(self, name):
        """
        Return the rec_dict associated with the provided name.
        :param description: str
        :return: dict
        """
        if self.has_rec(name):
            return self.recs[name]
        else:
            raise KeyError('QuickSim: get_rec: cannot find recording with name: %s' % name)

    def modify_rec(self, name, node=None, loc=None, object=None, param='_ref_v', ylabel=None, units=None,
                   description=None):
        """

        :param name: str
        :param node: class:'SHocNode'
        :param loc: float
        :param object: class:'HocObject'
        :param param: str
        :param ylabel: str
        :param units: str
        :param description: str
        """
        if not self.has_rec(name):
            raise KeyError('QuickSim: modify_rec: cannot find recording with name: %s' % name)
        if ylabel is not None:
            self.recs[name]['ylabel'] = ylabel
        if units is not None:
            self.recs[name]['units'] = units
        if node is not None:
            self.recs[name]['node'] = node
        if loc is not None:
            self.recs[name]['loc'] = loc
        if object is None:
            self.recs[name]['vec'].record(getattr(self.recs[name]['node'].sec(self.recs[name]['loc']), param), self.dt)
        elif param is None:
            self.recs[name]['vec'].record(object, self.dt)
        else:
            self.recs[name]['vec'].record(getattr(object, param), self.dt)
        if description is not None:
            self.recs[name]['description'] = description

    def append_stim(self, cell, node, name=None, loc=0.5, amp=0., delay=0., dur=0., description='IClamp'):
        """

        :param cell: :class:'BiophysCell'
        :param node: :class:'SHocNode'
        :param name: str
        :param loc: float
        :param amp: float
        :param delay: float
        :param dur: float
        :param description: str
        """
        if name is None:
            name = 'stim%i' % len(self.stims)
        elif name in self.stims:
            name = '%s%i' % (name, len(self.stims))
        self.stims[name]['cell'] = cell
        self.stims[name]['node'] = node
        self.stims[name]['stim'] = h.IClamp(node.sec(loc))
        self.stims[name]['stim'].amp = amp
        self.stims[name]['stim'].delay = delay
        self.stims[name]['stim'].dur = dur
        self.stims[name]['vec'] = h.Vector()
        self.stims[name]['vec'].record(self.stims[name]['stim']._ref_i, self.dt)
        self.stims[name]['description'] = description

    def has_stim(self, name):
        """
        Report whether a stimulus exists with the provided name.
        :param name: str
        :return: bool
        """
        return name in self.stims

    def get_stim(self, name):
        """
        Return the stim_dict associated with the provided name.
        :param description: str
        :return: dict
        """
        if self.has_stim(name):
            return self.stims[name]
        else:
            raise KeyError('QuickSim: get_stim: cannot find stimulus with name: %s' % name)

    def modify_stim(self, name, node=None, loc=None, amp=None, delay=None, dur=None, description=None):
        """

        :param name: str
        :param node: class:'SHocNode'
        :param loc: float
        :param amp: float
        :param delay: float
        :param dur: float
        :param description: str
        """
        if not (node is None and loc is None):
            if not node is None:
                self.stims[name]['node'] = node
            if loc is None:
                loc = self.stims[name]['stim'].get_segment().x
            self.stims[name]['stim'].loc(self.stims[name]['node'].sec(loc))
        if amp is not None:
            self.stims[name]['stim'].amp = amp
        if delay is not None:
            self.stims[name]['stim'].delay = delay
        if dur is not None:
            self.stims[name]['stim'].dur = dur
        if description is not None:
            self.stims[name]['description'] = description

    def plot(self):
        """

        """
        if len(self.recs) == 0:
            return
        fig, axes = plt.subplots()
        for name, rec_dict in self.recs.iteritems():
            description = str(rec_dict['description'])
            axes.plot(self.tvec, rec_dict['vec'],
                      label='%s: %s(%.2f) %s' % (name, rec_dict['node'].name, rec_dict['loc'], description))
            axes.set_xlabel('Time (ms)')
            axes.set_ylabel('%s (%s)' % (rec_dict['ylabel'], rec_dict['units']))
        axes.legend(loc='best', frameon=False, framealpha=0.5)
        title = None
        if 'title' in self.parameters:
            title = self.parameters['title']
        if 'description' in self.parameters:
            if title is not None:
                title = title + '; ' + self.parameters['description']
            else:
                title = self.parameters['description']
        if title is not None:
            axes.set_title(title)
        clean_axes(axes)
        fig.tight_layout()
        fig.show()

    def export_to_file(self, file_path, append=True):
        """
        Exports simulated data and metadata to an HDF5 file. Arrays are saved as datasets and metadata is saved as
        attributes. Repeated simulations are stored in enumerated groups.
        :param file_path: str (path)
        :param append: bool
        """
        if append:
            io_type = 'a'
        else:
            io_type = 'w'
        with h5py.File(file_path, io_type) as f:
            if 'sim_output' not in f:
                f.create_group('sim_output')
                f['sim_output'].attrs['enumerated'] = True
            target = f['sim_output']
            simiter = len(target)
            if str(simiter) not in target:
                target.create_group(str(simiter))
            target[str(simiter)].create_dataset('time', compression='gzip', data=self.tvec)
            target[str(simiter)]['time'].attrs['dt'] = self.dt
            for parameter in self.parameters:
                target[str(simiter)].attrs[parameter] = self.parameters[parameter]
            if len(self.stims) > 0:
                target[str(simiter)].create_group('stims')
                for name, stim_dict in self.stims.iteritems():
                    stim = target[str(simiter)]['stims'].create_dataset(name, compression='gzip', data=stim_dict['vec'])
                    cell = stim_dict['cell']
                    stim.attrs['cell'] = cell.gid
                    node = stim_dict['node']
                    stim.attrs['index'] = node.index
                    stim.attrs['type'] = node.type
                    loc = stim_dict['stim'].get_segment().x
                    stim.attrs['loc'] = loc
                    distance = get_distance_to_node(cell, cell.tree.root, node, loc)
                    stim.attrs['soma_distance'] = distance
                    distance = get_distance_to_node(cell, get_dendrite_origin(cell, node), node, loc)
                    stim.attrs['branch_distance'] = distance
                    stim.attrs['amp'] = stim_dict['stim'].amp
                    stim.attrs['delay'] = stim_dict['stim'].delay
                    stim.attrs['dur'] = stim_dict['stim'].dur
                    stim.attrs['description'] = stim_dict['description']
            target[str(simiter)].create_group('recs')
            for name, rec_dict in self.recs.iteritems():
                rec = target[str(simiter)]['recs'].create_dataset(name, compression='gzip', data=rec_dict['vec'])
                cell = rec_dict['cell']
                rec.attrs['cell'] = cell.gid
                node = rec_dict['node']
                rec.attrs['index'] = node.index
                rec.attrs['type'] = node.type
                loc = rec_dict['loc']
                rec.attrs['loc'] = loc
                distance = get_distance_to_node(cell, cell.tree.root, node, loc)
                rec.attrs['soma_distance'] = distance
                distance = get_distance_to_node(cell, get_dendrite_origin(cell, node), node, loc)
                node_is_terminal = is_terminal(node)
                branch_order = get_branch_order(cell, node)
                rec.attrs['branch_distance'] = distance
                rec.attrs['is_terminal'] = node_is_terminal
                rec.attrs['branch_order'] = branch_order
                rec.attrs['ylabel'] = rec_dict['ylabel']
                rec.attrs['units'] = rec_dict['units']
                rec.attrs['description'] = rec_dict['description']

    def get_cvode(self):
        """

        :return bool
        """
        return bool(h.CVode().active())

    def set_cvode(self, state):
        """

        :param state: bool
        """
        if state:
            h.CVode().active(1)
            h.CVode().atol(self.cvode_atol)
            h.CVode().use_daspk(int(self.daspk))
        else:
            h.CVode().active(0)
        self._cvode = state

    cvode = property(get_cvode, set_cvode)


def make_hoc_cell(env, gid, population):
    """

    :param env:
    :param gid:
    :param population:
    :return:
    """
    popName = population
    datasetPath = env.datasetPath
    dataFilePath = env.dataFilePath
    env.load_cell_template(popName)
    templateClass = getattr(h, env.celltypes[popName]['template'])

    if env.cellAttributeInfo.has_key(popName) and env.cellAttributeInfo[popName].has_key('Trees'):
        tree = select_tree_attributes(gid, env.comm, dataFilePath, popName)
        i = h.numCells
        hoc_cell = make_neurotree_cell(templateClass, neurotree_dict=tree, gid=gid, local_id=i,
                                             dataset_path=datasetPath)
        h.numCells = h.numCells + 1
    else:
        raise Exception('make_hoc_cell: data file: %s does not contain morphology for population: %s, gid: %i' %
                        dataFilePath, popName, gid)
    return hoc_cell


def configure_env(env):
    """

    :param env: :class:'Env'
    """
    h.load_file("nrngui.hoc")
    h.load_file("loadbal.hoc")
    h('objref fi_status, fi_checksimtime, pc, nclist, nc, nil')
    h('strdef datasetPath')
    h('numCells = 0')
    h('totalNumCells = 0')
    h.nclist = h.List()
    h.datasetPath = env.datasetPath
    h.pc = h.ParallelContext()
    env.pc = h.pc
    ## polymorphic value template
    h.load_file(env.hoclibPath + '/templates/Value.hoc')
    ## randomstream template
    h.load_file(env.hoclibPath + '/templates/ranstream.hoc')
    ## stimulus cell template
    h.load_file(env.hoclibPath + '/templates/StimCell.hoc')
    h.xopen(env.hoclibPath + '/lib.hoc')

    h('objref templatePaths, templatePathValue')
    h.templatePaths = h.List()
    for path in env.templatePaths:
        h.templatePathValue = h.Value(1, path)
        h.templatePaths.append(h.templatePathValue)


def get_biophys_cell(env, gid, pop_name):
    """
    TODO: Use Connections: distance attribute to compute and load netcon delays
    TODO: Consult env for weights namespaces, load_syn_weights
    :param env:
    :param gid:
    :param pop_name:
    :return:
    """
    hoc_cell = make_hoc_cell(env, gid, pop_name)
    cell = BiophysCell(gid=gid, pop_name=pop_name, hoc_cell=hoc_cell, env=env)
    syn_attrs = env.synapse_attributes
    if pop_name not in syn_attrs.select_cell_attr_index_map:
        syn_attrs.select_cell_attr_index_map[pop_name] = \
            get_cell_attributes_index_map(env.comm, env.dataFilePath, pop_name, 'Synapse Attributes')
    syn_attrs.load_syn_id_attrs(gid, select_cell_attributes(gid, env.comm, env.dataFilePath,
                                                            syn_attrs.select_cell_attr_index_map[pop_name], pop_name,
                                                            'Synapse Attributes'))

    for source_name in env.projection_dict[pop_name]:
        if source_name not in syn_attrs.select_edge_attr_index_map[pop_name]:
            syn_attrs.select_edge_attr_index_map[pop_name][source_name] = \
                get_edge_attributes_index_map(env.comm, env.connectivityFilePath, source_name, pop_name)
        source_indexes, edge_attr_dict = \
            select_edge_attributes(gid, env.comm, env.connectivityFilePath,
                                   syn_attrs.select_edge_attr_index_map[pop_name][source_name], source_name, pop_name,
                                   ['Synapses'])
        syn_attrs.load_edge_attrs(gid, source_name, edge_attr_dict['Synapses']['syn_id'], env)
    env.biophys_cells[pop_name][gid] = cell
    return cell


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

    init_biophysics(cell, reset_cable=True, from_file=True, mech_file_path=mech_file_path, correct_cm=True,
                    correct_g_pas=True, env=env)
    # TODO init_syn_mech_attrs
    config_syns_from_mech_attrs(gid, env, pop_name, insert=True)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)