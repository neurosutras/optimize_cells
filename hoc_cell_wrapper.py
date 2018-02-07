##
##  Dentate Gyrus model initialization script
##  Author: Ivan Raikov

"""
Example call: mpirun -n 4 python hoc_cell_wrapper.py --config-file ../dentate/config/Full_Scale_Control_log_normal_weights.yaml
--template-paths ../dgc/Mateos-Aparicio2014/ --dataset-prefix ../dentate/datasets/ --results-path data

"""

import sys, os
import os.path
import click
import itertools
from collections import defaultdict
from datetime import datetime
import numpy as np
from mpi4py import MPI  # Must come before importing NEURON
import h5py
from neuron import h
from specify_cells4 import *

"""
from neuroh5.io import read_projection_names, scatter_read_graph, bcast_graph, scatter_read_trees, \
    scatter_read_cell_attributes, write_cell_attributes
"""
from dentate.neuroh5_io_utils import *
from dentate.env import Env
import dentate.cells as cells
from nested.utils import *

context = Context()

syn_type_excitatory = 0
syn_type_inhibitory = 1

swc_type_soma = 1
swc_type_axon = 2
swc_type_basal  = 3
swc_type_apical = 4

swc_type_dict = {
      'soma':   1,
      'axon':   2,
      'ais':    2,
      'basal':  3,
      'apical': 4,
      'trunk':  5,
      'tuft':   6
    }

v_init = -75


def make_cell(env, gid, population):
    """

    :param env:
    :param gid:
    :param population:
    :return:
    """
    h('objref templatePaths, templatePathValue')

    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())

    v_sample_seed = int(env.modelConfig['Random Seeds']['Intracellular Voltage Sample'])
    ranstream_v_sample = np.random.RandomState()
    ranstream_v_sample.seed(v_sample_seed)

    datasetPath = os.path.join(env.datasetPrefix, env.datasetName)

    h.templatePaths = h.List()
    for path in env.templatePaths:
        h.templatePathValue = h.Value(1, path)
        h.templatePaths.append(h.templatePathValue)
    popName = population
    templateName = env.celltypes[popName]['template']
    h.find_template(env.pc, h.templatePaths, templateName)

    dataFilePath = os.path.join(datasetPath, env.modelConfig['Cell Data'])

    if rank == 0:
        print 'cell attributes: ', env.cellAttributeInfo

    if env.verbose:
        if env.pc.id() == 0:
            print "*** Creating population %s" % popName

    templateName = env.celltypes[popName]['template']
    templateClass = eval('h.%s' % templateName)

    if env.celltypes[popName].has_key('synapses'):
        synapses = env.celltypes[popName]['synapses']
    else:
        synapses = {}

    v_sample_set = set([])
    env.v_dict[popName] = {}
    """
    for gid in [gid]:
        if ranstream_v_sample.uniform() <= env.vrecordFraction:
            v_sample_set.add(gid)
    """
    #this is the gid we will use for testing purposes
    if env.cellAttributeInfo.has_key(popName) and env.cellAttributeInfo[popName].has_key('Trees'):
        if env.verbose:
            if env.pc.id() == 0:
                print "*** Reading trees for population %s" % popName
        """
        if env.nodeRanks is None:
            (trees, forestSize) = scatter_read_trees(env.comm, dataFilePath, popName, io_size=env.IOsize)
        else:
            (trees, forestSize) = scatter_read_trees(env.comm, dataFilePath, popName, io_size=env.IOsize,
                                                     node_rank_map=env.nodeRanks)
        """
        tree = select_tree_attributes(gid, env.comm, dataFilePath, popName)
        if env.verbose:
            if env.pc.id() == 0:
                print "*** Done reading trees for population %s" % popName

        h.numCells = 0
        i = 0
        if env.verbose:
            if env.pc.id() == 0:
                print "*** Creating gid %i" % gid

        verboseflag = 0
        model_cell = cells.make_neurotree_cell(templateClass, neurotree_dict=tree, gid=gid, local_id=i,
                                               dataset_path=datasetPath)
        """
        if env.verbose:
            if (rank == 0) and (i == 0):
                for sec in list(model_cell.all):
                    h.psection(sec=sec)
        env.gidlist.append(gid)
        env.cells.append(model_cell)
        env.pc.set_gid2node(gid, int(env.pc.id()))
        ## Tell the ParallelContext that this cell is a spike source
        ## for all other hosts. NetCon is temporary.
        nc = model_cell.connect2target(h.nil)
        env.pc.cell(gid, nc, 1)
        ## Record spikes of this cell
        env.pc.spike_record(gid, env.t_vec, env.id_vec)
        ## Record voltages from a subset of cells
        if gid in v_sample_set:
            v_vec = h.Vector()
            soma = list(model_cell.soma)[0]
            v_vec.record(soma(0.5)._ref_v)
            env.v_dict[popName][gid] = v_vec
        i = i + 1
        h.numCells = h.numCells + 1
        if env.verbose:
            if env.pc.id() == 0:
                print "*** Created %i cells" % i
        """
    return model_cell



def init(env, hoc_template_path):
    """

    :param env:
    """
    h.load_file("nrngui.hoc")
    h.load_file("loadbal.hoc")
    h('objref fi_status, fi_checksimtime, pc, nclist, nc, nil')
    h('strdef datasetPath')
    h('numCells = 0')
    h('totalNumCells = 0')
    h('max_walltime_hrs = 0')
    h('mkcellstime = 0')
    h('mkstimtime = 0')
    h('connectcellstime = 0')
    h('connectgjstime = 0')
    h('results_write_time = 0')
    h.nclist = h.List()
    datasetPath = os.path.join(env.datasetPrefix, env.datasetName)
    h.datasetPath = datasetPath
    ##  new ParallelContext object
    h.pc = h.ParallelContext()
    env.pc = h.pc
    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())
    print 'made pc'
    ## polymorphic value template
    h.load_file(hoc_template_path + "/templates/Value.hoc")
    ## randomstream template
    h.load_file(hoc_template_path + "/templates/ranstream.hoc")
    ## stimulus cell template
    h.load_file(hoc_template_path + "/templates/StimCell.hoc")

    h.xopen(hoc_template_path + "/lib.hoc")
    print 'opened hoc files'
    h.dt = env.dt
    h.tstop = env.tstop
    """
    if env.optldbal or env.optlptbal:
        lb = h.LoadBalance()
        if not os.path.isfile("mcomplex.dat"):
            lb.ExperimentalMechComplex()

    if (env.pc.id() == 0):
        mkspikeout(env, env.spikeoutPath)
    """
    env.pc.barrier()
    h.startsw()
    """
    env.mkcellstime = h.stopsw()
    env.pc.barrier()
    if (env.pc.id() == 0):
        print "*** Cells created in %g seconds" % env.mkcellstime
    print "*** Rank %i created %i cells" % (env.pc.id(), len(env.cells))
    h.startsw()
    mkstim(env)
    env.mkstimtime = h.stopsw()
    if (env.pc.id() == 0):
        print "*** Stimuli created in %g seconds" % env.mkstimtime
    env.pc.barrier()
    h.startsw()
    connectcells(env)
    env.connectcellstime = h.stopsw()
    env.pc.barrier()
    if (env.pc.id() == 0):
        print "*** Connections created in %g seconds" % env.connectcellstime
    print "*** Rank %i created %i connections" % (env.pc.id(), int(h.nclist.count()))
    h.startsw()
    # connectgjs(env)
    env.connectgjstime = h.stopsw()
    if (env.pc.id() == 0):
        print "*** Gap junctions created in %g seconds" % env.connectgjstime
    env.pc.setup_transfer()
    env.pc.set_maxstep(10.0)
    h.max_walltime_hrs = env.max_walltime_hrs
    h.mkcellstime = env.mkcellstime
    h.mkstimtime = env.mkstimtime
    h.connectcellstime = env.connectcellstime
    h.connectgjstime = env.connectgjstime
    h.results_write_time = env.results_write_time
    h.fi_checksimtime = h.FInitializeHandler("checksimtime(pc)")
    if (env.pc.id() == 0):
        print "dt = %g" % h.dt
        print "tstop = %g" % h.tstop
        h.fi_status = h.FInitializeHandler("simstatus()")
    h.v_init = env.v_init
    h.stdinit()
    h.finitialize(env.v_init)
    env.pc.barrier()
    if env.optldbal or env.optlptbal:
        cx(env)
        ld_bal(env)
        if env.optlptbal:
            lpt_bal(env)
    """

"""
@click.command()
@click.option("--gid", required=True, type=int)
@click.option("--pop-name", required=True, type=str)
@click.option("--config-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--template-paths", type=str)
@click.option("--hoc-templates-paths", type=str)
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--results-path", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--results-id", type=str, required=False, default='')
@click.option("--node-rank-file", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--io-size", type=int, default=1)
@click.option("--coredat", is_flag=True)
@click.option("--vrecord-fraction", type=float, default=0.001)
@click.option("--tstop", type=int, default=1)
@click.option("--v-init", type=float, default=-75.0)
@click.option("--max-walltime-hours", type=float, default=1.0)
@click.option("--results-write-time", type=float, default=360.0)
@click.option("--dt", type=float, default=0.025)
@click.option("--ldbal", is_flag=True)
@click.option("--lptbal", is_flag=True)
@click.option('--verbose', '-v', is_flag=True)
def main(config_file, template_paths, hoc_templates_path, dataset_prefix, results_path, results_id, node_rank_file, io_size, coredat,
         vrecord_fraction, tstop, v_init, max_walltime_hours, results_write_time, dt, ldbal, lptbal, verbose):
"""

def main(gid, pop_name, config_file, template_paths, hoc_templates_path, dataset_prefix=None, results_path=None, results_id='',
         node_rank_file=None, io_size=1, coredat=False, vrecord_fraction=0.001, tstop=1, v_init=-75.,
         max_walltime_hours=1, results_write_time=360., dt=0.025, ldbal=False, lptbal=False, verbose=False, **kwargs):
    print 'inside hoc cell wrapper'
    np.seterr(all='raise')
    comm = MPI.COMM_WORLD
    template_paths += ':%s/templates' %hoc_templates_path
    print 'will make env'
    print 'config file'
    print  config_file
    print 'template paths'
    print template_paths
    env = Env(comm, config_file,
              template_paths, dataset_prefix, results_path, results_id,
              node_rank_file, io_size,
              vrecord_fraction, coredat, tstop, v_init,
              max_walltime_hours, results_write_time,
              dt, ldbal, lptbal, verbose)
    print 'made env'
    init(env, hoc_templates_path)
    context.env = env
    print 'initialized env'
    hoc_cell = make_cell(env, gid, pop_name)
    print 'made hoc cell'
    cell = HocCell(existing_hoc_cell=hoc_cell)
    cell.load_morphology()
    print 'made python cell'
    context.update(locals())
    return cell

if __name__ == '__main__':
    # main(args=sys.argv[(sys.argv.index("main.py") + 1):])
    main(0, 'GC', '../dentate/config/Small_Scale_Control_log_normal_weights.yaml', '../dgc/Mateos-Aparicio2014', '../dentate',
         '../dentate/datasets', 'data')