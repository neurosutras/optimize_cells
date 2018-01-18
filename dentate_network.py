import sys, os
import os.path
import itertools
from collections import defaultdict
from datetime import datetime
import numpy as np
from mpi4py import MPI # Must come before importing NEURON
import h5py
from neuron import h
from neuroh5.io import read_projection_names, scatter_read_graph, bcast_graph, scatter_read_trees, scatter_read_cell_attributes, write_cell_attributes
from env import Env
import lpt, utils, synapses, cells


## Estimate cell complexity. Code by Michael Hines from the discussion thread
## https://www.neuron.yale.edu/phpBB/viewtopic.php?f=31&t=3628
def cx(env):
  rank   = int(env.pc.id())
  lb = h.LoadBalance()
  if os.path.isfile("mcomplex.dat"):
    lb.read_mcomplex()
  cxvec = h.Vector(len(env.gidlist))
  for i, gid in enumerate(env.gidlist):
    cxvec.x[i] = lb.cell_complexity(env.pc.gid2cell(gid))
  env.cxvec = cxvec
  return cxvec

# for given cxvec on each rank what is the fractional load balance.
def ld_bal(env):
  rank   = int(env.pc.id())
  nhosts = int(env.pc.nhost())
  cxvec  = env.cxvec
  sum_cx = sum(cxvec)
  max_sum_cx = env.pc.allreduce(sum_cx, 2)
  sum_cx = env.pc.allreduce(sum_cx, 1)
  if rank == 0:
    print ("*** expected load balance %.2f" % (sum_cx / nhosts / max_sum_cx))

# Each rank has gidvec, cxvec: gather everything to rank 0, do lpt
# algorithm and write to a balance file.
def lpt_bal(env):
  rank   = int(env.pc.id())
  nhosts = int(env.pc.nhost())

  cxvec  = env.cxvec
  gidvec = env.gidlist
  #gather gidvec, cxvec to rank 0
  src    = [None]*nhosts
  src[0] = zip(cxvec.to_python(), gidvec)
  dest   = env.pc.py_alltoall(src)
  del src

  if rank == 0:
    lb = h.LoadBalance()
    allpairs = sum(dest,[])
    del dest
    parts = lpt.lpt(allpairs, nhosts)
    lpt.statistics(parts)
    part_rank = 0
    with open('parts.%d' % nhosts, 'w') as fp:
      for part in parts:
        for x in part[1]:
          fp.write('%d %d\n' % (x[1],part_rank))
        part_rank = part_rank+1


def mkspikeout (env, spikeout_filename):
    datasetPath     = os.path.join(env.datasetPrefix,env.datasetName)
    forestFilePath  = os.path.join(datasetPath,env.modelConfig['Cell Data'])
    forestFile      = h5py.File(forestFilePath,'r')
    spikeoutFile    = h5py.File(spikeout_filename,'w')
    forestFile.copy('/H5Types',spikeoutFile)
    forestFile.close()
    spikeoutFile.close()


def init(env):
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
    datasetPath  = os.path.join(env.datasetPrefix, env.datasetName)
    h.datasetPath = datasetPath
    ##  new ParallelContext object
    h.pc   = h.ParallelContext()
    env.pc = h.pc
    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())
    ## polymorphic value template
    h.load_file("./templates/Value.hoc")
    ## randomstream template
    h.load_file("./templates/ranstream.hoc")
    ## stimulus cell template
    h.load_file("./templates/StimCell.hoc")
    h.xopen("./lib.hoc")
    h.dt = env.dt
    h.tstop = env.tstop
    if env.optldbal or env.optlptbal:
        lb = h.LoadBalance()
        if not os.path.isfile("mcomplex.dat"):
            lb.ExperimentalMechComplex()

    if (env.pc.id() == 0):
      mkspikeout (env, env.spikeoutPath)
    env.pc.barrier()
    h.startsw()
    mkcells(env)
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
    #connectgjs(env)
    env.connectgjstime = h.stopsw()
    if (env.pc.id() == 0):
        print "*** Gap junctions created in %g seconds" % env.connectgjstime
    env.pc.setup_transfer()
    env.pc.set_maxstep(10.0)
    h.max_walltime_hrs   = env.max_walltime_hrs
    h.mkcellstime        = env.mkcellstime
    h.mkstimtime         = env.mkstimtime
    h.connectcellstime   = env.connectcellstime
    h.connectgjstime     = env.connectgjstime
    h.results_write_time = env.results_write_time
    h.fi_checksimtime    = h.FInitializeHandler("checksimtime(pc)")
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


# Run the simulation
def run (env):
    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())

    env.pc.barrier()
    env.pc.psolve(h.tstop)

    if (rank == 0):
        print "*** Simulation completed"

    env.pc.barrier()
    if (rank == 0):
        print "*** Writing results data"
    spikeout(env, env.spikeoutPath, np.array(env.t_vec, dtype=np.float32), np.array(env.id_vec, dtype=np.uint32))

    # TODO:
    #if (env.vrecordFraction > 0):
    #    h.vrecordout("%s/%s_vrecord_%d.dat" % (env.resultsPath, env.modelName, env.pc.id(), env.indicesVrecord))

    comptime = env.pc.step_time()
    cwtime   = comptime + env.pc.step_wait()
    maxcw    = env.pc.allreduce(cwtime, 2)
    avgcomp  = env.pc.allreduce(comptime, 1)/nhosts
    maxcomp  = env.pc.allreduce(comptime, 2)

    if (env.pc.id() == 0):
        print "Execution time summary for host 0:"
        print "  created cells in %g seconds" % env.mkcellstime
        print "  connected cells in %g seconds" % env.connectcellstime
        print "  created gap junctions in %g seconds" % env.connectgjstime
        print "  ran simulation in %g seconds" % comptime
        if (maxcw > 0):
            print "  load balance = %g" % (avgcomp/maxcw)

    env.pc.runworker()
    env.pc.done()
    h.quit()
