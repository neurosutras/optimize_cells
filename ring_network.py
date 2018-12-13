from mpi4py import MPI
import sys
from neuron import h
from ring_cell import BallStick

# h.load_file('nrngui.hoc')
h.load_file('stdrun.hoc')


class Ring(object):
    """

    """
    def __init__(self, ncell, delay, pc, dt=None):
        # spiking script uses dt = 0.02
        self.pc = pc
        self.delay = delay
        self.ncell = int(ncell)
        self.mkring(self.ncell)
        self.mkstim()
        self.voltage_record(dt)
        self.spike_record()
        self.pydicts = {}

    def mkring(self, ncell):
        self.mkcells(ncell)
        self.connectcells(ncell)

    def mkcells(self, ncell):
        rank = int(self.pc.id())
        nhost = int(self.pc.nhost())
        self.cells = []
        self.gids = []
        for i in range(rank, ncell, nhost):
            cell = BallStick()
            self.cells.append(cell)
            self.gids.append(i)
            self.pc.set_gid2node(i, rank)
            nc = cell.connect2target(None)
            self.pc.cell(i, nc)
            test = self.pc.gid2cell(i)
            print "mkcell", type(test)
        # print self.gids

    def connectcells(self, ncell):
        rank = int(self.pc.id())
        nhost = int(self.pc.nhost())
        self.ncdict = {}
        # not efficient but demonstrates use of pc.gid_exists
        for pair in [(0, 1), (0, 2), (1, 2)]:  # connect only cell 0 to cell 1
            presyn_gid = pair[0] % ncell
            target_gid = pair[1] % ncell
            if self.pc.gid_exists(target_gid):
                target = self.pc.gid2cell(target_gid)
                syn = target.synlist[0]
                nc = self.pc.gid_connect(presyn_gid, syn)
                nc.delay = self.delay
                nc.weight[0] = 0.01
                self.ncdict.update({pair: nc})
        # print self.ncdict

    # Instrumentation - stimulation and recording
    def mkstim(self):
        if not self.pc.gid_exists(0):
            return
        self.stim = h.NetStim()
        self.stim.number = 10
        self.stim.start = 0
        self.ncstim = h.NetCon(self.stim, self.pc.gid2cell(0).synlist[0])
        self.ncstim.delay = 0
        self.ncstim.weight[0] = .01

    def update_stim_weight(self, new_weight):
        if 0 in self.gids:
            self.ncstim.weight[0] = new_weight

    def update_syn_weight(self, pair, new_weight):
        if pair in self.ncdict:
            self.ncdict[pair].weight[0] = new_weight

    def spike_record(self):
        self.spike_tvec = {}
        self.spike_idvec = {}
        for i, gid in enumerate(self.gids):
            tvec = h.Vector()
            idvec = h.Vector()
            nc = self.cells[i].connect2target(None)
            self.pc.spike_record(nc.srcgid(), tvec, idvec)
            # Alternatively, could use nc.record(tvec)
            self.spike_tvec[gid] = tvec
            self.spike_idvec[gid] = idvec
        li1 = [];
        li2 = []
        for x in tvec: li1.append(x)
        for x in idvec: li2.append(x)
        print gid, "t", li1
        print gid, "id", li2

    def voltage_record(self, dt=None):
        self.voltage_tvec = {}
        self.voltage_recvec = {}
        if dt is None:
            self.dt = h.dt
        else:
            self.dt = dt
        for i, cell in enumerate(self.cells):
            tvec = h.Vector()
            tvec.record(
                h._ref_t)  # dt is not accepted as an argument to this function in the PC environment -- may need to turn on cvode?
            rec = h.Vector()
            rec.record(getattr(cell.soma(0), '_ref_v'))  # dt is not accepted as an argument
            self.voltage_tvec[self.gids[i]] = tvec
            self.voltage_recvec[self.gids[i]] = rec

    def vecdict_to_pydict(self, vecdict, name):
        self.pydicts[name] = {}
        for key, value in vecdict.iteritems():
            self.pydicts[name][key] = value.to_python()


def runring(ring, pc, comm, tstop=100):
    pc.set_maxstep(10)
    h.stdinit()
    pc.psolve(tstop)
    ring.vecdict_to_pydict(ring.voltage_tvec, 't')
    ring.vecdict_to_pydict(ring.voltage_recvec, 'rec')
    nhost = int(pc.nhost())
    all_dicts = pc.py_alltoall([ring.pydicts for i in range(nhost)])
    # Use MPI Gather instead:
    # all_dicts = comm.gather(ring.pydicts, root=0)
    if int(pc.id()) == 0:
        t = {key: value for dict in all_dicts for key, value in dict['t'].iteritems()}
        rec = {key: value for dict in all_dicts for key, value in dict['rec'].iteritems()}
        li = []
        for i in range(len(rec[0])):
            if i % 300 == 0:
                li.append(rec[0][i])
        print li
        return {'t': t, 'rec': rec}
    # return None
