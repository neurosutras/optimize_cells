from mpi4py import MPI
from neuron import h
import numpy as np
import random

# for h.lambda_f
h.load_file('stdlib.hoc')
# for h.stdinit
h.load_file('stdrun.hoc')


# adopted from ring_network.py and ring_cell.py
NUM_POP = 3


"""FF ; I ; E"""


# =================== network class
class Network(object):

    def __init__(self, ncell, delay, pc, tstop, dt=None, ff_prob=1., e2e_prob=.05, e2i_prob=.05, \
                 i2i_prob=.05, i2e_prob=.05, ff_weight=2., e2e_weight=1., e2i_weight=1., i2i_weight=.5, \
                 i2e_weight=.5, ff_meanfreq=100):
        # spiking script uses dt = 0.02
        self.pc = pc
        self.delay = delay
        self.ncell = int(ncell)
        self.prob_dict = {'ff': ff_prob, 'e2e': e2e_prob, 'e2i': e2i_prob, 'i2i': i2i_prob, 'i2e': i2e_prob}
        self.weight_dict = {'ff' : ff_weight, 'e2e' : e2e_weight, 'e2i' : e2i_weight, 'i2i' : i2i_weight, \
                             'i2e' : i2e_weight}
        self.index_dict = {'ff': ((0, ncell), (ncell, ncell * NUM_POP)),  # exclusive [x, y)
                           'e2e': ((ncell, ncell * 2), (ncell, ncell * 2)),
                           'e2i': ((ncell, ncell * 2), (ncell * 2, ncell * NUM_POP)),
                           'i2i': ((ncell * 2, ncell * NUM_POP), (ncell * 2, ncell * NUM_POP)),
                           'i2e': ((ncell * 2, ncell * NUM_POP), (ncell, ncell * 2))}
        self.tstop = tstop
        self.ff_meanfreq = ff_meanfreq
        self.event_rec = {}
        self.mknetwork(self.ncell)
        #self.mkstim(self.ncell)
        self.voltage_record(dt)
        self.spike_record()
        self.pydicts = {}

    def mknetwork(self, ncell):
        self.mkcells(ncell)
        self.connectcells(ncell)

    def mkcells(self, ncell):
        rank = int(self.pc.id())
        nhost = int(self.pc.nhost())
        self.cells = []
        self.gids = []
        for i in range(rank, ncell * NUM_POP, nhost):
            if i < ncell:
                cell = FFCell(self.tstop, self.ff_meanfreq)
            else: 
                if i not in list(range(ncell * 2, ncell * 3)):
                    cell_type = 'RS'
                else:
                    cell_type = 'FS'
                cell = IzhiCell(cell_type)
            self.cells.append(cell)
            self.gids.append(i)
            self.pc.set_gid2node(i, rank)
            nc = cell.connect2target(None)
            self.pc.cell(i, nc)
            test = self.pc.gid2cell(i)
            if not cell.is_art():
                rec = h.Vector()
                nc.record(rec)
                self.event_rec[i] = rec


    def createpairs(self, prob, input_indices, output_indices):
        pair_list = []
        for i in input_indices:
            for o in output_indices:
                if random.random() <= prob:
                    pair_list.append((i, o))
        for elem in pair_list:
            x, y = elem
            if x == y: pair_list.remove(elem)
        return pair_list

    def get_cell_type(self, gid):
        if gid < self.ncell:
            return None
        elif gid < self.ncell * 2:
            return 'FS'
        else:
            return 'RS'

    def connectcells(self, ncell):
        rank = int(self.pc.id())
        nhost = int(self.pc.nhost())
        self.ncdict = {}
        # not efficient but demonstrates use of pc.gid_exists
        for connection in ['ff', 'e2e', 'e2i', 'i2i', 'i2e']:
            indices = self.index_dict[connection]
            inp = indices[0]
            out = indices[1]
            pair_list = self.createpairs(self.prob_dict[connection], list(range(inp[0], \
                                                                                inp[1])), list(range(out[0], out[1])))
            for pair in pair_list:
                presyn_gid = pair[0]
                target_gid = pair[1]
                if self.pc.gid_exists(target_gid):
                    target = self.pc.gid2cell(target_gid)
                    pre_type = self.get_cell_type(presyn_gid)
                    target_type = self.get_cell_type(target_gid)
                    if pre_type is None: #E
                        syn = target.synlist[0]
                        weight = self.weight_dict['ff']
                    elif pre_type == 'RS': #E
                        syn = target.synlist[0]
                        if target_type == 'RS':
                            weight = self.weight_dict['e2e']
                        else:
                            weight = self.weight_dict['e2i']
                    else: #I
                        syn = target.synlist[1]
                        if target_type == 'RS':
                            weight = self.weight_dict['i2e']
                        else:
                            weight = self.weight_dict['i2i']
                    nc = self.pc.gid_connect(presyn_gid, syn)
                    nc.delay = self.delay
                    nc.weight[0] = weight
                    self.ncdict[pair] = nc

    # Instrumentation - stimulation and recording
    def mkstim(self, ncell):
        self.ns = h.NetStim()
        self.ns.number = 100
        self.ns.start = 0
        for i in range(ncell):
            if not self.pc.gid_exists(i):  # or random.random() >= .3:  # stimulate only 30% of FF
                continue
            self.nc = h.NetCon(self.ns, self.pc.gid2cell(i).synlist[0])
            self.nc.delay = 0
            self.nc.weight[0] = 2
            self.ncdict[('stim', i)] = self.nc
            print self.ncdict

    def spike_record(self):
        self.spike_tvec = {}
        self.spike_idvec = {}
        for i, gid in enumerate(self.gids):
            tvec = h.Vector()
            idvec = h.Vector()
            nc = self.cells[i].connect2target(None)
            self.pc.spike_record(nc.srcgid(), tvec, idvec)# Alternatively, could use nc.record(tvec)
            self.spike_tvec[gid] = tvec
            self.spike_idvec[gid] = idvec

    def event_record(self, dt = None):
        for i, cell in enumerate(self. cells):
            if cell.is_art(): continue
            nc = h.NetCon(cell.sec(.5)._ref_v, None)
            rec = h.Vector()
            nc.record(rec)
            self.event_rec[i] = rec

            
    def voltage_record(self, dt=None):
        self.voltage_tvec = {}
        self.voltage_recvec = {}
        if dt is None:
            self.dt = h.dt
        else:
            self.dt = dt
        for i, cell in enumerate(self.cells):
            if cell.is_art(): continue
            tvec = h.Vector()
            tvec.record(
                h._ref_t)  # dt is not accepted as an argument to this function in the PC environment -- may need to turn on cvode?
            rec = h.Vector()
            rec.record(getattr(cell.sec(.5), '_ref_v'))  # dt is not accepted as an argument
            self.voltage_tvec[self.gids[i]] = tvec
            self.voltage_recvec[self.gids[i]] = rec

    def vecdict_to_pydict(self, vecdict, name):
        self.pydicts[name] = {}
        for key, value in vecdict.iteritems():
            self.pydicts[name][key] = value.to_python()
            # Alternatively, could use nc.record(tvec)

    def compute_isi(self, vecdict):
        self.ratedict = {}
        self.peakdict = {}
        for key, vec in vecdict.iteritems():
            isivec = h.Vector()
            if len(vec) > 1: 
                isivec.deriv(vec, 1, 1)
                rate = 1. / (isivec.mean() / 1000)
                self.ratedict[key] = rate
                self.peakdict[key] = 1. / (isivec.min() / 1000)

    def summation(self, vecdict):
        self.osc_E = h.Vector(self.tstop + 2)
        self.osc_I = h.Vector(self.tstop + 2)
        for key, vec in vecdict.iteritems():
            cell_type = self.get_cell_type(key)
            binned = vec.histogram(0, self.tstop, 1)
            if cell_type == 'RS':  # E
                self.osc_E.add(binned)
            elif cell_type == 'FS':  # F
                self.osc_I.add(binned)

    def compute_peak_osc_freq(self, osc):
        sparse_t = []
        peak_sum = osc.max()
        for i, v in enumerate(osc):
            if v >= peak_sum * .8:
                sparse_t.append(i)
        tmp = h.Vector()
        if len(sparse_t) > 1:
            tmp.deriv(h.Vector(sparse_t), 1, 1)
            peak = 1 / (tmp.min() / 1000)
        else:
            peak = -1
        return peak

    
    def remake_syn(self):
        if int(self.pc.id() == 0):
            for pair, nc in self.ncdict.iteritems():
                nc.weight[0] = 0.
                self.ncdict.pop(pair)
                print pair
        self.connectcells(self.ncell)


def run_network(network, pc, comm, tstop=600):
    pc.set_maxstep(10)
    h.stdinit()
    pc.psolve(tstop)
    nhost = int(pc.nhost())
    all_events = pc.py_alltoall([network.spike_tvec for _ in range(nhost)]) # list
    all_events = {key : val for dict in all_events for key, val in dict.iteritems()} #collapse list into dict
    network.compute_isi(all_events)
    # Use MPI Gather instead:
    rate_dicts = pc.py_alltoall([network.ratedict for _ in range(nhost)])
    peak_dicts = pc.py_alltoall([network.peakdict for _ in range(nhost)])
    processed_rd = {key: val for dict in rate_dicts for key, val in dict.iteritems()}
    processed_p = {key: val for dict in peak_dicts for key, val in dict.iteritems()}
    # all_dicts = pc.py_alltoall([network.voltage_recvec for i in range(nhost)])
    network.vecdict_to_pydict(network.voltage_recvec, 'rec')
    test = pc.py_alltoall([network.pydicts for i in range(nhost)])
    if int(pc.id()) == 0:
        network.summation(all_events)
        osc_E = network.compute_peak_osc_freq(network.osc_E)
        osc_I = network.compute_peak_osc_freq(network.osc_I)
        rec = {key: val for dict in test for key, val in dict['rec'].iteritems()}
        
        peak_voltage = float("-inf") #print list(rec.keys())
        if network.ncell in list(rec.keys()):
            for i, x in enumerate(rec[network.ncell]):
                if i % 500 == 0: 
                    peak_voltage = max(peak_voltage, x)
            #print li

        E_mean = 0
        I_mean = 0
        I_max = 0
        E_max = 0
        uncounted = 0

        for i in range(network.ncell, network.ncell * 2):
            if i not in processed_rd: 
                uncounted += 1
                continue
            E_mean += processed_rd[i]
            E_max += processed_p[i]
        if network.ncell - uncounted != 0:
            E_mean = E_mean / float(network.ncell - uncounted)
            E_max = E_max / float(network.ncell - uncounted)
        uncounted = 0
        for i in range(network.ncell * 2, network.ncell * 3):
            if i not in processed_rd: 
                uncounted = 0 
                continue
            I_mean += processed_rd[i] 
            I_max += processed_p[i]
        if network.ncell - uncounted != 0:
            I_mean = I_mean / float(network.ncell - uncounted)
            I_max = I_max / float(network.ncell - uncounted)

        # t = {key: value for dict in all_dicts for key, value in dict['t'].iteritems()}
        # rec = {key: value for dict in all_dicts for key, value in dict['rec'].iteritems()}
        return {'E_mean_rate': E_mean, 'E_peak_rate': E_max, 'I_mean_rate': I_mean, "I_peak_rate": I_max, \
                'peak': peak_voltage, 'peak_osc_freq_I': osc_I, 'peak_osc_freq_E': osc_E}


# ==================== cell class                                                                                                                                                                                                                                                                                                                                                        # single cell

class IzhiCell(object):
    # derived from modelDB
    def __init__(self, type='RS'):  # RS = excit or FS = inhib
        self.type = type
        self.sec = h.Section(cell=self)
        self.sec.L, self.sec.diam, self.sec.cm = 10, 10, 31.831
        self.izh = h.Izhi2007b(.5, sec=self.sec)
        self.vinit = -60
        self.sec(0.5).v = self.vinit
        self.sec.insert('pas')

        if type == 'RS': self.izh.a = .1
        if type == 'FS': self.izh.a = .02

        self.synapses()

    def __del__(self):
        # print 'delete ', self
        pass

    # from Ball_Stick
    def synapses(self):
        synlist = []
        s = h.ExpSyn(self.sec(0.8))  # E
        s.tau = 2
        synlist.append(s)
        s = h.ExpSyn(self.sec(0.1))  # I1
        s.tau = 5
        s.e = -80
        synlist.append(s)

        self.synlist = synlist

    # also from Ball Stick
    def connect2target(self, target):
        nc = h.NetCon(self.sec(1)._ref_v, target, sec=self.sec)
        nc.threshold = 10
        return nc

    def is_art(self):
        return 0

class FFCell(object):
    def __init__(self, tstop, mean_freq):
        self.pp = h.VecStim()
        #tstop in ms and mean_rate in s
        n_spikes = tstop * mean_freq / 1000.
        length = int(tstop / n_spikes)
        spikes = [x * float(tstop) / length for x in range(length)]
        vec = h.Vector(spikes)
        #vec = h.Vector([5, 200])
        self.pp.play(vec)

    def connect2target(self, target):
        nc = h.NetCon(self.pp, target)
        return nc

    def is_art(self):
        return 1

