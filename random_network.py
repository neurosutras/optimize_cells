from neuron import h
import random
h.load_file('stdlib.hoc')
h.load_file('nrngui.hoc')
#for h.lambda_f

# adopted from ring_network.py and ring_cell.py
NUM_POP = 3
import numpy as np

"""FF ; I ; E"""


#=================== network class
class Network(object):

  def __init__(self, ncell, delay, pc, dt=None, ff=1., e2e=.05, e2i=.05, i2i=.05, i2e=.05):
    # spiking script uses dt = 0.02
    self.pc = pc
    self.delay = delay
    self.ncell = int(ncell)
    self.prob_dict = {'ff' : ff, 'e2e' : e2e, 'e2i' : e2i, 'i2i' : i2i, 'i2e' : i2e}
    self.index_dict = {'ff' : ((0, ncell), (ncell, ncell * NUM_POP)), # exclusive [x, y)
                       'e2e' : ((ncell, ncell * 2), (ncell, ncell * 2)),
                       'e2i' : ((ncell, ncell * 2), (ncell *2, ncell * NUM_POP)),
                       'i2i' : ((ncell * 2, ncell * NUM_POP), (ncell * 2, ncell * NUM_POP)),
                       'i2e' : ((ncell * 2, ncell * NUM_POP), (ncell, ncell * 2))}
    self.mknetwork(self.ncell)
    self.mkstim(self.ncell)
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
      cell_type = 'FS'
      if i not in list(range(ncell* 2, ncell * 3)):
        cell_type = 'RS'
      cell = IzhiCell(cell_type)
      self.cells.append(cell)
      self.gids.append(i)
      self.pc.set_gid2node(i, rank)
      nc = cell.connect2target(None)
      self.pc.cell(i, nc)
      test = self.pc.gid2cell(i)
      print "mkcell :", type(test)
    # print self.gids

  def createpairs(self, prob, input_indices, output_indices):
    pair_list = []
    for i in input_indices:
      for o in output_indices:
        if random.random() >= prob:
          pair_list.append((i, o))
    for elem in pair_list:
        x, y = elem
        if x == y:
            pair_list.remove(elem)
    return pair_list

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
          if target.type == 'FS':  
            syn = target.synlist[1]
          else:
            syn = target.synlist[0]
          nc = self.pc.gid_connect(presyn_gid, syn)
          nc.delay = self.delay
          nc.weight[0] = 0.8
          self.ncdict[pair] = nc

  # Instrumentation - stimulation and recording
  def mkstim(self, ncell):
    ns = h.NetStim()
    ns.number = 10
    ns.interval = 1
    ns.start = 0
    for i in range(ncell):
      if not self.pc.gid_exists(i): #or random.random() >= .3:  # stimulate only 30% of FF
        continue
      nc = h.NetCon(ns, self.pc.gid2cell(i).synlist[0], 0, 1, 10)
      nc.delay = 0
      nc.weight[0] = 2
      self.ncdict[('stim', i)] = nc

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

  def voltage_record(self, dt=None):
    self.voltage_tvec = {}
    self.voltage_recvec = {}
    if dt is None:
      self.dt = h.dt
    else:
      self.dt = dt
    for i, cell in enumerate(self.cells):
      tvec = h.Vector()
      tvec.record(h._ref_t)  # dt is not accepted as an argument to this function in the PC environment -- may need to turn on cvode?
      rec = h.Vector()
      rec.record(getattr(cell.sec(.5), '_ref_v'))  # dt is not accepted as an argument
      self.voltage_tvec[self.gids[i]] = tvec
      self.voltage_recvec[self.gids[i]] = rec

  def vecdict_to_pydict(self, vecdict, name):
    self.pydicts[name] = {}
    for key, value in vecdict.iteritems():
      self.pydicts[name][key] = value.to_python()
      #Alternatively, could use nc.record(tvec)

  def compute_isi(self, vecdict):
      self.ratedict = {}
      self.peakdict = {}
      for key, vec in vecdict.iteritems():
        li = []
        for i, x in enumerate(vecdict[key]):
            if i % 5000 == 0: li.append(x)
        if key == 0: print li
        isivec = h.Vector()
        isivec.deriv(vec, 1, 1)
        rate = 1. / (isivec.mean() * 1000)
        self.ratedict[key] = rate
        self.peakdict[key] = 1. / (isivec.min() * 1000)

  def remake_syn(self):
    if int(self.pc.id() == 0):
        for pair, nc in self.ncdict.iteritems():
            nc.weight[0] = 0.
            self.ncdict.pop(pair)
            print pair
    self.connectcells(self.ncell)

def run_network(network, pc, comm, tstop=3000):
  pc.set_maxstep(10)
  h.stdinit()
  pc.psolve(tstop)
  nhost = int(pc.nhost())
  network.compute_isi(network.voltage_recvec)
  #Use MPI Gather instead:
  rate_dicts = pc.py_alltoall([network.ratedict for i in range(nhost)])
  peak_dicts = pc.py_alltoall([network.peakdict for i in range(nhost)])
  processed_rd = {key : val for dict in rate_dicts for key, val in dict.iteritems()}
  processed_p = {key : val for dict in peak_dicts for key, val in dict.iteritems()}
  #all_dicts = pc.py_alltoall([network.voltage_recvec for i in range(nhost)])
  network.vecdict_to_pydict(network.voltage_recvec, 'rec')
  test = pc.py_alltoall([network.pydicts for i in range(nhost)])
  if int(pc.id()) == 0:
    rec = {key : val for dict in test for key, val in dict['rec'].iteritems()}
    #print "pydict", rec[0]
    E_mean = 0; I_mean = 0; I_max = 0; E_max = 0
    for i in range(network.ncell, network.ncell * 2):
      E_mean += processed_rd[i] / float(network.ncell)
      E_max += processed_p[i] / float(network.ncell)
    for i in range(network.ncell * 2, network.ncell * 3):
      I_mean += processed_rd[i] / float(network.ncell)
      I_max += processed_p[i] / float(network.ncell)

    #t = {key: value for dict in all_dicts for key, value in dict['t'].iteritems()}
    #rec = {key: value for dict in all_dicts for key, value in dict['rec'].iteritems()}
    return {'E_mean_rate': E_mean, 'E_peak_rate' : E_max,'I_mean_rate': I_mean, "I_peak_rate" : I_max}
  return None

#==================== cell class                                                                                                                                                                                                                                                                                                                                                        # single cell

class IzhiCell(object):
    # derived from modelDB
    def __init__(self, type='RS'): #RS = excit or FS = inhib
        self.type=type
        self.sec = h.Section()
        self.sec.L, self.sec.diam, self.sec.cm = 10, 10, 31.831
        self.izh = h.Izhi2007b(.5, sec=self.sec)
        self.vinit= -60
        self.sec(0.5).v = self.vinit
        self.sec.insert('pas')

        if type == 'RS' : self.izh.a = .1
        if type=='FS' : self.izh.a = .02

        self.synapses()

    def __del__(self):
      # print 'delete ', self
      pass

    # from Ball_Stick
    def synapses(self):
      synlist = []
      s = h.ExpSyn(self.sec(0.8))  # E0
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
