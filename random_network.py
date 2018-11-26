from neuron import h
import random
h.load_file('stdlib.hoc')
h.load_file('nrngui.hoc')
#for h.lambda_f

# adopted from ring_network.py and ring_cell.py
NUM_POP = 3

#=================== network class
class Network(object):

  def __init__(self, ncell, delay, pc, dt=None, ff=.2, e2e=.05, e2i=.05, i2i=.05, i2e=.05):
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
    if int(self.pc.id() == 0):
        print "fin"

  def mkcells(self, ncell):
    rank = int(self.pc.id())
    nhost = int(self.pc.nhost())
    self.cells = []
    self.gids = []
    for i in range(rank, ncell * NUM_POP, nhost):
      excitatory = False
      if i not in list(range(ncell* 2, ncell * 3)):
        excitatory = True
      cell = BallStick(excitatory)
      self.cells.append(cell)
      self.gids.append(i)
      self.pc.set_gid2node(i, rank)
      nc = cell.connect2target(None)
      self.pc.cell(i, nc)
    # print self.gids

  def createpairs(self, prob, input_indices, output_indices):
    pair_list = []
    for i in input_indices:
      for o in output_indices:
        if random.random() >= prob:
          pair_list.append((i, o))
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
      pair_list = self.createpairs(self.prob_dict[connection], list(range(inp[0], inp[1])), list(range(out[0], out[1])))
      for pair in pair_list:
        presyn_gid = pair[0]
        target_gid = pair[1]
        if self.pc.gid_exists(target_gid):
          target = self.pc.gid2cell(target_gid)
          syn = target.syn
          nc = self.pc.gid_connect(presyn_gid, syn)
          nc.delay = self.delay
          nc.weight[0] = 0.01
          self.ncdict.update({pair: nc})
    # print self.ncdict

  # Instrumentation - stimulation and recording
  def mkstim(self, ncell):
    if not self.pc.gid_exists(0):
      return
    self.stim = h.NetStim()
    self.stim.number = 1
    self.stim.start = 0
    for i in range(ncell):
      if random.random() >= .3:  # stimulate only 30% of FF
        continue
      nc = h.NetCon(self.stim, self.pc.gid2cell(i).syn)
      nc.delay = 0
      nc.weight[0] = 0.01

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
      rec.record(getattr(cell.soma(0), '_ref_v'))  # dt is not accepted as an argument
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
          isivec = h.Vector()
          isivec.deriv(vec, 1, 1)
          rate = 1. / (isivec.mean() * 1000)
          self.ratedict[key] = rate
          histvec = vec.histogram(0, 100, 20)  #tstop
          self.peakdict[key] = histvec.max() / float(20 * 1000)


def run_network(network, pc, comm, tstop=100):
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
  if int(pc.id()) == 0:
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
class BallStick(object):
  def __init__(self, excitatory = True):
    self.topol()
    self.subsets()
    self.geom()
    self.biophys()
    self.geom_nseg()
    self.syn = None
    self.excitatory = excitatory
    self.synapses()

  def __del__(self):
    #print 'delete ', self
    pass

  def topol(self):
    self.soma = h.Section(name='soma', cell=self)
    self.dend = h.Section(name='dend', cell= self)
    self.dend.connect(self.soma(1))
    self.basic_shape()

  def basic_shape(self):
    self.soma.push()
    h.pt3dclear()
    h.pt3dadd(0, 0, 0, 1)
    h.pt3dadd(15, 0, 0, 1)
    h.pop_section()
    self.dend.push()
    h.pt3dclear()
    h.pt3dadd(15, 0, 0, 1)
    h.pt3dadd(105, 0, 0, 1)
    h.pop_section()

  def subsets(self):
    self.all = h.SectionList()
    self.all.append(sec=self.soma)
    self.all.append(sec=self.dend)

  def geom(self):
    self.soma.L = self.soma.diam = 12.6157
    self.dend.L = 200
    self.dend.diam = 1

  def geom_nseg(self):
    for sec in self.all:
      sec.nseg = int((sec.L/(0.1*h.lambda_f(100)) + .9)/2.)*2 + 1

  def biophys(self):
    for sec in self.all:
      sec.Ra = 100
      sec.cm = 1
    self.soma.insert('hh')
    self.soma.gnabar_hh = 0.12
    self.soma.gkbar_hh = 0.036
    self.soma.gl_hh = 0.0003
    self.soma.el_hh = -54.3

    self.dend.insert('pas')
    self.dend.g_pas = 0.001
    self.dend.e_pas = -65

  def connect2target(self, target):
    nc = h.NetCon(self.soma(1)._ref_v, target, sec = self.soma)
    nc.threshold = 10
    return nc

  def synapses(self):
    if self.excitatory: #RS
        s = h.Izhi2003b(self.dend(0.8))
        s.a = .1
        self.syn = s
    else: #FS
        s = h.Izhi2003b(self.dend(0.8))
        s.a = .02
        self.syn = s
