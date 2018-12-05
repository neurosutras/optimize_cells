from neuron import h
import random
from random_network import BallStick

h.load_file('stdlib.hoc')
h.load_file('nrngui.hoc')

#cell = BallStick(True)
"""from the modelDB"""
class IzhiCell(): 
    def __init__(self, type='RS', host=None, cellid=-1):
        self.type=type
        if host is None:
            self.sec = h.Section(name='izhi2007'+type+str(cellid))
            self.sec.L, self.sec.diam, self.sec.cm = 10, 10, 31.831
            self.izh = h.Izhi2007b(.5, sec=self.sec)
            self.vinit= -60
        else:
            self.sec= host
            self.izh = h.Izhi2007a(.5, sec=host)
        self.izh.C,self.izh.k,self.izh.vr,self.izh.vt,self.izh.vpeak,self.izh.a,self.izh.b,self.izh.c,self.izh.d,self.izh.celltype = 1.0, 0.7,  -60, -40, 35, 0.03,   -2, -50,  100,  1
        self.izh.cellid = cellid
    def init (self): self.sec(0.5).v = self.vinit

cell = IzhiCell()
izh = cell.izh
ns = h.NetStim()
nc = h.NetCon(ns,izh,0,1,10)
ns.start, ns.interval, ns.number = 10, 10, 10
nc.weight[0] = 2

vloc = cell.sec(0.5)._ref_v
recvec = h.Vector()
recvec.record(vloc)

h.tstop=500; h.cvode.active(0); h.dt=.01
h.run()

for i, v in enumerate(recvec):
    if i % 5000:
        print v

"""def izhstim () :
  stim=h.NetStim(0.5)
  stim.number = stim.start = 1
  nc = h.NetCon(stim,cell.syn)
  nc.delay = 2
  nc.weight = 0.1
  izh.erev = -5"""
