from dentate.biophysics_utils import *


class IzhiCell(): 
    def __init__(self, type='RS', host=None, cellid=-1):
        self.type=type
        self.name = 'izhi2007' + type + str(cellid)
        if host is None:
            self.sec = h.Section(self.name)
            self.sec.L, self.sec.diam, self.sec.cm = 10, 10, 31.831
            self.izh = h.Izhi2007b(.5, sec=self.sec)
            self.vinit= -60
        else:
            self.sec= host
            self.izh = h.Izhi2007a(.5, sec=host)
        self.izh.C, self.izh.k, self.izh.vr, self.izh.vt, self.izh.vpeak, self.izh.a, self.izh.b, self.izh.c, \
        self.izh.d, self.izh.celltype = 1.0, 0.7,  -60, -40, 35, 0.03,   -2, -50,  100,  1
        self.izh.cellid = cellid

    def init (self):
        self.sec(0.5).v = self.vinit

cell = IzhiCell()
cell.init()

sim = QuickSim()
sim.append_rec(cell=cell, node=cell, loc=0.5, name='soma')

syn = h.EPSC(cell.sec(0.5))
syn.i_unit = -0.1

ns = h.NetStim()
nc = h.NetCon(ns,syn)
ns.start, ns.interval, ns.number = 10, 10, 10
nc.weight[0] = 1.

sim.append_rec(cell, cell, loc=0.5, object=syn, param='_ref_i', name='EPSC')

#sim.append_stim(cell, cell, name='step', loc=0.5, amp=0.1, delay=10., dur=100.)

sim.run(v_init=cell.vinit)
sim.plot()

"""def izhstim () :
  stim=h.NetStim(0.5)
  stim.number = stim.start = 1
  nc = h.NetCon(stim,cell.syn)
  nc.delay = 2
  nc.weight = 0.1
  izh.erev = -5"""
