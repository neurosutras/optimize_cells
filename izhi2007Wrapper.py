"""
IZHI

Python wrappers for the different celltypes of Izhikevich neuron. 
Equations and parameter values taken from
  Izhikevich EM (2007). "Dynamical systems in neuroscience", MIT Press
Equation for synaptic inputs taken from
  Izhikevich EM, Edelman GM (2008). "Large-scale model of mammalian thalamocortical systems." PNAS 105(9) 3593-3598.

Cell types available are based on Izhikevich, 2007 book:
    1. RS - Layer 5 regular spiking pyramidal cell (fig 8.12 from 2007 book)
    2. IB - Layer 5 intrinsically bursting cell (fig 8.19 from 2007 book)
    3. CH - Cat primary visual cortex chattering cell (fig8.23 from 2007 book)
    4. LTS - Rat barrel cortex Low-threshold  spiking interneuron (fig8.25 from 2007 book)
    5. FS - Rat visual cortex layer 5 fast-spiking interneuron (fig8.27 from 2007 book)
    6. TC - Cat dorsal LGN thalamocortical (TC) cell (fig8.31 from 2007 book)
    7. RTN - Rat reticular thalamic nucleus (RTN) cell  (fig8.32 from 2007 book)
"""

import collections
from neuron import h

type2007 = collections.OrderedDict([
    #              C    k     vr  vt vpeak   a      b   c    d  celltype
    ('RS', (1.0, 0.7, -60, -40, 35, 0.03, -2, -50, 100, 1)),
    ('IB', (1.5, 1.2, -75, -45, 50, 0.01, 5, -56, 130, 2)),
    ('CH', (0.5, 1.5, -60, -40, 25, 0.03, 1, -40, 150, 3)),
    ('LTS', (1.0, 1.0, -56, -42, 40, 0.03, 8, -53, 20, 4)),
    ('FS', (0.2, 1.0, -55, -40, 25, 0.2, -2, -45, -55, 5)),
    ('TC', (2.0, 1.6, -60, -50, 35, 0.01, 15, -60, 10, 6)),
    ('TC_burst', (2.0, 1.6, -60, -50, 35, 0.01, 15, -60, 10, 6)),
    ('RTN', (0.4, 0.25, -65, -45, 0, 0.015, 10, -55, 50, 7)),
    ('RTN_burst', (0.4, 0.25, -65, -45, 0, 0.015, 10, -55, 50, 7))])


# class of basic Izhikevich neuron based on parameters in type2007
class IzhiCell(object):
    """
    Create an izhikevich cell based on 2007 parameterization using izhi2007bS.mod
    Uses state vars v, u where v is the hoc section voltage.
    Incompatible with IClamp or external synaptic point processes as current sources to the hoc section.
    Note: Capacitance 'C' differs from sec.cm. 'C' varies across cell types and effectively scales the section
    capacitance and, consequently, the membrane time constant.
    """
    def __init__(self, celltype='RS', cellid=None):
        """

        :param type: str
        :param cellid: int
        """
        if cellid is None:
            self.cellid = -1
        else:
            self.cellid = cellid
        if celltype not in type2007.keys():
            raise KeyError('izhi2007 cell type not recognized: %s' % str(celltype))
        self.sec = h.Section(name='izhi2007' + celltype + str(self.cellid))
        self.sec.L, self.sec.diam, self.sec.cm = 10, 10, 31.831  # empirically tuned
        self.izh = h.Izhi2007bS(0.5, sec=self.sec)
        self.reparam(celltype=celltype, cellid=self.cellid)

    def reparam(self, celltype='RS', cellid=None):
        self.celltype = celltype
        self.izh.C, self.izh.k, self.izh.vr, self.izh.vt, self.izh.vpeak, self.izh.a, self.izh.b, self.izh.c, \
        self.izh.d, self.izh.celltype = type2007[celltype]
        if cellid is not None:
            self.cellid = cellid
        self.izh.cellid = self.cellid  # for tracking cell identity
        self.name = '%s%i' % (self.celltype, self.cellid)

