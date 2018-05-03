: gsyn.mod

NEURON {
  POINT_PROCESS GSyn
  RANGE tau1, tau2, e, i
  RANGE Gtau1, Gtau2, Ginc
  NONSPECIFIC_CURRENT i
  RANGE g
}

UNITS {
  (nA) = (nanoamp)
  (mV) = (millivolt)
  (umho) = (micromho)
}

PARAMETER {
  tau1    =   1   (ms)
  tau2    =   1.05    (ms)
  Gtau1   = 20 (ms)
  Gtau2   = 21 (ms)
  Ginc    = 1
  e       = 0 (mV)
}

ASSIGNED {
  v  (mV)
  i  (nA)
  g  (umho)
  factor
  Gfactor
}

STATE {
  A  (umho)
  B  (umho)
}

INITIAL {
  LOCAL tp
  A=0
  B=0
  tp = (tau1*tau2)/(tau2 - tau1) * log(tau2/tau1)
  factor = -exp(-tp/tau1) + exp(-tp/tau2)
  factor = 1/factor
  tp = (Gtau1*Gtau2)/(Gtau2 - Gtau1) * log(Gtau2/Gtau1)
  Gfactor = -exp(-tp/Gtau1) + exp(-tp/Gtau2)
  Gfactor = 1/Gfactor
}

BREAKPOINT {
SOLVE state METHOD cnexp
  g=B-A
  i = g*(v - e)
}

DERIVATIVE state {
  A' = -A/tau1
  B' = -B/tau2
}

NET_RECEIVE(weight (umho), w, G1, G2, t0 (ms)) {
  G1 = G1*exp(-(t-t0)/Gtau1)
  G2 = G2*exp(-(t-t0)/Gtau2)
  G1 = G1 + Ginc*Gfactor
  G2 = G2 + Ginc*Gfactor
  t0 = t
  w = weight*(1 + G2 - G1)
  A = A + w*factor
  B = B + w*factor
}
