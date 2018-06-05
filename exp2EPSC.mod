COMMENT

Adapted exp2syn to generate a current rather than a conductance (Milstein 2015):

--------------------------------------------------------------------------------

Two state kinetic scheme synapse described by rise time tau_rise,
and decay time constant tau_decay. The normalized peak current is 1.
Decay time MUST be greater than rise time.

The solution of A->G->bath with rate constants 1/tau_rise and 1/tau_decay is
 A = a*exp(-t/tau_rise) and
 G = a*tau_decay/(tau_decay-tau_rise)*(-exp(-t/tau_rise) + exp(-t/tau_decay))
	where tau_rise < tau_decay

If tau_decay-tau_rise -> 0 then we have a alphasynapse.
and if tau_rise -> 0 then we have just single exponential decay.

The factor is evaluated in the
initial block such that an event of weight 1 generates a
peak conductance of 1.

Because the solution is a sum of exponentials, the
coupled equations can be solved as a pair of independent equations
by the more efficient cnexp method.

ENDCOMMENT

NEURON {
	POINT_PROCESS EPSC
	RANGE tau_rise, tau_decay, i, i_unit
	NONSPECIFIC_CURRENT i
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
}

PARAMETER {
	tau_rise = 0.2   (ms)    <1e-9,1e9>
	tau_decay = 5.0    (ms)   <1e-9,1e9>
    i_unit = -1.0           <0,1>        : to scale amplitude
}

ASSIGNED {
	v (mV)
	i (nA)
	factor
}

STATE {
	A (nA)
	B (nA)
}

INITIAL {
	LOCAL tp
	if (tau_rise/tau_decay > .9999) {
		tau_rise = .9999*tau_decay
	}
	A = 0
	B = 0
	tp = (tau_rise*tau_decay)/(tau_decay - tau_rise) * log(tau_decay/tau_rise)
	factor = -exp(-tp/tau_rise) + exp(-tp/tau_decay)
	factor = 1/factor
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	i = i_unit * (B - A)
}

DERIVATIVE state {
	A' = -A/tau_rise
	B' = -B/tau_decay
}

NET_RECEIVE(weight (nA)) {
	A = A + weight*factor
    B = B + weight*factor
}
