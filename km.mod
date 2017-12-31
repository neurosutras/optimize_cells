: K-Strom, I_M, Warman 94
: T-dependence. from Halliwell, Adams 82

UNITS 
{
    (mA) = (milliamp)
    (mV) = (millivolt)
	(S) = (siemens)
}
 
NEURON {
    SUFFIX km
	USEION k READ ek WRITE ik
    RANGE gmbar, gm, ik, timesTau, plusTau
    RANGE uinf, utau
}
 
PARAMETER 
{
    gmbar = 0.00034 (S/cm2)	<0,1e9>
	timesTau=1
	plusTau=0
}

STATE 
{
    u
}
 
ASSIGNED 
{
    v (mV)
    celsius (degC)
	gm (S/cm2)
    ik (mA/cm2)
    ek (mV)
    uinf
	utau (ms)
}

INITIAL
{
	rates(v)
	u = uinf
}

BREAKPOINT
{
    SOLVE states METHOD cnexp
    gm = gmbar*u*u
	ik = gm*(v - ek)
}

DERIVATIVE states 
{  
    rates(v)
    u' =  (uinf-u)/utau
}
 
PROCEDURE rates(v(mV))   :Computes rate and other constants at current v.
{
    LOCAL  alpha, beta, sum, q10

    UNITSOFF
    : see Warman
    q10 = 5^((celsius - 23)/10)
    :"u" potassium activation system
    alpha = 0.016 / exp((v+52.7)/-23)
    beta =  0.016 / exp((v+52.7)/18.8)
    sum = alpha + beta
	
    uinf = alpha/sum
	utau = timesTau/(sum*q10)+plusTau
    UNITSON
}
