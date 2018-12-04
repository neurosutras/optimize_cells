: M conductance

NEURON {
	SUFFIX DGC_KM
	USEION k READ ek WRITE ik
	RANGE gbar, minf, tau1, tau2, i, g, m1, m2, ginf
	RANGE tadjtau, Vhalf, Vshift, k, v0erev, kV, gamma
	RANGE Dtaumult1, Dtaumult2, tau0mult, taudiv
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(S) = (siemens)
	(um) = (micron)
} 

PARAMETER {
	gbar = 0.001  	    				(S/cm2)
	k = 9           				(mV)
	Vhalf = -50             (mV)  :for minf(V)
	Vshift = 0              (mV)	:for g(V) and minf(V)     
	v0erev = 65             (mV)     :50-80
	kV = 40                 (mV)     
	gamma = 0.5                      :0.5,1

	temptau = 22	          (degC) :tau reference temperature 	
	q10tau  = 5
	taudiv = 1
	Dtaumult1 = 30 		: Mateos-Aparecio (2014) set this default value in hoc
	Dtaumult2 = 30		: Mateos-Aparecio (2014) set this default value in hoc
	tau0mult = 1

	vmin = -100	            (mV)
	vmax = 100	            (mV)
	ten = 10		            (degC)
	temp0 = 273		          (degC)
	FoverR = 11.6045039552	(degC/mV)
} 
 
ASSIGNED {
	v 	     	(mV)
	ek		(mV)
	celsius		(degC)
	ginf			(S/cm2)
	Vhalf1    (mV) 
	Dtau1     (ms)
	z1               
	tau01   	(ms)	 
	Vhalf2  	(mV)	  
	Dtau2   	(ms)  
	z2               
	tau02   	(ms)	  
	alpha1				  
	beta1	  		  
	alpha2		
	beta2	
	i 	    	(mA/cm2)
	ik 	     	(mA/cm2)
	g		      (S/cm2)
	minf
	v0        (mV)      
	tau1			(ms)
	tau2			(ms)
	tadjtau
	frt		    (/mV)
}
 
STATE { m1 m2 }

INITIAL { 
	rates(v)
	m1 = minf
	m2 = minf
}

BREAKPOINT {
  SOLVE states METHOD cnexp
	g = gbar*gsat(v)*(m1^2)*m2
	ik = g*(v - ek)
	i = ik
} 

DERIVATIVE states {
	rates(v)
	m1' = (minf - m1)/tau1
	m2' = (minf - m2)/tau2
}

PROCEDURE rates(v (mV)) {
  IF (gamma == 0.5) {
  	z1 = 2.8
		Vhalf1 = -49.8+Vshift 	:(mV)  shifted - 20 mV (when Vshift = 0)
		tau01 = 20.7*tau0mult	  :(ms)
		Dtau1 = 176.1*Dtaumult1	:(ms)
		z2 = 8.9	              
		Vhalf2 = -55.5+Vshift 					:(mV)  shifted - 20 mV
		tau02 = 149*tau0mult   					:(ms)
		Dtau2 = 1473*Dtaumult2 	  			:(ms)
	}	
	IF (gamma == 1) {
  	z1 = 3.6
		Vhalf1 = -25.3+Vshift		:(mV)  shifted - 20 mV
		tau01 = 29.2*tau0mult	  :(ms)
		Dtau1 = 74.6*Dtaumult1	:(ms)
		z2 = 9.8	
		Vhalf2 = -44.7+Vshift 					:(mV)  shifted - 20 mV
		tau02 = 155*tau0mult   					:(ms)
		Dtau2 = 549*Dtaumult2  	  			:(ms)
	}
  tadjtau = q10tau^((celsius - temptau)/ten)
	frt = FoverR/(temp0 + celsius)

  alpha1 = exptrap(1,z1*gamma*frt*(v - Vhalf1))
  beta1 = exptrap(2,-z1*(1-gamma)*frt*(v - Vhalf1))
  tau1 = (Dtau1/(alpha1 + beta1) + tau01)/(tadjtau*taudiv)
  
  alpha2 = exptrap(3,z2*gamma*frt*(v - Vhalf2))
  beta2 = exptrap(4,-z2*(1-gamma)*frt*(v - Vhalf2))
  tau2 = (Dtau2/(alpha2 + beta2) + tau02)/(tadjtau*taudiv)

  minf = 1/(1 + exptrap(5,-(v - Vhalf - Vshift)/k))
  ginf = gbar*minf^3
}

FUNCTION gsat (v (mV)) {
	gsat = 1
	v0 = v0erev + ek
	IF (v > v0) {
		gsat = 1+(v0-v+kV*(1-exptrap(0,-(v-v0)/kV)))/(v-ek)
	}
}

FUNCTION exptrap(loc,x) {
  if (x>=700.0) {
    :printf("exptrap DGC_M [%d]: x = %g\n", loc, x)
    exptrap = exp(700.0)
  } else {
    exptrap = exp(x)
  }
}

