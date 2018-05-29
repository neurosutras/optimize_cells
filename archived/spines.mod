COMMENT
Introduces a spine area scale factor as a range variable.
This doesn't do magic: it will NOT automatically correct
for spines. You will still have to multiply surface-dependent
variables (capacitance, conductances) with scale_spines.
Copied from 
http://www.northwestern.edu/neurobiology/faculty/spruston/sk_models/JP_2005/Attenuation.zip
last retrieved 02-2007
Golding NL, Mickus TJ, Katz Y, Kath WL, Spruston N (2005)
Factors mediating powerful voltage attenuation along CA1 
pyramidal neuron dendrites. J Physiol (Lond) 568:69-82.

2007-05-15, CSH, University of Freiburg 
ENDCOMMENT

NEURON 
{
    SUFFIX spines
    RANGE scale, count
}

PARAMETER 
{
    scale = 1
    count = 0
}

