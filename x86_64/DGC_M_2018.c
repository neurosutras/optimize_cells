/* Created by Language version: 7.5.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "scoplib_ansi.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__DGC_KM
#define _nrn_initial _nrn_initial__DGC_KM
#define nrn_cur _nrn_cur__DGC_KM
#define _nrn_current _nrn_current__DGC_KM
#define nrn_jacob _nrn_jacob__DGC_KM
#define nrn_state _nrn_state__DGC_KM
#define _net_receive _net_receive__DGC_KM 
#define rates rates__DGC_KM 
#define states states__DGC_KM 
 
#define _threadargscomma_ _p, _ppvar, _thread, _nt,
#define _threadargsprotocomma_ double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt,
#define _threadargs_ _p, _ppvar, _thread, _nt
#define _threadargsproto_ double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 /* Thread safe. No static _p or _ppvar. */
 
#define t _nt->_t
#define dt _nt->_dt
#define gbar _p[0]
#define k _p[1]
#define Vhalf _p[2]
#define Vshift _p[3]
#define v0erev _p[4]
#define kV _p[5]
#define gamma _p[6]
#define taudiv _p[7]
#define Dtaumult1 _p[8]
#define Dtaumult2 _p[9]
#define tau0mult _p[10]
#define ginf _p[11]
#define i _p[12]
#define g _p[13]
#define minf _p[14]
#define tau1 _p[15]
#define tau2 _p[16]
#define tadjtau _p[17]
#define m1 _p[18]
#define m2 _p[19]
#define ek _p[20]
#define Vhalf1 _p[21]
#define Dtau1 _p[22]
#define z1 _p[23]
#define tau01 _p[24]
#define Vhalf2 _p[25]
#define Dtau2 _p[26]
#define z2 _p[27]
#define tau02 _p[28]
#define alpha1 _p[29]
#define beta1 _p[30]
#define alpha2 _p[31]
#define beta2 _p[32]
#define ik _p[33]
#define v0 _p[34]
#define frt _p[35]
#define Dm1 _p[36]
#define Dm2 _p[37]
#define v _p[38]
#define _g _p[39]
#define _ion_ek	*_ppvar[0]._pval
#define _ion_ik	*_ppvar[1]._pval
#define _ion_dikdv	*_ppvar[2]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  -1;
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 extern double celsius;
 /* declaration of user functions */
 static void _hoc_exptrap(void);
 static void _hoc_gsat(void);
 static void _hoc_rates(void);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _extcall_prop = _prop;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_DGC_KM", _hoc_setdata,
 "exptrap_DGC_KM", _hoc_exptrap,
 "gsat_DGC_KM", _hoc_gsat,
 "rates_DGC_KM", _hoc_rates,
 0, 0
};
#define exptrap exptrap_DGC_KM
#define gsat gsat_DGC_KM
 extern double exptrap( _threadargsprotocomma_ double , double );
 extern double gsat( _threadargsprotocomma_ double );
 /* declare global and static user variables */
#define FoverR FoverR_DGC_KM
 double FoverR = 11.6045;
#define q10tau q10tau_DGC_KM
 double q10tau = 5;
#define temp0 temp0_DGC_KM
 double temp0 = 273;
#define ten ten_DGC_KM
 double ten = 10;
#define temptau temptau_DGC_KM
 double temptau = 22;
#define vmax vmax_DGC_KM
 double vmax = 100;
#define vmin vmin_DGC_KM
 double vmin = -100;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "temptau_DGC_KM", "degC",
 "vmin_DGC_KM", "mV",
 "vmax_DGC_KM", "mV",
 "ten_DGC_KM", "degC",
 "temp0_DGC_KM", "degC",
 "FoverR_DGC_KM", "degC/mV",
 "gbar_DGC_KM", "S/cm2",
 "k_DGC_KM", "mV",
 "Vhalf_DGC_KM", "mV",
 "Vshift_DGC_KM", "mV",
 "v0erev_DGC_KM", "mV",
 "kV_DGC_KM", "mV",
 "ginf_DGC_KM", "S/cm2",
 "i_DGC_KM", "mA/cm2",
 "g_DGC_KM", "S/cm2",
 "tau1_DGC_KM", "ms",
 "tau2_DGC_KM", "ms",
 0,0
};
 static double delta_t = 0.01;
 static double m20 = 0;
 static double m10 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "temptau_DGC_KM", &temptau_DGC_KM,
 "q10tau_DGC_KM", &q10tau_DGC_KM,
 "vmin_DGC_KM", &vmin_DGC_KM,
 "vmax_DGC_KM", &vmax_DGC_KM,
 "ten_DGC_KM", &ten_DGC_KM,
 "temp0_DGC_KM", &temp0_DGC_KM,
 "FoverR_DGC_KM", &FoverR_DGC_KM,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(_NrnThread*, _Memb_list*, int);
static void nrn_state(_NrnThread*, _Memb_list*, int);
 static void nrn_cur(_NrnThread*, _Memb_list*, int);
static void  nrn_jacob(_NrnThread*, _Memb_list*, int);
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(_NrnThread*, _Memb_list*, int);
static void _ode_matsol(_NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[3]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.5.0",
"DGC_KM",
 "gbar_DGC_KM",
 "k_DGC_KM",
 "Vhalf_DGC_KM",
 "Vshift_DGC_KM",
 "v0erev_DGC_KM",
 "kV_DGC_KM",
 "gamma_DGC_KM",
 "taudiv_DGC_KM",
 "Dtaumult1_DGC_KM",
 "Dtaumult2_DGC_KM",
 "tau0mult_DGC_KM",
 0,
 "ginf_DGC_KM",
 "i_DGC_KM",
 "g_DGC_KM",
 "minf_DGC_KM",
 "tau1_DGC_KM",
 "tau2_DGC_KM",
 "tadjtau_DGC_KM",
 0,
 "m1_DGC_KM",
 "m2_DGC_KM",
 0,
 0};
 static Symbol* _k_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 40, _prop);
 	/*initialize range parameters*/
 	gbar = 0.001;
 	k = 9;
 	Vhalf = -50;
 	Vshift = 0;
 	v0erev = 65;
 	kV = 40;
 	gamma = 0.5;
 	taudiv = 1;
 	Dtaumult1 = 30;
 	Dtaumult2 = 30;
 	tau0mult = 1;
 	_prop->param = _p;
 	_prop->param_size = 40;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 4, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_k_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0]._pval = &prop_ion->param[0]; /* ek */
 	_ppvar[1]._pval = &prop_ion->param[3]; /* ik */
 	_ppvar[2]._pval = &prop_ion->param[4]; /* _ion_dikdv */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, _NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _DGC_M_2018_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("k", -10000.);
 	_k_sym = hoc_lookup("k_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
  hoc_register_prop_size(_mechtype, 40, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 DGC_KM /Users/wyin/optimize_cells/x86_64/DGC_M_2018.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int rates(_threadargsprotocomma_ double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[2], _dlist1[2];
 static int states(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {int _reset = 0; {
   rates ( _threadargscomma_ v ) ;
   Dm1 = ( minf - m1 ) / tau1 ;
   Dm2 = ( minf - m2 ) / tau2 ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {
 rates ( _threadargscomma_ v ) ;
 Dm1 = Dm1  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tau1 )) ;
 Dm2 = Dm2  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tau2 )) ;
  return 0;
}
 /*END CVODE*/
 static int states (double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) { {
   rates ( _threadargscomma_ v ) ;
    m1 = m1 + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tau1)))*(- ( ( ( minf ) ) / tau1 ) / ( ( ( ( - 1.0 ) ) ) / tau1 ) - m1) ;
    m2 = m2 + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tau2)))*(- ( ( ( minf ) ) / tau2 ) / ( ( ( ( - 1.0 ) ) ) / tau2 ) - m2) ;
   }
  return 0;
}
 
static int  rates ( _threadargsprotocomma_ double _lv ) {
   if ( gamma  == 0.5 ) {
     z1 = 2.8 ;
     Vhalf1 = - 49.8 + Vshift ;
     tau01 = 20.7 * tau0mult ;
     Dtau1 = 176.1 * Dtaumult1 ;
     z2 = 8.9 ;
     Vhalf2 = - 55.5 + Vshift ;
     tau02 = 149.0 * tau0mult ;
     Dtau2 = 1473.0 * Dtaumult2 ;
     }
   if ( gamma  == 1.0 ) {
     z1 = 3.6 ;
     Vhalf1 = - 25.3 + Vshift ;
     tau01 = 29.2 * tau0mult ;
     Dtau1 = 74.6 * Dtaumult1 ;
     z2 = 9.8 ;
     Vhalf2 = - 44.7 + Vshift ;
     tau02 = 155.0 * tau0mult ;
     Dtau2 = 549.0 * Dtaumult2 ;
     }
   tadjtau = pow( q10tau , ( ( celsius - temptau ) / ten ) ) ;
   frt = FoverR / ( temp0 + celsius ) ;
   alpha1 = exptrap ( _threadargscomma_ 1.0 , z1 * gamma * frt * ( _lv - Vhalf1 ) ) ;
   beta1 = exptrap ( _threadargscomma_ 2.0 , - z1 * ( 1.0 - gamma ) * frt * ( _lv - Vhalf1 ) ) ;
   tau1 = ( Dtau1 / ( alpha1 + beta1 ) + tau01 ) / ( tadjtau * taudiv ) ;
   alpha2 = exptrap ( _threadargscomma_ 3.0 , z2 * gamma * frt * ( _lv - Vhalf2 ) ) ;
   beta2 = exptrap ( _threadargscomma_ 4.0 , - z2 * ( 1.0 - gamma ) * frt * ( _lv - Vhalf2 ) ) ;
   tau2 = ( Dtau2 / ( alpha2 + beta2 ) + tau02 ) / ( tadjtau * taudiv ) ;
   minf = 1.0 / ( 1.0 + exptrap ( _threadargscomma_ 5.0 , - ( _lv - Vhalf - Vshift ) / k ) ) ;
   ginf = gbar * pow( minf , 3.0 ) ;
    return 0; }
 
static void _hoc_rates(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r = 1.;
 rates ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double gsat ( _threadargsprotocomma_ double _lv ) {
   double _lgsat;
 _lgsat = 1.0 ;
   v0 = v0erev + ek ;
   if ( _lv > v0 ) {
     _lgsat = 1.0 + ( v0 - _lv + kV * ( 1.0 - exptrap ( _threadargscomma_ 0.0 , - ( _lv - v0 ) / kV ) ) ) / ( _lv - ek ) ;
     }
   
return _lgsat;
 }
 
static void _hoc_gsat(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  gsat ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double exptrap ( _threadargsprotocomma_ double _lloc , double _lx ) {
   double _lexptrap;
 if ( _lx >= 700.0 ) {
     _lexptrap = exp ( 700.0 ) ;
     }
   else {
     _lexptrap = exp ( _lx ) ;
     }
   
return _lexptrap;
 }
 
static void _hoc_exptrap(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  exptrap ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ return 2;}
 
static void _ode_spec(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ek = _ion_ek;
     _ode_spec1 (_p, _ppvar, _thread, _nt);
  }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 2; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 (_p, _ppvar, _thread, _nt);
 }
 
static void _ode_matsol(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ek = _ion_ek;
 _ode_matsol_instance1(_threadargs_);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_k_sym, _ppvar, 0, 0);
   nrn_update_ion_pointer(_k_sym, _ppvar, 1, 3);
   nrn_update_ion_pointer(_k_sym, _ppvar, 2, 4);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {
  int _i; double _save;{
  m2 = m20;
  m1 = m10;
 {
   rates ( _threadargscomma_ v ) ;
   m1 = minf ;
   m2 = minf ;
   }
 
}
}

static void nrn_init(_NrnThread* _nt, _Memb_list* _ml, int _type){
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
  ek = _ion_ek;
 initmodel(_p, _ppvar, _thread, _nt);
 }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   g = gbar * gsat ( _threadargscomma_ v ) * ( pow( m1 , 2.0 ) ) * m2 ;
   ik = g * ( v - ek ) ;
   i = ik ;
   }
 _current += ik;

} return _current;
}

static void nrn_cur(_NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
  ek = _ion_ek;
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ double _dik;
  _dik = ik;
 _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
  _ion_dikdv += (_dik - ik)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ik += ik ;
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}
 
}

static void nrn_jacob(_NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}
 
}

static void nrn_state(_NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
  ek = _ion_ek;
 {   states(_p, _ppvar, _thread, _nt);
  } }}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = &(m1) - _p;  _dlist1[0] = &(Dm1) - _p;
 _slist1[1] = &(m2) - _p;  _dlist1[1] = &(Dm2) - _p;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif
