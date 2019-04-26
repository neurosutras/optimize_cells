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
 
#define nrn_init _nrn_init__CadepK
#define _nrn_initial _nrn_initial__CadepK
#define nrn_cur _nrn_cur__CadepK
#define _nrn_current _nrn_current__CadepK
#define nrn_jacob _nrn_jacob__CadepK
#define nrn_state _nrn_state__CadepK
#define _net_receive _net_receive__CadepK 
#define state state__CadepK 
 
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
#define gbkbar _p[0]
#define gskbar _p[1]
#define gcakmult _p[2]
#define ik _p[3]
#define isk _p[4]
#define ibk _p[5]
#define gbk _p[6]
#define gsk _p[7]
#define q _p[8]
#define r _p[9]
#define s _p[10]
#define ek _p[11]
#define cai _p[12]
#define gbar _p[13]
#define Dq _p[14]
#define Dr _p[15]
#define Ds _p[16]
#define v _p[17]
#define _g _p[18]
#define _ion_cai	*_ppvar[0]._pval
#define _ion_ek	*_ppvar[1]._pval
#define _ion_ik	*_ppvar[2]._pval
#define _ion_dikdv	*_ppvar[3]._pval
 
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
 /* declaration of user functions */
 static void _hoc_alphaq(void);
 static void _hoc_betar(void);
 static void _hoc_betaq(void);
 static void _hoc_exp1(void);
 static void _hoc_sinf(void);
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
 "setdata_CadepK", _hoc_setdata,
 "alphaq_CadepK", _hoc_alphaq,
 "betar_CadepK", _hoc_betar,
 "betaq_CadepK", _hoc_betaq,
 "exp1_CadepK", _hoc_exp1,
 "sinf_CadepK", _hoc_sinf,
 0, 0
};
#define alphaq alphaq_CadepK
#define betar betar_CadepK
#define betaq betaq_CadepK
#define exp1 exp1_CadepK
#define sinf sinf_CadepK
 extern double alphaq( _threadargsprotocomma_ double );
 extern double betar( _threadargsprotocomma_ double );
 extern double betaq( _threadargsprotocomma_ double );
 extern double exp1( _threadargsprotocomma_ double , double , double , double );
 extern double sinf( _threadargsprotocomma_ double );
 /* declare global and static user variables */
#define alphar alphar_CadepK
 double alphar = 7.5;
#define stau stau_CadepK
 double stau = 10;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "alphar_CadepK", "/ms",
 "stau_CadepK", "ms",
 "gbkbar_CadepK", "S/cm2",
 "gskbar_CadepK", "S/cm2",
 "ik_CadepK", "mA/cm2",
 "isk_CadepK", "mA/cm2",
 "ibk_CadepK", "mA/cm2",
 "gbk_CadepK", "S/cm2",
 "gsk_CadepK", "S/cm2",
 0,0
};
 static double delta_t = 0.01;
 static double q0 = 0;
 static double r0 = 0;
 static double s0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "alphar_CadepK", &alphar_CadepK,
 "stau_CadepK", &stau_CadepK,
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
 
#define _cvode_ieq _ppvar[4]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.5.0",
"CadepK",
 "gbkbar_CadepK",
 "gskbar_CadepK",
 "gcakmult_CadepK",
 0,
 "ik_CadepK",
 "isk_CadepK",
 "ibk_CadepK",
 "gbk_CadepK",
 "gsk_CadepK",
 0,
 "q_CadepK",
 "r_CadepK",
 "s_CadepK",
 0,
 0};
 static Symbol* _ca_sym;
 static Symbol* _k_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 19, _prop);
 	/*initialize range parameters*/
 	gbkbar = 9e-05;
 	gskbar = 0.0001;
 	gcakmult = 1;
 	_prop->param = _p;
 	_prop->param_size = 19;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 5, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_ca_sym);
 nrn_promote(prop_ion, 1, 0);
 	_ppvar[0]._pval = &prop_ion->param[1]; /* cai */
 prop_ion = need_memb(_k_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[1]._pval = &prop_ion->param[0]; /* ek */
 	_ppvar[2]._pval = &prop_ion->param[3]; /* ik */
 	_ppvar[3]._pval = &prop_ion->param[4]; /* _ion_dikdv */
 
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

 void _CadepK2018_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("ca", -10000.);
 	ion_reg("k", -10000.);
 	_ca_sym = hoc_lookup("ca_ion");
 	_k_sym = hoc_lookup("k_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
  hoc_register_prop_size(_mechtype, 19, 5);
  hoc_register_dparam_semantics(_mechtype, 0, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 4, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 CadepK /Users/wyin/optimize_cells/x86_64/CadepK2018.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[3], _dlist1[3];
 static int state(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {int _reset = 0; {
   Dq = alphaq ( _threadargscomma_ cai ) * ( 1.0 - q ) - betaq ( _threadargscomma_ cai ) * q ;
   Dr = alphar * ( 1.0 - r ) - betar ( _threadargscomma_ v ) * r ;
   Ds = ( sinf ( _threadargscomma_ cai ) - s ) / stau ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {
 Dq = Dq  / (1. - dt*( ( alphaq ( _threadargscomma_ cai ) )*( ( ( - 1.0 ) ) ) - ( betaq ( _threadargscomma_ cai ) )*( 1.0 ) )) ;
 Dr = Dr  / (1. - dt*( ( alphar )*( ( ( - 1.0 ) ) ) - ( betar ( _threadargscomma_ v ) )*( 1.0 ) )) ;
 Ds = Ds  / (1. - dt*( ( ( ( - 1.0 ) ) ) / stau )) ;
  return 0;
}
 /*END CVODE*/
 static int state (double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) { {
    q = q + (1. - exp(dt*(( alphaq ( _threadargscomma_ cai ) )*( ( ( - 1.0 ) ) ) - ( betaq ( _threadargscomma_ cai ) )*( 1.0 ))))*(- ( ( alphaq ( _threadargscomma_ cai ) )*( ( 1.0 ) ) ) / ( ( alphaq ( _threadargscomma_ cai ) )*( ( ( - 1.0 ) ) ) - ( betaq ( _threadargscomma_ cai ) )*( 1.0 ) ) - q) ;
    r = r + (1. - exp(dt*(( alphar )*( ( ( - 1.0 ) ) ) - ( betar ( _threadargscomma_ v ) )*( 1.0 ))))*(- ( ( alphar )*( ( 1.0 ) ) ) / ( ( alphar )*( ( ( - 1.0 ) ) ) - ( betar ( _threadargscomma_ v ) )*( 1.0 ) ) - r) ;
    s = s + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / stau)))*(- ( ( ( sinf ( _threadargscomma_ cai ) ) ) / stau ) / ( ( ( ( - 1.0 ) ) ) / stau ) - s) ;
   }
  return 0;
}
 
double exp1 ( _threadargsprotocomma_ double _lA , double _ld , double _lk , double _lx ) {
   double _lexp1;
 if ( _lx > 1e-7 ) {
     _lexp1 = _lA / exp ( ( 12.0 * log10 ( _lx ) + _ld ) / _lk ) ;
     }
   else {
     _lexp1 = _lA / exp ( ( 12.0 * ( - 7. ) + _ld ) / _lk ) ;
     }
   
return _lexp1;
 }
 
static void _hoc_exp1(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  exp1 ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) , *getarg(3) , *getarg(4) );
 hoc_retpushx(_r);
}
 
double alphaq ( _threadargsprotocomma_ double _lx ) {
   double _lalphaq;
 _lalphaq = exp1 ( _threadargscomma_ 0.00246 , 28.48 , - 4.5 , _lx ) ;
   
return _lalphaq;
 }
 
static void _hoc_alphaq(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  alphaq ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double betaq ( _threadargsprotocomma_ double _lx ) {
   double _lbetaq;
 _lbetaq = exp1 ( _threadargscomma_ 0.006 , 60.4 , 35.0 , _lx ) ;
   
return _lbetaq;
 }
 
static void _hoc_betaq(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  betaq ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double betar ( _threadargsprotocomma_ double _lv ) {
   double _lbetar;
 _lbetar = 0.11 / exp ( ( _lv - 35.0 ) / 14.9 ) ;
   
return _lbetar;
 }
 
static void _hoc_betar(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  betar ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double sinf ( _threadargsprotocomma_ double _lx ) {
   double _lsinf;
 if ( _lx > 1e-7 ) {
     _lsinf = 1.0 / ( 1.0 + 4.0 / ( 1000.0 * _lx ) ) ;
     }
   else {
     _lsinf = 1.0 / ( 1.0 + 4.0 / ( 1000.0 * ( 1e-7 ) ) ) ;
     }
   
return _lsinf;
 }
 
static void _hoc_sinf(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  sinf ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ return 3;}
 
static void _ode_spec(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  cai = _ion_cai;
  ek = _ion_ek;
     _ode_spec1 (_p, _ppvar, _thread, _nt);
  }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 3; ++_i) {
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
  cai = _ion_cai;
  ek = _ion_ek;
 _ode_matsol_instance1(_threadargs_);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_ca_sym, _ppvar, 0, 1);
   nrn_update_ion_pointer(_k_sym, _ppvar, 1, 0);
   nrn_update_ion_pointer(_k_sym, _ppvar, 2, 3);
   nrn_update_ion_pointer(_k_sym, _ppvar, 3, 4);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {
  int _i; double _save;{
  q = q0;
  r = r0;
  s = s0;
 {
   q = alphaq ( _threadargscomma_ cai ) / ( alphaq ( _threadargscomma_ cai ) + betaq ( _threadargscomma_ cai ) ) ;
   r = alphar / ( alphar + betar ( _threadargscomma_ v ) ) ;
   s = sinf ( _threadargscomma_ cai ) ;
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
  cai = _ion_cai;
  ek = _ion_ek;
 initmodel(_p, _ppvar, _thread, _nt);
 }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   gbk = gbkbar * gcakmult * r * s * s ;
   gsk = gskbar * gcakmult * q * q ;
   isk = gsk * ( v - ek ) ;
   ibk = gbk * ( v - ek ) ;
   ik = isk + ibk ;
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
  cai = _ion_cai;
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
  cai = _ion_cai;
  ek = _ion_ek;
 {   state(_p, _ppvar, _thread, _nt);
  } }}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = &(q) - _p;  _dlist1[0] = &(Dq) - _p;
 _slist1[1] = &(r) - _p;  _dlist1[1] = &(Dr) - _p;
 _slist1[2] = &(s) - _p;  _dlist1[2] = &(Ds) - _p;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif
