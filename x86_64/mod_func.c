#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _Ca2018_reg(void);
extern void _Cacum2018_reg(void);
extern void _CadepK2018_reg(void);
extern void _DGC_M_2018_reg(void);
extern void _ampa_kin_reg(void);
extern void _exp2EPSC_reg(void);
extern void _exp2EPSG_reg(void);
extern void _exp2EPSG_NMDA_reg(void);
extern void _facil_NMDA_reg(void);
extern void _facil_exp2syn_reg(void);
extern void _gaba_a_kin_reg(void);
extern void _h_reg(void);
extern void _izhi2007bS_reg(void);
extern void _izhi2019_reg(void);
extern void _kad_reg(void);
extern void _kap_reg(void);
extern void _kdr_reg(void);
extern void _km3_reg(void);
extern void _lin_exp2syn_reg(void);
extern void _nas_reg(void);
extern void _nax_reg(void);
extern void _nmda_kin5_reg(void);
extern void _pr_reg(void);
extern void _sat_exp2syn_reg(void);
extern void _vecevent_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," Ca2018.mod");
    fprintf(stderr," Cacum2018.mod");
    fprintf(stderr," CadepK2018.mod");
    fprintf(stderr," DGC_M_2018.mod");
    fprintf(stderr," ampa_kin.mod");
    fprintf(stderr," exp2EPSC.mod");
    fprintf(stderr," exp2EPSG.mod");
    fprintf(stderr," exp2EPSG_NMDA.mod");
    fprintf(stderr," facil_NMDA.mod");
    fprintf(stderr," facil_exp2syn.mod");
    fprintf(stderr," gaba_a_kin.mod");
    fprintf(stderr," h.mod");
    fprintf(stderr," izhi2007bS.mod");
    fprintf(stderr," izhi2019.mod");
    fprintf(stderr," kad.mod");
    fprintf(stderr," kap.mod");
    fprintf(stderr," kdr.mod");
    fprintf(stderr," km3.mod");
    fprintf(stderr," lin_exp2syn.mod");
    fprintf(stderr," nas.mod");
    fprintf(stderr," nax.mod");
    fprintf(stderr," nmda_kin5.mod");
    fprintf(stderr," pr.mod");
    fprintf(stderr," sat_exp2syn.mod");
    fprintf(stderr," vecevent.mod");
    fprintf(stderr, "\n");
  }
  _Ca2018_reg();
  _Cacum2018_reg();
  _CadepK2018_reg();
  _DGC_M_2018_reg();
  _ampa_kin_reg();
  _exp2EPSC_reg();
  _exp2EPSG_reg();
  _exp2EPSG_NMDA_reg();
  _facil_NMDA_reg();
  _facil_exp2syn_reg();
  _gaba_a_kin_reg();
  _h_reg();
  _izhi2007bS_reg();
  _izhi2019_reg();
  _kad_reg();
  _kap_reg();
  _kdr_reg();
  _km3_reg();
  _lin_exp2syn_reg();
  _nas_reg();
  _nax_reg();
  _nmda_kin5_reg();
  _pr_reg();
  _sat_exp2syn_reg();
  _vecevent_reg();
}
