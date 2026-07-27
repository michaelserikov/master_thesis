********************************************************************************
*  MINDTOY_rectgrid_learning_2130.gms — Cost-Risk Analysis toy model WITH learning
*
*  Built on the rectgrid (Almond-grid) structure:
*    - scenario points indexed by i, with x(i), lam(i), w(i) from GDX
*    - one-box climate model (no SO2), horizon 2025-2130
*    - utility based on FTotal (cumulative emissions) as in MINDTOY_rectgrid
*
*  Learning logic (the new part):
*    - BEFORE the learning point: all scenarios share ONE common emission
*      path, described by the scalar reduction speed v (exactly as in the
*      no-learning rectgrid model).
*    - AFTER the learning point: the true climate sensitivity is revealed and
*      each scenario i optimises its OWN emission path, subject to an inertia
*      limit on how fast emissions may change between periods.
*
*  Calibration is always done without learning (use cal_toy.cal from
*  CALIBRATE_TOY, same grid).
*
*  Run order:
*    1. gams CALIBRATE_TOY_..._2130.gms   (finds ScaleRA, writes cal_toy.cal)
*    2. gams MINDTOY_rectgrid_learning_2130.gms
********************************************************************************

* ============================================================
* USER SETTINGS
* ============================================================

Scalars
toggle_risk  "0 = BAU  |  1 = CRA"                          /0/

GR           "temperature guard rail (degrees C)"            /2.5/
ScaleRA      "risk weight - set by CALIBRATE_TOY.gms"        /0/

* Learning point: GAMS time index at which uncertainty resolves.
* t=1 is 2020, t=2 is 2025, t=3 is 2030, ... , t=23 is 2130.
* learningpoint = 3 means the true sensitivity is revealed in 2030.
* learningpoint = 23 effectively means "never learn" (no branching).
learningpoint "learning point (time index 1-22)"            /22/

* Cost function parameters (same as no-learning rectgrid)
kappa        "welfare/cost scale"                            /2.06/
g            "cumulative-emissions coefficient"              /2.07/
T0           "temperature/emissions offset"                  /0.78/
p            "curvature parameter"                           /2.17/
cume_scale   "convert emissions from GtC to TtC/EgC"         /1000/

Fhist        "historical cumulative emissions since 1750 (EgC)" /0.740/

disrate      "discount rate (per year)"                      /0.01/

* Emission inertia for the post-learning phase:
maxgrowth    "max relative emission growth per period"       /0.10/
maxdecline   "max relative emission decline per period"      /0.393/
;

* Load calibrated ScaleRA when available
$if exist 'cal_toy.cal' $include 'cal_toy.cal'


* ============================================================
* FIXED PARAMETERS
* ============================================================

Scalars
e0      "initial emissions 2025 (GtC/yr)"              /11.7/
cume0   "initial cumulative emissions 2025 (GtC)"      /740/
zeit    "length of one time step (years)"              /5/

c0_clim  "pre-industrial CO2 (ppmv)"                   /278.3/
c2025    "CO2 concentration 2025 (ppmv)"               /425.6/
b        "ocean-biosphere source"                      /0.00151/
cconvi   "GtC to ppmv conversion"                      /0.47/
sigma    "ocean-biosphere sink rate"                   /0.02150/
fcodb    "forcing for CO2 doubling (W/m2)"             /3.93/

w1 "risk shape" /2.8/
h1 "risk shape" /600/
m1 "risk shape" /0.03/

LN2 "natural log of 2" /0.6931471805599453/
;


* ============================================================
* SETS
* ============================================================

Sets
tall        /1*22/
t(tall)     "optimisation periods" /1*22/
tfirst(t)   /1/
tlast(t)    /22/
tprior(t)   "periods before learning - common path"
tpost(t)    "periods at/after learning - own path"

i           "scenario points inside Almond area"
k           "alpha strips"
;


* ============================================================
* PARAMETERS
* ============================================================

Parameters
x(i)          "ocean thermal inertia parameter"
lam(i)        "climate sensitivity"
w(i)          "probability weight of each state"
wstrip(k)     "probability weight of each alpha strip"
map(i,k)      "mapping from scenario point i to alpha strip k"
disc(t)       "discount factor"
;

$gdxin prior_data_rectgrid_25x25_learning.gdx
$load i k x lam w wstrip map
$gdxin

display i, k, x, lam, w, wstrip, map;

Scalar wsum;
wsum = sum(i, w(i));
display wsum;

disc(t)      = zeit * exp(-zeit * disrate * (ord(t) - 1));
disc(tlast)  = disc(tlast) / (1 - exp(-zeit * disrate));

* Split the horizon into "before learning" and "at/after learning"
tprior(t) = yes$(ord(t) lt learningpoint);
tpost(t)  = yes$(ord(t) ge learningpoint);


* ============================================================
* VARIABLES
* ============================================================

Positive Variables
v               "common emissions reduction speed BEFORE learning"
Emissions(t,i)  "CO2 emissions path (GtC/yr)"
CumEmi(t,i)     "cumulative CO2 emissions (GtC)"
CO2Conc(t,i)    "atmospheric CO2 concentration (ppmv)"
Temp(t,i)       "global mean temperature above pre-industrial (degrees C)"
fco2(t,i)       "CO2 radiative forcing (W/m2)"
HeavyS(t,i)     "smooth exceedance indicator at the guard rail"
Risk(t,i)       "risk penalty"
FTotal(i)       "total cumulative emissions entering static utility (EgC), per scenario"
;

Variables
UtilityLeft(i)  "static utility/cost part, per scenario"
goal            "objective: discounted expected welfare"
;


* ============================================================
* EQUATIONS
* ============================================================

Equations
GoalFct
UtilityLeftFct(i)
FTotalFct(i)
CommonPath(t,i)      "before learning: v-path, identical across scenarios"
InertiaUp(t,i)       "after learning: emissions cannot grow too fast"
InertiaDown(t,i)     "after learning: emissions cannot fall too fast"
AccumEmi(t,i)
CC(t,i)
ForcCO2(t,i)
CliSys(t,i)
HeavySideFct(t,i)
RiskFct(t,i)
;


* Objective: expected static utility minus expected discounted risk.
* Utility is now per-scenario (each i has its own FTotal after learning),
* so we take the probability-weighted average with w(i).
GoalFct..
    goal =E=
        sum(i, w(i) * UtilityLeft(i))
        - toggle_risk * exp(-ScaleRA) *
          sum(t, disc(t) * sum(i, w(i) * Risk(t,i)));


* Static utility per scenario
UtilityLeftFct(i)..
    UtilityLeft(i) =E= -kappa * rPower(g * FTotal(i) - T0, -p);


* Total cumulative emissions per scenario, taken numerically from the
* accumulated path (works for ANY path, not only the v-exponential).
* CumEmi is in GtC; divide by cume_scale to get EgC, add historical Fhist.
FTotalFct(i)..
    FTotal(i) =E= Fhist + (sum(tlast, CumEmi(tlast,i)) - cume0) / cume_scale;


* BEFORE learning: every scenario follows the SAME v-exponential path
* (this reproduces the no-learning rectgrid behaviour up to the learning point)
CommonPath(t,i)$tprior(t)..
    Emissions(t,i) =E= e0 * exp(-v * zeit * (ord(t)-1));


* AFTER learning: emissions are free per scenario, but limited by inertia.
* The relative change between consecutive periods is bounded.
InertiaUp(t+1,i)$tpost(t+1)..
    Emissions(t+1,i) =L= Emissions(t,i) * (1 + maxgrowth);

InertiaDown(t+1,i)$tpost(t+1)..
    Emissions(t+1,i) =G= Emissions(t,i) * (1 - maxdecline);


* Cumulative emissions accumulate each period
AccumEmi(t+1,i)..
    CumEmi(t+1,i) =E= CumEmi(t,i) + zeit * Emissions(t,i);


* CO2 concentration: trapezoid integration of net carbon flux
CC(t+1,i)..
    CO2Conc(t+1,i) =E= CO2Conc(t,i)
        + (zeit/2)*(b*CumEmi(t,i)   + cconvi*Emissions(t,i)   - sigma*(CO2Conc(t,i)   - c0_clim))
        + (zeit/2)*(b*CumEmi(t+1,i) + cconvi*Emissions(t+1,i) - sigma*(CO2Conc(t+1,i) - c0_clim));


* CO2 radiative forcing
ForcCO2(t,i)..
    fco2(t,i) =E= fcodb * log(CO2Conc(t,i) / c0_clim) / log(2);


* Temperature response to forcing: trapezoid integration
CliSys(t+1,i)..
    Temp(t+1,i) =E= Temp(t,i)
        + (zeit/2)*((1/x(i))*LN2/fcodb * fco2(t,i)   - LN2/(x(i)*lam(i))*Temp(t,i))
        + (zeit/2)*((1/x(i))*LN2/fcodb * fco2(t+1,i) - LN2/(x(i)*lam(i))*Temp(t+1,i));


* Smooth Heaviside: probability mass above the guard rail
HeavySideFct(t,i)..
    HeavyS(t,i) =E= 0.5 * (1 + errorf((Temp(t,i) - GR) * w1));


* Risk penalty
RiskFct(t,i)..
    Risk(t,i) =G= (1 / (h1*(HeavyS(t,i)*(1-HeavyS(t,i)) + m1)))
                   * log(1 + exp((h1*(HeavyS(t,i)*(1-HeavyS(t,i)) + m1)) * (Temp(t,i) - GR)));


* ============================================================
* BOUNDS AND INITIAL CONDITIONS
* ============================================================

* Common reduction speed
v.lo = 1e-6;
v.up = 0.10;
v.l  = 0.02;

* v is minimal, because the model starts the learning from the very begging
*v.fx = 1e-6;

* Emission level bounds
Emissions.lo(t,i) = 0.001;
Emissions.up(t,i) = e0;

* Minimum lower bounds to keep equations well-defined
CumEmi.lo(t,i)  = cume0;
CO2Conc.lo(t,i) = 1;
Temp.lo(t,i)    = 0.01;
fco2.lo(t,i)    = 0.01;
FTotal.lo(i)    = T0/g + 1e-6;

* Starting values to help the solver
Emissions.l(t,i) = e0 * exp(-v.l * zeit * (ord(t)-1));
CumEmi.l(t,i)    = cume0;
CO2Conc.l(t,i)   = c2025;
Temp.l(t,i)      = 1.37;
FTotal.l(i)      = 1.0;

* Fix initial conditions to 2025 observed values
Emissions.fx(tfirst,i) = e0;
CumEmi.fx(tfirst,i)    = cume0;
CO2Conc.fx(tfirst,i)   = c2025;
Temp.fx(tfirst,i)      = 1.37;


* ============================================================
* SOLVE - full
* ============================================================

model mindtoy_learn /all/;
option nlp = conopt3;
option domlim = 1000;
option iterlim = 100000;
option reslim  = 3600;

* ============================================================
* SOLVE  —  MULTISTART on the learningpoint set in USER SETTINGS
* ============================================================
* FULL learning model: post-learning each scenario i has its own free
* Emissions(t,i) (no class sharing). Set 'learningpoint' at the top:
*    22 -> no-learning base N        1 -> FULL learning F
* Sweeps pre-learning speed v (lp>1) and the initial emission-decay guess
* (all lp), keeps the best ModelStat 1/2 solution, then re-solves at that
* best seed so every .L trajectory is the best optimum.

tprior(t) = yes$(ord(t) lt learningpoint);
tpost(t)  = yes$(ord(t) ge learningpoint);

Set ms "multistart seeds" /1*7/;
Parameter
  vseed(ms)    / 1 1e-6, 2 0.02, 3 0.04, 4 0.06, 5 0.08, 6 0.094, 7 0.10 /
  rateseed(ms) / 1 0.02, 2 0.04, 3 0.06, 4 0.08, 5 0.10, 6 0.12, 7 0.15 / ;
Scalar bestgoal /-1e30/, bestv /0/, bestrate /0/, found /0/;
Parameter results(ms,*);

loop(ms,
   if(learningpoint = 1,
       v.fx = 1e-6;
   else
       v.lo = 1e-6; v.up = 0.10; v.l = vseed(ms);
   );
   Emissions.l(t,i) = e0 * exp(-rateseed(ms) * zeit * (ord(t)-1));

   solve mindtoy_learn maximizing goal using nlp;

   results(ms,'goal')  = goal.l;
   results(ms,'mstat') = mindtoy_learn.modelstat;
   results(ms,'sstat') = mindtoy_learn.solvestat;
   results(ms,'v')     = v.l;

   if((mindtoy_learn.modelstat = 2 or mindtoy_learn.modelstat = 1) and goal.l > bestgoal,
       bestgoal = goal.l;  bestv = vseed(ms);  bestrate = rateseed(ms);  found = 1;
   );
);

* re-solve at the best seed so all .L values correspond to the best optimum
if(found = 1,
   if(learningpoint = 1,
       v.fx = 1e-6;
   else
       v.lo = 1e-6; v.up = 0.10; v.l = bestv;
   );
   Emissions.l(t,i) = e0 * exp(-bestrate * zeit * (ord(t)-1));
   solve mindtoy_learn maximizing goal using nlp;
);

display results, bestgoal, bestv, bestrate;
display mindtoy_learn.modelstat, mindtoy_learn.solvestat;
display goal.l, v.l;
display Emissions.l, Temp.l, CO2Conc.l, CumEmi.l, FTotal.l;
display x, lam, w, UtilityLeft.L, Risk.L;
