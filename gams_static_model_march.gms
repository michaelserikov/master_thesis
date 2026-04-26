* ______________________________________________________________________
* Step 0. Load data from GDX

Sets
    i
    k;

Parameters
    x(i)
    lam(i)
    w(i)
    map(i,k)
    wstrip(k);

$gdxin prior_data.gdx
$load i k x lam w map wstrip
$gdxin


* ______________________________________________________________________
* Step 4. Welfare function
*
* Static version:
* - E is the decision variable = additional cumulative emissions after the start year
* - Ehist is fixed historical cumulative emissions up to the start year
* - Etot = Ehist + E
*
* Left part:
*   -kappa * ( g * Etot - T0 )^(-p)
*
* Right part (kept as static proxy for now):
*   -B * [ Etot * lam(i) * (1 - exp(-LN2*dt/(x(i)*lam(i)))) ]^2

Scalars
    LN2   /0.6931471805599453/
    kappa /2.06/
    g     /2.07/
    T0    /0.78/
    p     /2.17/
    Ehist /1.013/
    B     /1.0/
    dt    /1.0/;

Variables
    E
    z;

Positive Variable E;

* E is now additional cumulative emissions (EgC)
E.lo = 0;
E.up = 3;

Equation obj;

obj..
    z =e=
        sum(i,
            w(i) * (
                - kappa * rPower(g * (Ehist + E) - T0, -p)
                - B * sqr(
                    (Ehist + E) * lam(i)
                    * (1 - exp(-LN2 * dt / (x(i) * lam(i))))
                )
            )
        );

Model priorModel /obj/;


* ______________________________________________________________________
* Step 5. Find E* = argmax E[W(E)]

solve priorModel maximizing z using nlp;

Scalars
    E_star
    EW_prior;

E_star   = E.l;
EW_prior = z.l;

display E_star, EW_prior;


* ______________________________________________________________________
* Step 6. For each strip k, find E**(k)

Parameters
    active(i)
    E_dbstar(k);

Scalar wkcur;

Variables
    E_k
    z_k;

Positive Variable E_k;

* E_k is also additional cumulative emissions (EgC)
E_k.lo = 0;
E_k.up = 3;

Equation obj_k;

obj_k..
    z_k =e=
        sum(i,
            active(i) * w(i) * (
                - kappa * rPower(g * (Ehist + E_k) - T0, -p)
                - B * sqr(
                    (Ehist + E_k) * lam(i)
                    * (1 - exp(-LN2 * dt / (x(i) * lam(i))))
                )
            )
        ) / wkcur;

Model stripModel /obj_k/;

loop(k$(wstrip(k) > 0),
    active(i) = map(i,k);
    wkcur = wstrip(k);
    solve stripModel maximizing z_k using nlp;
    E_dbstar(k) = E_k.l;
);

display E_dbstar;


* ______________________________________________________________________
* Step 6. Compute E[W](E**)

Parameters
    Echosen(i)
    W_prior(i)
    W_learn(i);

Echosen(i) = sum(k$map(i,k), E_dbstar(k));

W_prior(i) =
    - kappa * rPower(g * (Ehist + E_star) - T0, -p)
    - B * sqr(
        (Ehist + E_star) * lam(i)
        * (1 - exp(-LN2 * dt / (x(i) * lam(i))))
    );

W_learn(i) =
    - kappa * rPower(g * (Ehist + Echosen(i)) - T0, -p)
    - B * sqr(
        (Ehist + Echosen(i)) * lam(i)
        * (1 - exp(-LN2 * dt / (x(i) * lam(i))))
    );

Scalars
    EW_learn
    EVPI;

EW_learn = sum(i, w(i) * W_learn(i));

display EW_learn;


* ______________________________________________________________________
* Step 7. EVPI

EVPI = sum(i, w(i) * (W_learn(i) - W_prior(i)));

display EVPI;
