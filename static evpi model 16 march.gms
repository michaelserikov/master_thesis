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
* Here, c_val plays the role of E (decision variable): E === c_val

Scalars
    LN2 /0.6931471805599453/
    E0  /1.0/
    B   /1.0/
    dt  /1.0/;

Variables
    E
    z;

Positive Variable E;
E.lo = 0;
E.up = 1000;

Equation obj;

obj..
    z =e=
        sum(i,
            w(i) * (
                - sqr(E - E0)
                - B * sqr(E * lam(i) * (1 - exp(-LN2 * dt / (x(i) * lam(i)))))
            )
        );

Model priorModel /obj/;


* ______________________________________________________________________
* Step 5. Let's find E* = argmax E[W(E)] over E in [0,1000]

solve priorModel maximizing z using nlp;

Scalar
    E_star
    EW_prior;

E_star   = E.l;
EW_prior = z.l;

display E_star, EW_prior;


* ______________________________________________________________________
* Step 6. For each strip k, let's find E**(k)

Parameters
    active(i)
    E_dbstar(k);

Scalar wkcur;

Variables
    E_k
    z_k;

Positive Variable E_k;
E_k.lo = 0;
E_k.up = 1000;

Equation obj_k;

obj_k..
    z_k =e=
        sum(i,
            active(i) * w(i) * (
                - sqr(E_k - E0)
                - B * sqr(E_k * lam(i) * (1 - exp(-LN2 * dt / (x(i) * lam(i)))))
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
    - sqr(E_star - E0)
    - B * sqr(E_star * lam(i) * (1 - exp(-LN2 * dt / (x(i) * lam(i)))));

W_learn(i) =
    - sqr(Echosen(i) - E0)
    - B * sqr(Echosen(i) * lam(i) * (1 - exp(-LN2 * dt / (x(i) * lam(i)))));

Scalars
    EW_learn
    EVPI;

EW_learn = sum(i, w(i) * W_learn(i));

display EW_learn;


* ______________________________________________________________________
* Step 7. EVPI

EVPI = sum(i, w(i) * (W_learn(i) - W_prior(i)));

display EVPI;
