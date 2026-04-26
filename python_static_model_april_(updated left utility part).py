import numpy as np
from scipy.optimize import minimize_scalar

# ______________________________________________________________________
# Step 1. Define the Almond Area: boundaries, intersections, Area, and m=1/Area

a, b, c = 0.068718309227142967, -0.7042877380888113, 1.8938727238235138
A, B, C = -0.080844837894888034, 1.2909933755066116, 1.2621122665955362

def lam_low(x):
    return a * x * x + b * x + c

def lam_up(x):
    return A * x * x + B * x + C

p2 = A - a
p1 = B - b
p0 = C - c

r = np.real(np.real_if_close(np.roots([p2, p1, p0]), tol=1000))
xmin = float(min(r))
xmax = float(max(r))
print("xmin, xmax =", xmin, xmax)

def area_between(x1, x2):
    def F(x):
        return (p2 / 3.0) * x**3 + (p1 / 2.0) * x**2 + p0 * x
    return float(F(x2) - F(x1))

area = area_between(xmin, xmax)
m = 1.0 / area
print("Area =", area)
print("m = 1/Area =", m)

# ______________________________________________________________________
# Step 2. Prior points (equal-weight sampling over the almond area)

N_x = 800
N_l = 800

xs = []
lmb = []

dx = (xmax - xmin) / N_x

for i in range(N_x):
    x = xmin + (i + 0.5) * dx

    lo = lam_low(x)
    hi = lam_up(x)

    for j in range(N_l):
        v = (j + 0.5) / N_l
        lam = lo + v * (hi - lo)
        xs.append(x)
        lmb.append(lam)

N = len(xs)
w = 1.0 / N
print("N =", N)

# ______________________________________________________________________
# Step 3. alpha strips (equal-mass via quantiles)

LN2 = np.log(2.0)
K = 21

xs_arr = np.asarray(xs, dtype=float)
lmb_arr = np.asarray(lmb, dtype=float)

alpha = LN2 / (xs_arr * lmb_arr)

a_min = float(alpha.min())
a_max = float(alpha.max())

edges = np.quantile(alpha, np.linspace(0.0, 1.0, K + 1))

strip_idx = np.searchsorted(edges, alpha, side="right") - 1
strip_idx = np.clip(strip_idx, 0, K - 1)

counts = np.bincount(strip_idx, minlength=K).astype(int)
w_per_strip = counts * w

print("alpha min/max =", a_min, a_max)
print("first strip weights =", w_per_strip[:5])

# ______________________________________________________________________
# Step 4. Welfare function: updated static version

# Held parameters
kappa = 2.06
g = 2.07
T0 = 0.78
p = 2.17

# Historical cumulative emissions (EgC)
Ehist = 1.013

# Static proxy parameters
B = 1.0
dt = 1.0

def W_welfare(E_add, x, lam):
    """
    E_add = additional cumulative emissions after the optimization start (EgC)
    """
    E_tot = Ehist + E_add

    # Left part
    left_term = -kappa * (g * E_tot - T0) ** (-p)

    # Right part (same static proxy as in GAMS)
    a_loc = LN2 / (x * lam)
    right_term = -B * (
        E_tot * lam * (1.0 - np.exp(-a_loc * dt))
    ) ** 2

    return left_term + right_term

# ______________________________________________________________________
# Step 5. Find E* = argmax E[W(E)] over E_add in [0,3]

def EW(E_add):
    s = 0.0
    for i in range(N):
        s += w * W_welfare(E_add, xs[i], lmb[i])
    return s

res = minimize_scalar(lambda E_add: -EW(E_add), bounds=(0.0, 3.0), method="bounded")
E_star = float(res.x)
best_EW = float(EW(E_star))

print("E* (additional EgC) =", E_star)
print("E_total* (EgC) =", Ehist + E_star)
print("E[W|prior](E*) =", best_EW)

# ______________________________________________________________________
# Step 6. For each strip k, find E**(k)

indices_in_strip = [[] for _ in range(K)]
for i in range(N):
    indices_in_strip[int(strip_idx[i])].append(i)

E_dbstar = [E_star] * K

for k in range(K):
    if w_per_strip[k] == 0.0:
        continue

    def EW_k(E_add):
        s = 0.0
        for i in indices_in_strip[k]:
            s += w * W_welfare(E_add, xs[i], lmb[i])
        return s / w_per_strip[k]

    resk = minimize_scalar(lambda E_add: -EW_k(E_add), bounds=(0.0, 3.0), method="bounded")
    E_dbstar[k] = float(resk.x)

print("E** first 10 =", E_dbstar[:10])
print("E** min/max =", min(E_dbstar), max(E_dbstar))

W_Edbstar = sum(
    w * W_welfare(E_dbstar[int(strip_idx[i])], xs[i], lmb[i])
    for i in range(N)
)
print("E[W](E**) =", float(W_Edbstar))

# ______________________________________________________________________
# Step 7. EVPI

EVPI = 0.0
for i in range(N):
    E_dd = E_dbstar[int(strip_idx[i])]
    EVPI += w * (
        W_welfare(E_dd, xs[i], lmb[i]) - W_welfare(E_star, xs[i], lmb[i])
    )

print("EVPI =", float(EVPI))
