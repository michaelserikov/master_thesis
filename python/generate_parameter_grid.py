import numpy as np
import gams.transfer as gt

# ============================================================
# CONFIG  -- change only here when switching grids
# ============================================================
# 10x10 -> 38 points -> Sturges K = 7
# 20x20 -> 171 points -> Sturges K = 13
# 25x25 -> 281 points -> Sturges K = 19
N_x = 25
N_lam = 25
K = 19
GDX_OUT = "prior_data_rectgrid_25x25_learning.gdx"
GAMS_SYSDIR = r"C:\GAMS\45"

# ============================================================
# 1. Almond area
# ============================================================
a, b, c = 0.068718309227142967, -0.7042877380888113, 1.8938727238235138
A, B, C = -0.080844837894888034, 1.2909933755066116, 1.2621122665955362

def lam_low(x):
    return a*x*x + b*x + c

def lam_up(x):
    return A*x*x + B*x + C

# intersections: lam_low = lam_up
roots = np.real(np.real_if_close(np.roots([A - a, B - b, C - c]), tol=1000))
xmin = float(np.min(roots))
xmax = float(np.max(roots))
print("xmin, xmax =", xmin, xmax)

# ============================================================
# 2. Rectangular grid in original (x, lambda) coordinates
# ============================================================
x_fine = np.linspace(xmin, xmax, 1000)
lam_min = min(lam_low(x_fine).min(), lam_up(x_fine).min())
lam_max = max(lam_low(x_fine).max(), lam_up(x_fine).max())

x_grid = np.linspace(xmin, xmax, N_x)
lam_grid = np.linspace(lam_min, lam_max, N_lam)
X, LAM = np.meshgrid(x_grid, lam_grid)
x_all = X.ravel()
lam_all = LAM.ravel()

# ============================================================
# 3. Keep only points inside Almond area
# ============================================================
inside = (lam_all >= lam_low(x_all)) & (lam_all <= lam_up(x_all))
xs = x_all[inside]
lmb = lam_all[inside]
print("Total rectangular grid points =", len(x_all))
print("Points inside Almond area =", len(xs))

# ============================================================
# 3b. Filter out scenarios with non-monotonic BAU temperature
#     (numerical artefact at low climate sensitivity)
#
#     NOTE: validated for the 10x10, 20x20 and 25x25 grids only.
#     For a different / larger grid (e.g. 30x30) this fixed
#     threshold may misclassify borderline scenarios -- in that
#     case switch to a direct BAU-monotonicity simulation instead
#     of relying on this static threshold.
# ============================================================
LAM_THRESHOLD = 1.850
keep = lmb >= LAM_THRESHOLD
xs = xs[keep]
lmb = lmb[keep]
N = len(xs)
print(f"Kept {N} scenarios after BAU-monotonicity filter (lam >= {LAM_THRESHOLD})")

if N == 0:
    raise ValueError("No grid points left after filtering. Check grid size / threshold.")

# ============================================================
# 4. Alpha and weights
# ============================================================
LN2 = np.log(2.0)
alpha = LN2 / (xs * lmb)
w = np.ones(N) / N
print("sum w     =", w.sum())
print("alpha min =", alpha.min())
print("alpha max =", alpha.max())

# ============================================================
# 5. Alpha classes (Variant A: sort by alpha, equal-mass classes)
#
#    Quantile cut: sorted point j (1..N) -> class ceil(j*K/N).
#    This spreads the leftover points across the range instead of
#    piling the heavy classes at the start (which np.array_split does).
#    For N=38, K=7 this gives sizes [5,5,6,5,6,5,6].
#
#    No Jacobian needed: (x, lambda) coordinates are unchanged,
#    alpha is used only as a label for grouping; class weight is
#    just the sum of point weights inside the class.
# ============================================================
if K > N:
    raise ValueError("K is larger than the number of retained points. Reduce K or increase grid size.")

order = np.argsort(alpha)                      # indices sorted by ascending alpha
pos = np.arange(1, N + 1)                       # 1-based position in sorted order
cls_for_sorted = np.ceil(pos * K / N).astype(int) - 1   # 0-based class id

strip_idx = np.empty(N, dtype=int)
strip_idx[order] = cls_for_sorted

counts = np.bincount(strip_idx, minlength=K)
wstrip = np.array([w[strip_idx == k].sum() for k in range(K)])

print("strip counts =", counts.tolist())
print("wstrip       =", wstrip)
print("sum wstrip   =", wstrip.sum())

# alpha boundaries between classes (useful for plotting the strips)
bnds = []
alpha_sorted = alpha[order]
for k in range(1, K):
    last = alpha_sorted[cls_for_sorted == k - 1][-1]
    first = alpha_sorted[cls_for_sorted == k][0]
    bnds.append(0.5 * (last + first))
print("alpha class boundaries =", [round(b, 5) for b in bnds])

# ============================================================
# 6. Write GDX for GAMS
# ============================================================
m = gt.Container(system_directory=GAMS_SYSDIR)

i_labels = [f"i{n+1}" for n in range(N)]
k_labels = [f"k{j+1}" for j in range(K)]

set_i = gt.Set(m, "i", description="scenario points inside Almond area")
set_i.setRecords(i_labels)

set_k = gt.Set(m, "k", description="alpha classes (equal-mass)")
set_k.setRecords(k_labels)

par_x = gt.Parameter(m, "x", domain=[set_i])
par_x.setRecords([(i_labels[n], float(xs[n])) for n in range(N)])

par_lam = gt.Parameter(m, "lam", domain=[set_i])
par_lam.setRecords([(i_labels[n], float(lmb[n])) for n in range(N)])

par_alpha = gt.Parameter(m, "alpha", domain=[set_i])
par_alpha.setRecords([(i_labels[n], float(alpha[n])) for n in range(N)])

par_w = gt.Parameter(m, "w", domain=[set_i])
par_w.setRecords([(i_labels[n], float(w[n])) for n in range(N)])

par_wstrip = gt.Parameter(m, "wstrip", domain=[set_k])
par_wstrip.setRecords([(k_labels[j], float(wstrip[j])) for j in range(K)])

par_map = gt.Parameter(m, "map", domain=[set_i, set_k])
par_map.setRecords([
    (i_labels[n], k_labels[int(strip_idx[n])], 1.0)
    for n in range(N)
])

m.write(GDX_OUT)
print(f"{GDX_OUT} written")

