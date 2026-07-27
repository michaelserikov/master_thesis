"""
Priority-1 analysis: economics of the alpha extremes.

Reads TWO GAMS listings of the partial-learning model on the same grid / same
calibration:
    PARTIAL_FILE : learning run   (lp = 1,  all steps uncommented)
    NOLEARN_FILE : no-learning run(lp = 22, STEP 2 commented out)

Both listings must contain (add to the .gms display if missing):
    display Emissions.L, Temp.L, CO2Conc.L, CumEmi.L, x, lam, w, UtilityLeft.L, Risk.L;

Outputs:
  * a 6-panel figure (3 alpha levels x [emissions, temperature]), before-learning
    (dashed) overlaid on after-learning (solid);
  * a printed welfare decomposition (mitigation vs risk) for each level,
    before vs after learning.

alpha = ln2 / (x * lam).

--------------------------------------------------------------------------------
COSMETIC EMISSION SMOOTHING (variant A, from the CRA figure script)
--------------------------------------------------------------------------------
The post-learning emission paths are underdetermined and look kinky. Following
Held's suggestion each emission branch is projected onto a smooth family
        E(tau) = a*exp(-v*tau) + b*tau*exp(-w*tau)      (decay + gamma bump)
purely for display. Applied AFTER optimisation, so welfare / risk / the welfare
split below are untouched. Only the Emissions panels are smoothed; Temperature
and everything else use the raw solver output. Set SMOOTH_EMISSIONS = False to
recover the original raw emission panels.

Change only the CONFIG block.
"""

import os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit   # needed only for the cosmetic fit

# ============ CONFIG ============
DATA_DIR     = r"C:\Users\u301907\Desktop\models runs results\2,5"
PARTIAL_FILE = "2,5partiallearning-cra-lp1-2025-2130-25x25-2_and_half_degree.txt"
NOLEARN_FILE = "2,5partiallearning-cra-lp22-2025-2130-25x25-2_and_half_degree-2130.txt"
GRID_LABEL   = "25x25"
GUARDRAIL    = 2.5
SCALERA      = 2.5
YEAR0, DT    = 2025, 5
DISRATE, ZEIT, N_T = 0.01, 5, 22
SMOOTH_EMISSIONS = True     # cosmetic smoothing of emission panels (display only)
# ================================


def read(p):
    with open(p, "r", errors="ignore") as f:
        return f.read()


def parse_scalar(text, name):
    m = re.search(rf"{re.escape(name)}\.L\s*=\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)", text)
    return float(m.group(1)) if m else None


def parse_var_2d(text, varname):
    """columns-style block: rows = time step, cols = i1 i2 ... (may be grouped)."""
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if re.search(rf"VARIABLE\s+{re.escape(varname)}\.L\b", ln):
            start = i; break
    if start is None:
        return None, None, None
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if re.match(r"----\s+\d+\s+(VARIABLE|PARAMETER)", lines[i]):
            end = i; break
    data, cols = {}, []
    hdr = re.compile(r"^\s*\+?\s*(i\d+(?:\s+i\d+)*)\s*$")
    for ln in lines[start:end]:
        h = hdr.match(ln)
        if h:
            cols = h.group(1).split(); continue
        parts = ln.split()
        if not parts or not parts[0].isdigit():
            continue
        step = int(parts[0])
        for c, v in zip(cols, parts[1:]):
            try:
                data.setdefault(c, {})[step] = float(v)
            except ValueError:
                pass
    if not data:
        return None, None, None
    scen = sorted(data.keys(), key=lambda s: int(s[1:]))
    steps = sorted({s for d in data.values() for s in d})
    arr = np.array([[data[c].get(s, np.nan) for c in scen] for s in steps])
    return np.array(steps), arr, scen


def parse_inline_1d(text, name, kind="VARIABLE"):
    """inline block: 'i1 -0.729,  i2 -0.729, ...'  ->  {scen: value}."""
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if re.search(rf"{kind}\s+{re.escape(name)}(\.L)?\b", ln):
            start = i; break
    if start is None:
        return None
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if re.match(r"----\s+\d+\s+(VARIABLE|PARAMETER)", lines[i]):
            end = i; break
    chunk = " ".join(lines[start:end])
    pairs = re.findall(r"(i\d+)\s+(-?\d+\.?\d*(?:[eE][+-]?\d+)?)", chunk)
    return {k: float(v) for k, v in pairs} if pairs else None


def load(path):
    t = read(path)
    d = {"goal": parse_scalar(t, "goal")}
    for v in ["Emissions", "Temp", "CO2Conc", "CumEmi", "Risk"]:
        s, m, sc = parse_var_2d(t, v)
        d[v] = m
        if v == "Temp":
            d["steps"], d["scen"] = s, sc
    d["UtilityLeft"] = parse_inline_1d(t, "UtilityLeft", "VARIABLE")
    d["x"]   = parse_inline_1d(t, "x", "PARAMETER")
    d["lam"] = parse_inline_1d(t, "lam", "PARAMETER")
    d["w"]   = parse_inline_1d(t, "w", "PARAMETER")
    d["_text"] = t
    return d


def risk_for(text, scen_list, n_steps):
    """Return Risk matrix [steps x scen] aligned to scen_list (zeros where absent)."""
    _, mat, cols = parse_var_2d(text, "Risk")
    full = np.zeros((n_steps, len(scen_list)))
    if mat is None:
        return full
    colidx = {c: j for j, c in enumerate(cols)}
    steps_r, _, _ = parse_var_2d(text, "Risk")
    for j, c in enumerate(scen_list):
        if c in colidx:
            src = mat[:, colidx[c]]
            for si, st in enumerate(steps_r):
                if 1 <= st <= n_steps and np.isfinite(src[si]):
                    full[st - 1, j] = src[si]
    return full


# ----------------------------------------------------------------------
# Cosmetic held-like smoothing (variant A) — display only
# ----------------------------------------------------------------------
def _held(tau, a, v, b, w):
    return a * np.exp(-v * tau) + b * tau * np.exp(-w * tau)


def _smooth_branch(years, E):
    E = np.asarray(E, dtype=float)
    if np.all(np.isnan(E)):
        return E
    tau = years - years[0]
    E0 = max(np.nan_to_num(E[0]), 1e-3)
    p0 = [E0, 0.05, 0.1 * E0, 0.02]
    bounds = ([0.0, 1e-4, 0.0, 1e-4],
              [5 * E0 + 1.0, 1.0, 50 * E0 + 1.0, 1.0])
    try:
        popt, _ = curve_fit(_held, tau, E, p0=p0, bounds=bounds, maxfev=20000)
        fit = _held(tau, *popt)
        return np.clip(fit, 0.0, max(np.nanmax(E), E0) * 1.05)
    except Exception:
        return E


def smooth_emissions(years, emis):
    out = np.empty_like(emis, dtype=float)
    for j in range(emis.shape[1]):
        out[:, j] = _smooth_branch(years, emis[:, j])
    return out


def main():
    part = load(os.path.join(DATA_DIR, PARTIAL_FILE))
    nol  = load(os.path.join(DATA_DIR, NOLEARN_FILE))
    scen  = part["scen"]
    years = YEAR0 + (part["steps"] - 1) * DT
    XLIM = (2025, 2130)

    LN2 = np.log(2)
    alpha = np.array([LN2 / (part["x"][s] * part["lam"][s]) for s in scen])
    order = np.argsort(alpha)
    lo  = int(order[0])                 # lowest alpha
    hi  = int(order[-1])                # highest alpha
    mid = int(order[len(order)//2])     # median alpha
    sel = [("HIGH alpha", hi), ("MID alpha", mid), ("LOW alpha", lo)]
    for tag, idx in sel:
        print(f"{tag} {scen[idx]}: a={alpha[idx]:.4f}  "
              f"x={part['x'][scen[idx]]:.3f} lam={part['lam'][scen[idx]]:.3f}")

    # cosmetic: smooth ONLY the emissions for display (both runs); temperature raw
    emis_part = smooth_emissions(years, part["Emissions"]) if SMOOTH_EMISSIONS else part["Emissions"]
    emis_nol  = smooth_emissions(years, nol["Emissions"])  if SMOOTH_EMISSIONS else nol["Emissions"]

    # figure: 3 rows (one per alpha level), 2 cols (emissions, temperature)
    fig, ax = plt.subplots(3, 2, figsize=(12, 12))
    smooth_note = " " if SMOOTH_EMISSIONS else ""
    fig.suptitle(f"Alpha levels: before vs after partial learning ({GRID_LABEL}, GR={GUARDRAIL}C){smooth_note}",
                 fontsize=13, fontweight="bold")

    def ov(a, before, after, idx, yl, ti, gl=False):
        a.plot(years, before[:, idx], "k--", lw=1.8, label="before learning")
        a.plot(years, after[:, idx], "C0-", lw=2.2, label="after learning")
        if gl: a.axhline(GUARDRAIL, color="red", ls=":", lw=1)
        a.set_xlim(XLIM)
        a.set_xticks([2025, 2050, 2075, 2100, 2130])
        a.set_title(ti); a.set_xlabel("Years"); a.set_ylabel(yl)
        a.grid(True, alpha=.3); a.legend(fontsize=8)

    for row, (tag, idx) in enumerate(sel):
        ov(ax[row,0], emis_nol, emis_part, idx, "Emissions, GtC/yr",
           f"{tag} ({scen[idx]}, a={alpha[idx]:.3f}) - Emissions")
        ov(ax[row,1], nol["Temp"], part["Temp"], idx, "Temperature, C",
           f"{tag} - Temperature", gl=True)

    fig.tight_layout(rect=[0,0,1,.97])
    out = os.path.join(DATA_DIR, f"alpha_extremes_{GRID_LABEL}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("saved ->", out)

    # welfare split (uses RAW quantities — never the smoothed emissions)
    disc = ZEIT * np.exp(-ZEIT * DISRATE * np.arange(N_T))
    disc[-1] /= (1 - np.exp(-ZEIT * DISRATE))
    Rp = risk_for(part["_text"], scen, N_T)
    Rn = risk_for(nol["_text"],  scen, N_T)

    def parts(util_dict, Rmat, idx):
        u = util_dict[scen[idx]]
        r = np.exp(-SCALERA) * np.sum(disc * Rmat[:, idx])
        return u, r

    print("\nWelfare split (mitigation = UtilityLeft, risk = discounted risk term):")
    for tag, idx in sel:
        ub, rb = parts(nol["UtilityLeft"], Rn, idx)
        ua, ra = parts(part["UtilityLeft"], Rp, idx)
        print(f"\n{tag} ({scen[idx]}, a={alpha[idx]:.4f}):")
        print(f"  mitigation U:  before={ub:+.4f}  after={ua:+.4f}  dU={ua-ub:+.4f}")
        print(f"  risk term  R:  before={rb:+.4f}  after={ra:+.4f}  dR(gain)={rb-ra:+.4f}")
        print(f"  net welfare gain from learning: {(ua-ra)-(ub-rb):+.4f}")


if __name__ == "__main__":
    main()
