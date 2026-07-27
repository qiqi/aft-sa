"""Analyze the drag-crisis PILOT cylinder URANS case (Re_D=1e5, M=0.1).

Extracts, from the production window (default: physical step 6000 on):
  - mean Cd, mean CL, rms CL, rms CD from total_forces_v2.csv
    (last pseudo-step row per physical step = the converged step value);
  - St from the CL spectrum peak (dt = 0.1 solver units, U = M = 0.1
    => f_D/U = f_solver * D / U = f_solver / 0.1);
  - laminar separation (and any reattachment) angles from the sign of the
    TIME-AVERAGED tangential Cf on surface_fluid_cylinder_time_avg.pvtu
    (CfVec projected on the front-stagnation-to-base tangent, per side);
  - the base-pressure coefficient and mean Cp(phi) from the same file;
  - the max-chi(phi) front from slice_centerSpan_time_avg.pvtu
    (near-wall band, chi = nuHat/nu, solver chi=1 crossing convention).

Angle convention: phi = 0 at the FRONT stagnation point, 180 deg at the rear;
upper (z>0) and lower (z<0) sides reported separately. Subcritical targets:
Cd 1.2 +/- 0.05 (2D URANS biases HIGH), St 0.19-0.20, laminar separation
~78-80 deg (feasibility note Sec. 1-2). Writes a diagnostic figure to
paper/repro/cfd/figs_explore/ (exploratory only, NOT a paper figure) and a
JSON summary next to it.

Usage: python dragcrisis_pilot_forces.py [CASE_DIR] [--avg-start 6000]
"""
import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy

HERE = os.path.dirname(os.path.abspath(__file__))
FIGD = os.path.join(HERE, "figs_explore")
DT = 0.1          # solver time units per physical step
U_INF = 0.1       # = Mach (a_inf = 1)
CENTER = (0.5, 0.0)


# ---------------------------------------------------------------------------
def load_forces(case_dir):
    """(step, CL, CD) using the LAST pseudo-step row of each physical step."""
    rows = {}
    with open(os.path.join(case_dir, "total_forces_v2.csv")) as f:
        r = csv.reader(f)
        hdr = [h.strip() for h in next(r)]
        iPS, iCL, iCD = (hdr.index(k) for k in ("physical_step", "CL", "CD"))
        for row in r:
            if len(row) <= max(iPS, iCL, iCD):
                continue
            try:
                rows[int(float(row[iPS]))] = (float(row[iCL]), float(row[iCD]))
            except ValueError:
                continue
    steps = np.array(sorted(rows))
    cl = np.array([rows[s][0] for s in steps])
    cd = np.array([rows[s][1] for s in steps])
    return steps, cl, cd


def strouhal(steps, cl, avg_start):
    m = steps >= avg_start
    t = steps[m] * DT
    y = cl[m] - cl[m].mean()
    if len(t) < 64:
        return float("nan"), None
    # uniform sampling: dt per physical step
    freq = np.fft.rfftfreq(len(y), d=DT)
    amp = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    ipk = 1 + np.argmax(amp[1:])
    St = freq[ipk] / U_INF          # f D / U, D = 1
    return float(St), (freq / U_INF, amp)


# ---------------------------------------------------------------------------
def _read_pvtu(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(path)
    r.Update()
    return r.GetOutput()


def surface_mean_cfcp(case_dir, avg=True):
    """phi[deg], Cf_s (attached>0), Cp per side from the (time-avg) surface file.

    Returns {'upper': (phi, cf_s, cp), 'lower': ...}. Cf sign convention:
    tangential Cf projected on the front->rear flow direction of that side.
    """
    name = ("surface_fluid_cylinder_time_avg.pvtu" if avg
            else "surface_fluid_cylinder.pvtu")
    g = _read_pvtu(os.path.join(case_dir, name))
    pd = g.GetPointData()
    P = vtk_to_numpy(g.GetPoints().GetData())
    names = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
    cfv = vtk_to_numpy(pd.GetArray(next(n for n in names if n.startswith("CfVec"))))
    cp = vtk_to_numpy(pd.GetArray(next(n for n in names if n.startswith("Cp"))))
    # dedup the two span planes by (x, z)
    key = np.stack([np.round(P[:, 0], 9), np.round(P[:, 2], 9)], axis=1)
    _, idx = np.unique(key, axis=0, return_index=True)
    x, z = P[idx, 0], P[idx, 2]
    cfv, cp = cfv[idx], cp[idx]
    th = np.arctan2(z, x - CENTER[0])            # 0 at rear, +-pi at front
    phi = 180.0 - np.abs(np.degrees(th))         # 0 front stagnation -> 180 base
    # ccw tangent (d/dtheta): flow front->rear is -theta_hat on upper, +theta_hat on lower
    tccw = np.stack([-np.sin(th), np.cos(th)], axis=1)
    cf_t = cfv[:, 0] * tccw[:, 0] + cfv[:, 2] * tccw[:, 1]
    out = {}
    for side, m in (("upper", z >= 0), ("lower", z < 0)):
        sgn = -1.0 if side == "upper" else 1.0
        o = np.argsort(phi[m])
        out[side] = (phi[m][o], (sgn * cf_t[m])[o], cp[m][o])
    return out


def cf_knee(phi, cf_s, frac=0.2, phi_search_min=50.0):
    """Separation marker validated on the pilot: the angle where cf_s first
    falls below ``frac`` of the laminar peak (mean-Cf zero crossings are
    rectified away under shedding; see the 2006 pilot record)."""
    pk = cf_s[(phi > 30) & (phi < 70)].max()
    m = phi > phi_search_min
    if not (cf_s[m] < frac * pk).any():
        return None
    return float(phi[m][np.argmax(cf_s[m] < frac * pk)])


def crossings(phi, cf_s, phi_min=10.0):
    """Sign-change angles of cf_s (aft of phi_min), linearly interpolated."""
    res = []
    for i in range(len(phi) - 1):
        if phi[i] < phi_min:
            continue
        a, b = cf_s[i], cf_s[i + 1]
        if a == 0 or a * b >= 0:
            continue
        w = a / (a - b)
        res.append((float(phi[i] + w * (phi[i + 1] - phi[i])),
                    "separation" if a > 0 else "reattachment"))
    return res


def chi_front(case_dir, nu, band=2.0e-3, avg=True):
    """max chi in the near-wall band per 1-deg phi bin (both sides pooled)."""
    name = ("slice_centerSpan_time_avg.pvtu" if avg else "slice_centerSpan.pvtu")
    g = _read_pvtu(os.path.join(case_dir, name))
    pd = g.GetPointData()
    P = vtk_to_numpy(g.GetPoints().GetData())
    names = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
    nuh = vtk_to_numpy(pd.GetArray(next(n for n in names if n.startswith("nuHat"))))
    d = vtk_to_numpy(pd.GetArray(next(n for n in names if n.startswith("wallDistance"))))
    m = d < band
    x, z = P[m, 0], P[m, 2]
    chi = nuh[m] / nu
    th = np.arctan2(z, x - CENTER[0])
    phi = 180.0 - np.abs(np.degrees(th))
    out = {}
    for side, mm in (("upper", z >= 0), ("lower", z < 0)):
        bins = np.arange(0, 181, 1.0)
        idx = np.digitize(phi[mm], bins)
        prof = np.full(len(bins), np.nan)
        for b in np.unique(idx):
            if 0 < b <= len(bins):
                prof[b - 1] = chi[mm][idx == b].max()
        out[side] = (bins, prof)
    return out


# ---------------------------------------------------------------------------
def load_pseudo_forces(case_dir):
    """(pseudo_step, CL, CD) rows of a STEADY run's force history."""
    ps, cl, cd = [], [], []
    with open(os.path.join(case_dir, "total_forces_v2.csv")) as f:
        r = csv.reader(f)
        hdr = [h.strip() for h in next(r)]
        iP, iL, iD = (hdr.index(k) for k in ("pseudo_step", "CL", "CD"))
        for row in r:
            try:
                ps.append(int(float(row[iP])))
                cl.append(float(row[iL]))
                cd.append(float(row[iD]))
            except (ValueError, IndexError):
                continue
    return np.array(ps), np.array(cl), np.array(cd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case_dir", nargs="?",
                    default="/local_data/qiqi/sa-ai/dragcrisis_pilot/cylL1_Re100k")
    ap.add_argument("--avg-start", type=int, default=6000)
    ap.add_argument("--steady", action="store_true",
                    help="steady pseudo-transient twin: report the tail MEDIAN "
                         "+ limit-cycle amplitude of the pseudo-step force "
                         "history; surface/slice = the final written snapshot")
    args = ap.parse_args()
    cd_dir = args.case_dir
    nu = json.load(open(os.path.join(cd_dir, "Flow360.json")))["freestream"]["muRef"]

    if args.steady:
        ps, cl, cdrag = load_pseudo_forces(cd_dir)
        tail = max(5000, len(ps) // 4)
        clw, cdw = cl[-tail:], cdrag[-tail:]
        steps, spec, St = ps, None, float("nan")
        summary = {
            "steady": True, "n_pseudo_steps": int(ps[-1]),
            "tail_rows": int(tail),
            "Cd_median": float(np.median(cdw)), "Cd_tail_std": float(cdw.std()),
            "Cd_tail_p2p": float(cdw.max() - cdw.min()),
            "CL_median": float(np.median(clw)), "CL_tail_std": float(clw.std()),
            "CL_tail_p2p": float(clw.max() - clw.min()),
        }
        m = np.zeros(len(ps), bool)
        m[-tail:] = True
    else:
        steps, cl, cdrag = load_forces(cd_dir)
        m = steps >= args.avg_start
        St, spec = strouhal(steps, cl, args.avg_start)
        summary = {
            "n_steps": int(steps[-1] + 1),
            "avg_window_steps": [int(args.avg_start), int(steps[-1])],
            "avg_window_periods_at_St": float(
                (steps[-1] - args.avg_start) * DT * St * U_INF)
            if np.isfinite(St) else None,
            "Cd_mean": float(cdrag[m].mean()), "Cd_rms": float(cdrag[m].std()),
            "CL_mean": float(cl[m].mean()), "CL_rms": float(cl[m].std()),
            "St": St,
        }

    try:
        sides = surface_mean_cfcp(cd_dir, avg=not args.steady)
        for side in ("upper", "lower"):
            phi, cf_s, cp = sides[side]
            summary[f"cf_crossings_{side}"] = crossings(phi, cf_s)
            summary[f"Cp_base_{side}"] = float(np.interp(180.0, phi, cp))
            i = np.argmin(cp)
            summary[f"Cp_min_{side}"] = float(cp[i])
            summary[f"phi_Cp_min_{side}"] = float(phi[i])
    except (FileNotFoundError, StopIteration, OSError) as e:
        sides = None
        summary["surface_avg_note"] = f"surface file not usable: {e}"

    try:
        fronts = chi_front(cd_dir, nu, avg=not args.steady)
        for side in ("upper", "lower"):
            b, prof = fronts[side]
            ok = np.isfinite(prof)
            above = ok & (prof >= 1.0)
            summary[f"chi1_front_phi_{side}"] = (
                float(b[above].min()) if above.any() else None)
    except (FileNotFoundError, StopIteration, OSError) as e:
        fronts = None
        summary["chi_front_note"] = f"time-avg slice not usable: {e}"

    os.makedirs(FIGD, exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    ax = axs[0, 0]
    if args.steady:
        t = np.arange(len(cl))
        ax.set_xlabel("force-history row (pseudo-stepping)")
        ax.set_title("steady pseudo-transient force history")
    else:
        t = steps * DT * U_INF        # in D/U units
        ax.axvspan(args.avg_start * DT * U_INF, t[-1], color="0.9", zorder=0)
        ax.set_xlabel("t U / D")
        ax.set_title("force history (shaded = averaging window)")
    ax.plot(t, cdrag, lw=0.8, color="#1f77b4", label="Cd")
    ax.plot(t, cl, lw=0.8, color="#d62728", label="CL")
    ax.set_ylabel("coefficient")
    ax.legend(frameon=False)
    ax = axs[0, 1]
    if spec is not None:
        fSt, amp = spec
        ax.plot(fSt, amp, lw=1.0, color="#1f77b4")
        ax.axvline(St, color="#d62728", lw=0.8)
        ax.set_xlim(0, 1.0); ax.set_xlabel("f D / U"); ax.set_ylabel("|CL| spectrum")
        ax.set_title(f"St = {St:.3f}")
    ax = axs[1, 0]
    if sides:
        for side, c in (("upper", "#1f77b4"), ("lower", "#d62728")):
            phi, cf_s, cp = sides[side]
            ax.plot(phi, cf_s, lw=1.2, color=c, label=side)
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_xlabel("phi from front stagnation [deg]")
        ax.set_ylabel("mean tangential Cf (attached > 0)")
        ax.legend(frameon=False); ax.set_title("time-averaged Cf")
    ax = axs[1, 1]
    if fronts:
        for side, c in (("upper", "#1f77b4"), ("lower", "#d62728")):
            b, prof = fronts[side]
            ax.semilogy(b, prof, lw=1.2, color=c, label=side)
        ax.axhline(1.0, color="0.6", lw=0.6)
        ax.set_xlabel("phi [deg]"); ax.set_ylabel("max chi in near-wall band")
        ax.legend(frameon=False); ax.set_title("chi front (time-avg slice)")
    fig.suptitle(os.path.basename(cd_dir))
    fig.tight_layout()
    sfx = "_steady" if args.steady else ""
    fp = os.path.join(FIGD, f"dragcrisis_pilot_forces{sfx}.png")
    fig.savefig(fp, dpi=140)
    out = os.path.join(FIGD, f"dragcrisis_pilot_summary{sfx}.json")
    json.dump(summary, open(out, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print(f"figure: {fp}\nsummary: {out}")


if __name__ == "__main__":
    main()
