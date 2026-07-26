#!/usr/bin/env python3
"""Extract chi / Cf vs Re_theta from OpenFOAM SA-AI flat-plate cases and
compare against the paper's references (AGS correlation, Schubauer-Skramstad
band), mirroring sa-ai/paper/regen_flatplate_flow360.py.

Reads the latest time's fields via foamToVTK output (legacy .vtk or .vtu with
internal cell data). Requires the compute venv python (vtk) OR plain numpy if
using the raw-field reader below (structured single-block mesh => we can read
OpenFOAM ascii/binary fields directly with the known cell ordering:
i fastest (x), then j (y, 1 cell), then k (z)).

Outputs: transition_summary.csv + flatplate_openfoam.pdf in sa-ai/openfoam/.
"""
import glob
import os
import re
import struct
import sys

import numpy as np

CASE_ROOT = "/local_data/qiqi/openfoam-sa-ai/cases"
OUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NU = 1.0e-6
NX, NZ = 320, 80
LX = 6.0
OUTLET_MARGIN = 0.5
TU_LIST = [0.04, 0.08, 0.16, 0.30, 0.60]


def AGS_Reth(tu_pct):
    """Abu-Ghannam & Shaw (1980) zero-PG transition-onset Re_theta (Tu in %)."""
    return 163.0 + np.exp(6.91 - tu_pct)


SS_BAND = [(0.026, 2.78, 3.82), (0.04, 2.80, 3.85), (0.08, 2.80, 3.88),
           (0.12, 2.62, 3.72), (0.16, 2.10, 3.25), (0.20, 1.82, 3.00),
           (0.24, 1.66, 2.90), (0.28, 1.55, 2.83), (0.32, 1.47, 2.76),
           (0.342, 1.42, 2.70)]


def latest_time(cd):
    ts = []
    for d in os.listdir(cd):
        try:
            t = float(d)
            if t > 0:
                ts.append((t, d))
        except ValueError:
            pass
    return max(ts)[1] if ts else None


def read_scalar_field(path, n):
    """Read an OpenFOAM volScalarField internalField (ascii or binary)."""
    with open(path, "rb") as f:
        data = f.read()
    m = re.search(rb"internalField\s+nonuniform\s+List<scalar>\s*\n(\d+)\s*\n\(", data)
    if m is None:
        mu = re.search(rb"internalField\s+uniform\s+([0-9eE.+-]+)", data)
        if mu:
            return np.full(n, float(mu.group(1)))
        raise ValueError(f"no internalField in {path}")
    cnt = int(m.group(1))
    assert cnt == n, (cnt, n)
    start = m.end()
    if b"format      ascii" in data[:1000] or b"format ascii" in data[:1000]:
        body = data[start:data.index(b")", start)].split()
        return np.array([float(x) for x in body])
    return np.frombuffer(data, dtype="<f8", count=n, offset=start)


def read_vector_field(path, n):
    with open(path, "rb") as f:
        data = f.read()
    m = re.search(rb"internalField\s+nonuniform\s+List<vector>\s*\n(\d+)\s*\n\(", data)
    if m is None:
        mu = re.search(rb"internalField\s+uniform\s+\(([^)]*)\)", data)
        if mu:
            v = np.array([float(x) for x in mu.group(1).split()])
            return np.tile(v, (n, 1))
        raise ValueError(f"no internalField in {path}")
    cnt = int(m.group(1))
    assert cnt == n, (cnt, n)
    start = m.end()
    head = data[:1000]
    if b"ascii" in head:
        seg = data[start:data.index(b"\n)\n", start)]
        vals = re.findall(rb"\(([^)]*)\)", seg)
        return np.array([[float(x) for x in v.split()] for v in vals])
    return np.frombuffer(data, dtype="<f8", count=3 * n, offset=start).reshape(n, 3)


def cell_coords():
    """Cell-center coordinates from the known grading (matches blockMeshDict)."""
    def geom(d0, r, ncell):
        edges = np.concatenate([[0.0], np.cumsum(d0 * r ** np.arange(ncell))])
        return 0.5 * (edges[:-1] + edges[1:]), np.diff(edges)

    def solve_ratio(dx0, ncell, L, lo=1.0 + 1e-9, hi=1.5):
        f = lambda r: dx0 * (r ** ncell - 1.0) / (r - 1.0) - L
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            (lo, hi) = (mid, hi) if f(mid) < 0.0 else (lo, mid)
        return 0.5 * (lo + hi)

    rx = solve_ratio(8.0e-4, NX, LX)
    xc, dxc = geom(8.0e-4, rx, NX)
    zc, dzc = geom(7.0e-6, 1.12, NZ)
    return xc, zc, dzc


def profiles(cd):
    """Return x, Re_theta(x), Cf(x), maxchi(x) from the latest written time."""
    t = latest_time(cd)
    if t is None:
        return None
    n = NX * NZ
    U = read_vector_field(f"{cd}/{t}/U", n)
    nuT = read_scalar_field(f"{cd}/{t}/nuTilda", n)
    xc, zc, dzc = cell_coords()
    # OpenFOAM single hex block ordering: x fastest, then y (1 cell), then z
    u = U[:, 0].reshape(NZ, NX)
    chi = (nuT / NU).reshape(NZ, NX)
    # Edge velocity = U_inf = 1 (unit problem). Clip u to [0, 1] so the mild
    # numerical overshoot above the BL edge (u up to ~1.06 near the LE)
    # contributes zero to the momentum-thickness integrand instead of
    # negative area.
    uu = np.clip(u, 0.0, 1.0)
    theta = np.sum(uu * (1.0 - uu) * dzc[:, None], axis=0)
    Re_theta = theta / NU
    # Cf from wall-adjacent cell: tau_w = nu * u1 / z1 (low-Re resolved)
    Cf = 2.0 * NU * u[0, :] / zc[0]
    maxchi = chi.max(axis=0)
    return xc, Re_theta, Cf, maxchi


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    rows = []
    for Tu in TU_LIST:
        cd = os.path.join(CASE_ROOT, f"flatplate_Tu{int(round(Tu * 1000)):04d}")
        r = profiles(cd)
        if r is None:
            print(f"skip {cd} (no results)")
            continue
        x, Re_theta, Cf, maxchi = r
        keep = x < LX - OUTLET_MARGIN
        x, Re_theta, Cf, maxchi = x[keep], Re_theta[keep], Cf[keep], maxchi[keep]
        # transition metrics: chi=1 crossing; Cf-min (onset) and Cf-rise
        Reth_chi1 = np.nan
        ix = np.where(maxchi >= 1.0)[0]
        if len(ix) and ix[0] > 0:
            i = ix[0]
            w = (1.0 - maxchi[i - 1]) / (maxchi[i] - maxchi[i - 1])
            Reth_chi1 = Re_theta[i - 1] + w * (Re_theta[i] - Re_theta[i - 1])
        i_cfmin = int(np.argmin(Cf + 1e9 * (x < 0.05)))
        Rex_cfmin = x[i_cfmin] * 1e6
        rows.append((Tu, Reth_chi1, AGS_Reth(Tu), Rex_cfmin))
        axes[0].semilogy(Re_theta, maxchi, label=f"Tu={Tu}%")
        axes[1].semilogy(Re_theta, Cf, label=f"Tu={Tu}%")
        if np.isfinite(Reth_chi1):
            axes[0].axvline(AGS_Reth(Tu), ls=":", color="gray", lw=0.8)
    axes[0].axhline(1.0, color="k", lw=0.6, ls="--")
    axes[0].set_xlabel(r"$Re_\theta$"); axes[0].set_ylabel(r"max$_z\,\chi$")
    axes[1].set_xlabel(r"$Re_\theta$"); axes[1].set_ylabel(r"$C_f$")
    # Blasius + turbulent correlations on Cf panel (vs Re_theta):
    Reth = np.logspace(1.7, 3.7, 100)
    axes[1].plot(Reth, 0.4410 / Reth, "k--", lw=0.8, label="Blasius")
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "flatplate_openfoam.pdf")
    fig.savefig(out)
    print(f"wrote {out}")
    print(f"{'Tu%':>6} {'Reth(chi=1)':>12} {'AGS':>8} {'Rex(Cf min)':>12}")
    with open(os.path.join(OUT_DIR, "transition_summary.csv"), "w") as f:
        f.write("Tu_pct,Reth_chi1,Reth_AGS,Rex_cfmin\n")
        for Tu, r1, ags, rx in rows:
            print(f"{Tu:6.2f} {r1:12.1f} {ags:8.1f} {rx:12.3e}")
            f.write(f"{Tu},{r1:.1f},{ags:.1f},{rx:.4e}\n")


if __name__ == "__main__":
    main()
