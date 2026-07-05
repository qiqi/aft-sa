"""Build the PROPER Construct2D refinement ladder (L0..L4).
At each level, write a fresh airfoil .dat from the same shape source as cavity
(spline-interpolated to N_L cosine-clustered points), then script Construct2D
via stdin to set parameters and generate the grid.
"""
import os, sys, subprocess, shutil, time, json
sys.path.insert(0, "/home/qiqi/flexcompute/aft-sa/flow360")
from naca_contour import generate_contour

CONSTRUCT2D = "/home/qiqi/flexcompute/aft-sa/external/construct2d/construct2d"
WORKBASE = "/home/qiqi/flexcompute/aft-sa/external/construct2d"
OUTBASE = "/home/qiqi/flexcompute/aft-sa/flow360"
import numpy as np

# At each level we set:
# - nsrf (output surface points after Construct2D's redistribution; should match N for the level)
# - jmax (wall-normal points; tuned to give first-cell h0 with implied growth approaching 1)
# - radi (farfield radius, 100 chords)
# - ypls (target y+; we override with our explicit h0 via the cfrc parameter or by adjusting ypls)
# - lesp, tesp (LE/TE spacing on the surface)
# Construct2D expects ypls (target y+) and Re; from those it sets first-cell height
# To get our specific h0, we set ypls such that ypls / (Re·sqrt(Cf_flat/2)) ≈ h0
# Cf_flat = 0.026 / Re^(1/7) at Re=1e6 → 0.00440; sqrt(Cf/2) = 0.0469
# So h0 ≈ ypls / (1e6 · 0.0469) = ypls · 2.13e-5
# To get h0 = h0_desired: ypls = h0_desired / 2.13e-5
LEVELS = [
    # (tag, N, jmax, h0_target, lesp_target, far_r)
    ('L0',  200,  80,  14.0e-6, 0.0040, 100.0),
    ('L1',  400, 160,   7.0e-6, 0.0020, 100.0),
    ('L2',  800, 320,   3.5e-6, 0.0010, 100.0),
    ('L3', 1600, 640,   1.75e-6, 0.0005, 100.0),
    ('L4', 3200, 1280, 0.875e-6, 0.00025, 100.0),
]
RE = 1e6
def ypls_for_h0(h0): return h0 / 2.13e-5  # invert wall_h0 formula

def write_airfoil_dat(N, path):
    """Write Construct2D-format .dat file from refined NACA0012 contour."""
    c = generate_contour(N)
    with open(path, 'w') as f:
        f.write(f"naca0012_L{N}\n")
        for x, z in c:
            f.write(f"{x:12.8f}  {z:12.8f}\n")

def script_construct2d(project, lesp, tesp, nsrf_target, radi, jmax, ypls, recd, work_cwd):
    """Run Construct2D interactively via stdin. Send commands to set params + generate grid.
    SMTH mode: Construct2D smooths/redistributes the contour internally.
    NWKE=0 to remove the wake-region duplicate point pairs around TE."""
    cmds = [
        f"{project}.dat",  # load airfoil (with .dat extension required)
        "SOPT",            # surface options
        "NSRF", str(nsrf_target),
        "LESP", f"{lesp:.6e}",
        "TESP", f"{tesp:.6e}",
        "RADI", f"{radi:.4f}",
        "NWKE", "0",       # no wake-region addition
        "QUIT",            # back to main
        "VOPT",            # volume options
        "JMAX", str(jmax),
        "YPLS", f"{ypls:.6e}",
        "RECD", f"{recd:.1f}",
        "QUIT",            # back to main
        "GRID",            # generate
        "SMTH",            # smoothed airfoil (default; works with sharp TE)
        "QUIT",            # exit
    ]
    inp = "\n".join(cmds) + "\n"
    proc = subprocess.run([CONSTRUCT2D],
                          input=inp, capture_output=True, text=True,
                          cwd=work_cwd, timeout=600)
    return proc

ncells_record = {}
t_total = time.time()

for tag, N, jmax, h0_target, lesp_target, far_r in LEVELS:
    project = f"proper_struct_{tag}"
    t = time.time()
    print(f"\n=== building struct {tag}: N={N}, jmax={jmax}, h0_target={h0_target:.3e}, lesp={lesp_target}, R={far_r} ===", flush=True)
    # Write airfoil .dat in Construct2D's working dir
    dat_path = f"{WORKBASE}/{project}.dat"
    write_airfoil_dat(N, dat_path)
    # Use ypls = h0_target / 2.13e-5
    ypls = ypls_for_h0(h0_target)
    tesp = lesp_target * 0.063  # match construct2d default ratio tesp/lesp
    # nsrf = TOTAL points on airfoil after redistribution (around upper+lower)
    nsrf_target = N
    # Run Construct2D
    try:
        proc = script_construct2d(project, lesp_target, tesp, nsrf_target, far_r,
                                   jmax, ypls, RE, WORKBASE)
        out_p3d = f"{WORKBASE}/{project}.p3d"
        if not os.path.exists(out_p3d):
            print(f"  FAIL {tag}: .p3d not produced")
            print(f"  stdout tail: {proc.stdout[-500:] if proc.stdout else 'empty'}")
            print(f"  stderr tail: {proc.stderr[-500:] if proc.stderr else 'empty'}")
            continue
        # parse cells count
        with open(out_p3d) as f:
            head = f.readline().split() + f.readline().split()
        ni = int(head[0]); jmax_actual = int(head[1])
        ncells = (ni - 1) * (jmax_actual - 1)
        print(f"  built {tag}: ni={ni}, jmax={jmax_actual}, ncells2d={ncells}, t={time.time()-t:.0f}s", flush=True)
        ncells_record[tag] = ncells
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT building {tag}")
    except Exception as e:
        print(f"  EXC building {tag}: {e}")

json.dump(ncells_record, open(f"{OUTBASE}/proper_struct_ncells.json", 'w'), indent=1)
print(f"\nTotal time: {time.time()-t_total:.0f}s")
print(f"struct cell counts: {ncells_record}")
