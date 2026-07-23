"""Build the PROPER Construct2D refinement ladder L0..L2 for Eppler 387.

Mirrors build_nlf_proper_struct.py with E387-specific TE (h_te_L0 = 0.000833 c,
half the NLF h_te_L0 because E387's modified TE is half as thick).

After Construct2D writes proper_struct_eppler_L*.p3d, build_proper_struct_cases.py
turns each into a Flow360 CGNS case dir.
"""
import os, sys, subprocess, time, json
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/flow360")
from eppler_contour_te import generate_contour_te

CONSTRUCT2D = "/home/qiqi/flexcompute/sa-ai/external/construct2d/construct2d"
WORKBASE = "/home/qiqi/flexcompute/sa-ai/external/construct2d"
OUTBASE = "/home/qiqi/flexcompute/sa-ai/flow360_fr"

LEVELS = [
    # (tag, N,  jmax, h0_target, lesp_target, far_r, h_te,     r_te)
    ('L0',  200,  80,  14.0e-6, 0.0040, 100.0,  8.33e-4,  2.000),
    ('L1',  400, 160,   7.0e-6, 0.0020, 100.0,  4.17e-4,  1.500),
    ('L2',  800, 320,   3.5e-6, 0.0010, 100.0,  2.08e-4,  1.250),
]
RE = 2e5   # E387 design Re (LSB regime in the paper)

def ypls_for_h0(h0, Re):
    """Invert Construct2D's wall_h0 formula:  h0 ≈ ypls / (Re·sqrt(Cf_flat/2))."""
    Cf_flat = 0.026 / (Re ** (1/7))
    return h0 * Re * (Cf_flat / 2) ** 0.5

def write_airfoil_dat(N, h_te, r_te, path):
    c = generate_contour_te(N, h_te=h_te, r_te=r_te)
    with open(path, 'w') as f:
        f.write(f"eppler387_L{N}\n")
        for x, z in c:
            f.write(f"{x:12.8f}  {z:12.8f}\n")
    return len(c)

def script_construct2d(project, lesp, tesp, nsrf, radi, jmax, ypls, recd, cwd):
    cmds = [
        f"{project}.dat",
        "SOPT",
        "NSRF", str(nsrf),
        "LESP", f"{lesp:.6e}",
        "TESP", f"{tesp:.6e}",
        "RADI", f"{radi:.4f}",
        "NWKE", "0",
        "QUIT",
        "VOPT",
        "JMAX", str(jmax),
        "YPLS", f"{ypls:.6e}",
        "RECD", f"{recd:.1f}",
        "QUIT",
        "GRID",
        "SMTH",
        "QUIT",
    ]
    inp = "\n".join(cmds) + "\n"
    proc = subprocess.run([CONSTRUCT2D], input=inp, capture_output=True, text=True,
                         cwd=cwd, timeout=600)
    return proc

ncells_record = {}
t_total = time.time()
for tag, N, jmax, h0_target, lesp_target, far_r, h_te, r_te in LEVELS:
    project = f"proper_struct_eppler_{tag}"
    t = time.time()
    print(f"\n=== building E387 str {tag}: N={N}, jmax={jmax}, h0={h0_target:.3e}, "
          f"h_te={h_te:.3e}, r_te={r_te}, R={far_r} ===", flush=True)
    dat_path = f"{WORKBASE}/{project}.dat"
    n_written = write_airfoil_dat(N, h_te, r_te, dat_path)
    print(f"  wrote {dat_path} ({n_written} pts)", flush=True)
    ypls = ypls_for_h0(h0_target, RE)
    tesp = lesp_target * 0.063
    try:
        proc = script_construct2d(project, lesp_target, tesp, N, far_r, jmax, ypls, RE, WORKBASE)
        out_p3d = f"{WORKBASE}/{project}.p3d"
        if not os.path.exists(out_p3d):
            print(f"  FAIL {tag}: .p3d not produced")
            print(f"  stdout tail: {proc.stdout[-800:] if proc.stdout else 'empty'}")
            print(f"  stderr tail: {proc.stderr[-400:] if proc.stderr else 'empty'}")
            continue
        with open(out_p3d) as f:
            head = f.readline().split() + f.readline().split()
        ni, jmax_actual = int(head[0]), int(head[1])
        ncells = (ni - 1) * (jmax_actual - 1)
        print(f"  built {tag}: ni={ni}, jmax={jmax_actual}, ncells2d={ncells}, "
              f"t={time.time()-t:.0f}s", flush=True)
        ncells_record[tag] = ncells
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT {tag}")
    except Exception as e:
        print(f"  EXC {tag}: {e}")

json.dump(ncells_record, open(f"{OUTBASE}/eppler_struct_ncells.json", 'w'), indent=1)
print(f"\nTotal time: {time.time()-t_total:.0f}s")
print(f"E387 struct cell counts: {ncells_record}")
