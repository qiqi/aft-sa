"""External convergence monitor based on transition-location stability.

Strategy:
  1. Run a batch of pseudo-steps with restart from current state.
  2. Extract x_tr on upper + lower surface (first x where near-wall chi > 1).
  3. Save to history file.
  4. If |x_tr - x_tr_prev| < tol_x on BOTH sides for 2 consecutive batches,
     declare converged.  Otherwise run another batch.

Usage: python converge_by_xtr.py <case_dir> [--batch N] [--tol 0.01] [--max-batches K]

History file: <case_dir>/xtr_history.csv  (step, x_tr_upper, x_tr_lower)
"""
import os, sys, json, shutil, time, argparse, csv
import numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

sys.path.insert(0, '/home/qiqi/flexcompute/aft-sa/paper')
sys.path.insert(0, '/home/qiqi/flexcompute/flexfoil/rans')

# Defer heavy imports to runtime
def _import_runner():
    from regen_nlf_v2 import walk_contour_xz, load_slice_derived  # noqa
    from rans.env import make_env  # noqa
    from rans.solve import run_solver  # noqa
    return walk_contour_xz, make_env, run_solver

NU = 2.5e-8  # Flow360 stored ν for Re=4M cases (Mach=0.1, muRef=2.5e-8)

def get_current_step(cd):
    """Read last pseudo-step from nonlinear_residual_v2.csv."""
    p = f"{cd}/nonlinear_residual_v2.csv"
    if not os.path.exists(p): return 0
    last = None
    for ln in open(p):
        s = ln.strip()
        if not s or s.startswith('physical'): continue
        try: last = int(s.split(',')[1])
        except: pass
    return last or 0

def extract_xtr(cd, walk_contour_xz, chi_thresh=1.0, wd_max=1e-3):
    """Find x_tr on upper + lower of the airfoil case.
    Probe each surface point with a short wall-normal stencil; take the max
    chi over the stencil; find the first x where this max > chi_thresh."""
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(f"{cd}/volume.pvtu"); r.Update()
    g = r.GetOutput(); pd = g.GetPointData()
    pts = vtk_to_numpy(g.GetPoints().GetData())
    chi = vtk_to_numpy(pd.GetArray('nuHat')) / NU
    wd = vtk_to_numpy(pd.GetArray('wallDistance'))
    Xm, Zm, up_idx, lo_idx = walk_contour_xz(cd)
    x_up = Xm[up_idx]; z_up = Zm[up_idx]
    x_lo = Xm[lo_idx]; z_lo = Zm[lo_idx]
    out = {}
    for side, x_s, z_s in [('upper', x_up, z_up), ('lower', x_lo, z_lo)]:
        order = np.argsort(x_s)
        x_s, z_s = x_s[order], z_s[order]
        # x grid step 0.005 along surface
        x_grid = np.linspace(0.0, 1.0, 201)
        chi_max = np.full_like(x_grid, np.nan)
        for i, x_t in enumerate(x_grid):
            z_surf = float(np.interp(x_t, x_s, z_s))
            if side == 'upper':
                m = (np.abs(pts[:,0]-x_t)<0.005) & (pts[:,2]>z_surf-0.001) & \
                    (pts[:,2]<z_surf+wd_max) & (wd>0) & (wd<wd_max)
            else:
                m = (np.abs(pts[:,0]-x_t)<0.005) & (pts[:,2]<z_surf+0.001) & \
                    (pts[:,2]>z_surf-wd_max) & (wd>0) & (wd<wd_max)
            if m.sum() >= 3:
                chi_max[i] = chi[m].max()
        # First crossing
        cross = np.where(chi_max > chi_thresh)[0]
        if len(cross) == 0:
            out[side] = 1.0  # never transitions
        else:
            i = cross[0]
            # Refine via linear interpolation between i-1 and i if both valid
            if i > 0 and np.isfinite(chi_max[i-1]) and np.isfinite(chi_max[i]):
                lo_c, hi_c = chi_max[i-1], chi_max[i]
                if hi_c > lo_c:
                    frac = (chi_thresh - lo_c) / (hi_c - lo_c)
                    out[side] = x_grid[i-1] + frac*(x_grid[i] - x_grid[i-1])
                else:
                    out[side] = x_grid[i]
            else:
                out[side] = x_grid[i]
    return out

def run_batch(cd, n_steps, walk_contour_xz, make_env, run_solver, gpu=0):
    """Set up restart, run n_steps more from current step, capture x_tr."""
    p = f"{cd}/Flow360.json"
    d = json.load(open(p))
    current_step = get_current_step(cd)
    target = current_step + n_steps
    d['runControl']['restart'] = current_step > 0
    d['timeStepping']['maxPseudoSteps'] = target
    d['volumeOutput']['animationFrequency'] = -1
    json.dump(d, open(p, 'w'), indent=1)
    # Copy restartOutput/* to case root for restart
    ro = f"{cd}/restartOutput"
    if d['runControl']['restart'] and os.path.exists(ro):
        for f in os.listdir(ro):
            shutil.copy2(f"{ro}/{f}", f"{cd}/{f}")
    env, find = make_env()
    env["AFT_SA"] = "1"; env["AFT_LAMINAR_SLOWDOWN"] = "0.01"
    t0 = time.time()
    run_solver(cd, find, env, gpu=gpu, timeout=14400)
    dt = time.time() - t0
    xtr = extract_xtr(cd, walk_contour_xz)
    new_step = get_current_step(cd)
    return new_step, dt, xtr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('case_dir')
    ap.add_argument('--batch', type=int, default=5000, help='pseudo-steps per batch')
    ap.add_argument('--tol', type=float, default=0.01, help='x_tr stability tolerance')
    ap.add_argument('--max-batches', type=int, default=30)
    ap.add_argument('--consec', type=int, default=2,
                    help='# consecutive stable batches required to declare convergence')
    ap.add_argument('--gpu', type=int, default=0, help='GPU index')
    args = ap.parse_args()

    walk_contour_xz, make_env, run_solver = _import_runner()
    cd = args.case_dir

    hist_path = f"{cd}/xtr_history.csv"
    hist = []
    if os.path.exists(hist_path):
        for r in csv.DictReader(open(hist_path)):
            hist.append({'step': int(r['step']),
                         'xtr_upper': float(r['xtr_upper']),
                         'xtr_lower': float(r['xtr_lower']),
                         'dt_s': float(r['dt_s'])})
        print(f"loaded {len(hist)} prior batches from {hist_path}")

    stable = 0
    for it in range(args.max_batches):
        print(f"\n=== batch {it+1}/{args.max_batches} ===", flush=True)
        step, dt, xtr = run_batch(cd, args.batch, walk_contour_xz, make_env, run_solver, gpu=args.gpu)
        print(f"  step {step}: dt={dt:.0f}s  x_tr_upper={xtr['upper']:.4f}  x_tr_lower={xtr['lower']:.4f}", flush=True)
        hist.append({'step': step, 'xtr_upper': xtr['upper'],
                     'xtr_lower': xtr['lower'], 'dt_s': dt})
        with open(hist_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['step', 'xtr_upper', 'xtr_lower', 'dt_s'])
            w.writeheader(); w.writerows(hist)
        # Check stability
        if len(hist) >= 2:
            du = abs(hist[-1]['xtr_upper'] - hist[-2]['xtr_upper'])
            dl = abs(hist[-1]['xtr_lower'] - hist[-2]['xtr_lower'])
            print(f"  Δx_tr_upper = {du:.4f}  Δx_tr_lower = {dl:.4f}  (tol={args.tol})", flush=True)
            if du < args.tol and dl < args.tol:
                stable += 1
                print(f"  stable for {stable}/{args.consec} consecutive batches", flush=True)
                if stable >= args.consec:
                    print(f"\nCONVERGED after {step} pseudo-steps")
                    return 0
            else:
                stable = 0
    print(f"\nNOT converged in {args.max_batches} batches; last x_tr = upper {hist[-1]['xtr_upper']:.3f}, lower {hist[-1]['xtr_lower']:.3f}")
    return 1

if __name__ == '__main__':
    sys.exit(main())
