"""Augment each case's slice_centerSpan.pvtu with derived fields:
  - Re_Omega = d^2 * |omega| / nu
  - Gamma    = 2 (omega d)^2 / (|U|^2 + (omega d)^2)
  - lambda_p = -d^2 (u . grad_p) / (rho nu |u|^2)  -- local streamwise PG sensor
  - amp_rate = a(Re_Omega, Gamma, lambda_p)  -- kernel rate (FPG-cliff only, no sigma_FPG)
              (matches Flow360 SAAftTransition.h::__aftRate)
Writes slice_with_derived.pvtu next to slice_centerSpan.pvtu.
"""
import os, sys, json, numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

# Kernel constants — must stay in sync with Flow360 ModelConstants.h.
G_C, SLOPE, BARRIER_POWER, A_MAX = 1.572, 5.263, 4.0, 0.15
RE_OMEGA_FLOOR = 100.0
TILT_SLOPE = 1.0e6  # effectively disabled (no slanted cliff)
LAMBDA_STAR = 0.64
LAMBDA_SLOPE = 4.56
CLIFF_LAMBDA_SLOPE = 10.0  # K_λ — same as Flow360 aft_cliffLambdaSlope

def detect_nu(case_dir):
    """Stored ν = muRef = Mach/Re_chord from Flow360.json."""
    d = json.load(open(f"{case_dir}/Flow360.json"))
    return float(d['freestream']['muRef'])

def rate(Re_O, Gamma, lambda_p):
    """SA-AI kernel: FPG-modulated onset-delay cliff only (no sigma_FPG). Matches
    Flow360 SAAftTransition.h::__aftRate (modulo float32 rounding).
        Re_Ω_cliff(λ_p) = floor · exp(K_λ · max(0, λ_p))  (FPG-cliff)
        rate            = A_MAX · sigmoid(z)
        z               = SLOPE·(Γ - Γ_c) + ln(barrier)
    """
    log_floor = np.log10(RE_OMEGA_FLOOR)
    log_extra = np.where(Gamma < 1.0, (1.0 - Gamma) / TILT_SLOPE, 0.0)
    lambda_pos = np.maximum(lambda_p, 0.0)
    fpg_boost = CLIFF_LAMBDA_SLOPE * lambda_pos / np.log(10.0)
    Re_cliff = 10.0 ** (log_floor + log_extra + fpg_boost)
    safe = np.maximum(Re_O, Re_cliff + 1e-12)
    ratio = Re_cliff / safe
    bar_inside = np.maximum(1.0 - np.power(ratio, BARRIER_POWER), 1e-20)
    barr = np.where(Re_O > Re_cliff, np.log(bar_inside), -np.inf)
    z = SLOPE*(Gamma - G_C) + barr
    a_kernel = A_MAX/(1.0+np.exp(-z))
    # (sigma_FPG rate factor removed: favorable-PG is carried entirely by the
    #  onset-delay cliff Re_cliff(lambda_p) above; see paper Sec. calib.)
    return np.where(Re_O > Re_cliff, a_kernel, 0.0)

def augment(pvtu_path):
    case_dir = os.path.dirname(pvtu_path)
    nu = detect_nu(case_dir)
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(pvtu_path); r.Update()
    g = r.GetOutput()
    pd = g.GetPointData()
    have = {pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())}
    needed = {'vorticityMagnitude','wallDistance','velocity','p'}
    miss = needed - have
    if miss:
        return False, f"missing {miss}"
    # Compute grad(p) via VTK gradient filter.
    gf = vtk.vtkGradientFilter()
    gf.SetInputData(g)
    gf.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, 'p')
    gf.SetResultArrayName('grad_p'); gf.Update()
    g = gf.GetOutput(); pd = g.GetPointData()
    omega = vtk_to_numpy(pd.GetArray('vorticityMagnitude'))
    d = vtk_to_numpy(pd.GetArray('wallDistance'))
    v = vtk_to_numpy(pd.GetArray('velocity'))
    gradP = vtk_to_numpy(pd.GetArray('grad_p'))
    rho = vtk_to_numpy(pd.GetArray('rho'))
    U2 = v[:,0]**2 + v[:,1]**2 + v[:,2]**2
    U2_safe = np.maximum(U2, 1e-12)
    Re_O = d*d*omega / nu
    ome_d = omega * d
    U = np.sqrt(U2)
    Gamma = 2.0*ome_d**2 / (U**2 + ome_d**2 + 1e-30)
    u_dot_gp = v[:,0]*gradP[:,0] + v[:,1]*gradP[:,1] + v[:,2]*gradP[:,2]
    lambda_p = -d*d*u_dot_gp / (rho*nu*U2_safe)
    a = rate(Re_O, Gamma, lambda_p)
    for name, arr in [('Re_Omega', Re_O.astype(np.float32)),
                      ('Gamma', Gamma.astype(np.float32)),
                      ('lambda_p', lambda_p.astype(np.float32)),
                      ('amp_rate', a.astype(np.float32))]:
        va = numpy_to_vtk(arr, deep=True); va.SetName(name); pd.AddArray(va)
    out_pvtu = os.path.join(os.path.dirname(pvtu_path), 'slice_with_derived.pvtu')
    w = vtk.vtkXMLPUnstructuredGridWriter()
    w.SetFileName(out_pvtu); w.SetInputData(g)
    w.SetNumberOfPieces(1); w.SetStartPiece(0); w.SetEndPiece(0)
    w.SetDataModeToBinary()
    w.Write()
    return True, f"{g.GetNumberOfPoints()} pts"

def find_cases(base, family):
    """family is e.g. 'nlf0416_Re4M' or 'eppler387_Re200k'.  Returns dirs that
    actually exist on disk.  Hardcoded α-sets per family."""
    alpha_by_family = {
        'nlf0416_Re4M':   ['0','4','9','15'],
        'eppler387_Re200k': ['0','2','5','7'],
    }
    alphas = alpha_by_family.get(family, ['0','4','9','15'])
    cases = []
    for mesh in ['cav','str']:
        for L in ['L0','L1','L2']:
            for a in alphas:
                d = f'{base}/{mesh}{L}prop_{family}_a{a}'
                if os.path.exists(d): cases.append(d)
    return cases

if __name__ == '__main__':
    base = '/home/qiqi/flexcompute/aft-sa/flow360'
    families = sys.argv[1:] or ['nlf0416_Re4M', 'eppler387_Re200k']
    cases = []
    for fam in families:
        cases += find_cases(base, fam)
    print(f"found {len(cases)} cases ({', '.join(families)})\n")
    ok = 0
    for c in cases:
        p = os.path.join(c, 'slice_centerSpan.pvtu')
        if not os.path.exists(p):
            print(f"SKIP {os.path.basename(c)}: no slice_centerSpan.pvtu"); continue
        success, msg = augment(p)
        tag = "OK" if success else "FAIL"
        print(f"{tag:>4s} {os.path.basename(c):>42s}  {msg}")
        if success: ok += 1
    print(f"\n{ok}/{len(cases)} augmented")
