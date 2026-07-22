"""Augment each case's slice_centerSpan.pvtu with the sphere-kernel derived
fields used by the Cf figures:
  - Re_Omega = d^2 |omega| / nu                                   (onset gate)
  - sph_X = |u|, sph_Y = omega d, sph_Z = 1/2 d^2 (n.grad omega)  (indicators)
  - Shat = Y/sqrt(X^2+Y^2),  g_coord = (Y-X-Z)/R,  P = Shat*g     (rate coord)
Writes slice_with_derived.pvtu next to slice_centerSpan.pvtu.  (The retired
Gamma / lambda_p / amp_rate fields of the old kernel are no longer produced.)
"""
import os, sys, json, numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

def detect_nu(case_dir):
    """Stored ν = muRef = Mach/Re_chord from Flow360.json."""
    d = json.load(open(f"{case_dir}/Flow360.json"))
    return float(d['freestream']['muRef'])

def augment(pvtu_path):
    case_dir = os.path.dirname(pvtu_path)
    nu = detect_nu(case_dir)
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(pvtu_path); r.Update()
    g = r.GetOutput()
    pd = g.GetPointData()
    have = {pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())}
    needed = {'vorticityMagnitude','wallDistance','velocity'}
    miss = needed - have
    if miss:
        return False, f"missing {miss}"
    # grad omega and grad d for the sphere curvature indicator Z, via chained
    # VTK gradient filters.
    def add_grad(dset, field, out):
        gf = vtk.vtkGradientFilter(); gf.SetInputData(dset)
        gf.SetInputArrayToProcess(0, 0, 0,
                                  vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, field)
        gf.SetResultArrayName(out); gf.Update()
        return gf.GetOutput()
    g = add_grad(g, 'vorticityMagnitude', 'grad_omega')
    g = add_grad(g, 'wallDistance', 'grad_d')
    pd = g.GetPointData()
    omega = vtk_to_numpy(pd.GetArray('vorticityMagnitude'))
    d = vtk_to_numpy(pd.GetArray('wallDistance'))
    v = vtk_to_numpy(pd.GetArray('velocity'))
    gradW = vtk_to_numpy(pd.GetArray('grad_omega'))
    gradD = vtk_to_numpy(pd.GetArray('grad_d'))
    U = np.sqrt(v[:,0]**2 + v[:,1]**2 + v[:,2]**2)
    Re_O = d*d*omega / nu
    ome_d = omega * d
    # Sphere-kernel amplifying coordinate P = Shat * g.
    # X = |u|, Y = omega d, Z = 1/2 d^2 (n.grad omega); n = grad(wallDistance)
    # (outward wall-normal), so Z carries the physical curvature sign.
    nmag = np.sqrt(gradD[:,0]**2 + gradD[:,1]**2 + gradD[:,2]**2) + 1e-30
    domega_dn = (gradW[:,0]*gradD[:,0] + gradW[:,1]*gradD[:,1]
                 + gradW[:,2]*gradD[:,2]) / nmag
    X = U; Y = ome_d; Z = 0.5*d*d*domega_dn
    Rsph = np.sqrt(X*X + Y*Y + Z*Z) + 1e-30
    Shat = Y / (np.sqrt(X*X + Y*Y) + 1e-30)
    g_coord = (Y - X - Z) / Rsph
    P = Shat * g_coord
    for name, arr in [('Re_Omega', Re_O.astype(np.float32)),
                      ('Shat', Shat.astype(np.float32)),
                      ('g_coord', g_coord.astype(np.float32)),
                      ('P', P.astype(np.float32)),
                      ('sph_X', X.astype(np.float32)),        # |u|
                      ('sph_Y', Y.astype(np.float32)),        # omega d (shear)
                      ('sph_Z', Z.astype(np.float32)),        # 1/2 d^2 n.grad(omega) (curvature)
                      ('domega_dn', domega_dn.astype(np.float32))]:
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
    base = os.environ.get('SAAI_CFD_ROOT', '/home/qiqi/flexcompute/sa-ai/flow360')
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
