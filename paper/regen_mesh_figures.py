"""Mesh figures for the paper. For each airfoil (NLF, Eppler) and mesher
(structured O-grid, unstructured cavity): a 3x2 panel -- rows = L0/L1/L2
refinement, columns = leading-edge / trailing-edge zoom. The 2D wireframe is
the y_min span plane of the quasi-2D Flow360 mesh.cgns (chord = 1, LE at x~0,
TE at x~1). -> figs/mesh_{family}_{mesher}.pdf
"""
import warnings; warnings.filterwarnings('ignore')
import sys, numpy as np, vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

B = "/home/qiqi/flexcompute/aft-sa/flow360"
FAMILIES = {'nlf': ('nlf0416_Re4M', 'NLF(1)-0416'),
            'eppler': ('eppler387_Re200k', 'Eppler 387')}
MESHERS = {'str': 'structured O-grid', 'cav': 'unstructured cavity'}
LEVELS = ['L0', 'L1', 'L2']
LE_WIN = (-0.03, 0.12, -0.07, 0.07)   # xlo,xhi,zlo,zhi
TE_WIN = (0.88, 1.05, -0.05, 0.05)

def first_grid(mb):
    for i in range(mb.GetNumberOfBlocks()):
        b = mb.GetBlock(i)
        if b is None: continue
        if b.IsA('vtkMultiBlockDataSet') or 'Partitioned' in b.GetClassName():
            g = first_grid(b)
            if g is not None: return g
        elif b.GetNumberOfPoints() > 0:
            return b
    return None

def mesh_edges_2d(cgns):
    r = vtk.vtkCGNSReader(); r.SetFileName(cgns); r.Update()
    g = first_grid(r.GetOutput())
    ee = vtk.vtkExtractEdges(); ee.SetInputData(g); ee.Update()
    ed = ee.GetOutput()
    pts = vtk_to_numpy(ed.GetPoints().GetData())
    conn = vtk_to_numpy(ed.GetLines().GetData()).reshape(-1, 3)[:, 1:3]  # all 2-pt lines
    ymin = pts[:, 1].min()
    yplane = np.abs(pts[:, 1] - ymin) < 1e-4
    keep = yplane[conn[:, 0]] & yplane[conn[:, 1]]
    c = conn[keep]
    P = pts[:, [0, 2]]              # (x, z)
    return np.stack([P[c[:, 0]], P[c[:, 1]]], axis=1)   # (nedge, 2, 2)

def clip(segs, win):
    xlo, xhi, zlo, zhi = win
    a, b = segs[:, 0], segs[:, 1]
    inx = ((a[:, 0] >= xlo) & (a[:, 0] <= xhi) & (a[:, 1] >= zlo) & (a[:, 1] <= zhi)) | \
          ((b[:, 0] >= xlo) & (b[:, 0] <= xhi) & (b[:, 1] >= zlo) & (b[:, 1] <= zhi))
    return segs[inx]

def make_figure(fam_key):
    fam, fam_label = FAMILIES[fam_key]
    for mesh_key, mesh_label in MESHERS.items():
        fig, axes = plt.subplots(3, 2, figsize=(7.0, 8.2))
        for row, L in enumerate(LEVELS):
            cgns = f"{B}/{mesh_key}{L}prop_{fam}_a0/mesh.cgns"
            try:
                segs = mesh_edges_2d(cgns)
            except Exception as e:
                print(f"  {mesh_key}{L} {fam}: FAILED {e}"); segs = np.zeros((0, 2, 2))
            for col, (win, name) in enumerate([(LE_WIN, 'LE'), (TE_WIN, 'TE')]):
                ax = axes[row, col]
                sc = clip(segs, win)
                ax.add_collection(LineCollection(sc, colors='k', linewidths=0.35))
                ax.set_xlim(win[0], win[1]); ax.set_ylim(win[2], win[3])
                ax.set_aspect('equal'); ax.tick_params(labelsize=7)
                if row == 0:
                    ax.set_title(f"{'Leading' if name=='LE' else 'Trailing'} edge", fontsize=10)
                if col == 0:
                    ax.set_ylabel(f"{L}\n$z/c$", fontsize=9)
                if row == 2:
                    ax.set_xlabel('$x/c$', fontsize=9)
                print(f"  {mesh_key}{L} {fam} {name}: {len(sc)} edges")
        plt.tight_layout(rect=[0, 0, 1, 1])
        out = f"/home/qiqi/flexcompute/aft-sa/paper/figs/mesh_{fam_key}_{mesh_key}.pdf"
        plt.savefig(out); plt.savefig(f"/tmp/mesh_{fam_key}_{mesh_key}.png", dpi=120)
        plt.close(fig); print(f"wrote {out}")

if __name__ == '__main__':
    fams = sys.argv[1:] if len(sys.argv) > 1 else list(FAMILIES)
    for f in fams:
        make_figure(f)
