"""Convert Plot3D ASCII 2D structured grid (.p3d) to legacy VTK structured grid (.vtk)."""
import sys
import numpy as np

def read_p3d(path):
    toks = open(path).read().split()
    ni, nj = int(toks[0]), int(toks[1])
    n = ni * nj
    x = np.array([float(t) for t in toks[2:2+n]]).reshape(nj, ni).T
    z = np.array([float(t) for t in toks[2+n:2+2*n]]).reshape(nj, ni).T
    return ni, nj, x, z

def write_vtk_structured(path, ni, nj, x, z):
    """Write a Z=0 (in y, since Flow360 uses y=span axis) structured grid VTK.
    The grid is in the xz plane; we put y=0 for visualization."""
    with open(path, 'w') as f:
        f.write('# vtk DataFile Version 3.0\n')
        f.write('NLF structured C-grid (Construct2D)\n')
        f.write('ASCII\n')
        f.write('DATASET STRUCTURED_GRID\n')
        f.write(f'DIMENSIONS {ni} {nj} 1\n')
        f.write(f'POINTS {ni*nj} float\n')
        for j in range(nj):
            for i in range(ni):
                # Y=0 plane; X and Z from p3d
                f.write(f'{x[i,j]:.8e} 0.0 {z[i,j]:.8e}\n')

if __name__ == '__main__':
    for tag in ['L0', 'L1', 'L2']:
        p3d = f'/home/qiqi/flexcompute/aft-sa/external/construct2d/proper_struct_nlf_{tag}.p3d'
        vtk = f'/home/qiqi/flexcompute/aft-sa/flow360/nlfprop_str_{tag}.vtk'
        ni, nj, x, z = read_p3d(p3d)
        write_vtk_structured(vtk, ni, nj, x, z)
        print(f'{tag}: {ni}x{nj} = {ni*nj} pts ({(ni-1)*(nj-1)} cells) → {vtk}')
