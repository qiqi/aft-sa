"""Convert Construct2D-generated p3d structured grids for Eppler 387 to legacy
VTK structured grids for ParaView preview."""
import sys
sys.path.insert(0, '/home/qiqi/flexcompute/aft-sa/flow360')
from p3d_to_vtk import read_p3d, write_vtk_structured

if __name__ == '__main__':
    for tag in ['L0', 'L1', 'L2']:
        p3d = f'/home/qiqi/flexcompute/aft-sa/external/construct2d/proper_struct_eppler_{tag}.p3d'
        vtk = f'/home/qiqi/flexcompute/aft-sa/flow360/epprop_str_{tag}.vtk'
        ni, nj, x, z = read_p3d(p3d)
        write_vtk_structured(vtk, ni, nj, x, z)
        print(f'{tag}: {ni}x{nj} = {ni*nj} pts ({(ni-1)*(nj-1)} cells) -> {vtk}')
