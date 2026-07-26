#!/usr/bin/env python3
"""Build OpenFOAM SA-AI flat-plate natural-transition cases.

Mirrors sa-ai/flow360/build_flatplate_cases.py (paper Sec. IV grid):
  x in [0, 6], 320 cells, first dx = 8e-4, geometric (ratio solved exactly)
  z in [0, ~0.505], 80 cells, first dz = 7e-6, geometric ratio 1.12
  quasi-2D (OpenFOAM: 1 cell, empty front/back)
Unit Reynolds number 1e6 (U_inf = 1, nu = 1e-6), incompressible simpleFoam.

Five cases flatplate_Tu{0040,0080,0160,0300,0600}; freestream SA seed
  chi_inf = C_V1 * exp(-N_crit),  N_crit = -8.43 - 2.4*ln(Tu_frac)  (Mack 1977)
applied directly as nuTilda_inf = chi_inf * nu (no AI_LAMINAR_SLOWDOWN in the
OpenFOAM steady solve; SIMPLE under-relaxation plays that role).

Cases are written to CASE_ROOT (on /local_data -- large data), symlinked as
sa-ai/openfoam/data.
"""
import math
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
CASE_ROOT = "/local_data/qiqi/openfoam-sa-ai/cases"

TU_LIST = [0.04, 0.08, 0.16, 0.30, 0.60]  # percent
A_TU, B_TU, C_V1 = -8.43, 2.4, 7.1
NU = 1.0e-6                                # unit Re = 1e6, U_inf = 1
NX, LX, DX0 = 320, 6.0, 8.0e-4
NZ, DZ0, RZ = 80, 7.0e-6, 1.12


def chi_inf(Tu_pct):
    N_crit = A_TU - B_TU * math.log(Tu_pct / 100.0)
    return C_V1 * math.exp(-N_crit)


def solve_ratio(dx0, n, L, lo=1.0 + 1e-9, hi=1.5):
    f = lambda r: dx0 * (r ** n - 1.0) / (r - 1.0) - L
    assert f(lo) < 0 < f(hi)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        (lo, hi) = (mid, hi) if f(mid) < 0.0 else (lo, mid)
    return 0.5 * (lo + hi)


RX = solve_ratio(DX0, NX, LX)
LZ = DZ0 * (RZ ** NZ - 1.0) / (RZ - 1.0)
# blockMesh simpleGrading expansion = (last cell)/(first cell) = r^(n-1)
EXP_X = RX ** (NX - 1)
EXP_Z = RZ ** (NZ - 1)


def write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


FOAMFILE = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
"""


def blockmesh():
    return FOAMFILE.format(cls="dictionary", obj="blockMeshDict") + f"""
scale 1;

vertices
(
    (0  0    0)      // 0
    ({LX} 0   0)     // 1
    ({LX} 0.1 0)     // 2
    (0  0.1  0)      // 3
    (0  0    {LZ:.9g})   // 4
    ({LX} 0   {LZ:.9g})  // 5
    ({LX} 0.1 {LZ:.9g})  // 6
    (0  0.1  {LZ:.9g})   // 7
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({NX} 1 {NZ})
    simpleGrading ({EXP_X:.9g} 1 {EXP_Z:.9g})
);

boundary
(
    inlet    {{ type patch;  faces ((0 4 7 3)); }}
    outlet   {{ type patch;  faces ((1 2 6 5)); }}
    plate    {{ type wall;   faces ((0 3 2 1)); }}
    top      {{ type patch;  faces ((4 5 6 7)); }}
    front    {{ type empty;  faces ((0 1 5 4)); }}
    back     {{ type empty;  faces ((3 7 6 2)); }}
);
"""


def field_U():
    return FOAMFILE.format(cls="volVectorField", obj="U") + """
dimensions [0 1 -1 0 0 0 0];
internalField uniform (1 0 0);
boundaryField
{
    inlet  { type fixedValue; value uniform (1 0 0); }
    outlet { type zeroGradient; }
    plate  { type noSlip; }
    top    { type pressureInletOutletVelocity; value uniform (1 0 0); }
    front  { type empty; }
    back   { type empty; }
}
"""


def field_p():
    return FOAMFILE.format(cls="volScalarField", obj="p") + """
dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet  { type zeroGradient; }
    outlet { type fixedValue; value uniform 0; }
    plate  { type zeroGradient; }
    top    { type totalPressure; p0 uniform 0; }
    front  { type empty; }
    back   { type empty; }
}
"""


def field_nuTilda(nuTildaInf):
    v = f"{nuTildaInf:.6g}"
    return FOAMFILE.format(cls="volScalarField", obj="nuTilda") + f"""
dimensions [0 2 -1 0 0 0 0];
internalField uniform {v};
boundaryField
{{
    inlet  {{ type fixedValue; value uniform {v}; }}
    outlet {{ type zeroGradient; }}
    plate  {{ type fixedValue; value uniform 0; }}
    top    {{ type inletOutlet; inletValue uniform {v}; value uniform {v}; }}
    front  {{ type empty; }}
    back   {{ type empty; }}
}}
"""


def field_nut(nuTildaInf):
    # freestream chi << 1 => nut ~ 0; wall low-Re => 0
    return FOAMFILE.format(cls="volScalarField", obj="nut") + f"""
dimensions [0 2 -1 0 0 0 0];
internalField uniform 0;
boundaryField
{{
    inlet  {{ type calculated; value uniform 0; }}
    outlet {{ type calculated; value uniform 0; }}
    plate  {{ type nutLowReWallFunction; value uniform 0; }}
    top    {{ type calculated; value uniform 0; }}
    front  {{ type empty; }}
    back   {{ type empty; }}
}}
"""


def transportProperties():
    return FOAMFILE.format(cls="dictionary", obj="transportProperties") + f"""
transportModel Newtonian;
nu [0 2 -1 0 0 0 0] {NU};
"""


def turbulenceProperties():
    return FOAMFILE.format(cls="dictionary", obj="turbulenceProperties") + """
simulationType RAS;
RAS
{
    RASModel        SpalartAllmarasAI;
    turbulence      on;
    printCoeffs     on;
}
"""


def controlDict(nSteps):
    return FOAMFILE.format(cls="dictionary", obj="controlDict") + f"""
libs            ("libSAAIIncompressibleTurbulenceModel.so");
application     simpleFoam;
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         {nSteps};
deltaT          1;
writeControl    timeStep;
writeInterval   {nSteps};
purgeWrite      2;
writeFormat     binary;
writePrecision  12;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

functions
{{
    wallShear
    {{
        type            wallShearStress;
        libs            (fieldFunctionObjects);
        patches         (plate);
        writeControl    writeTime;
    }}
    minMax
    {{
        type            fieldMinMax;
        libs            (fieldFunctionObjects);
        fields          (nuTilda U);
        writeControl    timeStep;
        writeInterval   100;
    }}
    forces
    {{
        type            forceCoeffs;
        libs            (forces);
        patches         (plate);
        rho             rhoInf;
        rhoInf          1;
        liftDir         (0 0 1);
        dragDir         (1 0 0);
        CofR            (0 0 0);
        pitchAxis       (0 1 0);
        magUInf         1;
        lRef            1;
        Aref            0.1;
        writeControl    timeStep;
        writeInterval   100;
    }}
}}
"""


def fvSchemes():
    return FOAMFILE.format(cls="dictionary", obj="fvSchemes") + """
ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; }
divSchemes
{
    default                     none;
    div(phi,U)                  bounded Gauss linearUpwind grad(U);
    div(phi,nuTilda)            bounded Gauss linearUpwind grad(nuTilda);
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
wallDist        { method meshWave; }
"""


def fvSolution():
    return FOAMFILE.format(cls="dictionary", obj="fvSolution") + """
solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-10;
        relTol          0.01;
    }
    "(U|nuTilda)"
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-12;
        relTol          0.01;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent      yes;
    residualControl { }
}

relaxationFactors
{
    equations
    {
        U               0.9;
        nuTilda         0.7;
    }
}
"""


def build_case(Tu, nSteps=30000):
    chi = chi_inf(Tu)
    nuT = chi * NU
    name = f"flatplate_Tu{int(round(Tu * 1000)):04d}"
    cd = os.path.join(CASE_ROOT, name)
    if os.path.exists(cd):
        shutil.rmtree(cd)
    write(f"{cd}/system/blockMeshDict", blockmesh())
    write(f"{cd}/system/controlDict", controlDict(nSteps))
    write(f"{cd}/system/fvSchemes", fvSchemes())
    write(f"{cd}/system/fvSolution", fvSolution())
    write(f"{cd}/constant/transportProperties", transportProperties())
    write(f"{cd}/constant/turbulenceProperties", turbulenceProperties())
    write(f"{cd}/0/U", field_U())
    write(f"{cd}/0/p", field_p())
    write(f"{cd}/0/nuTilda", field_nuTilda(nuT))
    write(f"{cd}/0/nut", field_nut(nuT))
    print(f"{name}: Tu={Tu}%  chi_inf={chi:.4e}  nuTilda_inf={nuT:.4e}")
    return cd


if __name__ == "__main__":
    print(f"grid: RX={RX:.8f} EXP_X={EXP_X:.4f}  LZ={LZ:.6f} EXP_Z={EXP_Z:.2f}")
    for Tu in TU_LIST:
        build_case(Tu)
