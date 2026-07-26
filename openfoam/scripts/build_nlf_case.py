#!/usr/bin/env python3
"""Build an OpenFOAM SA-AI case for NLF(1)-0416 from the SAME mesh Flow360
used: the committed gmsh source (mesh.msh) in the flow360_fr case dir,
converted with gmshToFoam, span planes relabeled `empty` (true 2D).

Usage: build_nlf_case.py {str|cav} {0|1|2} {0|4|9|15}

Paper condition: Re = 4e6, M = 0.1 (incompressible here), chi_inf = 8.76e-4
(N_crit = 9). OpenFOAM: U_inf = 1, nu = 2.5e-7, nuTilda_inf = 2.19e-10.
The case dir is written to /local_data (cases symlink); mesh conversion is
done by the caller (needs the OpenFOAM env):
    gmshToFoam mesh.msh && python3 fix_boundary.py <case>
"""
import math
import os
import shutil
import sys

CASE_ROOT = "/local_data/qiqi/openfoam-sa-ai/cases"
F360_ROOT = "/home/qiqi/flexcompute/sa-ai/flow360_fr"

RE = 4.0e6
NU = 1.0 / RE
CHI_INF = 7.1 * math.exp(-9.0)          # 8.7623e-4
NUTILDA_INF = CHI_INF * NU

FOAMFILE = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
"""


def write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def build(fam, lvl, alpha, n_ranks=8, n_steps=30000):
    src = f"{F360_ROOT}/{fam}L{lvl}prop_nlf0416_Re4M_a{alpha}"
    assert os.path.isfile(f"{src}/mesh.msh"), f"no mesh.msh in {src}"
    name = f"nlf_{fam}L{lvl}_a{alpha}"
    cd = os.path.join(CASE_ROOT, name)
    if os.path.exists(cd):
        shutil.rmtree(cd)
    os.makedirs(cd)
    shutil.copy(f"{src}/mesh.msh", f"{cd}/mesh.msh")

    a = math.radians(alpha)
    ux, uz = math.cos(a), math.sin(a)
    v = f"{NUTILDA_INF:.6g}"

    write(f"{cd}/0/U", FOAMFILE.format(cls="volVectorField", obj="U") + f"""
dimensions [0 1 -1 0 0 0 0];
internalField uniform ({ux:.8f} 0 {uz:.8f});
boundaryField
{{
    farfield {{ type freestreamVelocity; freestreamValue uniform ({ux:.8f} 0 {uz:.8f}); }}
    nlf0416  {{ type noSlip; }}
    "(symmetry1|symmetry2)" {{ type empty; }}
}}
""")
    write(f"{cd}/0/p", FOAMFILE.format(cls="volScalarField", obj="p") + """
dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    farfield {  type freestreamPressure; freestreamValue uniform 0; }
    nlf0416  { type zeroGradient; }
    "(symmetry1|symmetry2)" { type empty; }
}
""")
    write(f"{cd}/0/nuTilda", FOAMFILE.format(cls="volScalarField", obj="nuTilda") + f"""
dimensions [0 2 -1 0 0 0 0];
internalField uniform {v};
boundaryField
{{
    farfield {{ type inletOutlet; inletValue uniform {v}; value uniform {v}; }}
    nlf0416  {{ type fixedValue; value uniform 0; }}
    "(symmetry1|symmetry2)" {{ type empty; }}
}}
""")
    write(f"{cd}/0/nut", FOAMFILE.format(cls="volScalarField", obj="nut") + """
dimensions [0 2 -1 0 0 0 0];
internalField uniform 0;
boundaryField
{
    farfield { type calculated; value uniform 0; }
    nlf0416  { type nutLowReWallFunction; value uniform 0; }
    "(symmetry1|symmetry2)" { type empty; }
}
""")
    write(f"{cd}/constant/transportProperties",
          FOAMFILE.format(cls="dictionary", obj="transportProperties") + f"""
transportModel Newtonian;
nu [0 2 -1 0 0 0 0] {NU:.6g};
""")
    write(f"{cd}/constant/turbulenceProperties",
          FOAMFILE.format(cls="dictionary", obj="turbulenceProperties") + """
simulationType RAS;
RAS
{
    RASModel        SpalartAllmarasAI;
    turbulence      on;
    printCoeffs     on;
}
""")
    write(f"{cd}/system/controlDict",
          FOAMFILE.format(cls="dictionary", obj="controlDict") + f"""
libs            ("libSAAIIncompressibleTurbulenceModel.so");
application     simpleFoam;
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         {n_steps};
deltaT          1;
writeControl    timeStep;
writeInterval   {n_steps};
purgeWrite      2;
writeFormat     binary;
writePrecision  12;
runTimeModifiable true;

functions
{{
    forces
    {{
        type            forceCoeffs;
        libs            (forces);
        patches         (nlf0416);
        rho             rhoInf;
        rhoInf          1;
        liftDir         ({-uz:.8f} 0 {ux:.8f});
        dragDir         ({ux:.8f} 0 {uz:.8f});
        CofR            (0.25 0 0);
        pitchAxis       (0 1 0);
        magUInf         1;
        lRef            1;
        Aref            0.1;
        writeControl    timeStep;
        writeInterval   100;
    }}
    wallShear
    {{
        type            wallShearStress;
        libs            (fieldFunctionObjects);
        patches         (nlf0416);
        writeControl    writeTime;
    }}
}}
""")
    write(f"{cd}/system/fvSchemes",
          FOAMFILE.format(cls="dictionary", obj="fvSchemes") + """
ddtSchemes      { default steadyState; }
gradSchemes     { default cellLimited Gauss linear 1; grad(nuTilda) Gauss linear; }
divSchemes
{
    default                     none;
    div(phi,U)                  bounded Gauss linearUpwind grad(U);
    div(phi,nuTilda)            bounded Gauss linearUpwind grad(nuTilda);
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear limited corrected 0.5; }
interpolationSchemes { default linear; }
snGradSchemes   { default limited corrected 0.5; }
wallDist        { method meshWave; }
""")
    write(f"{cd}/system/fvSolution",
          FOAMFILE.format(cls="dictionary", obj="fvSolution") + """
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
    nNonOrthogonalCorrectors 1;
    consistent      yes;
    residualControl { }
}

relaxationFactors
{
    equations
    {
        U               0.85;
        nuTilda         0.6;
    }
}
""")
    write(f"{cd}/system/decomposeParDict",
          FOAMFILE.format(cls="dictionary", obj="decomposeParDict") + f"""
numberOfSubdomains {n_ranks};
method          hierarchical;
coeffs {{ n (4 2 1); }}
""")
    print(f"{name}: nuTilda_inf={v}  ranks={n_ranks}  src={src}")
    return cd


if __name__ == "__main__":
    fam, lvl, alpha = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    build(fam, lvl, alpha)
