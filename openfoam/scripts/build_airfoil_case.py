#!/usr/bin/env python3
"""Build an OpenFOAM SA-AI case (NLF or Eppler, any structured level, any
alpha incl. negative) from the SAME mesh Flow360 used.

Usage: build_airfoil_case.py {nlf|eppler} {0|1|2} <alpha> [conservative]

Mesh source: flow360_fr/strL{lvl}prop_..._a{alpha} (or am{|alpha|}); falls
back to the a0 dir of the same level (the O-grid is alpha-independent).
Freestream: chi_inf = 8.76e-4 (N_crit = 9), U_inf = 1, alpha via velocity
rotation. Two-stage protocol handled by the driver (this writes stage-1
endTime).

'conservative' switches fvSolution to SIMPLE (not SIMPLEC), p 0.3 / U 0.7 /
nuTilda 0.4 -- retry setting for LSB-marginal FPE cases.
"""
import math
import os
import shutil
import sys

CASE_ROOT = "/local_data/qiqi/openfoam-sa-ai/cases"
F360_ROOT = "/home/qiqi/flexcompute/sa-ai/flow360_fr"

AIRFOILS = {
    "nlf": dict(re=4.0e6, wall="nlf0416",
                src="strL{lvl}prop_nlf0416_Re4M_{atag}"),
    "eppler": dict(re=2.0e5, wall="eppler387",
                   src="strL{lvl}prop_eppler387_Re200k_{atag}"),
}
# per-level: ranks, decomp, stage-1 iters
LEVELS = {0: (6, "(3 2 1)", 15000),
          1: (8, "(4 2 1)", 15000),
          2: (24, "(6 4 1)", 20000)}

CHI_INF = 7.1 * math.exp(-9.0)

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


def atag(alpha):
    return f"a{alpha}" if alpha >= 0 else f"am{-alpha}"


def build(airfoil, lvl, alpha, conservative=False):
    cfg = AIRFOILS[airfoil]
    RE, WALL = cfg["re"], cfg["wall"]
    NU = 1.0 / RE
    NUTILDA_INF = CHI_INF * NU
    n_ranks, decomp, n_steps = LEVELS[lvl]

    src = None
    for root in (F360_ROOT, F360_ROOT.replace("_fr", "_fv1")):
        for at in (atag(alpha), "a0"):
            cand = f"{root}/{cfg['src'].format(lvl=lvl, atag=at)}"
            if os.path.isfile(f"{cand}/mesh.msh"):
                src = cand
                break
        if src:
            break
    assert src, f"no mesh.msh found for {airfoil} L{lvl} {atag(alpha)}"

    name = f"{airfoil}_strL{lvl}_{atag(alpha)}"
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
    {WALL}  {{ type noSlip; }}
    "(symmetry1|symmetry2)" {{ type empty; }}
}}
""")
    write(f"{cd}/0/p", FOAMFILE.format(cls="volScalarField", obj="p") + f"""
dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{{
    farfield {{ type freestreamPressure; freestreamValue uniform 0; }}
    {WALL}  {{ type zeroGradient; }}
    "(symmetry1|symmetry2)" {{ type empty; }}
}}
""")
    write(f"{cd}/0/nuTilda", FOAMFILE.format(cls="volScalarField", obj="nuTilda") + f"""
dimensions [0 2 -1 0 0 0 0];
internalField uniform {v};
boundaryField
{{
    farfield {{ type inletOutlet; inletValue uniform {v}; value uniform {v}; }}
    {WALL}  {{ type fixedValue; value uniform 0; }}
    "(symmetry1|symmetry2)" {{ type empty; }}
}}
""")
    write(f"{cd}/0/nut", FOAMFILE.format(cls="volScalarField", obj="nut") + f"""
dimensions [0 2 -1 0 0 0 0];
internalField uniform 0;
boundaryField
{{
    farfield {{ type calculated; value uniform 0; }}
    {WALL}  {{ type nutLowReWallFunction; value uniform 0; }}
    "(symmetry1|symmetry2)" {{ type empty; }}
}}
""")
    write(f"{cd}/constant/transportProperties",
          FOAMFILE.format(cls="dictionary", obj="transportProperties") + f"""
transportModel Newtonian;
nu [0 2 -1 0 0 0 0] {NU:.8g};
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
        patches         ({WALL});
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
        patches         ({WALL});
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
    if conservative:
        simple = """SIMPLE
{
    nNonOrthogonalCorrectors 1;
    consistent      no;
    residualControl { }
}

relaxationFactors
{
    fields { p 0.3; }
    equations
    {
        U               0.7;
        nuTilda         0.4;
    }
}"""
    else:
        simple = """SIMPLE
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
}"""
    write(f"{cd}/system/fvSolution",
          FOAMFILE.format(cls="dictionary", obj="fvSolution") + f"""
solvers
{{
    p
    {{
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-10;
        relTol          0.01;
    }}
    "(U|nuTilda)"
    {{
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-12;
        relTol          0.01;
    }}
    Phi
    {{
        $p;
    }}
}}

potentialFlow
{{
    nNonOrthogonalCorrectors 10;
}}

{simple}
""")
    write(f"{cd}/system/decomposeParDict",
          FOAMFILE.format(cls="dictionary", obj="decomposeParDict") + f"""
numberOfSubdomains {n_ranks};
method          hierarchical;
coeffs {{ n {decomp}; }}
""")
    print(f"{name}: ranks={n_ranks} stage1={n_steps} src={os.path.basename(src)}"
          f"{' CONSERVATIVE' if conservative else ''}")
    return cd


if __name__ == "__main__":
    build(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),
          conservative=(len(sys.argv) > 4 and sys.argv[4] == "conservative"))
