#!/usr/bin/env python3
"""After gmshToFoam: set patch types in constant/polyMesh/boundary --
airfoil -> wall, symmetry1/2 -> empty (true 2D)."""
import re
import sys

cd = sys.argv[1]
wall = sys.argv[2] if len(sys.argv) > 2 else "nlf0416"
p = f"{cd}/constant/polyMesh/boundary"
txt = open(p).read()


def set_type(txt, patch, newtype):
    # patch block: name { type X; physicalType Y; ... }
    pat = re.compile(r"(\n\s+" + patch + r"\n\s+\{[^}]*?type\s+)(\w+)(;)")
    out, n = pat.subn(r"\g<1>" + newtype + r"\g<3>", txt)
    assert n == 1, f"{patch}: {n} matches"
    return out


txt = set_type(txt, wall, "wall")
txt = set_type(txt, "symmetry1", "empty")
txt = set_type(txt, "symmetry2", "empty")
open(p, "w").write(txt)
print(f"fixed {p}")
