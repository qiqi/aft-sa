"""Compile the systematic Tu=0.2 up/down Cd + theta_tr tables from
systematic_Tu0.2_summary.jsonl (deduped, last row per case wins)."""
import json
from pathlib import Path

OUT = Path("/local_data/qiqi/sa-ai/dragcrisis_matrix")
S = OUT / "systematic_Tu0.2_summary.jsonl"


def load():
    by = {}
    for ln in open(S):
        ln = ln.strip()
        if not ln:
            continue
        d = json.loads(ln)
        by[d["case"]] = d
    return list(by.values())


def main():
    rows = load()
    up = {r["re"]: r for r in rows if r["dir"] == "up"}
    dn = {r["re"]: r for r in rows if r["dir"] == "dn"}
    seam = [r for r in rows if r["dir"] == "seam"]
    res = sorted(set(up) | set(dn))
    hdr = (f"{'Re':>11s} {'mesh':7s} | {'Cd_up':>7s} {'vu':>3s} {'ttrU':>6s}"
           f" | {'Cd_dn':>7s} {'vd':>3s} {'ttrD':>6s} | {'dCd':>7s}")
    print(hdr); print("-" * len(hdr))
    for re in res:
        u, d = up.get(re), dn.get(re)
        mesh = (u or d)["mesh"]
        def g(r, k, f="{:.4f}"):
            return f.format(r[k]) if r and r.get(k) is not None else "   -  "
        vu = (u["verdict"][:3] if u else "-")
        vd = (d["verdict"][:3] if d else "-")
        dcd = (f"{u['Cd']-d['Cd']:+.4f}" if u and d and
               u.get("Cd") is not None and d.get("Cd") is not None else "   -  ")
        print(f"{re:>11.4g} {mesh:7s} | {g(u,'Cd')} {vu:>3s} "
              f"{g(u,'theta_tr_chi1','{:.1f}'):>6s} | {g(d,'Cd')} {vd:>3s} "
              f"{g(d,'theta_tr_chi1','{:.1f}'):>6s} | {dcd}")
    print(f"\n{len(up)} up, {len(dn)} dn, {len(seam)} seam cases")
    if seam:
        print("\nSEAM overlaps:")
        for r in seam:
            print(f"  Re={r['re']:.4g} {r['mesh']:7s} Cd={r.get('Cd')} "
                  f"verdict={r['verdict']} warm_src={r.get('warm_src')}")
    tot = sum(r.get("wall_s", 0) for r in rows)
    print(f"\ntotal wall (all sys cases): {tot/3600:.2f} GPU-h")


if __name__ == "__main__":
    main()
