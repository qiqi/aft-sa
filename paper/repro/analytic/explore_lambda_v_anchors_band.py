"""Anchor re-solve for the FINAL band-form composite gate (lv3: smooth Q1 x
parabola-loop Q2, c2=8). Only cV=4 (Q1 unchanged; Q2's anchor effect is
small, sep-limit raw 1.01 at the previous constants)."""
from multiprocessing import Pool
from explore_lambda_v_anchors import one

if __name__ == '__main__':
    CANDS = [("cB4", 'lv3', 4.0, (243.7, 0.9874, 10.68))]
    with Pool(1) as pool:
        results = pool.map(one, CANDS)
    print("\n===== BAND-FORM COMPOSITE (anchors re-solved) =====")
    for name, gate, gw, floor, gc, s, o in results:
        print(f"{name}: floor={floor:.1f} gc={gc:.4f} s={s:.2f} | "
              f"bl_int={o['bl_intdev']:.3f} bl_late={o['bl_late']:.3f} "
              f"m(-.09)={o['mean-0.09']:.3f} m(-.15)={o['mean-0.15']:.3f}")
