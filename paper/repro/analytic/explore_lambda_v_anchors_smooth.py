"""Anchor re-solve for the SMOOTH Lambda_v gate (sqrt(1+(cV Lv)^2) form)."""
from multiprocessing import Pool
from explore_lambda_v_anchors import one

if __name__ == '__main__':
    CANDS = [
        ("cVs1", 'lvs', 1.0, (180.0, 0.97, 13.5)),
        ("cVs2", 'lvs', 2.0, (200.0, 1.00, 11.0)),
        ("cVs4", 'lvs', 4.0, (220.0, 1.05, 9.0)),
    ]
    with Pool(3) as pool:
        results = pool.map(one, CANDS)
    print("\n===== SMOOTH LAMBDA_V GATE (anchors re-solved) =====")
    print(f"{'cand':>7} {'gate':>5} {'w':>5} | {'floor':>7} {'g_c':>7} {'s':>6} | "
          f"{'bl_int':>7} {'bl_late':>8} {'m(-.09)':>8} {'m(-.15)':>8}")
    for name, gate, gw, floor, gc, s, o in results:
        print(f"{name:>7} {gate:>5} {gw:5.1f} | {floor:7.1f} {gc:7.4f} {s:6.2f} | "
              f"{o['bl_intdev']:7.3f} {o['bl_late']:8.3f} {o['mean-0.09']:8.3f} "
              f"{o['mean-0.15']:8.3f}")
