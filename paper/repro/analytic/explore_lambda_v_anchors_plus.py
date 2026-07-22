"""Anchor re-solve for the ONE-SIDED Lambda_v gate (|Lv| -> max(0, Lv)),
user-proposed. The sub-inflection regions lose their gate opening, so the
separation-limit anchor march slows (raw 1.00 -> 0.56); this re-solves
(floor, g_c, s) per the three-anchor protocol to see what the compensation
does to the rest of the family."""
from multiprocessing import Pool
from explore_lambda_v_anchors import one

if __name__ == '__main__':
    CANDS = [
        ("cVp1", 'lvp', 1.0, (200.0, 0.97, 16.0)),
        ("cVp2", 'lvp', 2.0, (220.0, 1.00, 12.0)),
        ("cVp4", 'lvp', 4.0, (230.0, 1.05, 9.0)),
    ]
    with Pool(3) as pool:
        results = pool.map(one, CANDS)
    print("\n===== ONE-SIDED LAMBDA_V GATE (anchors re-solved) =====")
    print(f"{'cand':>7} {'gate':>5} {'w':>5} | {'floor':>7} {'g_c':>7} {'s':>6} | "
          f"{'bl_int':>7} {'bl_late':>8} {'m(-.09)':>8} {'m(-.15)':>8}")
    for name, gate, gw, floor, gc, s, o in results:
        print(f"{name:>7} {gate:>5} {gw:5.1f} | {floor:7.1f} {gc:7.4f} {s:6.2f} | "
              f"{o['bl_intdev']:7.3f} {o['bl_late']:8.3f} {o['mean-0.09']:8.3f} "
              f"{o['mean-0.15']:8.3f}")
