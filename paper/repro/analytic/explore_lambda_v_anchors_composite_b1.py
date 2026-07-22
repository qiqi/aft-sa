"""Anchor re-solve for the composite two-pinch gate with FAST band opening
(band_pow=1: B^1 = Gamma(2-Gamma), quadratic opening away from Gamma=1;
user-proposed). Fully open at Gamma = 0, 2; wall pinch unchanged."""
from multiprocessing import Pool
from explore_lambda_v_anchors import one

if __name__ == '__main__':
    CANDS = [
        ("cB1_1", 'lv2b', 1.0, (200.0, 0.99, 12.0)),
        ("cB1_2", 'lv2b', 2.0, (220.0, 1.02, 10.0)),
        ("cB1_4", 'lv2b', 4.0, (240.0, 1.05, 8.0)),
    ]
    with Pool(3) as pool:
        results = pool.map(one, CANDS)
    print("\n===== COMPOSITE GATE, FAST BAND (B^1, anchors re-solved) =====")
    print(f"{'cand':>7} {'gate':>5} {'w':>5} | {'floor':>7} {'g_c':>7} {'s':>6} | "
          f"{'bl_int':>7} {'bl_late':>8} {'m(-.09)':>8} {'m(-.15)':>8}")
    for name, gate, gw, floor, gc, s, o in results:
        print(f"{name:>7} {gate:>5} {gw:5.1f} | {floor:7.1f} {gc:7.4f} {s:6.2f} | "
              f"{o['bl_intdev']:7.3f} {o['bl_late']:8.3f} {o['mean-0.09']:8.3f} "
              f"{o['mean-0.15']:8.3f}")
