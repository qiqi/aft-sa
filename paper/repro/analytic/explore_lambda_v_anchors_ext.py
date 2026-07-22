"""explore-lambda-v, step A extension: the first scan (cV in {1..16}) showed
the mid-adverse mean-rate ratio improving monotonically as cV decreases
(0.58 -> 0.63 -> 0.74 at cV=4 -> 2 -> 1), so bracket below 1."""
from multiprocessing import Pool
from explore_lambda_v_anchors import one

if __name__ == '__main__':
    CANDS = [
        ("cV0.25", 'lv', 0.25, (230.0, 1.00, 15.0)),
        ("cV0.5",  'lv', 0.50, (210.0, 0.98, 14.5)),
        ("cV0.71", 'lv', 0.71, (190.0, 0.97, 14.0)),
    ]
    with Pool(3) as pool:
        results = pool.map(one, CANDS)
    print("\n===== LAMBDA_V GATE SCAN EXT (cV < 1) =====")
    print(f"{'cand':>7} {'gate':>5} {'w':>5} | {'floor':>7} {'g_c':>7} {'s':>6} | "
          f"{'bl_int':>7} {'bl_late':>8} {'m(-.09)':>8} {'m(-.15)':>8}")
    for name, gate, gw, floor, gc, s, o in results:
        print(f"{name:>7} {gate:>5} {gw:5.2f} | {floor:7.1f} {gc:7.4f} {s:6.2f} | "
              f"{o['bl_intdev']:7.3f} {o['bl_late']:8.3f} {o['mean-0.09']:8.3f} "
              f"{o['mean-0.15']:8.3f}")
