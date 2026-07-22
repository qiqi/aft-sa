"""Upper bracketing of c_V for the FINAL band-form composite gate (paper
sec:calib_bg): anchors re-solved at c_V = 8 and 16."""
from multiprocessing import Pool
from explore_lambda_v_anchors import one

if __name__ == '__main__':
    CANDS = [("cB8", 'lv3', 8.0, (250.0, 1.02, 9.0)),
             ("cB16", 'lv3', 16.0, (260.0, 1.05, 8.0))]
    with Pool(2) as pool:
        results = pool.map(one, CANDS)
    print("\n===== BAND-FORM COMPOSITE, UPPER cV BRACKET =====")
    for name, gate, gw, floor, gc, s, o in results:
        print(f"{name}: floor={floor:.1f} gc={gc:.4f} s={s:.2f} | "
              f"bl_int={o['bl_intdev']:.3f} bl_late={o['bl_late']:.3f} "
              f"m(-.09)={o['mean-0.09']:.3f} m(-.15)={o['mean-0.15']:.3f}")
