"""Sensitivity studies for the paper-v2 triple (band-form composite gate):
(1) departure-level study: re-solve the triple with the Blasius departure
    at N = 0.5 / 2.0 (moved consistently through the conditions), report
    triple + mid-adverse family response vs the N = 1 solve;
(2) valley (conditioning) study: tight-solve from a displaced start,
    report the (g_c, s) slide, anchor residuals, and mid-adverse delta."""
import numpy as np
from multiprocessing import Pool
import explore_lambda_v_anchors as A

CANON = (243.7, 0.9874, 10.68)


def solve_with_departure(N1, x0, tag):
    import lib.correlations as C
    from lib.boundary_layer import FalknerSkanWedge
    fs = FalknerSkanWedge(0.0)
    I = np.trapezoid(fs.u*(1-fs.u), fs.eta)
    Hb = np.trapezoid(1-fs.u, fs.eta)/I
    Rtc = float(C.Re_theta0(Hb)); slope = float(C.dN_dRe_theta(Hb))
    targets = (Rtc + N1/slope, Rtc + 9.0/slope)
    # clone of solve_triple with custom Blasius targets
    orig = A.solve_triple.__defaults__
    def blas_res(floor, gc, s, gw):
        return A.blasius_residuals(floor, gc, s, 'lv3', gw, targets)
    # monkeypatch targets through a custom inner loop: reuse A.solve_triple
    # by temporarily overriding its target computation is invasive; instead
    # run the secant structure inline:
    floor, gc, s = x0
    import lib.correlations as _c
    fs_sep = FalknerSkanWedge(A.BETA_SEP)
    Is = np.trapezoid(fs_sep.u*(1-fs_sep.u), fs_sep.eta)
    Hs = np.trapezoid(1-fs_sep.u, fs_sep.eta)/Is
    d_sep = float(C.dN_dRe_theta(Hs))
    def inner(s_val, floor0, gc0):
        floor, gc = floor0, gc0
        for it in range(10):
            r = blas_res(floor, gc, s_val, 4.0)
            if r[0] is None:
                floor *= 0.7; continue
            r1, r9 = r
            if abs(r1) < 1.0 and abs(r9) < 3.0:
                return floor, gc, True
            dfl = max(4.0, 0.05*floor)
            rb = blas_res(floor+dfl, gc, s_val, 4.0)
            if rb[0] is not None and abs(rb[0]-r1) > 1e-9:
                floor = min(max(floor - r1*dfl/(rb[0]-r1), 30.0), 1500.0)
            rc = blas_res(floor, gc, s_val, 4.0)
            if rc[0] is None:
                floor = min(max(floor*1.2, 30.0), 1500.0); continue
            rd = blas_res(floor, gc+0.01, s_val, 4.0)
            if rd[1] is not None and abs(rd[1]-rc[1]) > 1e-9:
                gc = min(max(gc - rc[1]*0.01/(rd[1]-rc[1]), 0.80), 1.40)
        return floor, gc, False
    def sep_ratio(s_val, floor, gc):
        Rt, N, _ = A.envelope(A.BETA_SEP, 1.2e6, floor, gc, s_val, 'lv3', 4.0)
        if N[-1] < 9.0: return None
        Rt1 = float(np.interp(1.0, N, Rt)); Rt9 = float(np.interp(9.0, N, Rt))
        return (8.0/(Rt9-Rt1))/d_sep
    hist = []
    for it in range(8):
        floor, gc, ok = inner(s, floor, gc)
        r = sep_ratio(s, floor, gc)
        hist.append((s, floor, gc, r, ok))
        print(f"[{tag}] it{it}: s={s:.2f} floor={floor:.1f} gc={gc:.4f} "
              f"sep={r if r else float('nan'):.4f} ok={ok}", flush=True)
        if r is not None and abs(r-1.0) < 0.005:
            break
        if len(hist) >= 2 and hist[-2][3] is not None and r is not None \
                and abs(r-hist[-2][3]) > 1e-6:
            s = min(max(s - (r-1.0)*(s-hist[-2][0])/(r-hist[-2][3]), 2.0), 40.0)
        else:
            s = s*(1.25 if (r or 0) < 1.0 else 0.8)
    o = A.family_observables(floor, gc, s, 'lv3', 4.0)
    return tag, floor, gc, s, o


def job(args):
    kind = args[0]
    if kind == 'dep':
        _, N1, x0, tag = args
        return solve_with_departure(N1, x0, tag)
    _, x0, tag = args
    return solve_with_departure(1.0, x0, tag)


if __name__ == '__main__':
    jobs = [('dep', 0.5, (235.0, 0.96, 12.5), 'N0.5'),
            ('dep', 2.0, (255.0, 1.06, 7.5), 'N2.0'),
            ('valley', (300.0, 1.05, 8.0), 'tightB')]
    with Pool(3) as pool:
        results = pool.map(job, jobs)
    print("\n===== V2 SENSITIVITY (band-form composite) =====")
    print(f"canon: floor={CANON[0]} gc={CANON[1]} s={CANON[2]}")
    for tag, floor, gc, s, o in results:
        print(f"{tag:>7}: floor={floor:.1f} gc={gc:.4f} s={s:.2f} | "
              f"bl_int={o['bl_intdev']:.3f} m(-.09)={o['mean-0.09']:.3f} "
              f"m(-.15)={o['mean-0.15']:.3f}")
