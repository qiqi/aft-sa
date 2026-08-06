"""Test of the proposed acceleration-quench (K) gate on the frozen-Hiemenz rig.

Executes the verification the proposal (blindspots/07, agent-paper-review
2026-07-27-1250) asks for as its step 2: put q(K) on the SA production in the
frozen stagnation-point model problem and ask whether the attachment-anchored
turbulent branch survives.

    K_loc = nu (u.grad|u|) / |u|^3 ,   q(K) = 1/(1 + (max(K,0)/K_c)^n)

applied to the SA production only.  The gate lives in
stagnation_bistability.run_case behind K_crit=None, so the ungated path is
unchanged; test_ungated_is_bit_identical() below asserts that.

Constants: K_c and n are the proposal's, K_c from the relaminarization
literature (Launder & Jones; Narasimha & Sreenivasan give 2.5-3.5e-6 depending
on definition and facility), so the run sweeps that range rather than adopting a
point value.  Nothing else is chosen: the L ladder and the near-critical bracket
are the ones the published ungated study used, read from its own output file.

Run from paper/: python3 repro/analytic/stagnation_kgate_test.py
  -> data/stagnation_kgate_test.json
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stagnation_bistability as sb                            # noqa: E402

PAPER = sb.PAPER

# The proposal's gate constants.  K_C_RANGE is the literature spread, not a fit.
K_C_NOMINAL = 3.0e-6
K_C_RANGE = (2.5e-6, 3.5e-6)
N_GATE = 2.0


def published_bracket():
    """The ungated critical bracket, read from the published rebisection."""
    p = os.path.join(PAPER, 'data', 'stagnation_bistability_rebisect.json')
    d = json.load(open(p))
    return d['L_lo'], d['L_hi']


def test_ungated_is_bit_identical():
    """Adding the gate must not change the ungated path at all."""
    a = sb.run_case(300.0, niter=1500)
    b = sb.run_case(300.0, niter=1500, K_crit=None)
    assert a['maxchi'] == b['maxchi'], (a['maxchi'], b['maxchi'])
    print(f"  ungated path unchanged: maxchi = {a['maxchi']:.12e} both ways")


def test_K_matches_closed_form():
    """On the inviscid edge this field has K = nu/(k x^2) = 1/x^2 exactly."""
    r = sb.run_case(300.0, niter=1, K_crit=K_C_NOMINAL, return_gate=True)
    x, y, _q, K = r['gate']
    j = int(np.argmin(np.abs(y - 8.0)))           # outside the layer
    worst = 0.0
    for xi in (100.0, 200.0, 290.0):
        i = int(np.argmin(np.abs(x - xi)))
        worst = max(worst, abs(K[i, j]*xi**2 - 1.0))
    assert worst < 0.02, f"K/(1/x^2) off by {worst:.3f}"
    print(f"  K matches the closed form 1/x^2 to {worst:.1%} at the edge")


def gate_coverage(L, K_crit=K_C_NOMINAL):
    """Where the gate is active: the quench station, and how much of the layer.

    Reports the fraction of the amplifying region (inside the layer, y < 3,
    which is where f' rises to ~1) at which q < 0.99 and q < 0.5.
    """
    r = sb.run_case(L, niter=1, K_crit=K_crit, return_gate=True)
    x, y, q, K = r['gate']
    inlayer = y < 3.0
    xpos = x > 0.0
    sub = q[np.ix_(xpos, inlayer)]
    x_quench = 1.0/np.sqrt(K_crit)               # edge value K = 1/x^2 = K_c
    return dict(L=L, x_quench_edge=x_quench,
                frac_q_below_099=float(np.mean(sub < 0.99)),
                frac_q_below_050=float(np.mean(sub < 0.50)),
                q_at_outer_edge=float(q[-2, int(np.argmin(np.abs(y - 8.0)))]),
                q_min=float(sub.min()), q_max=float(sub.max()))


def verdict(L, K_crit, chunks=8, chunk_iters=20000):
    """Cap-proof classification, same rule as the published study."""
    chi = None
    prev = None
    t0 = time.time()
    for k in range(chunks):
        r = sb.run_case(L, chi_init=chi, return_field=True,
                        niter=chunk_iters, K_crit=K_crit)
        m = r['maxchi']
        print(f"      chunk {k+1}: maxchi = {m:11.5f}  ({time.time()-t0:5.1f}s)",
              flush=True)
        if m < 1e-3:
            return 'collapsed', m, k+1
        chi = r['field'][2]
        if prev is not None and m > 2.0 and abs(m - prev) < 0.02*m:
            return 'sustained', m, k+1
        prev = m
    return 'unresolved', prev, chunks


def main():
    print("=" * 74)
    print("REGRESSION AND INSTRUMENT CHECKS")
    print("=" * 74)
    test_ungated_is_bit_identical()
    test_K_matches_closed_form()

    L_lo, L_hi = published_bracket()
    print(f"\n  published ungated critical bracket: L in ({L_lo:.1f}, {L_hi:.1f}]"
          f"  (Re_r {L_lo**2:.2e}..{L_hi**2:.2e})")

    print("\n" + "=" * 74)
    print("WHERE THE GATE IS ACTIVE")
    print("=" * 74)
    print(f"  quench station on the edge, x = 1/sqrt(K_c) = "
          f"{1.0/np.sqrt(K_C_NOMINAL):.0f} delta\n")
    print(f"    {'L':>7s} {'q<0.99':>8s} {'q<0.50':>8s} {'q_min':>9s} "
          f"{'q_max':>8s} {'q at outer edge':>16s}")
    cov = []
    for L in (300.0, L_hi, 3000.0, 10000.0):
        c = gate_coverage(L)
        cov.append(c)
        print(f"    {L:7.1f} {c['frac_q_below_099']:8.3f} "
              f"{c['frac_q_below_050']:8.3f} {c['q_min']:9.2e} "
              f"{c['q_max']:8.4f} {c['q_at_outer_edge']:16.4f}")

    print("\n" + "=" * 74)
    print("DOES THE BRANCH SURVIVE?")
    print("=" * 74)
    runs = []
    for L in (round(L_hi, 1), 3000.0):
        print(f"\n  L = {L:.1f} (Re_r = {L*L:.2e})  UNGATED control:", flush=True)
        v, m, k = verdict(L, None)
        print(f"    -> {v}  maxchi = {m:.4f}")
        runs.append(dict(L=L, K_crit=None, verdict=v, maxchi=m, chunks=k))

        for Kc in (K_C_NOMINAL,) + K_C_RANGE:
            print(f"  L = {L:.1f}  GATED, K_c = {Kc:.1e}:", flush=True)
            v, m, k = verdict(L, Kc)
            print(f"    -> {v}  maxchi = {m:.4f}")
            runs.append(dict(L=L, K_crit=Kc, verdict=v, maxchi=m, chunks=k))

    out = dict(
        what="acceleration-quench gate q(K)=1/(1+(K/K_c)^n) on SA production, "
             "frozen-Hiemenz rig",
        gate="K_loc = nu (u.grad|u|)/|u|^3; n = %g" % N_GATE,
        K_c_nominal=K_C_NOMINAL, K_c_range=list(K_C_RANGE),
        published_ungated_bracket=[L_lo, L_hi],
        x_quench_edge=1.0/np.sqrt(K_C_NOMINAL),
        coverage=cov, runs=runs)
    p = os.path.join(PAPER, 'data', 'stagnation_kgate_test.json')
    json.dump(out, open(p, 'w'), indent=1)
    print(f"\nwrote {p}")


if __name__ == '__main__':
    main()
