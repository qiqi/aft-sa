# Plan: where each characterization point goes in the two papers

Requested by the user, 2026-08-05: *"for all the things we are saying, we will
plan to put into the 'discussion and future work' section of the paper, and each
point probably is worth one sentence in remarks (or whatever section that's at
the end) of the whitepaper too."*

This is the **plan only** — no `.tex` was edited. Numbers referenced here are in
file `12`; the literature values they are compared against still carry the
verification debt listed at the end of that file, and **none of this should be
written into the paper until those are checked**.

## Structural decision needed first

- **`sa-ai.tex`** has a `\section{Conclusion}` whose penultimate paragraph
  already begins *"Several extensions follow."* That paragraph is the natural
  host: it currently runs limitation → extension in one sweep. The material
  below roughly doubles it, which argues for splitting the conclusion into
  `\subsection{The operating envelope}` (what we now know the kernel does
  outside its calibration) and `\subsection{Extensions}` (what to do about it),
  keeping the closing *Daodejing* quote where it is.
- **`whitepaper.tex`** has **no closing section at all** — it runs Model →
  Results → Appendix (`whitepaper.tex:1164`). So the one-sentence remarks need a
  new `\section{Remarks on the operating envelope}` inserted **before**
  `\clearpage\appendix`, i.e. after the spheroid subsection ends. Ten to twelve
  sentences, no figures, no tables.

## The whitepaper remarks section — one sentence each

Drafted at the intended length so the section can be assembled directly. Order
is by how surely we know the answer, which also reads as strongest-first.

1. **Solid-body rotation.** *A frame or fluid in solid-body rotation is read as
   stable exactly, because the velocity term in the amplifying coordinate
   dominates the rotational shear (`Y/X = 2d/r`), so the rate is clipped to
   zero — the counterpart, by a different route, of the irrotational-freestream
   property baseline SA obtains from building production on vorticity.*
2. **Rotation on a real layer.** *Frame rotation at `Omega ~ U_inf/L` perturbs
   the amplification rate by `41 (ell/L)`, i.e. `O(Re_theta/Re_L)`: 1.3 % on the
   NLF(1)-0416 and 6–12 % at the Eppler 387's Reynolds numbers for the
   pessimistic `L = c`, ten times less at a rotor-like `L = 10c`.*
3. **Kernel conditioning.** *Because the amplifying coordinate is a
   near-cancellation, `g = (Y-X-Z)/R`, relative perturbations of the shear
   indicator reach the rate magnified about fourfold — which is why terms as
   small as `ell/R ~ 10^-3` are visible in it at all.*
4. **Wall curvature, convex.** *Streamwise wall curvature enters the curvature
   indicator through the metric term `dZ = (d/2R) Y` and is therefore already
   present in every computed case, contributing −0.17 % of the amplification
   rate on the NLF(1)-0416 and −0.4 to −0.5 % on the Eppler 387 — stabilizing,
   and far below the mesh-to-mesh scatter.*
5. **Wall curvature at the low-`Re` cylinder.** *The same term is not a
   perturbation at the bottom of the drag-crisis traverse: it removes 10 % of the
   rate at `Re_D = 10^4` and essentially all of it at `Re_D <= 3 x 10^2`
   (`ell/R = 0.13`–`0.23`), so at the traverse's low-Reynolds end the reported
   transition is set as much by the wall's curvature metric as by the profile.*
6. **Bodies of revolution.** *For transverse curvature the two candidate
   curvature realizations do not agree, so the spheroid campaign carries a
   3–7 % realization uncertainty in the rate — unlike streamwise curvature,
   where the realizations coincide and the airfoil numbers are unambiguous.*
7. **Görtler.** *On a concave wall the metric term changes sign and destabilizes,
   so the kernel is not blind to centrifugal geometry, but its response is
   linear in `delta/R` where the instability is governed by
   `G = Re_delta sqrt(delta/R)` — the right sign under the wrong scaling law, and
   the gap widens with Reynolds number.*
8. **Attachment line.** *Evaluated on the exact swept-Hiemenz spanwise profile
   the kernel amplifies at 82 % of the Blasius rate and switches on at
   `Rbar = 601`, against the linear-stability critical `Rbar ≈ 583`, with no
   fitting — agreement on the threshold, though not on the Görtler–Hämmerlin
   mechanism, and not separable in a computed case from standard SA's own
   attachment-anchored branch.*
9. **Suction layers.** *The asymptotic suction profile is read as stable at every
   Reynolds number (`P < 0` away from the wall) against a true critical
   `Re_delta* ~ 5.4 x 10^4`, so the model errs conservatively and will
   over-credit hybrid-laminar-flow suction; the same evaluation shows the
   feared positive amplifying coordinate at a transpiring wall does not occur
   under suction.*
10. **Upstream wakes and turbomachinery.** *Because SA's destruction scales as
    `(nu_tilde/d)^2` and is negligible where `d` is large, a wake's
    `nu_tilde` is transported without decaying — unlike `k` in a two-equation
    model, which its dissipation drains — so a blade row ingesting an upstream
    wake will receive `chi > 1` and complete its handover before its own layer
    can amplify; the two-element section at negative incidence is that mechanism
    in isolation.*
11. **Smoothness.** *The assembled model is continuous everywhere and
    differentiable almost everywhere, the exceptions being the amplifying
    coordinate's clip at `P = 0` and `P = 1`, the handover switch at `chi = 1`,
    and the gated maximum at handover — all replaceable by the soft forms the
    rate already uses, should a gradient-based application require it.*

Remark 10 is the one that should also appear **in the body**, one sentence in the
two-element section (`sa-ai.tex:2455`), since that section is where the mechanism
is demonstrated. It is also the item the user identified as making a further
turbomachinery campaign unnecessary — so the sentence has to carry enough
mechanism to stand in for the case that was not run.

## The `sa-ai.tex` discussion / future-work paragraph

Three groups. Each item is one to three sentences, with the numbers from file
`12`, and each states plainly whether it is a *bound* (we know the size), a
*defect* (we know the sign and not the size), or an *untested regime*.

**Group A — bounds now quantified, no case needed.** Rotation (§1), streamwise
curvature (§2 items 1–2), transverse-curvature realization uncertainty (§2 item
3), the suction layer (§4), and the wake-transport statement. Frame these as the
envelope being *characterized* rather than the model being *limited*: each has a
number that scales, and each was obtained analytically. This is also where the
kernel-conditioning factor of 4.3 belongs, because it explains why several of the
other numbers are as large as they are.

**Group B — defects with a known sign and no size yet.** The Görtler scaling
mismatch (right sign, wrong law). The attachment line (agreement on threshold,
mechanism unrepresented, inseparable from SA's spurious branch without file
`07`'s quench — so it should be written as a *conditional* result). Blowing, as
opposed to suction, at a transpiring wall. The `chi = 1` kink's consequences for
implicit solution, and the clipped Jacobian, both of which belong here rather
than in Group A because their practical cost is unmeasured.

**Group C — untested regimes, in the order file `10` now ranks them.** Steps and
gaps; heat transfer as a quantity of interest and hence the transition-zone
closure; max-lift with free transition; the community benchmark sets. Then, with
its reason for deferral attached rather than implied: turbomachinery cascades,
where remark 10 says what the answer will be and why running the case would
mostly confirm a structural statement.

Also fold in, since they are corrections rather than additions:

- The two `02` §5 statements that turned out wrong (curvature "enters no sensor";
  attachment line "no concept of it"). If any earlier draft of the paper carries
  either claim, it must be pulled — worth a `grep` for "curvature" and
  "attachment" in `sa-ai.tex` before writing.
- File `10`'s withdrawal of the A-airfoil and the bar-passing case, if the
  future-work list is meant to reflect the current plan rather than the first
  one.

## What must happen before any of this is written

1. **Check the literature values** — swept-Hiemenz `Rbar_crit`, Poll's
   criterion, ASBL `Re_delta*_crit`, Blasius `Re_delta*_crit`, and Spalart's
   freestream-vorticity wording (file `12`, verification debt).
2. **Second-read the attachment-line result.** It is positive, it is the kind of
   agreement that can be coincidence, and it rests on the `x=0` reduction.
3. **Decide the conclusion's structure** (split or not) before drafting, because
   the material is large enough that the choice changes the sentences.
4. **Decide whether the whitepaper remarks carry numbers or only signs.** The
   drafts above carry numbers, which is the more useful choice but commits the
   whitepaper to the same verification debt as the paper.
