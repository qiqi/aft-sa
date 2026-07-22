# Validation-section rewrite plan (sphere kernel)

Prepared 2026-07-21. The model/calibration section (now the single §II "The
amplification model") is rewritten to the sphere kernel. The validation sections
(flat plate, NLF, Eppler, Daedalus), Conclusion, and appendices still describe
and use the OLD ("predecessor") kernel. **All their coupled-RANS results are
predecessor-kernel and must be re-run with the sphere kernel — Flow360-gated
(task #66).** This is the prose+figure rewrite plan for when that CFD is unblocked.

## Global decision needed first
- **λ_p / Γ diagnostic panels.** Row 2 of every airfoil C_f figure (fig:nlfcflow,
  fig:nlfcfhigh, fig:eppcflow, fig:eppcfhigh) plots λ_p and max Γ — both removed
  from the model. Decide once: (a) replace with sphere diagnostics (P, or Ŝ & g,
  and Re_Ω vs Re_Ω^c), or (b) keep λ_p only as an external pressure-field
  descriptor explicitly flagged "not a model input." Drives the shared row-2
  layout prose (was ~L1321-1353) and 4 captions.

## §III Blasius flat plate — LIGHTEST
- Prose clean (no removed mechanisms; only χ=1/c_v1 handover, σ_t, Mack map).
- Re-run: fig:flatplate_batch (χ, C_f, crossings, all 5 Tu); the AGS-agreement
  numbers (~11%, per-Tu). Keep: grid, AGS/acoustic discussion, Tu→χ_∞ map,
  bypass out-of-scope argument.

## §IV NLF(1)-0416
- Rewrite: λ_p/cliff rooftop prose → P<0 (favorable held, no PG term); the
  indicator list "(Re_Ω, Γ, λ_p)" → (X,Y,Z; Re_Ω for onset); row-2 layout;
  fig:nlfcflow L0-artifact "Γ-sigmoid throttles" → P-collapse/onset; the
  favorable paragraph's SECOND half (cliff/f_λ, "factor moves fronts 0.003c",
  "factor inert on Eppler") — delete (no factor now).
- Re-run: fig:nlfpolar (+ the 58-78% / +27% / +10% ratios), drag numbers,
  fig:nlfcflow/fig:nlfcfhigh (all rows; row 2 changes content), tab:nlftrans
  SA-AI column, Appendix-A χ sheets.
- Keep: meshes (tab:nlfmesh), adversarial-mesh discussion (update symbols),
  Somers data, e^9 refs, ∂x_tr/∂N framing.

## §V Eppler 387
- Rewrite: Γ-sigmoid ignition paragraph → sphere ignition (g,P rise, Re_Ω^c(P)
  falls); the Q_2 ablation ("predecessor kernel without Q_2 pocket, same forces
  to 0.007 c_l") — delete/recast (no Q_2); the low-α drag deficit attributed to
  "under-predicted bubble length (sec:calib_rate)" — sphere now UNDER-amplifies
  at separation (0.65×), which LENGTHENS bubbles, so the sign may flip; re-derive
  after re-run.
- Re-run: fig:epppolar (+39-51%, ΔC_d ratios), fig:eppcflow/high, tab:eppxtr,
  tab:eppresweep (SA-AI cols), bubble-progression narrative, fig:eppresweep_low/high.
- Keep: meshes, McGhee/LTPT data, e^9 cross-check, out-of-scope reverse-C_f caveat.

## §VI Daedalus wing — MOST STALE
- Whole narrative is Q₁-only-predecessor + "second gate pinch" (Q₂). Rewrite:
  "predate the second gate pinch / Q₁-only / Q₂ regulates" framing (obsolete;
  difference is now old-vs-sphere wholesale); the "blind spot / second gate pinch
  built to close" paragraph — CONTRADICTS the rewritten reversed-flow result:
  text says predecessor reversed-flow ran 1.3-1.8× HOT, but the sphere kernel
  runs 0.6-0.7× COLD (fig:calibrate). Re-reason the compressed-bubble prediction
  from the sphere kernel's actual separated behavior.
- Re-run: fig:daepolar, fig:daesurf4/5/6, bubble/front numbers, tab:daetotals
  (App B), a_eff temporal-units numbers (App B, sigmoid-anchored → re-measure).
- Keep: geometry, meshes, AVL+XFOIL/FlexFoil reference framework, the two
  spanwise-phenomena descriptions.

## §VII Conclusion
- Rewrite the anchor catalog: remove "the gate", "Re_Ω^f envelope condition"
  (survives as the floor 65), "s, g_c" (removed sigmoid triple), "c_V, c_2, p"
  (removed), "favorable-gradient pair cliff Re_Ω^c(λ_p) + f_λ / K_λ / K_r"
  (removed), "second gate pinch / avanci_2019" (removed). Recast around: a_max=0.19
  (Michalke), the parabola great circle / g, P=Ŝg, Re_Ω^c(P)=max(65,88 P^-0.6)
  on the Blasius N=1,9 envelope. Also "logistic switch S(z) and the gate Q" →
  S(z) survives, gate Q gone.

## Appendices
- App A: re-run all 20 χ-sheet figures (SA-AI velocity/log χ fields). Keep mesh
  views + layout prose.
- App B: re-scope "predecessor kernel" note; the a_eff derivation formula stays,
  but sigmoid-anchored numbers (0.35 a_max, saturated-sigmoid 1.45×) re-measure;
  tab:daetotals SA-AI cols re-run.
