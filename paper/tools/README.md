# Digitization & verification tools

Helpers for reading the **experimental Eppler-387 surface-pressure ($C_p$) data**
out of the scanned NASA TM-4062 report (`../eppler.pdf`) and verifying every digit
before it lands in `../data/exp_cp_tables.json`.

All scripts `os.chdir()` to the `paper/` directory on startup, so you can run them
from anywhere (e.g. `python paper/tools/mk_cp_verify.py ...`). They read
`eppler.pdf` + `data/exp_cp_tables.json` and write `figs/verify/`.

## The golden rule: read the TABLES, not the plots

`C_p` comes from the **Appendix D tables** ("... PRESSURE COEFFICIENTS FOR VARIOUS
ALPHAS"), which give 4-decimal numbers at 29 fixed `x/c` stations. Do **not**
pixel-digitize the Fig. 22 line plots — we tried that (`extract_cp.py`, kept only
for reference) and it is far less accurate than reading the tabulated digits.

## ⚠️ Two different table orientations

The scan lays the tables out **differently by Reynolds number** (all pages are
`/Rotate=0`, so this is baked into the scanned image):

| Re | Layout in the render | Tool |
|----|----------------------|------|
| 60k, 100k, 300k, 460k | α = **horizontal rows**, `x/c` runs left→right (.000→.950) | `mk_cp_verify.py` |
| 200k | α = **vertical columns**, `x/c` runs top→bottom; title printed sideways | `mk_cp_verify_col.py` (renders the page rotated 90° CW) |

If you use the wrong tool the crop is meaningless. When in doubt, render the page
and look (`showrow.py`, or `fitz` + `ROTATE_270`).

## Workflow (per column/row, per surface)

1. **Find the page.** The user gives printed page numbers; `fitz` is 0-indexed and
   the printed number is roughly `fitz_index − 3` (varies — confirm by eye). Upper
   and lower surfaces are on **separate pages**. The `.000` `x/c` row is the shared
   leading-edge point and carries the **same** value on both surfaces.
2. **Read** the target α column/row left-to-right (or top-to-bottom for 200k) into
   29 values. `x/c` stations: `.000 .005 .010 .015 .020 .025 .030 .040 .050 .060
   .075 .100 .150 .200 .250 .300 .350 .400 .450 .500 .550 .600 .650 .700 .750 .800
   .850 .900 .950`. (200k upper uses `.030/.500`; 200k lower uses `.031/.505`.)
3. **Store** in `data/exp_cp_tables.json` under `[Re][surface][alpha_col]`.
4. **Verify**: run the matching tool to render an ORIGINAL-vs-MINE side-by-side, then
   **confirm every digit matches**. Fix and re-verify on any mismatch.

```
# row-layout Re (60/100/300/460k): args = <page0idx> <rowIdx> <Re> <surf> <colkey> <name>
python tools/mk_cp_verify.py 143 2 100 upper 5.01 100k_a5_upper

# column-layout Re (200k): args = <page0idx> <Re> <surf> <colkey> <name>
python tools/mk_cp_verify_col.py 133 200 upper 7.01 200k_a7_upper
```

Verified outputs live in `figs/verify/`. OCR (tesseract, easyocr) was tried and
**fails** on this 1988 scan — hand-read + digit-match verification is the method.

## Files
- `mk_cp_verify.py` — verifier for the row-layout pages (60/100/300/460k).
- `mk_cp_verify_col.py` — verifier for the 200k column-layout pages (rotates CW,
  detects the 29 `x/c` rows from the label column).
- `showrow.py` — high-zoom single-row reader with `x/c` gridlines (row-layout pages).
- `extract_cp.py` — **deprecated** pixel-based Fig. 22 extractor; superseded by
  reading the tables. Kept for reference only.
