"""Pixel-calibrated extractor for TM-4062 Fig.22 Cp panels (scanned).
Detects the L-axes + tick marks, calibrates pixel<->data from KNOWN tick values,
extracts the two marker traces (upper=topmost, lower=next) per x-column, and
writes a validation overlay (extracted points drawn back onto the panel image).

Usage: python extract_cp.py <panel.png> <ytick_vals_csv> <out_prefix>
  ytick_vals_csv: known Cp tick values top->bottom, e.g. "-0.8,-0.4,0,0.4,0.8,1.2"
Prints (x/c, Cp_upper, Cp_lower); saves <out_prefix>_overlay.png.
"""
import sys, numpy as np
from PIL import Image, ImageDraw

img_path, ytick_csv, outpref = sys.argv[1], sys.argv[2], sys.argv[3]
yvals = [float(v) for v in ytick_csv.split(",")]           # top -> bottom
xvals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
im = np.array(Image.open(img_path).convert("L")).astype(float)
H, W = im.shape
dark = im < 110

# --- axes ---
XA = int(np.argmax(dark[:, :int(0.35*W)].sum(0)))          # left vertical axis
sub = dark[int(0.60*H):, :].sum(1)
YB = int(0.60*H) + int(np.argmax(sub))                     # bottom horizontal axis

def cluster(idxs, gap):
    cl = []
    for v in idxs:
        if not cl or v - cl[-1][-1] > gap: cl.append([v])
        else: cl[-1].append(v)
    return [int(np.mean(c)) for c in cl]

# --- y ticks: dark protrusions just LEFT of the axis ---
ys = [y for y in range(H) if dark[y, max(0,XA-24):XA-5].sum() >= 5]
ytick_px = cluster(ys, 12)
# --- x ticks: just BELOW the axis ---
xs = [x for x in range(W) if dark[YB+5:min(H,YB+24), x].sum() >= 5]
xtick_px = cluster(xs, 12)
print(f"axes XA={XA} YB={YB}; {len(ytick_px)} yticks {ytick_px}; {len(xtick_px)} xticks {xtick_px}")

# --- calibration: fit pixel->value using matched ticks (use as many as match) ---
def fit(px, vals):
    n = min(len(px), len(vals))
    px, vals = np.array(px[:n], float), np.array(vals[:n], float)
    A = np.polyfit(px, vals, 1); return A  # slope,intercept
cy = fit(ytick_px, yvals)      # row_px -> Cp
cx = fit(xtick_px, xvals)      # col_px -> x/c
def to_cp(rp): return cy[0]*rp + cy[1]
def to_xc(cp): return cx[0]*cp + cx[1]
def px_row(cpv): return int((cpv - cy[1]) / cy[0])
def px_col(xc): return int((xc - cx[1]) / cx[0])

# --- markers: dark pixels inside the box, per x-column bin, cluster into up/low ---
up, lo = [], []
x0, x1 = px_col(0.02), px_col(0.97)
for xc in np.arange(0.02, 0.97, 0.02):
    c = px_col(xc)
    band = dark[:YB-3, c-4:c+5].sum(1)                     # dark count per row at this column
    rows = [r for r in range(5, YB-3) if band[r] >= 3]
    if not rows: continue
    grp = []
    for r in rows:
        if not grp or r - grp[-1][-1] > 22: grp.append([r])
        else: grp[-1].append(r)
    cents = [np.mean(g) for g in grp]
    up.append((xc, to_cp(cents[0])))
    if len(cents) >= 2: lo.append((xc, to_cp(cents[-1])))

up = np.array(up); lo = np.array(lo)
print("UPPER Cp:", " ".join(f"{x:.2f}:{c:+.2f}" for x, c in up))
print("LOWER Cp:", " ".join(f"{x:.2f}:{c:+.2f}" for x, c in lo))

# --- overlay for visual validation ---
rgb = Image.open(img_path).convert("RGB"); dr = ImageDraw.Draw(rgb)
for x, c in up:
    px, py = px_col(x), px_row(c); dr.ellipse([px-6,py-6,px+6,py+6], outline=(255,0,0), width=3)
for x, c in lo:
    px, py = px_col(x), px_row(c); dr.ellipse([px-6,py-6,px+6,py+6], outline=(0,120,255), width=3)
rgb.save(f"{outpref}_overlay.png"); print("wrote", f"{outpref}_overlay.png")
