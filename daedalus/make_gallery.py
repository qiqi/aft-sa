"""Build a browsing gallery for the mesh-view PNGs in
/tmp/daedalus_mesh_views. Sections per mesh directory, ordered surface views
then slices; click any image for a lightbox with arrow-key navigation.

Two outputs:
  index.html          relative image paths -- must stay inside the views dir
  index_embedded.html single self-contained file (images base64-embedded),
                      works anywhere (download/copy just this one file)

Usage: python3 make_gallery.py [views_dir]
"""
import base64
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else '/tmp/daedalus_mesh_views'


def embed(path, max_px=None):
    if max_px:
        from PIL import Image
        import io
        im = Image.open(path)
        if max(im.size) > max_px:
            im.thumbnail((max_px, max_px), Image.LANCZOS)
        # line art: 16 grayscale levels quantize cleanly and shrink ~4x
        buf = io.BytesIO()
        im.convert('L').quantize(16).save(buf, 'PNG', optimize=True)
        data = buf.getvalue()
    else:
        data = open(path, 'rb').read()
    return 'data:image/png;base64,' + base64.b64encode(data).decode()

ORDER = [
    ('surf_upper_global', 'Upper surface, global (8k: zoom in for cells)'),
    ('surf_upper_tip', 'Upper surface, tip zoom'),
    ('surf_lower_global', 'Lower surface, global (8k: zoom in for cells)'),
    ('surf_lower_tip', 'Lower surface, tip zoom'),
    ('surf_iso_tip', 'Isometric, tip zoom'),
    ('surf_front_root', 'Front view (looking downstream), root'),
    ('surf_front_tip', 'Front view, tip'),
    ('surf_back_root', 'Back view (looking upstream), root'),
    ('surf_back_tip', 'Back view, tip'),
    ('surf_side_from_tip', 'Side view from beyond the tip'),
    ('slice_span_root', 'Spanwise cut near the root'),
    ('slice_span_mid', 'Spanwise cut at mid-span'),
    ('slice_span_neartip', 'Spanwise cut 12 cm inboard of the tip'),
    ('slice_span_on_tip', 'Spanwise cut exactly on the tip (+1e-4 m)'),
    ('slice_xcut_qc_full', 'Quarter-chord x-cut, whole wing'),
    ('slice_xcut_qc_tip', 'Quarter-chord x-cut, tip zoom'),
    ('slice_farfield_global', 'Far field (spanwise cut near root)'),
    ('size_vs_dist_v1', 'Cell size vs wall distance (v1, average-edge seeding)'),
    ('size_vs_dist_v2', 'Cell size vs wall distance (v2, area-equivalent seeding)'),
    ('sectional_cl', 'Sectional cl(eta), wind axes: RANS native strips vs AVL'),
    ('sectional_cd', 'Sectional cd(eta), wind axes: RANS vs AVL cdi + XFOIL cdp'),
    ('transition_eta', 'Transition location + separation bubbles: SA-AI vs XFOIL'),
    ('polar', 'L0 SA-AI polar (CL-CD, CL-alpha): RANS vs AVL + XFOIL'),
    ('sectional_cl_sweep', 'Sectional cl(eta) across the alpha sweep'),
    ('transition_eta_sweep', 'Upper-surface transition vs alpha: SA-AI vs XFOIL'),
    ('l1_vs_l0_sectional_cl', 'Grid refinement: sectional cl L0 vs L1 (alpha=4), tip zoom'),
    ('l1_vs_l0_transition', 'Grid refinement: upper transition front L0 vs L1 (alpha=4)'),
    ('polar_levels', 'Grid-refinement polar trails: L0-L2 both families vs AVL+XFOIL'),
    ('transition_levels', 'Transition front + bubble length vs grid level (alpha=4,5)'),
    ('totals_table', 'SA-AI CL/CD totals by grid level and alpha'),
    ('section_chi_eta030_a4', 'Section eta=0.305 alpha=4: SA-AI chi(x) L1/L2 vs mfoil e^N envelope'),
    ('section_chi_eta030_a5', 'Section eta=0.305 alpha=5: SA-AI chi(x) L1/L2 vs mfoil e^N envelope'),
    ('chi_map_ogrid_L1_a4', 'O-grid L1 alpha=4: near-wall max chi color map (upper/lower)'),
    ('chi_contours_ogrid_L1_a4', 'O-grid L1 alpha=4: max chi contour lines, top view'),
    ('chi_map_cavity_L1_a4', 'Cavity L1 alpha=4: near-wall max chi color map (upper/lower)'),
    ('chi_contours_cavity_L1_a4', 'Cavity L1 alpha=4: max chi contour lines, top view'),
    ('chi_map_ogrid_L2_a4', 'O-grid L2 alpha=4: near-wall max chi color map (upper/lower)'),
    ('chi_contours_ogrid_L2_a4', 'O-grid L2 alpha=4: max chi contour lines, top view'),
    ('chi_map_ogrid_L2_a5', 'O-grid L2 alpha=5: near-wall max chi color map (upper/lower)'),
    ('chi_contours_ogrid_L2_a5', 'O-grid L2 alpha=5: max chi contour lines, top view'),
    ('chi_map_ogrid_L2_a6', 'O-grid L2 alpha=6: near-wall max chi color map (upper/lower)'),
    ('chi_contours_ogrid_L2_a6', 'O-grid L2 alpha=6: max chi contour lines, top view'),
    ('chi_map_cavity_L2_a4', 'Cavity L2 alpha=4: near-wall max chi color map (upper/lower)'),
    ('chi_contours_cavity_L2_a4', 'Cavity L2 alpha=4: max chi contour lines, top view'),
    ('chi_map_cavity_L2_a5', 'Cavity L2 alpha=5: near-wall max chi color map (upper/lower)'),
    ('chi_contours_cavity_L2_a5', 'Cavity L2 alpha=5: max chi contour lines, top view'),
    ('chi_map_cavity_L2_a6', 'Cavity L2 alpha=6: near-wall max chi color map (upper/lower)'),
    ('chi_contours_cavity_L2_a6', 'Cavity L2 alpha=6: max chi contour lines, top view'),
    ('chi_map_ogrid_L1_a5', 'O-grid L1 alpha=5: near-wall max chi color map (upper/lower)'),
    ('chi_contours_ogrid_L1_a5', 'O-grid L1 alpha=5: max chi contour lines, top view'),
    ('chi_map_ogrid_L1_a6', 'O-grid L1 alpha=6: near-wall max chi color map (upper/lower)'),
    ('chi_contours_ogrid_L1_a6', 'O-grid L1 alpha=6: max chi contour lines, top view'),
    ('chi_map_cavity_L1_a5', 'Cavity L1 alpha=5: near-wall max chi color map (upper/lower)'),
    ('chi_contours_cavity_L1_a5', 'Cavity L1 alpha=5: max chi contour lines, top view'),
    ('chi_map_cavity_L1_a6', 'Cavity L1 alpha=6: near-wall max chi color map (upper/lower)'),
    ('chi_contours_cavity_L1_a6', 'Cavity L1 alpha=6: max chi contour lines, top view'),
]

SECTIONS = [
    ('ogrid', 'Mesh — stacked O-grid (722k nodes, all hexes)'),
    ('cavity', 'Mesh — Flynn360 cavity/tet (area-equivalent sizing)'),
    ('sectional', 'Sectional comparison — RANS vs AVL + XFOIL strips'),
    ('chi', 'Transition diagnostic — near-wall max chi surface maps (SA-AI)'),
]

cards, items = [], []
for dirname, title in SECTIONS:
    d = os.path.join(ROOT, dirname)
    if not os.path.isdir(d):
        continue
    present = set(os.listdir(d))
    cards.append(f'<h2 id="{dirname}">{title}</h2><div class="grid">')
    for stem, caption in ORDER:
        fn = f'{stem}.png'
        if fn not in present:
            continue
        rel = f'{dirname}/{fn}'
        idx = len(items)
        items.append((rel, f'{title} — {caption}'))
        cards.append(
            f'<figure><a href="{rel}" onclick="return openLb({idx})">'
            f'<img loading="lazy" src="{rel}" alt=""></a>'
            f'<figcaption>{caption}</figcaption></figure>')
    cards.append('</div>')

items_js = ',\n'.join(f'["{r}",{c!r}]' for r, c in items)
nav = ' &middot; '.join(f'<a href="#{d}">{d}</a>' for d, _ in SECTIONS
                        if os.path.isdir(os.path.join(ROOT, d)))

html = f"""<!doctype html>
<meta charset="utf-8">
<title>Daedalus L0 wing meshes — view gallery</title>
<style>
  body {{ font: 14px/1.45 system-ui, sans-serif; margin: 24px auto; max-width: 1500px;
         padding: 0 16px; color: #1d2530; background: #fafbfc; }}
  h1 {{ font-size: 22px; }} h2 {{ margin-top: 36px; border-bottom: 1px solid #d5dbe2;
         padding-bottom: 6px; font-size: 17px; }}
  .nav {{ color: #5a6572; margin-bottom: 4px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
           gap: 14px; }}
  figure {{ margin: 0; background: #fff; border: 1px solid #e2e7ec; border-radius: 6px;
            padding: 8px; }}
  figure img {{ width: 100%; height: 200px; object-fit: contain; display: block;
                background: #fff; }}
  figcaption {{ font-size: 12.5px; color: #45505c; padding-top: 6px; }}
  a {{ color: #23558c; }}
  #lb {{ position: fixed; inset: 0; background: rgba(15,18,22,.93); display: none;
         flex-direction: column; align-items: center; justify-content: center; z-index: 9; }}
  #lb img {{ max-width: 96vw; max-height: 88vh; background: #fff; }}
  #lb .cap {{ color: #cfd6dd; padding: 10px 16px; font-size: 13.5px; text-align: center; }}
  #lb .hint {{ color: #7d8894; font-size: 12px; }}
</style>
<h1>Daedalus L0 wing meshes &mdash; view gallery</h1>
<p class="nav">Sections: {nav}. Click an image; then &larr;/&rarr; to step through, Esc to close.</p>
{''.join(cards)}
<div id="lb" onclick="closeLb()">
  <img id="lbimg" src="">
  <div class="cap" id="lbcap"></div>
  <div class="hint">&larr;/&rarr; navigate &middot; Esc or click to close</div>
</div>
<script>
const ITEMS = [{items_js}];
let cur = -1;
function show() {{
  document.getElementById('lbimg').src = ITEMS[cur][0];
  document.getElementById('lbcap').textContent = (cur+1) + ' / ' + ITEMS.length + ' — ' + ITEMS[cur][1];
}}
function openLb(i) {{ cur = i; show();
  document.getElementById('lb').style.display = 'flex'; return false; }}
function closeLb() {{ document.getElementById('lb').style.display = 'none'; }}
document.addEventListener('keydown', e => {{
  if (document.getElementById('lb').style.display !== 'flex') return;
  if (e.key === 'Escape') closeLb();
  if (e.key === 'ArrowRight') {{ cur = (cur + 1) % ITEMS.length; show(); }}
  if (e.key === 'ArrowLeft') {{ cur = (cur - 1 + ITEMS.length) % ITEMS.length; show(); }}
}});
</script>
"""
out = os.path.join(ROOT, 'index.html')
open(out, 'w').write(html)
print(f'wrote {out} with {len(items)} images')

# single-file variants: swap every image reference for a data URI
for suffix, max_px in (('embedded', None), ('lite', 1400)):
    emb = html
    for rel, _ in items:
        u = embed(os.path.join(ROOT, rel), max_px)
        emb = emb.replace(f'href="{rel}"', 'href="#"')
        emb = emb.replace(f'src="{rel}"', f'src="{u}"')
        emb = emb.replace(f'["{rel}",', f'["{u}",')
    out2 = os.path.join(ROOT, f'index_{suffix}.html')
    open(out2, 'w').write(emb)
    print(f'wrote {out2} ({os.path.getsize(out2)/1e6:.1f} MB, self-contained'
          + (f', images downscaled to {max_px}px)' if max_px else ')'))
