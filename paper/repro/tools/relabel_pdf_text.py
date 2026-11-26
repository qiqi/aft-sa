"""Replace words in the text of a matplotlib PDF, shifting the rest of each line.

Every character of an edited line is text in DejaVu Sans (upright or oblique). The
line is removed character by character and written again with the new word, so sub-
and superscripts keep their size and offset. Symbols that matplotlib drew as filled
outlines (Greek letters, relations, the degree ring) are moved with the text.

All edits of a page are made in ONE redaction pass (relabel_many). Removing text in
several passes shifts unrelated text: after the first pass the page stream is
rewritten, and a later removal at the end of a line then moves the line that follows
it in the stream. Even in one pass a removal can displace the next line by a few
points; relabel_file detects such lines and writes them anew at their own place.

    from relabel_pdf_text import relabel_file
    relabel_file(src, dst, [dict(old='FlexFoil', new='Flexcompute Foil')])
"""
import collections
import os

import fitz
import matplotlib
import numpy as np

TTF = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf')
FILES = {'DejaVuSans': os.path.join(TTF, 'DejaVuSans.ttf'),
         'DejaVuSans-Oblique': os.path.join(TTF, 'DejaVuSans-Oblique.ttf')}
FONTS = {k: fitz.Font(fontfile=v) for k, v in FILES.items()}
ALIAS = {'DejaVuSans': 'djr', 'DejaVuSans-Oblique': 'djo'}


def lines(pg):
    out = []
    for b in pg.get_text('rawdict')['blocks']:
        for l in b.get('lines', []):
            ch = []
            for s in l['spans']:
                fam = 'DejaVuSans-Oblique' if 'Oblique' in s['font'] or (s['flags'] & 2) else 'DejaVuSans'
                c = s['color']
                col = ((c >> 16 & 255)/255, (c >> 8 & 255)/255, (c & 255)/255)
                for q in s['chars']:
                    ch.append(dict(c=q['c'], o=q['origin'], b=q['bbox'], font=fam, size=s['size'], col=col))
            out.append((l['dir'], ch))
    return out


def _fresh_alias(pg):
    # a font written by an earlier edit of this file is a subset; a new edit needs
    # resource names of its own, or its letters may be missing from that subset
    used = {f[4] for f in pg.get_fonts()}
    k = 0
    while any(v + str(k) in used for v in ALIAS.values()):
        k += 1
    return {key: v + str(k) for key, v in ALIAS.items()}


def _plan(pg, all_lines, taken, old, new, drop='', centre=False, strip_tail='', tail_sub=None, only=None):
    """Plans for every line that contains `old`: what to remove and what to write."""
    plans = []
    for idx, (direc, ch) in enumerate(all_lines):
        txt = ''.join(q['c'] for q in ch)
        i = txt.find(old + drop)
        if i < 0 or (only is not None and (txt != only or idx in taken)):
            continue
        assert idx not in taken, 'two edits on one line: combine them into one edit'
        used = {idx}
        vert = abs(direc[1]) > 0.5
        along = (lambda p: -p[1]) if vert else (lambda p: p[0])
        across = (lambda p: p[0]) if vert else (lambda p: p[1])
        word = ch[i:i + len(old + drop)]
        tail = ch[i + len(old + drop):]
        head = ch[:i]
        size, font = word[0]['size'], word[0]['font']
        # pieces of the same entry that the reader split off: same direction, same
        # baseline band, farther along
        for j, (d2, ch2) in enumerate(all_lines):
            # a line rewritten in place (only=...) takes no pieces; a piece belongs to one edit
            if j == idx or d2 != direc or not ch2 or vert or only is not None or j in taken:
                continue
            if abs(across(ch2[0]['o']) - across(word[0]['o'])) < 0.6*size and \
               0 < along(ch2[0]['o']) - along(word[-1]['o']) < 12*size:
                tail = tail + ch2
                used.add(j)
        tail.sort(key=lambda q: along(q['o']))
        cut = []
        if strip_tail:
            ttxt = ''.join(q['c'] for q in tail)
            assert ttxt.endswith(strip_tail), (ttxt, strip_tail)
            cut, tail = tail[len(tail) - len(strip_tail):], tail[:len(tail) - len(strip_tail)]
        if tail:
            end_old = along(tail[0]['o'])
        elif cut:
            end_old = along(cut[0]['o'])
        else:
            end_old = along(word[-1]['o']) + FONTS[font].text_length(word[-1]['c'], size)
        delta = FONTS[font].text_length(new, size) - (end_old - along(word[0]['o']))
        back = delta/2 if centre else 0.0
        gone = (head if centre else []) + word + tail + cut
        rects = []
        for q in gone:
            r = fitz.Rect(q['b'])
            if vert:
                r = fitz.Rect(r.x0 + 0.4*r.width, r.y0 + 0.15*r.height, r.x1 - 0.4*r.width, r.y1 - 0.15*r.height)
            else:
                r = fitz.Rect(r.x0 + 0.15*r.width, r.y0 + 0.4*r.height, r.x1 - 0.15*r.width, r.y1 - 0.4*r.height)
            rects.append(r)
        # outlined symbols of this entry: small filled paths in the band of the line
        moved = []
        if not vert and tail:
            y0 = min(q['b'][1] for q in word + tail); y1 = max(q['b'][3] for q in word + tail)
            x0 = word[-1]['b'][2] - 0.5; x1 = max(q['b'][2] for q in tail) + 0.5
            for dr in pg.get_drawings():
                r = fitz.Rect(dr['rect'])
                if dr['type'] == 'f' and r.width < 1.2*size and r.height < 1.2*size and \
                   r.x0 >= x0 and r.x1 <= x1 and r.y0 >= y0 - 1 and r.y1 <= y1 + 1:
                    moved.append(dr)
        writes = []                      # (text, char record, shift along the line)
        if centre:
            writes += [(q['c'], q, -back) for q in head]
        if new:
            writes.append((new, word[0], -back))
        extra = 0.0
        for q in tail:
            c = q['c']
            if tail_sub and c in tail_sub and q['font'] == 'DejaVuSans-Oblique':
                writes.append((tail_sub[c], q, delta - back + extra))
                extra += FONTS[q['font']].text_length(tail_sub[c], q['size']) - FONTS[q['font']].text_length(c, q['size'])
            else:
                writes.append((c, q, delta - back + extra))
        plans.append(dict(vert=vert, rects=rects, moved=moved, writes=writes, delta=delta, used=used,
                          baseline=across(word[0]['o']), size=size))
        taken |= used
    return plans


def relabel_many(pg, edits):
    """Apply all `edits` (dicts of the keyword arguments of relabel) to the page in
    one redaction pass. Returns the number of lines changed."""
    alias = _fresh_alias(pg)
    all_lines = lines(pg)
    taken, plans = set(), []
    for e in edits:
        got = _plan(pg, all_lines, taken, **e)
        assert got or e.get('only') is not None, ('no line contains', e['old'])
        plans += got
    any_moved = False
    for p in plans:
        for r in p['rects']:
            pg.add_redact_annot(r, fill=False)
        for dr in p['moved']:
            pg.add_redact_annot(fitz.Rect(dr['rect']) + (-1.0, -1.0, 1.0, 1.0), fill=False)
            any_moved = True
    pg.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE,
                        graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED if any_moved
                        else fitz.PDF_REDACT_LINE_ART_NONE)
    # Removing text can displace a line that was not edited (the line that follows an
    # edited one in the page stream). Such lines are reported, so that relabel_file can
    # start again and write them anew at their own place.
    now = [(''.join(q['c'] for q in ch), ch[0]['o']) for _, ch in lines(pg) if ch]
    displaced = []
    for idx, (_, ch) in enumerate(all_lines):
        if idx in taken or not ch:
            continue
        txt, o = ''.join(q['c'] for q in ch), ch[0]['o']
        # the same wording on the same baseline (a displaced line moves along its own direction)
        vert = abs(all_lines[idx][0][1]) > 0.5
        same = [n for n in now if n[0] == txt and abs(n[1][0 if vert else 1] - o[0 if vert else 1]) < 1.0]
        if same and min(abs(n[1][1 if vert else 0] - o[1 if vert else 0]) for n in same) > 0.3:
            displaced.append(txt)
    # a very small outline (the degree ring) can survive the removal; it is then
    # covered with the white of the legend before the line is written again
    left = [fitz.Rect(q['rect']) for q in pg.get_drawings() if q['type'] == 'f']
    for p in plans:
        for dr in p['moved']:
            r = fitz.Rect(dr['rect'])
            if any(abs(r.x0 - s.x0) < 0.05 and abs(r.y0 - s.y0) < 0.05 and abs(r.x1 - s.x1) < 0.05 for s in left):
                sh = pg.new_shape()
                sh.draw_rect(r + (-0.25, -0.25, 0.25, 0.25))
                sh.finish(fill=(1, 1, 1), color=None)
                sh.commit()
    for p in plans:
        vert = p['vert']
        for text, q, shift in p['writes']:
            x, y = q['o']
            pt = (x, y - shift) if vert else (x + shift, y)
            pg.insert_text(pt, text, fontsize=q['size'], fontname=alias[q['font']], fontfile=FILES[q['font']],
                           color=q['col'], rotate=90 if vert else 0)
        for dr in p['moved']:
            sh = pg.new_shape()
            mv = fitz.Point(p['delta'], 0)
            for it in dr['items']:
                if it[0] == 'l':
                    sh.draw_line(it[1] + mv, it[2] + mv)
                elif it[0] == 'c':
                    sh.draw_bezier(it[1] + mv, it[2] + mv, it[3] + mv, it[4] + mv)
                elif it[0] == 're':
                    sh.draw_rect(fitz.Rect(it[1]) + (p['delta'], 0, p['delta'], 0))
                elif it[0] == 'qu':
                    sh.draw_quad(fitz.Quad([pp + mv for pp in it[1]]))
                else:
                    raise ValueError(it[0])
            sh.finish(fill=dr['fill'], color=None, even_odd=dr.get('even_odd', False), closePath=True)
            sh.commit()
    return len(plans), displaced


def relabel(pg, old, new, drop='', centre=False, strip_tail='', tail_sub=None):
    """One edit; see relabel_many. Use relabel_many for more than one edit of a page."""
    return relabel_many(pg, [dict(old=old, new=new, drop=drop, centre=centre, strip_tail=strip_tail,
                                  tail_sub=tail_sub)])[0]


def relabel_file(src, dst, edits, after=None):
    """Apply `edits` to the first page of `src` and save `dst`. If the removal displaces
    lines that were not edited, start again and write those lines anew as well.
    `after(page)` is called before saving, for additions to the page."""
    extra = []
    for _ in range(20):
        d = fitz.open(src)
        n, displaced = relabel_many(d[0], list(edits) + extra)
        if not displaced:
            if after:
                after(d[0])
            d.subset_fonts()
            d.save(dst + '.tmp', garbage=3, deflate=True)
            os.replace(dst + '.tmp', dst)
            return n - len(extra), [e['only'] for e in extra]
        for txt in displaced:
            if any(e['only'] == txt for e in extra):
                continue
            first = next(c for c in txt if c.strip())
            extra.append(dict(old=first, new=first, only=txt))
    raise RuntimeError('lines keep moving: ' + repr(displaced))


def _spans(pdf):
    out = collections.defaultdict(list)
    for b in fitz.open(pdf)[0].get_text('rawdict')['blocks']:
        for l in b.get('lines', []):
            for s in l['spans']:
                for c in s['chars']:
                    if c['c'].strip():
                        out[round(c['origin'][1], 0) if abs(l['dir'][0]) > 0.5 else ('v', round(c['origin'][0], 0))].append(
                            (c['c'], round(c['origin'][0], 1), round(c['origin'][1], 1)))
    return out


def check_untouched(ref, new, tol=0.6):
    """Baselines whose text is the same in both files but sits at different places.
    A baseline whose wording changed is an edited line and is not reported."""
    a, b = _spans(ref), _spans(new)
    bad = []
    for key, ca in a.items():
        cb = b.get(key)
        if cb is None:
            continue
        ta = ''.join(c[0] for c in sorted(ca, key=lambda c: (c[1], c[2])))
        tb = ''.join(c[0] for c in sorted(cb, key=lambda c: (c[1], c[2])))
        if ta != tb:
            continue
        for p, q in zip(sorted(ca, key=lambda c: (c[1], c[2])), sorted(cb, key=lambda c: (c[1], c[2]))):
            if abs(p[1] - q[1]) > tol or abs(p[2] - q[2]) > tol:
                bad.append((key, ta[:40], p, q))
                break
    return bad


def changed_box(a, b, dpi=100):
    pa, pb = a.get_pixmap(dpi=dpi), b.get_pixmap(dpi=dpi)
    A = np.frombuffer(pa.samples, dtype=np.uint8).astype(int).reshape(pa.height, pa.width, pa.n)
    B = np.frombuffer(pb.samples, dtype=np.uint8).astype(int).reshape(pb.height, pb.width, pb.n)
    D = (abs(A - B) > 40).any(axis=2)
    ys, xs = np.nonzero(D)
    s = 72.0/dpi
    return None if not len(xs) else (round(xs.min()*s, 1), round(ys.min()*s, 1), round(xs.max()*s, 1), round(ys.max()*s, 1))
