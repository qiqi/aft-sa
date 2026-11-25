"""Replace a word in a text line of a matplotlib PDF, shifting the rest of the line.

Every character of the line is text in DejaVu Sans (upright or oblique); the line is
removed character by character and written again with the new word, so sub- and
superscripts keep their size and offset. Nothing but that line is touched."""
import os, sys
import fitz, matplotlib
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


def relabel(pg, old, new, drop='', centre=False, strip_tail='', tail_sub=None, alias=ALIAS):
    """In every line containing `old`, write `new` in its place and shift the rest of
    the line. `drop` is text right after `old` that is removed as well; `strip_tail`
    is text removed from the end of the line; `tail_sub` maps single characters of the
    rest of the line to replacements. Symbols that matplotlib drew as filled outlines
    (Greek letters, relations, the degree ring) are moved with the text. Returns the
    number of lines changed."""
    n = 0
    all_lines = lines(pg)
    for direc, ch in all_lines:
        txt = ''.join(q['c'] for q in ch)
        i = txt.find(old + drop)
        if i < 0:
            continue
        vert = abs(direc[1]) > 0.5
        along = (lambda p: -p[1]) if vert else (lambda p: p[0])
        across = (lambda p: p[0]) if vert else (lambda p: p[1])
        word = ch[i:i + len(old + drop)]
        tail = ch[i + len(old + drop):]
        head = ch[:i]
        size, font, col = word[0]['size'], word[0]['font'], word[0]['col']
        # pieces of the same entry that the reader split off: same direction, same
        # baseline band, farther along
        for d2, ch2 in all_lines:
            if ch2 is ch or d2 != direc or not ch2:
                continue
            if abs(across(ch2[0]['o']) - across(word[0]['o'])) < 0.6*size and \
               0 < along(ch2[0]['o']) - along(word[-1]['o']) < 12*size and not vert:
                tail = tail + ch2
        tail.sort(key=lambda q: along(q['o']))
        if strip_tail:
            ttxt = ''.join(q['c'] for q in tail)
            assert ttxt.endswith(strip_tail), (ttxt, strip_tail)
            cut, tail = tail[len(tail) - len(strip_tail):], tail[:len(tail) - len(strip_tail)]
        else:
            cut = []
        end_old = along(tail[0]['o']) if tail else along(word[-1]['o']) + FONTS[font].text_length(word[-1]['c'], size)
        if strip_tail and not tail:
            end_old = along(cut[0]['o'])
        delta = FONTS[font].text_length(new, size) - (end_old - along(word[0]['o']))
        back = delta/2 if centre else 0.0
        gone = (head if centre else []) + word + tail + cut
        for q in gone:
            r = fitz.Rect(q['b'])
            if vert:
                r = fitz.Rect(r.x0 + 0.4*r.width, r.y0 + 0.15*r.height, r.x1 - 0.4*r.width, r.y1 - 0.15*r.height)
            else:
                r = fitz.Rect(r.x0 + 0.15*r.width, r.y0 + 0.4*r.height, r.x1 - 0.15*r.width, r.y1 - 0.4*r.height)
            pg.add_redact_annot(r, fill=False)
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
                    pg.add_redact_annot(r + (-1.0, -1.0, 1.0, 1.0), fill=False)
        pg.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE,
                            graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED if moved
                            else fitz.PDF_REDACT_LINE_ART_NONE)

        # a very small outline (the degree ring) can survive the removal; it is then
        # covered with the white of the legend before the line is written again
        left = [fitz.Rect(q['rect']) for q in pg.get_drawings() if q['type'] == 'f']
        for dr in moved:
            r = fitz.Rect(dr['rect'])
            if any(abs(r.x0 - s.x0) < 0.05 and abs(r.y0 - s.y0) < 0.05 and abs(r.x1 - s.x1) < 0.05 for s in left):
                sh = pg.new_shape()
                sh.draw_rect(r + (-0.25, -0.25, 0.25, 0.25))
                sh.finish(fill=(1, 1, 1), color=None)
                sh.commit()

        def put(text, q, shift):
            x, y = q['o']
            p = (x, y - shift) if vert else (x + shift, y)
            pg.insert_text(p, text, fontsize=q['size'], fontname=alias[q['font']], fontfile=FILES[q['font']],
                           color=q['col'], rotate=90 if vert else 0)
        if centre:
            for q in head:
                put(q['c'], q, -back)
        put(new, word[0], -back)
        extra = 0.0
        for q in tail:
            c = q['c']
            if tail_sub and c in tail_sub and q['font'] == 'DejaVuSans-Oblique':
                put(tail_sub[c], q, delta - back + extra)
                extra += FONTS[q['font']].text_length(tail_sub[c], q['size']) - FONTS[q['font']].text_length(c, q['size'])
            else:
                put(c, q, delta - back + extra)
        for dr in moved:
            sh = pg.new_shape()
            mv = fitz.Point(delta, 0)
            for it in dr['items']:
                if it[0] == 'l':
                    sh.draw_line(it[1] + mv, it[2] + mv)
                elif it[0] == 'c':
                    sh.draw_bezier(it[1] + mv, it[2] + mv, it[3] + mv, it[4] + mv)
                elif it[0] == 're':
                    sh.draw_rect(fitz.Rect(it[1]) + (delta, 0, delta, 0))
                elif it[0] == 'qu':
                    sh.draw_quad(fitz.Quad([p + mv for p in it[1]]))
                else:
                    raise ValueError(it[0])
            sh.finish(fill=dr['fill'], color=None, even_odd=dr.get('even_odd', False), closePath=True)
            sh.commit()
        n += 1
    return n


def changed_box(a, b, dpi=100):
    pa, pb = a.get_pixmap(dpi=dpi), b.get_pixmap(dpi=dpi)
    A = np.frombuffer(pa.samples, dtype=np.uint8).astype(int).reshape(pa.height, pa.width, pa.n)
    B = np.frombuffer(pb.samples, dtype=np.uint8).astype(int).reshape(pb.height, pb.width, pb.n)
    D = (abs(A - B) > 40).any(axis=2)
    ys, xs = np.nonzero(D)
    s = 72.0/dpi
    return None if not len(xs) else (round(xs.min()*s, 1), round(ys.min()*s, 1), round(xs.max()*s, 1), round(ys.max()*s, 1))
