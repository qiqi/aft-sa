#!/usr/bin/env python3
"""Daodejing epigraphs -> paper/figs/laozi_<chapter>.pdf, and the title
inscription -> paper/figs/title_inscription.pdf (vector graphics).

The nine epigraphs of sa-ai.tex are set in clerical script (lishu), the hand
of Han-dynasty documents, in the classical layout: columns read top to bottom
and run right to left, one phrase per column, with no punctuation. This TeX
installation cannot load a TrueType CJK font or set vertical text, so each
epigraph is drawn here and included by sa-ai.tex as a graphic at its natural
size.

Font: TW-MOE-Li 3.0, the clerical-script font of the Ministry of Education,
Republic of China (Taiwan). Licence: Creative Commons Attribution-NoDerivatives
3.0 Taiwan; credit "Ministry of Education, Republic of China". The font file is
NOT in the repository. Download and unpack
    https://language.moe.gov.tw/001/Upload/Files/site_content/M0001/MoeLI-3.0.zip
and save the .ttf as repro/tools/fonts/MoeLI.ttf (that folder is git-ignored).
The font has 4,808 characters; it lacks the first two of the full chapter-11
line, which is why that epigraph is the shortened form.

Run from paper/:  python3 repro/tools/make_laozi_epigraphs.py
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = os.path.join(HERE, 'fonts', 'MoeLI.ttf')
SIZE_PT = 11.0                      # a little above the 10 pt body type
CHAR_PITCH = 1.12*SIZE_PT           # distance between characters in a column
COL_PITCH = 1.50*SIZE_PT            # distance between columns
PAD = 1.0                           # margin around the ink, in points

# Classical layout: each phrase is one column, read top to bottom, and the
# columns run right to left. Spaces and punctuation are not needed.
EPIGRAPHS = {                       # chapter -> phrases, in reading order
    25: ['人法地', '地法天', '天法道', '道法自然'],
    11: ['當其無', '有器之用'],
    42: ['道生一', '一生二', '二生三', '三生萬物'],
    77: ['天之道', '損有餘', '而補不足'],
    40: ['天下萬物生於有', '有生於無'],
    51: ['道生之', '德畜之', '物形之', '勢成之'],
    64: ['合抱之木', '生於毫末'],
    9: ['功遂身退', '天之道'],
    41: ['大方無隅', '大器晚成', '大音希聲', '大象無形'],
}

# The model's name as a title inscription, alone on the page before the title
# page: Spalart / Allmaras / autogenous inception / laminar and turbulent in
# one equation. Sixteen characters, four columns of four.
TITLE = ['斯帕拉特', '奧馬拉斯', '自然肇始', '層湍一式']
TITLE_SIZE_PT = 16.0
# The inscription is framed like a seal: a rounded rectangle around the block.
FRAME_GAP = 0.55                    # space between ink and frame, in ems
FRAME_LINE = 0.075                  # line width of the frame, in ems
FRAME_ROUND = 0.60                  # corner radius, in ems


def draw(phrases, out, size, frame=False):
    """One phrase per column, top to bottom; columns right to left."""
    char_pitch, col_pitch = CHAR_PITCH*size/SIZE_PT, COL_PITCH*size/SIZE_PT
    pad = PAD*size/SIZE_PT
    lw = FRAME_LINE*size
    if frame:
        pad = (FRAME_GAP + FRAME_LINE)*size
    ncol, nrow = len(phrases), max(len(p) for p in phrases)
    w = (ncol - 1)*col_pitch + size + 2*pad
    h = (nrow - 1)*char_pitch + size + 2*pad
    fig = plt.figure(figsize=(w/72.0, h/72.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis('off')
    for c, phrase in enumerate(phrases):
        x = w - pad - size/2 - c*col_pitch            # first phrase on the right
        for r, ch in enumerate(phrase):
            y = h - pad - size/2 - r*char_pitch       # first character on top
            ax.text(x, y, ch, fontproperties=PROP, fontsize=size,
                    ha='center', va='center')
    if frame:
        from matplotlib.patches import FancyBboxPatch
        r = FRAME_ROUND*size
        ax.add_patch(FancyBboxPatch(
            (lw/2 + r, lw/2 + r), w - lw - 2*r, h - lw - 2*r,
            boxstyle='round,pad=%g,rounding_size=%g' % (r, r),
            linewidth=lw, edgecolor='black', facecolor='none'))
    fig.savefig(out, transparent=True)
    plt.close(fig)
    print('wrote %s  (%.0f x %.0f pt)' % (out, w, h))


def main():
    global PROP
    if not os.path.exists(FONT):
        sys.exit('font not found: %s (see the docstring for where to get it)' % FONT)
    PROP = FontProperties(fname=FONT)
    matplotlib.rcParams['pdf.fonttype'] = 42
    for chapter, phrases in EPIGRAPHS.items():
        draw(phrases, os.path.join('figs', 'laozi_%d.pdf' % chapter), SIZE_PT)
    draw(TITLE, os.path.join('figs', 'title_inscription.pdf'), TITLE_SIZE_PT,
         frame=True)


if __name__ == '__main__':
    main()
