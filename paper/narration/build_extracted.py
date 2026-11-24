#!/usr/bin/env python3
"""Step 2: build extracted.md, clean Markdown with intact LaTeX math.

The math and prose come from the paper's LaTeX source (sa-ai.tex and the files
it inputs), which is exact, instead of from a PDF parser's reading of the
typeset page. Numbers (equations, figures, tables, sections, citations) are
resolved from sa-ai.aux / sa-ai.bbl so the text reads as the PDF does.

What is kept: title block, abstract, epigraphs, nomenclature, every section
heading, all body text, display equations (tagged with their numbers), every
figure and table caption, table notes, footnotes, appendix, acknowledgments.
What is dropped: the reference list, the graphics, and the bodies of the
numbered data tables (two small tables that the prose leans on are kept as
rows).

Usage: build_extracted.py  (run with the narration venv's python; needs pandoc
from pypandoc_binary).
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)


def read(name):
    with open(os.path.join(PAPER, name)) as f:
        return f.read()


def strip_comments(s):
    return re.sub(r'(?<!\\)%.*\n[ \t]*', '', s)


def inline_inputs(s):
    def rep(m):
        name = m.group(1)
        if not name.endswith('.tex'):
            name += '.tex'
        return inline_inputs(strip_comments(read(name)))
    return re.sub(r'\\input\{([^}]*)\}', rep, s)


def grab(s, i):
    """s[i] == '{'. Return (content, index after the closing brace)."""
    assert s[i] == '{', s[i:i + 20]
    depth = 0
    j = i
    while True:
        c = s[j]
        if c == '\\':
            j += 2
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1


# ---------------------------------------------------------------- numbers
aux = read('sa-ai.aux')
LABEL = {m.group(1): m.group(2)
         for m in re.finditer(r'\\newlabel\{([^}]*)\}\{\{([^}]*)\}\{', aux)}
bbl = read('sa-ai.bbl')
CITE = {k: i + 1 for i, k in
        enumerate(re.findall(r'\\bibitem\[.*?\]\{([^}]*)\}', bbl, re.S))}


def cite_text(keys):
    nums = sorted(CITE[k.strip()] for k in keys.split(','))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        if j - i >= 2:
            out.append('%d--%d' % (nums[i], nums[j]))
        else:
            out.extend(str(n) for n in nums[i:j + 1])
        i = j + 1
    return '[' + ', '.join(out) + ']'


# ---------------------------------------------------------------- source
src = inline_inputs(strip_comments(read('sa-ai.tex')))
# the paper's own macro for the onset factor
src = re.sub(r'\\con(?![A-Za-z])', r'c_{\\mathrm{on}}', src)
src = re.sub(r'\\cite\{([^}]*)\}', lambda m: cite_text(m.group(1)), src)

title, _ = grab(src, src.index(r'\title') + len(r'\title'))
title = ' '.join(title.split())
a0 = src.index(r'\author[1,2]') + len(r'\author[1,2]')
author_raw, _ = grab(src, a0)
f0 = author_raw.index(r'\footnote') + len(r'\footnote')
author_note, _ = grab(author_raw, f0)
author = author_raw[:author_raw.index(r'\footnote')].strip()
affils = [' '.join(m.group(1).split())
          for m in re.finditer(r'\\affil\[\d\]\{([^}]*)\}', src)]
abstract = src[src.index(r'\begin{abstract}') + len(r'\begin{abstract}'):
               src.index(r'\end{abstract}')].strip()

body = src[src.index(r'\end{abstract}') + len(r'\end{abstract}'):
           src.index(r'\bibliography')]

# labels checked against the running counters below
counters = {'eq': 0, 'fig': 0, 'tab': 0}


def check(env_text, kind, number):
    for m in re.finditer(r'\\label\{([^}]*)\}', env_text):
        want = LABEL.get(m.group(1))
        assert want == str(number), (kind, m.group(1), want, number)


def strip_labels(s):
    return re.sub(r'\\label\{[^}]*\}', '', s)


# ---------------------------------------------------------------- equations
def eq_rep(m):
    env, inner = m.group(1), m.group(2)
    if env == 'equation*':
        return '\n\\[' + inner.strip() + '\\]\n'
    if env == 'equation':
        counters['eq'] += 1
        check(inner, 'eq', counters['eq'])
        return '\n\\[' + strip_labels(inner).strip() + \
            ' \\tag{%d}\\]\n' % counters['eq']
    rows = re.split(r'\\\\(?:\[[^\]]*\])?', inner)
    out = []
    for row in rows:
        if not row.strip():
            continue
        if r'\notag' in row:
            row = row.replace(r'\notag', '')
            out.append('\n\\[' + strip_labels(row).strip() + '\\]\n')
        else:
            counters['eq'] += 1
            check(row, 'eq', counters['eq'])
            out.append('\n\\[' + strip_labels(row).strip() +
                       ' \\tag{%d}\\]\n' % counters['eq'])
    return ''.join(out)


body = re.sub(r'\\begin\{(equation\*?|gather)\}(.*?)\\end\{\1\}', eq_rep, body,
              flags=re.S)


# ---------------------------------------------------------------- floats
def captions(inner):
    """Yield (command, text) for each \\caption / \\figcont, in order."""
    for m in re.finditer(r'\\(caption|figcont)(?=\{)', inner):
        text, _ = grab(inner, m.end())
        yield m.group(1), ' '.join(strip_labels(text).split())


def fig_rep(m):
    inner = m.group(1)
    out = []
    for cmd, text in captions(inner):
        if cmd == 'caption':
            counters['fig'] += 1
            check(inner, 'fig', counters['fig'])
            out.append('\n\n\\textbf{Figure %d.} %s\n\n' % (counters['fig'], text))
        else:
            out.append('\n\n\\textbf{Figure %d, continued.} %s\n\n'
                       % (counters['fig'], text))
    assert out, inner[:200]
    return ''.join(out)


def split_rows(tab):
    """Rows of a tabular as lists of cells; rules and spacing dropped."""
    tab = re.sub(r'\\(toprule|midrule|bottomrule|addlinespace)(\[[^\]]*\])?', '', tab)
    tab = re.sub(r'\\cmidrule(\([^)]*\))?\{[^}]*\}', '', tab)
    rows = []
    for row in re.split(r'\\\\(?:\[[^\]]*\])?', tab):
        if row.strip():
            rows.append([' '.join(c.split()) for c in row.split('&')])
    return rows


def tabular_parts(inner):
    m = re.search(r'\\begin\{tabular\}', inner)
    spec, j = grab(inner, m.end())
    e = inner.index(r'\end{tabular}')
    return inner[:m.start()], inner[j:e], inner[e + len(r'\end{tabular}'):]


def rows_to_list(rows):
    """First row is the header; each later row becomes one list item."""
    head = rows[0]
    items = []
    for r in rows[1:]:
        items.append('\\item ' + '; '.join(
            (h + ': ' + c) for h, c in zip(head, r)) + '.')
    return '\n\\begin{itemize}\n' + '\n'.join(items) + '\n\\end{itemize}\n'


def columns_to_list(rows):
    """A transposed table: the first row names the constants, and each later
    row gives one property of all of them. One list item per constant."""
    head = rows[0]
    items = []
    for j in range(1, len(head)):
        items.append('\\item ' + head[j] + ': ' + '; '.join(
            (r[0] + ' ' + r[j]) for r in rows[1:]) + '.')
    return '\n\\begin{itemize}\n' + '\n'.join(items) + '\n\\end{itemize}\n'


def tab_rep(m):
    inner = m.group(1)
    counters['tab'] += 1
    check(inner, 'tab', counters['tab'])
    caps = list(captions(inner))
    assert len(caps) == 1, inner[:200]
    out = '\n\n\\textbf{Table %d.} %s\n\n' % (counters['tab'], caps[0][1])
    before, tab, after = tabular_parts(inner)
    if 't:constants' in inner or 't:saconstants' in inner:
        out += columns_to_list(split_rows(tab))
    # a note set under the tabular is prose: keep it
    note = re.search(r'\{\\footnotesize\s+(.*)\}\s*$', after, re.S)
    if note:
        out += '\n\n' + note.group(1).strip() + '\n\n'
    return out


body = re.sub(r'\\begin\{figure\}(?:\[[^\]]*\])?(.*?)\\end\{figure\}', fig_rep,
              body, flags=re.S)
body = re.sub(r'\\begin\{table\}(?:\[[^\]]*\])?(.*?)\\end\{table\}', tab_rep,
              body, flags=re.S)


# the one unnumbered tabular set inside the running text: keep it as rows
def loose_tab(m):
    _, tab, _ = tabular_parts(m.group(0))
    return rows_to_list(split_rows(tab))


body = re.sub(r'\\begin\{center\}\\small\s*\\begin\{tabular\}.*?\\end\{tabular\}\s*\\end\{center\}',
              loose_tab, body, flags=re.S)
assert r'\begin{tabular}' not in body

# ---------------------------------------------------------------- nomenclature
n0 = body.index(r'\noindent\begin{longtable*}')
n1 = body.index(r'\end{longtable*}}') + len(r'\end{longtable*}}')
nom = body[n0:n1]
nom = nom[nom.index('\n') + 1:nom.index(r'\end{longtable*}')]
nrows = []
for row in re.split(r'\\\\', nom):
    if row.strip():
        sym, definition = row.split('&', 1)
        nrows.append('\\item %s = %s' % (sym.strip(), ' '.join(definition.split())))
k = body.rfind(r'{\renewcommand\arraystretch', 0, n0)
assert k >= 0
body = body[:k] + '\\begin{itemize}\n' + '\n'.join(nrows) + '\n\\end{itemize}\n' + body[n1:]

# ---------------------------------------------------------------- headings
ROMAN = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII']
sec = {'s': 0, 'ss': 0}


out, i = [], 0
for m in re.finditer(r'\\(section|subsection)(\*?)(?=\{)', body):
    text, end = grab(body, m.end())
    text = ' '.join(text.split())
    out.append(body[i:m.start()])
    if m.group(2):
        out.append('\n\n\\section*{%s}\n\n' % text)
    elif m.group(1) == 'section':
        sec['s'] += 1
        sec['ss'] = 0
        out.append('\n\n\\section*{%s. %s}\n\n' % (ROMAN[sec['s'] - 1], text))
    else:
        sec['ss'] += 1
        out.append('\n\n\\subsection*{%s. %s}\n\n' % (chr(64 + sec['ss']), text))
    i = end
out.append(body[i:])
body = ''.join(out)

# ---------------------------------------------------------------- references
body = re.sub(r'\\eqref\{([^}]*)\}', lambda m: '(' + LABEL[m.group(1)] + ')', body)
body = re.sub(r'\\ref\{([^}]*)\}', lambda m: LABEL[m.group(1)], body)
body = strip_labels(body)

# footnotes become parenthetical notes where they are called
while r'\footnote' in body:
    k = body.index(r'\footnote')
    text, end = grab(body, k + len(r'\footnote'))
    body = body[:k] + ' (Footnote: ' + ' '.join(text.split()) + ')' + body[end:]

# layout-only commands
body = re.sub(r'\\begin\{CJK\}\{UTF8\}\{bsmi\}(.*?)\\end\{CJK\}', r'\1', body)
body = re.sub(r'\\begin\{chinese\}(.*?)\\end\{chinese\}', r'\1', body)
# the epigraphs are graphics now; they carry no text to read
body = re.sub(r'\\laozi(\[[^\]]*\])?\{\d+\}', '', body)
body = re.sub(r'\\begin\{minipage\}\{[^}]*\}|\\end\{minipage\}', '', body)
body = re.sub(r'\\(begin|end)\{center\}', '\n\n', body)
body = re.sub(r'\\phantomsection|\\addcontentsline\{[^}]*\}\{[^}]*\}\{[^}]*\}', '', body)
body = re.sub(r'\\suppressfloats(\[[^\]]*\])?|\\clearpage|\\centering|\\itemsep\s*\d+pt|\\noindent', '', body)
body = re.sub(r'\\\\\[\d+pt\]', r'\\\\', body)

front = ('\\section*{%s}\n\n%s\n\n%s\n\n(Footnote: %s)\n\n\\section*{Abstract}\n\n%s\n\n'
         % (title, author, '\\\\\n'.join(affils), ' '.join(author_note.split()),
            abstract))
tex = front + body

with open(os.path.join(HERE, 'extracted_source.tex'), 'w') as f:
    f.write(tex)

import pypandoc  # noqa: E402
md = pypandoc.convert_text(
    tex, 'markdown-smart-raw_tex-raw_attribute-fenced_divs-bracketed_spans'
         '-header_attributes-auto_identifiers+tex_math_dollars',
    format='latex', extra_args=['--wrap=none'])
with open(os.path.join(HERE, 'extracted.md'), 'w') as f:
    f.write(md)
print('equations %(eq)d  figures %(fig)d  tables %(tab)d' % counters)
print('wrote extracted.md: %d lines, %d characters' % (md.count('\n'), len(md)))
