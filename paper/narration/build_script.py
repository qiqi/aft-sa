#!/usr/bin/env python3
"""Step 3: extracted.md -> narrated_script.txt.

Prose is carried over word for word. The only things touched outside math:
  * bracketed citation numbers are removed,
  * Markdown markup (#, *, `, list bullets, line-break backslashes) is removed,
  * the two lines of Chinese in the epigraphs are dropped (the English voice
    cannot read them; their printed English translations stay),
  * headings are announced: "II. Model" -> "Section 2. Model."
Every $...$ and $$...$$ is replaced by its spoken form (math2speech.speak, or a
hand-written entry in math_overrides.py).

The symbol a is written with the mathematical italic letter (U+1D44E) so the
synthesizer can tell it from the article.

The run ends with a word-for-word check of the prose against extracted.md.
"""
import os
import re
import sys

from math2speech import speak, Unhandled, norm
from math_overrides import INLINE, DISPLAY

HERE = os.path.dirname(os.path.abspath(__file__))
SYMBOL_A = '\U0001D44E'
ROMAN = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6}

md = open(os.path.join(HERE, 'extracted.md'), encoding='utf-8').read()

# ---------------------------------------------------------------- math
used_inline, missing = set(), {}
display_i = [0]
stats = {'inline': 0, 'display': 0, 'by_rule': 0, 'by_hand': 0}


def say_display(m):
    tex = norm(m.group(1))
    i = display_i[0]
    display_i[0] += 1
    assert i < len(DISPLAY), 'more display equations than DISPLAY entries'
    needle, spoken = DISPLAY[i]
    assert norm(needle) in tex, 'DISPLAY[%d] does not match: %s' % (i, tex[:80])
    stats['display'] += 1
    return '\n\n⟦D⟧' + spoken + '⟦/D⟧\n\n'


def say_inline(m):
    tex = norm(m.group(1))
    stats['inline'] += 1
    if tex in INLINE:
        used_inline.add(tex)
        stats['by_hand'] += 1
        return '⟦M⟧' + INLINE[tex] + '⟦/M⟧'
    try:
        spoken = speak(tex)
    except Unhandled as e:
        missing[tex] = str(e)
        return '⟦M⟧?⟦/M⟧'
    stats['by_rule'] += 1
    return '⟦M⟧' + spoken + '⟦/M⟧'


text = re.sub(r'\$\$(.*?)\$\$', say_display, md, flags=re.S)
text = re.sub(r'(?<!\\)\$(.+?)(?<!\\)\$', say_inline, text, flags=re.S)
# "the wall-distance vector $\mathbf d$": do not say the word twice
text = text.replace('vector ⟦M⟧vector ', 'vector ⟦M⟧')

if missing:
    print('NO SPOKEN FORM for %d expressions:' % len(missing))
    for k, v in sorted(missing.items()):
        print('   %s    [%s]' % (k, v))
    sys.exit(1)
unused = set(INLINE) - used_inline
if unused:
    print('INLINE entries that match nothing in the paper:')
    for k in sorted(unused):
        print('   ' + k)
    sys.exit(1)
assert display_i[0] == len(DISPLAY), (display_i[0], len(DISPLAY))


# ---------------------------------------------------------------- prose cleanup
def clean_prose(s):
    """Markup and citations out; wording untouched."""
    s = re.sub(r'[  ]*\\\[[\d,\s–-]+\\\]', '', s)      # [12], [3, 4], [4–6]
    s = s.replace('\\[', '[').replace('\\]', ']')
    s = re.sub(r'[⺀-鿿　-〿＀-￯]+', '', s)   # CJK
    s = s.replace('**', '').replace('*', '').replace('`', '')
    s = s.replace('­', '')                              # soft hyphen
    s = s.replace(' ', ' ')
    return s


paras = []
for block in re.split(r'\n\s*\n', text):
    block = block.strip()
    if not block:
        continue
    block = clean_prose(block)
    block = re.sub(r'\\\n', '\n', block)                      # hard line breaks
    block = re.sub(r'^- ', '', block)                          # list bullet
    block = ' '.join(block.split())
    block = block.strip('\\ ').strip()
    if not block:
        continue
    h = re.match(r'^(#+) (.*)$', block)
    if h:
        title = h.group(2)
        m = re.match(r'^(I|II|III|IV|V|VI)\. (.*)$', title)
        s = re.match(r'^([A-J])\. (.*)$', title)
        if h.group(1) == '#' and m:
            block = 'Section %d. %s.' % (ROMAN[m.group(1)], m.group(2))
        elif h.group(1) == '##' and s:
            block = 'Subsection %s. %s.' % (s.group(1), s.group(2))
        else:
            block = title + '.'
    paras.append(block)


def finish(p):
    p = p.replace('⟦D⟧', '').replace('⟦/D⟧', '').replace('⟦M⟧', '').replace('⟦/M⟧', '')
    return p.replace('{a}', SYMBOL_A)


script = '\n\n'.join(finish(p) for p in paras) + '\n'
with open(os.path.join(HERE, 'narrated_script.txt'), 'w', encoding='utf-8') as f:
    f.write(script)


# ---------------------------------------------------------------- the check
def prose_words(s):
    """Words outside math, markup-free, for comparing the two files."""
    s = re.sub(r'⟦D⟧.*?⟦/D⟧|⟦M⟧.*?⟦/M⟧', ' ', s, flags=re.S)
    return s.split()


src = re.sub(r'\$\$(.*?)\$\$', ' ', md, flags=re.S)
src = re.sub(r'(?<!\\)\$(.+?)(?<!\\)\$', ' ', src, flags=re.S)
src_words = []
for block in re.split(r'\n\s*\n', src):
    block = clean_prose(block.strip())
    block = re.sub(r'\\\n', '\n', block)
    block = re.sub(r'^- ', '', block)
    block = block.strip('\\ ').strip()
    h = re.match(r'^(#+) (.*)$', ' '.join(block.split()))
    if h:
        block = h.group(2)
    src_words += block.split()

out_words = []
for p in paras:
    w = prose_words(p)
    # undo the heading announcement for the comparison
    joined = ' '.join(w)
    m = re.match(r'^Section (\d)\. (.*)\.$', joined)
    s = re.match(r'^Subsection ([A-J])\. (.*)\.$', joined)
    if m:
        inv = {v: k for k, v in ROMAN.items()}
        joined = '%s. %s' % (inv[int(m.group(1))], m.group(2))
    elif s:
        joined = '%s. %s' % (s.group(1), s.group(2))
    out_words += joined.split()


def strip_tail(ws):
    return [w.rstrip('.') for w in ws]


# math sitting flush against punctuation leaves that punctuation as its own
# "word" on one side only; compare on letters-and-digits words
def core(ws):
    out = []
    for w in ws:
        w = re.sub(r'^[^\w]+|[^\w]+$', '', w)
        if w:
            out.append(w)
    return out


a, b = core(src_words), core(out_words)
headings_fix = 0
if a != b:
    # headings carry one added full stop each; align on that
    import difflib
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    diffs = [op for op in sm.get_opcodes() if op[0] != 'equal']
    print('PROSE MISMATCH in %d places:' % len(diffs))
    for tag, i1, i2, j1, j2 in diffs[:20]:
        print('  ', tag, a[i1:i2], '->', b[j1:j2])
    sys.exit(1)

total = len(script.split())
print('prose words: paper %d, script %d (identical sequence)' % (len(a), len(b)))
print('math: %(inline)d inline (%(by_rule)d by rule, %(by_hand)d by hand), '
      '%(display)d display' % stats)
print('narrated_script.txt: %d paragraphs, %d words' % (len(paras), total))
