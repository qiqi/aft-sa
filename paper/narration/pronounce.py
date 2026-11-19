"""Pronunciation layer applied at synthesis time.

narrated_script.txt keeps the author's wording and punctuation. This layer only
changes how written forms are voiced: abbreviations (Fig., Eq., Sec.), en and
em dashes, ratios, model and airfoil designations, file paths, and the symbol a.
"""
import re

SYMBOL_A = '\U0001D44E'
ROMAN = {'I': '1', 'II': '2', 'III': '3', 'IV': '4', 'V': '5', 'VI': '6'}

RULES = [
    # ---- abbreviations
    (r'\bFigs\.\s*', 'Figures '),
    (r'\bFig\.\s*', 'Figure '),
    (r'\bEqs\.\s*', 'Equations '),
    (r'\bEq\.\s*', 'Equation '),
    (r'\bSec\.\s*', 'Section '),
    (r'\bch\.\s*(?=\d)', 'chapter '),
    (r'\bvs\b\.?', 'versus'),
    (r'\brms\b', 'R M S'),
    (r'\bm/s\b', 'meters per second'),
    # equation numbers in parentheses: "Equation (13)" -> "Equation 13";
    # a bare "(18)" standing for an equation -> "equation 18"
    (r'\b(Equations?) \((\d+)\)', r'\1 \2'),
    (r'(?<=\d)–\((\d+)\)', r' through \1'),
    (r'\b(Equations \d+ and) \((\d+)\)', r'\1 \2'),
    (r'(?<=[a-z,] )\((1[89]|2[01])\)(?= (is|at|gives)\b)', r'equation \1'),
    # section references: "Section II" -> "Section 2", "II.A" -> "2 A"
    (r'\bSection (I|II|III|IV|V|VI)\b', lambda m: 'Section ' + ROMAN[m.group(1)]),
    # ---- designations
    (r'NLF\(1\)-0416', 'N L F 1, 0 4 1 6'),
    (r'\bTP-1861\b', 'T P 18 61'),
    (r'\bTM-4062\b', 'T M 40 62'),
    (r'\bDAE-11/21/31\b', 'D A E 11, 21, 31'),
    (r'\bDAE-11\b', 'D A E 11'),
    (r'\bXFOIL\b', 'X-foil'),
    (r'\bmfoil\b', 'M-foil'),
    (r'\bOpenFOAM\b', 'Open Foam'),
    (r'\bFlow360\b', 'Flow 3 60'),
    (r'\bDFVLR\b', 'D F V L R'),
    (r'\bLTPT\b', 'L T P T'),
    (r'\bCST\b', 'C S T'),
    (r'\bDaodejing\b', 'Dao De Jing'),
    (r'G[öo]rtler', 'Gertler'),
    (r'\b(\d+):1\b', r'\1-to-1'),
    (r'\bL0/L1/L2\b', 'L 0, L 1, L 2'),
    # ---- nomenclature lines and headings
    (r' = ', ', '),
    # ---- file paths
    (r'[\w./-]+\.(?:json|py)\b',
     lambda m: m.group(0).replace('/', ' slash ').replace('_', ' underscore ')
     .replace('.', ' dot ')),
    # ---- slashes: lists of values, then alternatives in words
    (r'(?<=[\d)])/(?=[\d(a-z])', ', '),
    (r'(?<=[a-z])/(?=\d)', ', '),
    (r'(?<=[A-Za-z])/(?=[A-Za-z])', ' slash '),
    (r' / ', ' slash '),
    # ---- dashes
    (r'\b(L\d)–(L\d)\b', r'\1 through \2'),
    (r'“–”', 'a dash'),
    (r' – ', ', '),
    (r'–(?=\s?[\d(+-])', ' through '),      # ranges
    (r'(?<=\w)–(?=\w)', '-'),               # Falkner–Skan
    (r'\s*—\s*', ', '),
    (r'\(Footnote: ', '(Footnote. '),
    # ---- the symbol a: say the letter name
    (SYMBOL_A, '[a](/ˈA/)'),
]


def pronounce(text):
    for pat, rep in RULES:
        text = re.sub(pat, rep, text)
    return text


if __name__ == '__main__':
    import sys
    for line in sys.stdin:
        print(pronounce(line.rstrip('\n')))
