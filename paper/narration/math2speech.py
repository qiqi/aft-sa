#!/usr/bin/env python3
"""LaTeX math -> unambiguous spoken English, for the paper's notation.

Two layers:
  1. OVERRIDES (math_overrides.py): hand-written readings for every display
     equation and every inline expression with grouping, fractions, roots,
     integrals, operators, sets or tuples.
  2. speak(): a small rule-based reader for the simple, regular expressions
     (a symbol with subscripts and superscripts, a relation, a number with
     units). It refuses anything it does not fully understand, so nothing is
     ever guessed: an expression is either read by rule, or by hand.

Reading conventions
  * Upper-case Latin letters are announced "capital"; lower-case are bare.
    Upper-case Omega and Delta are "capital omega", "capital delta".
  * Subscripts: "sub ...". A comma closes the subscript before a superscript:
    "u sub tau, squared".
  * Juxtaposed symbols are read with an explicit "times".
  * Re and Tu are spelled "R e", "T u".
  * The letter a, as a symbol, is written {a} so the synthesizer can say the
    letter name and not the article.
"""
import re


class Unhandled(Exception):
    pass


GREEK = {
    'alpha': 'alpha', 'beta': 'beta', 'gamma': 'gamma', 'delta': 'delta',
    'eta': 'eta', 'theta': 'theta', 'kappa': 'kappa', 'lambda': 'lambda',
    'mu': 'mu', 'nu': 'nu', 'xi': 'xi', 'rho': 'rho', 'sigma': 'sigma',
    'tau': 'tau', 'phi': 'phi', 'chi': 'chi', 'omega': 'omega', 'zeta': 'zeta',
    'Omega': 'capital omega', 'Delta': 'capital delta', 'ell': 'script ell',
    'infty': 'infinity',
}
WORDS = {  # roman labels in subscripts and superscripts
    'crit': 'crit', 'loc': 'loc', 'max': 'max', 'sep': 'sep', 'root': 'root',
    'eq': 'e q', 'visc': 'visc', 'peak': 'peak', 'Mack': 'Mack', 'ai': '{a} i',
    'te': 't e', 'tr': 't r', 'up': 'up', 'SA': 'S A', 'AI': 'A I', 'TS': 'T S',
    'LS': 'L S', 'reatt': 'reattach', 'c': 'c',
}
RELATIONS = {
    '=': 'equals', '<': 'is less than', '>': 'is greater than',
    r'\le': 'is less than or equal to', r'\ge': 'is greater than or equal to',
    r'\approx': 'is approximately', r'\sim': 'is of order',
    r'\gtrsim': 'is greater than about', r'\lesssim': 'is less than about',
    r'\equiv': 'is identically', r'\to': 'goes to', r'\neq': 'is not equal to',
    ':=': 'is defined as',
}
PREFIX_REL = {  # a relation with nothing on its left
    '<': 'less than', '>': 'greater than', r'\sim': 'about',
    r'\approx': 'approximately', r'\le': 'at most', r'\ge': 'at least',
    r'\gtrsim': 'more than about', r'\lesssim': 'less than about',
}
OPERATORS = {r'\max': 'the maximum', r'\min': 'the minimum',
             r'\ln': 'the natural log', r'\log': 'log'}
ACCENTS = {r'\hat': 'hat', r'\tilde': 'tilde', r'\bar': 'bar'}

TOKEN = re.compile(r'''
    (?P<num>\d+(?:\.\d+)?)
  | (?P<id>Re(?![a-z])|Tu(?![a-z]))
  | (?P<cmd>\\[a-zA-Z]+)
  | (?P<esc>\\[%{}|])
  | (?P<defeq>:=)
  | (?P<ch>\S)
''', re.X)


def tokenize(s):
    s = re.sub(r'(\d)\\,(\d{3})(?!\d)', r'\1\2', s)   # 45\,000
    s = s.replace('{,}', '')                           # 200{,}000
    s = re.sub(r'\\[!,;: ]|~', ' ', s)
    out = []
    for m in TOKEN.finditer(s):
        out.append((m.lastgroup, m.group(0)))
    return out


def letter(c):
    if c == 'a':
        return '{a}'
    return ('capital ' + c) if c.isupper() else c


class Reader:
    def __init__(self, toks):
        self.t = toks
        self.i = 0

    def peek(self, k=0):
        return self.t[self.i + k] if self.i + k < len(self.t) else (None, None)

    def next(self):
        tok = self.peek()
        self.i += 1
        return tok

    # ---- a brace group or a single token, as a token list
    def arg(self):
        kind, v = self.next()
        if v == '{':
            depth, out = 1, []
            while True:
                kind, v = self.next()
                if v is None:
                    raise Unhandled('unbalanced')
                if v == '{':
                    depth += 1
                elif v == '}':
                    depth -= 1
                    if depth == 0:
                        return out
                out.append((kind, v))
        if v is None:
            raise Unhandled('missing argument')
        return [(kind, v)]

    # ---- subscript / superscript labels
    def label(self, toks):
        """Read a label: letters and digits one by one, words from WORDS."""
        out, i = [], 0
        while i < len(toks):
            kind, v = toks[i]
            if v == r'\mathrm':
                j = i + 1
                if toks[j][1] == '{':
                    word = ''
                    j += 1
                    while toks[j][1] != '}':
                        word += toks[j][1]
                        j += 1
                    j += 1
                else:
                    word = toks[j][1]
                    j += 1
                for part in word.split(','):
                    if part not in WORDS:
                        raise Unhandled('label word ' + part)
                    out.append(WORDS[part])
                    out.append('comma')
                out.pop()
                i = j
                continue
            if kind == 'num':
                out.append(v)
            elif kind == 'cmd' and v[1:] in GREEK:
                out.append(GREEK[v[1:]])
            elif kind == 'cmd' and v == r'\max':
                out.append('max')
            elif kind == 'id':
                out.append(' '.join(v))
            elif kind == 'ch' and v.isalpha():
                # runs of capitals are spelled plain (L S, T S); one is "capital"
                run = v
                j = i + 1
                while j < len(toks) and toks[j][0] == 'ch' and toks[j][1].isalpha():
                    run += toks[j][1]
                    j += 1
                if len(run) > 1 and run.isupper():
                    out.append(' '.join(run))
                elif run in WORDS and len(run) > 1:
                    out.append(WORDS[run])
                else:
                    out.extend(letter(c) for c in run)
                i = j
                continue
            elif v == ',':
                out.append('comma')
            elif v == '*':
                out.append('star')
            else:
                raise Unhandled('label token ' + v)
            i += 1
        return ' '.join(out)

    def superscript(self, toks, base_is_number):
        vals = [v for _, v in toks]
        s = ''.join(vals)
        if s == '2' and not base_is_number:
            return ', squared' if self.had_sub else ' squared'
        if s == '3' and not base_is_number:
            return ', cubed' if self.had_sub else ' cubed'
        if s == r'\circ':
            return ' degrees'
        if s == '*':
            return ' star'
        if s == '+':
            return ' plus'
        if s == r'\dagger':
            return ' dagger'
        if s == r'\ddagger':
            return ' double dagger'
        sep = ',' if self.had_sub else ''
        if vals and vals[0] == r'\mathrm':
            return sep + ' superscript ' + self.label(toks)
        sign = ''
        if vals and vals[0] == '-':
            sign, toks = 'minus ', toks[1:]
        if len(toks) == 1 and (toks[0][0] == 'num' or
                               (toks[0][0] == 'ch' and toks[0][1].isalpha())):
            v = toks[0][1]
            return sep + ' to the power ' + sign + (v if toks[0][0] == 'num' else letter(v))
        raise Unhandled('superscript ' + s)

    # ---- one atom with its primes, subscript and superscript
    def atom(self):
        kind, v = self.next()
        self.had_sub = False
        is_num = False
        if kind == 'num':
            text, is_num = v, True
        elif kind == 'id':
            text = ' '.join(v)
        elif kind == 'ch' and v.isalpha():
            text = letter(v)
        elif kind == 'cmd' and v == r'\Delta' and (
                self.peek()[0] in ('id', 'cmd') or
                (self.peek()[0] == 'ch' and self.peek()[1].isalpha())):
            inner, _ = self.atom()
            return 'capital delta ' + inner, 'sym'
        elif kind == 'cmd' and v[1:] in GREEK:
            text = GREEK[v[1:]]
        elif kind == 'cmd' and v in ACCENTS:
            inner = Reader(self.arg())
            text = inner.atom_only() + ' ' + ACCENTS[v]
        elif kind == 'cmd' and v in (r'\mathbf', r'\boldsymbol'):
            inner = Reader(self.arg())
            text = 'vector ' + inner.atom_only()
        elif kind == 'cmd' and v == r'\mathrm':
            text = self.label([(kind, v), ('ch', '{')] + self.arg() + [('ch', '}')])
        elif v == '{':
            self.i -= 1
            inner = Reader(self.arg())
            text = inner.sequence()
            return text, 'group'
        else:
            raise Unhandled('atom ' + str(v))
        # primes
        n = 0
        while self.peek()[1] == "'":
            self.next()
            n += 1
        if n:
            text += ' ' + ['prime', 'double prime', 'triple prime'][n - 1]
        sub = sup = None
        while self.peek()[1] in ('_', '^'):
            which = self.next()[1]
            a = self.arg()
            if a == [('cmd', r'\mathrm')]:      # x_\mathrm{tr}: unbraced label
                a = a + [('ch', '{')] + self.arg() + [('ch', '}')]
            if which == '_':
                sub = a
            else:
                sup = a
        if sub is not None:
            text += ' sub ' + self.label(sub)
            self.had_sub = True
        if sup is not None:
            text += self.superscript(sup, is_num)
        return text, ('num' if is_num and sup is None else 'sym')

    def atom_only(self):
        text, _ = self.atom()
        if self.peek()[1] is not None:
            raise Unhandled('accent argument')
        return text

    # ---- a flat sequence: atoms, relations, + - / , ;
    def sequence(self):
        out = []          # list of (text, kind)
        while self.peek()[1] is not None:
            kind, v = self.peek()
            prev = out[-1][1] if out else None
            if v in RELATIONS or kind == 'defeq':
                self.next()
                if prev in (None, 'sep'):
                    if v not in PREFIX_REL:
                        raise Unhandled('leading relation ' + v)
                    out.append((PREFIX_REL[v], 'rel'))
                else:
                    out.append((RELATIONS[v], 'rel'))
            elif v == '{' and self.peek(1)[1] in PREFIX_REL and self.peek(2)[1] == '}':
                self.next()
                out.append((PREFIX_REL[self.next()[1]], 'rel'))
                self.next()
            elif v in ('+', '-', r'\pm'):
                self.next()
                word = {'+': 'plus', '-': 'minus', r'\pm': 'plus or minus'}[v]
                out.append((word, 'op'))
            elif v == '/':
                self.next()
                out.append(('over', 'op'))
            elif v == r'\times':
                self.next()
                nk, nv = self.peek()
                if nv == '10' and self.peek(1)[1] == '^':
                    out.append(('times', 'op'))
                elif nk == 'num':
                    out.append(('by', 'op'))
                elif nv is None:
                    out.append(('times', 'sep'))
                else:
                    raise Unhandled('times')
            elif v in (',', ';'):
                self.next()
                out.append((',', 'sep'))
            elif v == r'\%':
                self.next()
                out.append(('percent', 'unit'))
            elif v == r'\in':
                raise Unhandled('set membership')
            elif kind == 'cmd' and v in OPERATORS:
                self.next()
                text = OPERATORS[v]
                if self.peek()[1] == '_':
                    self.next()
                    lab = self.label(self.arg())
                    text += (' base ' + lab) if v == r'\log' else (' over ' + lab)
                out.append((text + ' of', 'opr'))
            elif v == '^':            # a bare superscript mark: ^\dagger
                self.next()
                self.had_sub = False
                out.append((self.superscript(self.arg(), False).strip(), 'sym'))
            else:
                if kind == 'num' and v == '10' and self.peek(1)[1] == '^':
                    self.next()
                    self.next()
                    self.had_sub = False
                    a = self.arg()
                    out.append(('10' + self.superscript(a, True), 'sym'))
                    continue
                text, k = self.atom()
                if prev in ('sym', 'group') and k in ('sym', 'group'):
                    out.append(('times', 'op'))
                elif prev in ('sym', 'group') and k == 'num':
                    raise Unhandled('symbol then number')
                out.append((text, k))
        # assemble
        words = []
        nrel = 0
        for text, k in out:
            if k == 'rel':
                nrel += 1
                if nrel > 1 and words and not words[-1].endswith(','):
                    words[-1] += ','
            if k == 'sep':
                if text == ',':
                    words[-1:] = [words[-1] + ','] if words else []
                else:
                    words.append(text)
            else:
                words.append(text)
        s = ' '.join(words)
        s = re.sub(r'(?<![\d.])1 degrees', '1 degree', s)
        return s


def speak(latex):
    """Rule-based reading, or Unhandled."""
    r = Reader(tokenize(latex))
    s = r.sequence()
    if not s.strip():
        raise Unhandled('empty')
    return s


def norm(latex):
    return ' '.join(latex.split())


if __name__ == '__main__':
    import sys
    for a in sys.argv[1:]:
        print(a, '->', speak(a))
