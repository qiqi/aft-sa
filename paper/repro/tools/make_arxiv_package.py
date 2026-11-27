#!/usr/bin/env python3
"""Build the arXiv source package of sa-ai.tex and test it.

    python3 repro/tools/make_arxiv_package.py            (run in paper/)

Writes paper/arxiv_sa-ai.tar.gz and paper/arxiv_sa-ai_metadata.txt.

What goes in: sa-ai.tex and the tables it inputs, with their comments removed
and each epigraph written with its file name (see literal_figure_names);
sa-ai.bbl; new-aiaa.cls; and exactly the figure files that pdflatex opens
(taken from its -recorder list, so nothing unused is shipped).  arXiv compiles
from the package root and uses sa-ai.bbl as it is, so references.bib and the
.bst stay out.

What is checked before the archive is written:
  * sa-ai.bbl is what bibtex produces from the current .aux and references.bib;
  * the package compiles alone in an empty directory with no error, no
    undefined reference and no overfull box;
  * every \\includegraphics names a file of the package literally, and every
    figure file is named by one (arXiv checks this before it compiles);
  * the PDF it gives has the same text and the same number of included
    graphics, page for page, as paper/sa-ai.pdf.
The script stops on the first failed check.

Build sa-ai.pdf first (pdflatex, bibtex, pdflatex x2); the script compares
against it and reads sa-ai.aux.
"""
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

sys.path.append(os.path.expanduser('~/.local/lib/python3.10/site-packages'))
import fitz  # noqa: E402

PAPER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAIN = 'sa-ai'
OUT = os.path.join(PAPER, 'arxiv_sa-ai.tar.gz')
META = os.path.join(PAPER, 'arxiv_sa-ai_metadata.txt')


def run(cmd, cwd, env=None):
    return subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)


def strip_comments(text):
    """Remove TeX comments.  A line that is only a comment is deleted.  A
    comment after code is cut: where a space precedes the %, the line end then
    supplies that space; where the % touches the code, a bare % stays, because
    it is there to remove the line end."""
    out = []
    for line in text.split('\n'):
        if line.lstrip().startswith('%'):
            continue
        m = re.search(r'(?<!\\)%', line)
        if m:
            code = line[:m.start()]
            line = code.rstrip() if code[-1] in ' \t' else code + '%'
        out.append(line)
    return '\n'.join(out)


LAOZI_DEF = ('\\newcommand{\\laozi}[2][0pt]{\\par\\vspace{#1}\\begin{center}%\n'
             '  \\includegraphics{laozi_#2.pdf}\\end{center}\\vspace{-#1}}\n')


def literal_figure_names(text):
    """Write every epigraph with its file name.  arXiv reads the source for
    \\includegraphics{...} before it compiles and takes the argument as a file
    name, so the name built inside the \\laozi macro, laozi_#2.pdf, is reported
    as a missing file.  Each call is replaced by the body of the macro."""
    assert text.count(LAOZI_DEF) == 1, 'the definition of \\laozi has changed'
    text = text.replace(LAOZI_DEF, '')
    text, n = re.subn(r'\\laozi\[([0-9.]+pt)\]\{(\d+)\}',
                      r'\\par\\vspace{\1}\\begin{center}\\includegraphics{laozi_\2.pdf}'
                      r'\\end{center}\\vspace{-\1}', text)
    assert n and '\\laozi' not in text, 'a \\laozi call was not replaced'
    return text


def page_text(pdf):
    d = fitz.open(pdf)
    return [' '.join(p.get_text().split()) for p in d]


def main():
    os.chdir(PAPER)
    for need in (MAIN + '.tex', MAIN + '.aux', MAIN + '.bbl', MAIN + '.pdf', MAIN + '.toc'):
        assert os.path.exists(need), 'build the paper first: missing ' + need
    assert os.path.getmtime(MAIN + '.pdf') >= os.path.getmtime(MAIN + '.tex'), \
        'sa-ai.pdf is older than sa-ai.tex: rebuild the paper first'

    work = tempfile.mkdtemp(prefix='arxiv_saai_')
    rec, bib, pkg = (os.path.join(work, d) for d in ('rec', 'bib', 'pkg'))
    for d in (rec, bib, pkg):
        os.mkdir(d)

    # 1. sa-ai.bbl is current
    for f in (MAIN + '.aux', 'references.bib', 'new-aiaa.bst'):
        shutil.copy(f, bib)
    r = run(['bibtex', MAIN], bib)
    assert os.path.exists(os.path.join(bib, MAIN + '.bbl')), r.stdout
    assert open(os.path.join(bib, MAIN + '.bbl')).read() == open(MAIN + '.bbl').read(), \
        'sa-ai.bbl is stale: run bibtex and pdflatex twice'
    nref = open(MAIN + '.bbl').read().count('\\bibitem')

    # 2. the files pdflatex opens
    for f in (MAIN + '.aux', MAIN + '.toc', MAIN + '.out', MAIN + '.bbl'):
        if os.path.exists(f):
            shutil.copy(f, rec)
    r = run(['pdflatex', '-recorder', '-interaction=nonstopmode',
             '-output-directory=' + rec, MAIN], PAPER)
    used = set()
    for line in open(os.path.join(rec, MAIN + '.fls')):
        if line.startswith('INPUT '):
            f = line[6:].strip()
            f = f if f.startswith('/') else os.path.normpath(os.path.join(PAPER, f))
            if f.startswith(PAPER + '/'):
                used.add(os.path.relpath(f, PAPER))
    used = sorted(used)
    odd = [f for f in used if re.search(r'[^A-Za-z0-9_./\-]', f)]
    assert not odd, 'file names arXiv may reject: %s' % odd

    # 3. stage
    for f in used:
        dst = os.path.join(pkg, f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if f.endswith('.tex'):
            text = strip_comments(open(f).read())
            if f == MAIN + '.tex':
                text = literal_figure_names(text)
            open(dst, 'w').write(text)
        else:
            shutil.copy(f, dst)
    shutil.copy(MAIN + '.bbl', pkg)
    staged = sorted(os.path.relpath(os.path.join(dp, f), pkg)
                    for dp, _, fs in os.walk(pkg) for f in fs)

    # 3b. what arXiv checks before it compiles: every \\includegraphics names a
    # file of the package literally, and every figure file is named by one
    named = set()
    for f in staged:
        if f.endswith('.tex'):
            for arg in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}', open(os.path.join(pkg, f)).read()):
                assert not re.search(r'[#\\]', arg), 'figure name built by a macro: %s' % arg
                named.add('figs/' + arg)
    figs = {f for f in staged if f.startswith('figs/')}
    assert named == figs, 'named but absent: %s; present but not named: %s' % (
        sorted(named - figs), sorted(figs - named))

    # 4. compile the package alone, the way arXiv does (no bibtex: the .bbl is used)
    env = dict(os.environ, TEXINPUTS='.:', BIBINPUTS='.:', BSTINPUTS='.:')
    for _ in range(4):
        r = run(['pdflatex', '-interaction=nonstopmode', MAIN], pkg, env)
    log = open(os.path.join(pkg, MAIN + '.log'), errors='replace').read()
    assert not re.search(r'(?m)^! ', log), 'LaTeX error in the package build'
    for bad in ('undefined', 'Overfull', 'Rerun to get', 'not found', 'LaTeX Error'):
        assert bad not in log, 'package build log contains %r' % bad
    new, old = page_text(os.path.join(pkg, MAIN + '.pdf')), page_text(MAIN + '.pdf')
    assert len(new) == len(old), 'page count %d != %d' % (len(new), len(old))
    diff = [i for i, (a, b) in enumerate(zip(new, old)) if a != b]
    assert not diff, 'pages whose text differs from paper/sa-ai.pdf: %s' % diff
    nx = [[len(p.get_xobjects()) for p in fitz.open(q)]
          for q in (os.path.join(pkg, MAIN + '.pdf'), MAIN + '.pdf')]
    assert nx[0] == nx[1], 'the number of included graphics differs on some page'
    pdf_mb = os.path.getsize(os.path.join(pkg, MAIN + '.pdf')) / 1e6

    # 5. archive: only the staged sources, none of the build products
    with tarfile.open(OUT, 'w:gz') as tar:
        for f in staged:
            tar.add(os.path.join(pkg, f), arcname=f)
    src_mb = sum(os.path.getsize(os.path.join(pkg, f)) for f in staged) / 1e6

    # 6. metadata for the submission form (ASCII only; arXiv limit 1920 characters)
    tex = open(MAIN + '.tex').read()
    title = re.search(r'\\title\{(.*?)\}\s*\n\s*\n', tex, re.S).group(1)
    abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', tex, re.S).group(1)

    def plain(s):
        s = ' '.join(s.split()).replace('--', '-')
        assert all(ord(c) < 128 for c in s), 'non-ASCII character in the metadata'
        return s
    title, abstract = plain(title), plain(abstract)
    assert len(abstract) <= 1920, 'abstract is %d characters; arXiv accepts 1920' % len(abstract)
    aux = open(MAIN + '.aux').read()
    nfig = max(int(n) for n in re.findall(r'\\newlabel\{f:[^}]*\}\{\{(\d+)\}', aux))
    ntab = max(int(n) for n in re.findall(r'\\newlabel\{t(?:ab)?:[^}]*\}\{\{(\d+)\}', aux))
    printed = len(old) - 1          # the inscription page before the title is unnumbered
    with open(META, 'w') as f:
        f.write('Title:\n%s\n\nAuthors:\nQiqi Wang\n\nAbstract (%d characters, limit 1920):\n%s\n\n'
                % (title, len(abstract), abstract))
        f.write('Comments:\n%d pages, %d figures, %d tables\n\n' % (printed, nfig, ntab))
        f.write('Package: %s\n%d files, %.1f MB of sources; the PDF is %d pages (%d numbered '
                'and one unnumbered inscription page) and %.1f MB; %d references.\n'
                % (os.path.basename(OUT), len(staged), src_mb, len(old), printed, pdf_mb, nref))

    kinds = {}
    for f in staged:
        kinds[os.path.splitext(f)[1]] = kinds.get(os.path.splitext(f)[1], 0) + 1
    print('package   %s  (%.1f MB compressed, %.1f MB of sources)' % (OUT, os.path.getsize(OUT) / 1e6, src_mb))
    print('files     %d: %s' % (len(staged), kinds))
    print('test PDF  %d pages, %.1f MB, text identical to paper/sa-ai.pdf' % (len(old), pdf_mb))
    print('counts    %d figures, %d tables, %d references, abstract %d characters' % (nfig, ntab, nref, len(abstract)))
    print('metadata  %s' % META)
    print('work dir  %s' % work)


if __name__ == '__main__':
    main()
