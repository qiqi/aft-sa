import re, sys
def prose(path):
    s=open(path).read()
    s=re.sub(r'(?s)\\begin\{(figure|table|tabbing|equation|gather|align|equation\*|gather\*|align\*|CJK|center)\*?\}.*?\\end\{\1\*?\}',' ',s)
    s=re.sub(r'(?m)^%.*','',s)
    s=re.sub(r'(?s)\$\$.*?\$\$',' MATH ',s)
    s=re.sub(r'\$[^$]*\$',' MATH ',s)
    s=re.sub(r'\\(cite|ref|eqref|label)\{[^}]*\}',' ',s)
    s=re.sub(r'\\[a-zA-Z]+\*?',' ',s)
    s=re.sub(r'[{}]',' ',s)
    s=re.sub(r'---',' ',s)
    s=re.sub(r'\s+',' ',s)
    return s
def sents(t):
    parts=re.split(r'(?<=[.!?])\s+(?=[A-Z“"\(])', t)
    return [p.strip() for p in parts if len(p.split())>2]
for path in sys.argv[1:]:
    S=sents(prose(path))
    n=[len(x.split()) for x in S]
    n.sort()
    tot=len(n)
    import statistics
    print("\n=== %s : %d sentences, median %d, mean %.1f" % (path, tot, statistics.median(n), sum(n)/tot))
    for lo,hi,lab in ((0,10,"very short <=10"),(11,20,"short 11-20"),(21,35,"medium 21-35"),(36,50,"long 36-50"),(51,999,"very long >50")):
        c=sum(1 for x in n if lo<=x<=hi)
        print("   %-16s %4d  %5.1f%%" % (lab,c,100*c/tot))
