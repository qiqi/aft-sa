import re,sys,bisect
path=sys.argv[1]; MIN=int(sys.argv[2]); LIM=int(sys.argv[3]) if len(sys.argv)>3 else 20
src=open(path).read(); lines=src.split('\n')
off=[];c=0
for l in lines: off.append(c); c+=len(l)+1
def line_of(p): return bisect.bisect_right(off,p)
# blank out environments and comments so we only see prose
mask=list(src)
for m in re.finditer(r'(?s)\\begin\{(figure|table|tabbing|equation|gather|align|CJK|center|abstract|enumerate|itemize)\*?\}.*?\\end\{\1\*?\}', src):
    for i in range(m.start(),m.end()): mask[i]='\0'
for m in re.finditer(r'(?m)^%.*$', src):
    for i in range(m.start(),m.end()): mask[i]='\0'
mask=''.join(mask)
def words(t):
    t=re.sub(r'\$[^$]*\$','M',t); t=re.sub(r'\\[a-zA-Z]+\*?',' ',t); t=re.sub(r'[{}~\\]',' ',t)
    return len([w for w in re.sub(r'[^A-Za-z0-9]',' ',t).split() if w])
res=[]
for para in re.finditer(r'(?s)(?:(?<=\n\n)|\A)(.+?)(?=\n\n|\Z)', mask):
    p=para.group(1)
    if '\0' in p or not p.strip(): continue
    base=para.start(1)
    for sm in re.finditer(r'.+?[.!?](?=\s|$)', p, re.S):
        t=sm.group(0).strip()
        n=words(t)
        if n>=MIN: res.append((n, line_of(base+sm.start()), t))
res.sort(reverse=True)
for n,ln,t in res[:LIM]:
    print("\n[%dw L%d] %s" % (n,ln,' '.join(t.split())))
print("\nTOTAL >=%d words: %d" % (MIN,len(res)))
