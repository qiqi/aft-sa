"""Show one alpha-row of an appendix table at high zoom with x/c gridlines+labels,
for accurate manual reading. Prints detected rows. Usage:
  python showrow.py <page0idx> <row_idx> <out.png>
"""
import sys, os, fitz, numpy as np
from PIL import Image, ImageDraw, ImageFont
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # -> paper/ so eppler.pdf resolves
page=int(sys.argv[1]); ridx=int(sys.argv[2]); out=sys.argv[3]
D=5.0
d=fitz.open("eppler.pdf"); pix=d[page].get_pixmap(matrix=fitz.Matrix(D,D))
im=Image.frombytes("RGB",(pix.width,pix.height),pix.samples); g=np.array(im.convert("L")); H,W=g.shape; dark=g<140
dx0,dx1=int(0.40*W),int(0.82*W)
prof=dark[:,dx0:dx1].sum(1); thr=0.015*(dx1-dx0)
bands=[]; inb=False
for y in range(H):
    if prof[y]>thr and not inb: s=y; inb=True
    elif prof[y]<=thr and inb:
        if y-s>15: bands.append([s,y])
        inb=False
if inb: bands.append([s,H])
merged=[]
for a,b in bands:
    if merged and ((a+b)//2-(merged[-1][0]+merged[-1][1])//2)<180: merged[-1][1]=b
    else: merged.append([a,b])
cens=[(a+b)//2 for a,b in merged]
print(f"p{page}: {len(cens)} rows y={cens}")
cen=cens[ridx]; lx0=int(0.35*W)      # data start (skip labels; we know x/c)
strip=im.crop((lx0,cen-92,dx1+8,cen+12))   # numbers sit above the detected baseline/markers
# column centers from the x/c-LABEL row (last band) -- always has all 29 labels incl ends
lab_cen=cens[-1]
lsg=np.array(im.crop((lx0,lab_cen-38,dx1+8,lab_cen+38)).convert("L"))<140
lc=np.where(lsg.sum(0)>2)[0]; xf=lc.min(); xl=lc.max()
centers=np.linspace(xf,xl,29)
xc=[".000",".005",".010",".015",".020",".025",".030",".040",".050",".060",".075",".100",".150",".200",".250",".300",".350",".400",".450",".500",".550",".600",".650",".700",".750",".800",".850",".900",".950"]
try: font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",30)
except: font=ImageFont.load_default()
def build(i0,i1):
    px0=int(centers[i0]-24); px1=int(centers[i1]+40)
    sub=strip.crop((max(0,px0),0,min(strip.width,px1),strip.height))
    scale=2000.0/sub.width; sub=sub.resize((2000,int(sub.height*scale)))
    cs=[(centers[i]-px0)*scale for i in range(i0,i1+1)]
    sd=ImageDraw.Draw(sub)
    for c in cs: sd.line([(c,0),(c,sub.height)],fill=(150,200,255),width=1)
    xl_=Image.new("RGB",(2000,40),"white"); xd=ImageDraw.Draw(xl_)
    for c,l in zip(cs,xc[i0:i1+1]):
        w=xd.textlength(l,font=font); xd.text((c-w/2,6),l,fill=(0,120,0),font=font)
    return sub,xl_
sA,lA=build(0,14); sB,lB=build(15,28)
canvas=Image.new("RGB",(2000,sA.height+lA.height+sB.height+lB.height+12),"white")
y=0
for img_ in (sA,lA,sB,lB): canvas.paste(img_,(0,y)); y+=img_.height+ (4 if img_ in (sA,sB) else 4)
canvas.save(out); print("wrote",out,canvas.size)
