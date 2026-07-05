"""Generate side-by-side verification images for the digitized experimental Cp.
Rotated-table layout: alpha = horizontal rows, x/c = columns (left->right .000..950).
For a given page + row index, crops the ORIGINAL alpha-row (with its label) and
renders my values (from data/exp_cp_tables.json) directly below, aligned to 29
even x/c column centers. Output -> figs/verify/<name>.png (readable, upscaled).
Usage: python mk_cp_verify.py <page0idx> <row_idx> <Re> <surface> <colkey> <name>
"""
import sys, os, json, fitz, numpy as np
from PIL import Image, ImageDraw, ImageFont
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # -> paper/ so eppler.pdf, data/, figs/ resolve
page=int(sys.argv[1]); ridx=int(sys.argv[2]); Re=sys.argv[3]; surf=sys.argv[4]; colk=sys.argv[5]; name=sys.argv[6]
vals=json.load(open("data/exp_cp_tables.json"))[Re][surf][colk]
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
# merge bands whose centers are within 180px (rows are ~370px apart; spurious splits merge)
merged=[]
for a,b in bands:
    if merged and ((a+b)//2 - (merged[-1][0]+merged[-1][1])//2) < 180: merged[-1][1]=b
    else: merged.append([a,b])
cens=[(a+b)//2 for a,b in merged]
print(f"p{page}: {len(cens)} rows y={cens}")
cen=cens[ridx]
lx0=int(0.275*W)
strip=im.crop((lx0,cen-58,dx1+8,cen+58))
# data extent for columns (within the strip, offset lx0)
sg=np.array(im.crop((dx0,cen-45,dx1,cen+45)).convert("L"))<140
cols=np.where(sg.sum(0)>1)[0]; xf=dx0+cols.min()-lx0; xl=dx0+cols.max()-lx0
UP=2.9
strip=strip.resize((int(strip.width*UP),int(strip.height*UP)))
centers=np.linspace(xf,xl,29)*UP
try: font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",26)
except: font=ImageFont.load_default()
mine=Image.new("RGB",(strip.width,64),"white"); dr=ImageDraw.Draw(mine)
xc_lab=[".000",".005",".010",".015",".020",".025",".03",".04",".05",".06",".075",".10",".15",".20",".25",".30",".35",".40",".45",".50",".55",".60",".65",".70",".75",".80",".85",".90",".95"]
for i,c in enumerate(centers):
    # faint gridline to help align original<->mine
    dr.line([(c,0),(c,mine.height)],fill=(210,210,210),width=1)
    if i<len(vals):
        v=f"{vals[i]:.4f}" if vals[i]>=0 else f"-{abs(vals[i]):.4f}"
        v=v.replace("0.",".",1) if v[:2]=="0." else v.replace("-0.","-.",1)
        w=dr.textlength(v,font=font); dr.text((c-w/2,8),v,fill=(200,0,0),font=font)
lab=Image.new("RGB",(strip.width,32),"white")
ImageDraw.Draw(lab).text((6,6),f"Re={Re}k {surf} col={colk}  TOP=ORIGINAL(with label)  BOTTOM=MINE(red)  x/c .000->.950 left-to-right",fill="black")
# gridlines over original strip too
sdr=ImageDraw.Draw(strip)
for c in centers: sdr.line([(c,0),(c,strip.height)],fill=(200,220,255),width=1)
canvas=Image.new("RGB",(strip.width,strip.height+mine.height+38),"white")
canvas.paste(lab,(0,0)); canvas.paste(strip,(0,32)); canvas.paste(mine,(0,32+strip.height+2))
os.makedirs("figs/verify",exist_ok=True)
canvas.save(f"figs/verify/{name}.png"); print("wrote figs/verify/"+name+".png",canvas.size)
