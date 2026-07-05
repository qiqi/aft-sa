"""Verification image for the 200k Appendix-D pages, where alpha runs in VERTICAL
columns (x/c down the rows). Renders the page rotated 90deg CW into the readable
orientation, crops the x/c-label + target alpha column, detects the 29 data rows,
and draws my stored values (red) beside each row for digit-by-digit checking.
Usage: python mk_cp_verify_col.py <page0idx> <Re> <surface> <colkey> <name>
"""
import sys, os, json, fitz, numpy as np
from PIL import Image, ImageDraw, ImageFont
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # -> paper/ so eppler.pdf, data/, figs/ resolve
page=int(sys.argv[1]); Re=sys.argv[2]; surf=sys.argv[3]; colk=sys.argv[4]; name=sys.argv[5]
vals=json.load(open("data/exp_cp_tables.json"))[Re][surf][colk]
D=5.0
d=fitz.open("eppler.pdf")
def load(pg):
    pix=d[pg].get_pixmap(matrix=fitz.Matrix(D,D))
    return Image.frombytes("RGB",(pix.width,pix.height),pix.samples).transpose(Image.ROTATE_270)
im=load(page); W,H=im.size

def detect_rows(img):
    """29 x/c data-row y-centres, from the x/c-label column (one dark band per row).
    The last 29 bands are .000..950 (header/dash bands sit above and are dropped)."""
    g=np.array(img.convert("L")); dark=g<140
    Wi,Hi=img.size
    dx0,dx1=int(0.105*Wi),int(0.175*Wi)
    prof=dark[:,dx0:dx1].sum(1); thr=0.03*(dx1-dx0)
    bands=[]; inb=False
    for y in range(Hi):
        if prof[y]>thr and not inb: s=y; inb=True
        elif prof[y]<=thr and inb:
            if y-s>6: bands.append((s+y)//2)
            inb=False
    cens=[c for c in bands if c>0.25*Hi]   # 0.25 catches .000 even on the higher-set lower page
    return cens[-29:]

rows=detect_rows(im)
print(f"p{page}: {len(rows)} data rows, span y=[{rows[0]},{rows[-1]}]")
# crop x: x/c label column through the target alpha column. Find alpha-column x via
# vertical projection of the header/first-row. Simpler: crop a fixed left window
# that includes x/c + first 2 alpha cols, plus a right margin for my values.
crop_x0,crop_x1=int(0.10*W),int(0.36*W)
strip=im.crop((crop_x0, rows[0]-40, crop_x1, rows[-1]+40))
UP=1.6
strip=strip.resize((int(strip.width*UP),int(strip.height*UP)))
# my-values panel to the right
panelw=520
canvas=Image.new("RGB",(strip.width+panelw, strip.height),"white")
canvas.paste(strip,(0,0))
dr=ImageDraw.Draw(canvas)
try: font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",30)
except: font=ImageFont.load_default()
xc_lab=[".000",".005",".010",".015",".020",".025",".030",".040",".050",".060",".075",".100",".150",".200",".250",".300",".350",".400",".450",".500",".550",".600",".650",".700",".750",".800",".850",".900",".950"]
y_off=(rows[0]-40)
for i,yc in enumerate(rows):
    yy=int((yc-y_off)*UP)
    dr.line([(strip.width,yy),(canvas.width,yy)],fill=(220,220,220),width=1)
    if i<len(vals):
        v=vals[i]
        s=f"{v:.4f}" if v>=0 else f"-{abs(v):.4f}"
        s=s.replace("0.",".",1) if s[:2]=="0." else s.replace("-0.","-.",1)
        dr.text((strip.width+14, yy-18), f"{xc_lab[i]}  {s}", fill=(200,0,0), font=font)
hdr=Image.new("RGB",(canvas.width,34),"white")
ImageDraw.Draw(hdr).text((6,6),f"Re={Re}k {surf} col={colk}  LEFT=ORIGINAL(x/c+6.50+{colk})  RIGHT=MINE(red: x/c value)",fill="black")
out=Image.new("RGB",(canvas.width,canvas.height+34),"white")
out.paste(hdr,(0,0)); out.paste(canvas,(0,34))
os.makedirs("figs/verify",exist_ok=True)
out.save(f"figs/verify/{name}.png"); print("wrote figs/verify/"+name+".png",out.size)
