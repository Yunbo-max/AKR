"""Precisely repair equations and routing on the supplied Figure 2 artwork.
No model inference, new experimental data, or generated artwork is used.
"""
from pathlib import Path
from io import BytesIO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
from matplotlib.mathtext import math_to_image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'AUTHOR_NOTES' / 'pipeline_before_annotation.png'
DST = ROOT / 'figures' / 'pipeline.png'
S = 3
base = Image.open(SRC).convert('RGB').resize((1648, 928), Image.Resampling.LANCZOS)
im = base.resize((base.width*S, base.height*S), Image.Resampling.LANCZOS).convert('RGBA')
draw = ImageDraw.Draw(im)
navy = '#172F48'; blue = '#145CAC'; orange = '#BC4F26'; green = '#315D3C'; violet = '#634985'
fontpath = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

def erase(box, color=None):
    x0,y0,x1,y1 = box
    if color is None:
        # The chosen reference is an unprinted patch of the same coloured panel.
        color = tuple(np.array(base.crop((x0,y0,x0+3,y0+3))).reshape(-1,3).mean(0).astype(int))
    draw.rectangle(tuple(int(v*S) for v in box), fill=color)

def formula(text, box, size=25, color='#071329', fill=None):
    if fill is not False:
        erase(box, fill)
    bio = BytesIO()
    math_to_image('$'+text+'$', bio, dpi=3*100, format='png', color=color)
    bio.seek(0)
    glyph = Image.open(bio).convert('RGBA')
    # Clear white background produced by math_to_image, retaining antialiased ink.
    arr=np.asarray(glyph).copy(); rgb=arr[...,:3].astype(float)
    ink=np.max(255-rgb,axis=-1)
    arr[...,3]=np.clip(ink*1.05,0,255).astype('uint8')
    glyph=Image.fromarray(arr)
    x0,y0,x1,y1=box
    wanted_h=size*S
    scale=min((x1-x0-3)*S/glyph.width, wanted_h/glyph.height, (y1-y0-2)*S/glyph.height)
    glyph=glyph.resize((max(1,int(glyph.width*scale)),max(1,int(glyph.height*scale))),Image.Resampling.LANCZOS)
    im.alpha_composite(glyph,(int((x0+x1)*S/2-glyph.width/2),int((y0+y1)*S/2-glyph.height/2)))

def line(points,color,width=2,dashed=False,arrow=True):
    pts=[(int(x*S),int(y*S)) for x,y in points]
    if dashed:
        for a,b in zip(pts,pts[1:]):
            dx,dy=b[0]-a[0],b[1]-a[1]; length=(dx*dx+dy*dy)**.5
            for start in np.arange(0,length,8*S):
                end=min(start+5*S,length)
                draw.line([(a[0]+dx*start/length,a[1]+dy*start/length),(a[0]+dx*end/length,a[1]+dy*end/length)],fill=color,width=int(width*S))
    else:
        draw.line(pts,fill=color,width=int(width*S),joint='curve')
    if arrow:
        a,b=pts[-2],pts[-1];v=np.array(b,dtype=float)-a;v/=np.linalg.norm(v)
        normal=np.array([-v[1],v[0]])
        tri=[tuple(b),tuple(np.array(b)-10*S*v+4*S*normal),tuple(np.array(b)-10*S*v-4*S*normal)]
        draw.polygon(tri,fill=color)

# Correct the sign and pooled target; coefficients follow a_i notation in main.tex.
formula(r'g_i=-\mathrm{Pool}_{\mathcal{A}_i}\,\nabla_Z\mathcal{L}_{\rm ans}',(585,392,819,463),size=26,fill=(252,226,206))
formula(r'g_i\approx\bar g+B a_i',(961,227,1258,266),size=31,fill=(236,246,228))
formula(r'a_i',(1240,338,1270,385),size=25,fill=(238,246,230))
formula(r'a_i\approx Wz_i+b',(1327,223,1601,267),size=31,fill=(237,246,229))
formula(r'z_i',(1320,342,1369,384),size=29,fill=(237,246,229))
formula(r'\widehat a_i',(1566,345,1611,385),size=25,fill=(237,246,229))
# There is no feature-to-SVD input: only gradients define the correction basis.
erase((884,179,939,414),(255,255,255))
line([(883,419),(938,419)],orange)
line([(882,190),(905,190),(905,116),(1466,116),(1466,153)],blue)
formula(r'z_i=\mathrm{std}(h_i)',(1000,84,1265,117),size=24,fill=(255,255,255))
# Both fitted components, plus their normalizer, are carried to query inference.
line([(1450,440),(1450,492),(1158,492),(1158,540),(1060,540)],green,dashed=True)
line([(1078,440),(1078,482),(962,482),(962,509)],green,dashed=True)
erase((618,519,1028,560),(225,232,246))
draw.text((642*S,528*S),'Fitted',font=ImageFont.truetype(fontpath,23*S),fill=navy)
formula(r'(\bar g,B,W,b,\mu,\sigma)',(742,522,1025,558),size=25,fill=False)
# Exact standardized inference equations.
erase((603,739,791,837),(242,234,253))
erase((871,738,1052,803),(242,234,253))
formula(r'\widehat a_q=Wz_q+b',(605,740,791,790),size=27,fill=(242,234,253))
formula(r'z_q=\mathrm{std}(h_q)',(610,794,780,823),size=19,fill=(242,234,253))
formula(r'\widehat a_q',(787,807,828,841),size=23,fill=(241,233,253))
formula(r'\widehat g_q=\bar g+B\widehat a_q',(871,738,1050,788),size=28,fill=(242,234,253))
erase((813,615,835,656),(255,255,255))
line([(824,570),(824,657)],navy)
line([(998,570),(998,610),(968,610),(968,657)],green,dashed=True)
# Second forward pass receives the original recording/prompt, not only a correction.
line([(106,851),(106,911),(1126,911),(1126,810),(1161,810)],navy,width=1.6)
formula(r'x_q,p',(774,884,911,909),size=20,fill=(255,255,255))
# Preserve the original states and add increments, rather than replacing K or V.
# Delta K and Delta V retain their original short annotation.

im.convert('RGB').save(DST,optimize=True)
print(f'Annotated pipeline saved: {DST.name} ({im.width} x {im.height})')
