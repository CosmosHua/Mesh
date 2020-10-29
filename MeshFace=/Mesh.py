# coding:utf-8
#!/usr/bin/python3
import numpy as np
# before import matplotlib.pyplot or pylab!
import os, cv2, matplotlib; matplotlib.use('Agg')
from pylab import * # import matplotlib & numpy


rand = np.random.rand # for short
# tp = [Type=0-3, Fix/Scale=0/1, Noise_Amplitude=0-1]
#####################################################################
# Create Normalized Curve:
def Curve(tp, p, t): # normalized curve
    if type(tp)!=str: tp = "sps" # tp=tp[0]
    s1 = p[2] * np.sin((p[0]*t+p[1])*np.pi)
    s2 = p[5] * np.sin((p[3]*t+p[4])*np.pi)
    if "sps" in tp: return (s1 + s2)/(abs(p[2]) + abs(p[5]))
    if "sms" in tp: return (s1 * s2)/(abs(p[2]) * abs(p[5]))
    if len(p)>=9: s3 = p[8] * np.sin((p[6]*t+p[7])*np.pi)
    if "sss" in tp: return (s1 + s2 + s3)/(abs(p[2])+abs(p[5])+abs(p[8]))


# Get Random [linewidth, alpha] Pairs:
def LwAl(tp, dx=180, n=1): # [linewidth, alpha] pair
    wa = rand(2*n); f = tp[1]*(dx/180-1)+1 # scale ratio
    wa[::2] = [round((1.8*i+2)*f,2) for i in wa[::2]] # linewidth
    wa[1::2] = [round(0.45*i+0.2,2) for i in wa[1::2]] # alpha
    if tp[0]==3: wa[::2] = round(f,2) # linewidth for tp=3
    return list(wa) # type: np.array->list


# Rotate or Affine the Curve:
def RoAf(t, y, ra=0, af=None): # rotate or affine the curve
    if type(ra)!=np.ndarray: # rotational angle -> matrix
        ra *= np.pi; ra = np.array([[cos(ra),-sin(ra)],[sin(ra),cos(ra)]])
    if type(af)==np.ndarray: ra = ra.dot(af); # affine & rotate
    y = ra.dot(np.array([t,y])) # rotate/affine the curve
    return y[0,:], y[1,:] # t'=y[0,:], y'=y[1,:]


# Draw a Curve with Annotation:
def DrawCu(tp, p=None, xi=0, dx=20, yo=0, A=1, ra=0, af=0, wa=[]):
    if not isinstance(tp,(list,tuple)): tp = [tp,0,0] # default
    if p==None or len(p)<6: # default: random curve parameters
        p = [round(2*i,2) for i in rand(9)]; p[2]=p[5]=p[8]=1
    
    tp, fs, no = tp # tp[0]=Type, tp[1]=Fix/Scale, tp[2]=Noise Scale
    t = np.linspace(xi-dx, xi+dx, round(2*dx*(rand()+1)), endpoint=True)
    no = no/5 * (1+(tp==3)) * (rand(len(t))-0.5) # noise
    y = A * (Curve(tp,p,t) + yo + no) # vertically scale + translate
    t,y = RoAf(t-xi, y, ra, af) # horizontally adjust -> rotate/affine
    
    if not wa or len(wa)<2: wa = LwAl([tp,fs], dx, 1); # get [linewidth,alpha] pair
    ann = str(tp)+": "+", ".join([str(i) for i in p])+"->"+", ".join([str(i) for i in wa])
    plot(t, y, color="k", lw=wa[0], alpha=wa[-1], label=ann)
    return t, y, wa, p


#####################################################################
# Extract sps Cell Parameters:
def Paras(tp, dx, A, f): # Extract sps Cell Parameters
    tp, yf = tp[0], tp[1]*(dx/180-1)+1 # Cell scale ratio
    if tp==0: # Reticulate Pattern Type0
        A = 42*yf; f = 12/dx; p = [0.2*f, 3/8, 0.5, 0.8*f, 0, 0.8]
    elif tp==1: # Reticulate Pattern Type1
        A = 30*yf; f = 8/dx; p = [0.2*f, 3/8, 0.5, 0.8*f, 0, 0.75]
    elif tp==2: # Reticulate Pattern Type2
        A = 55*yf; f = 8/dx; p = [0.2*f, 3/8, 0.5, 0.8*f, 0, 0.8]
        f = np.array([[1,-0.5],[-0.15,1]]); # Affine Matrix
    elif tp==3: # Reticulate Pattern Type3
        A = 10*yf; f = 7.5/dx; p = [0.2*f, 3/8, 0.5, 0.8*f, 0, 0.8]
        f = np.array([[1.15,1.1],[-0.45,0.7]]); # Affine Matrix
    else: A *= yf; f /= dx; p = [0.2*f, 3/8, 0.5, 0.8*f, 0, 0.8-tp%2/10]
    return A, p, f


# Draw Reticulate Pattern Cell(sps):
def DrawCell(tp, dx=180, yi=0, ra=0, wa=[], A=42, f=12):
    xi = round(dx*rand(), 1) # x-beginning offset
    dy = round(0.2+(rand()-0.5)/10, 3) # y-offset
    A, p1, f = Paras(tp, dx, A, f); p2 = p1.copy()
    p2[::-3] = [-i for i in p1[::-3]] # vertical flip
    
    # DrawCu set alpha=wa[-1], thus Curve Cells have same alpha:
    t1,y1,w1,p1 = DrawCu(tp, p1, xi, dx, yi+dy, A, ra, f, wa=wa[:])
    t2,y2,w2,p2 = DrawCu(tp, p2, xi, dx, yi-dy, A, ra, f, wa=wa[2:])
    return [t1,y1, t2,y2]


# Draw Reticulate Pattern Cell(sps):
def DrawCel_(tp, dx=180, yi=0, ra=0, wa=[], A=42, f=12):
    xi = round(dx*rand(), 1) # x-beginning offset
    dy = round(0.2+(rand()-0.5)/10, 3) # y-offset
    A, p1, f = Paras(tp, dx, A, f); p2 = p1.copy()
    p2[::-3] = [-i for i in p1[::-3]] # vertical flip
    
    t = np.linspace(xi-dx, xi+dx, round(2*dx*(rand()+1)), endpoint=True)
    no1 = tp[2]/5 * (1+(tp[0]==4)) * (rand(len(t))-0.5) # noise
    no2 = tp[2]/5 * (1+(tp[0]==4)) * (rand(len(t))-0.5) # noise
    y1 = A * (Curve(tp[0], p1, t) + (yi+dy) + no1); # vertically scale + trans
    y2 = A * (Curve(tp[0], p2, t) + (yi-dy) + no2); # vertically scale + trans
    t -= xi; t1,y1 = RoAf(t,y1,ra,f); t2,y2 = RoAf(t,y2,ra,f) # rotate/affine
    plot(t1, y1, color="b", lw=wa[0], alpha=wa[-1]) # same alpha
    plot(t2, y2, color="b", lw=wa[2], alpha=wa[-1]) # same alpha
    return [t1,y1, t2,y2]


#####################################################################
# Add Mesh to Image:
def Add2Im_(im, tp, ro=None, wa=None, gap=1.6, fun=DrawCell):
    if type(im)==str: im = cv2.imread(im)[:,:,::-1] # =imread(im)
    y,x = im.shape[:2]; n = y//15; net = []; dx = 180 # OR: dx=x
    if not wa or len(wa)<4: wa = LwAl(tp, dx, 2) # [lw,alpha]
    if not ro: ro = 2*rand()-1 # randomly rotation
    
    ofs = round(1.5*rand(), 2); gap = round(gap+(rand()-0.3)/10, 2)
    for i in range(2*n): net += fun(tp, dx, gap*(i-n)+ofs, ro, wa=wa)
    
    subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
    imshow(im, extent=(-x/2,x/2,-y/2,y/2)); axis("off"); xticks([]); yticks([])
    return net


# Save Mesh Image with Mask(by black.png):
def SaveIm_(im, out, tp, qt=20, ms=None, ro=None, wa=None, gap=1.6):
    if os.path.isfile(out): return # already exist->skip
    if type(im)==str: im = cv2.imread(im)[:,:,::-1] # =imread(im)
    dpi = 72; y,x = im.shape[:2] # (height, width, channel)
    figure(figsize=(x/dpi, y/dpi), dpi=dpi) # set figsize
    net = Add2Im_(im, tp, ro, wa, gap); savefig(out, dpi=dpi)
    
    if type(qt)!=int: qt = np.random.randint(qt[0],qt[1]) # quality
    cv2.imwrite(out, cv2.imread(out), [cv2.IMWRITE_JPEG_QUALITY, qt])
    
    if ms: # output mask image
        if type(ms)!=str: ms = "./Test/blank.png"
        imshow(imread(ms), extent=(-x/2,x/2,-y/2,y/2))
        out = out[:-4]+"_m.png"; savefig(out); im = cv2.imread(out,0)
        ret,im = cv2.threshold(im, 250, 255, cv2.THRESH_BINARY_INV)
        cv2.imwrite(out, im, [cv2.IMWRITE_PXM_BINARY, 1])
    close("all"); return net


#####################################################################
# Save Mesh Image with Mask:
def SaveIm(im, out, tp, qt=20, ms=None, ro=None, wa=None, gap=1.6):
    if os.path.isfile(out): return # already exist->skip
    if type(im)==str: im = cv2.imread(im)[:,:,::-1] # =imread(im)
    y,x = im.shape[:2]; n = y//15; net = []; dx = 180 # OR: dx=x
    if not wa or len(wa)<4: wa = LwAl(tp, dx, 2) # [lw,alpha]
    if not ro: ro = 2*rand()-1 # randomly rotation
    
    ofs = round(1.5*rand(), 2); gap = round(gap+(rand()-0.3)/10, 2)
    dpi = 72; figure(figsize=(x/dpi, y/dpi), dpi=dpi); axis("off")
    subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
    for i in range(2*n): net += DrawCell(tp, dx, gap*(i-n)+ofs, ro, wa=wa)
    
    if ms: # output mask image
        xlim(-x/2,x/2); ylim(-y/2,y/2); ms = out[:-4]+"_m.png"
        savefig(ms, facecolor="w", dpi=dpi); tmp = cv2.imread(ms,0)
        gap,tmp = cv2.threshold(tmp, 250, 255, cv2.THRESH_BINARY_INV)
        cv2.imwrite(ms, tmp, [cv2.IMWRITE_PXM_BINARY, 1])
    imshow(im, extent=(-x/2,x/2,-y/2,y/2)); savefig(out, dpi=dpi)
    
    if type(qt)!=int: qt = np.random.randint(qt[0],qt[1]) # quality
    cv2.imwrite(out, cv2.imread(out), [cv2.IMWRITE_JPEG_QUALITY, qt])
    close("all"); return net


# SSIM (after resize) in turn: 0.972636, 0.998395, 0.999770, 0.997944, 0.999939
# INTER_NEAREST=0, INTER_LINEAR=1, INTER_CUBIC=2, INTER_AREA=3, INTER_LANCZOS4=4
rsz = lambda im,hw: cv2.resize(im, tuple(hw[::-1]), interpolation=cv2.INTER_LANCZOS4)
#####################################################################
# Census Image Size and Probability:
def SizeProb(DIR="./Sample", out="SP.npz"):
    assert os.path.isdir(DIR); sp = {}
    for i in os.listdir(DIR):
        i = os.path.join(DIR,i); im = cv2.imread(i)
        if type(im)!=np.ndarray: os.remove(i); continue
        k = im.shape[:2]; sp[k] = sp[k]+1 if k in sp else 1
    pp = np.array([i for i in sp.values()]); pp /= sum(pp)
    sz = np.array([i for i in sp.keys()]) # (height,width)
    np.savez(out, s=sz, p=pp); return sz, pp


# Adjust Image Exposure:
def Expose(im, rt=1, b=0): # rt=linear ratio
    # Ref: https://www.jianshu.com/p/67adf1c5664e
    # Expose: (im.mean(axis=2)*rt-b).clip(0,255).astype("uint8")
    if type(im)==str: im = cv2.imread(im, -1) # 0=gray_mode
    if im.ndim>2: gy = im[:,:,:3].mean(axis=2) # 3|4 channels
        #gy = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # COLOR_BGRA2GRAY
        #gy = (0.11*im[:,:,0] + 0.59*im[:,:,1] + 0.3*im[:,:,2]) # BGR
    gy = (float(rt)*gy-b).clip(0,255).astype("uint8") # linear exposure
    #gy = ((2.0**rt)*gy-b).clip(0,255).astype("uint8") # exponential
    return cv2.cvtColor(gy, cv2.COLOR_GRAY2BGR) if im.ndim>2 else gy


# Crop Randomly with Prob=cp, then Resize Image:
def Crop(im, hw, cp=0.5, sd=(220,178)): #  cp=crop prob
    if type(im)==str: im = cv2.imread(im) # BGR
    sz, hw = np.array(im.shape[:2]), np.array(hw)
    sd = np.array(sd); ry, rx = sz/sd # normalized ratio
    if rand()<cp and (hw<sd).any(): # True if any True
        wd = round(rx*np.random.randint(125,145)) # face_width
        ht = round(ry*np.random.randint(85,105)*2) # face_height
        x1 = np.random.randint((sz[1]-wd)//2); x2 = sz[1]-x1 # right
        y1 = int(round(ry*np.random.randint(10))) # upper_limit
        y2 = sz[0]-np.random.randint(sz[0]-ht+y1) # bottom_limit
        im = im[y1:y2, x1:x2] # ROI
    return rsz(im, hw) # hw=(height,width)


# Batch Adding Meshes to Images and Saving:
def BatchSave(DIR, tp, qt=20, num=4, pg=0.2):
    tp, fs, no = tp # parse: Type, Fix/Scale, Noise
    tp = [tp] if type(tp)==int else tp; N = len(tp)
    num = num if type(num)==int else N # default=N
    
    ss = os.path.basename(DIR); DIR = os.path.abspath(DIR)
    dd = "_".join([str(i) for i in [ss,num,fs,no]]) # dst_dir
    out = lambda x,k="",t=".jpg": x.replace(ss,dd)[:-4]+"_"+str(k)+t
    
    tps = [i%N for i in range(num)] # for num>=N
    sp = np.load("SP.npz"); sz, pp = sp["s"], sp["p"]
    sd = sz[0]; #sz, pp = sz[1:], pp[1:]; pp /= sum(pp)
    for cwd, subs, files in os.walk(DIR): # traverse DIR
        os.chdir(cwd); dst = cwd.replace(ss,dd) # cd cwd
        if not os.path.isdir(dst): os.mkdir(dst) # dst_dir
        for im in files: # loop images in cwd
            sp = np.random.choice(len(sz), num, replace=True, p=pp) # sizes
            if num<N: tps = np.random.choice(N, num, replace=False) # unique
            #else: tps = np.random.choice(N, num, replace=True) # repetitive
            
            im = os.path.join(cwd, im); img = cv2.imread(im)
            for i,k in enumerate(tps): # loop for types and sizes
                ssz = sz[sp[i]]; rz = Crop(img, ssz, cp=0.5) # crop+resize
                wf = (sd/ssz).prod(); wa = LwAl((tp[k],fs), 180*wf, 2) # linewidth
                if (ssz!=sd).any() and rand()<pg: rz = Expose(rz, 1.6**rand(), 128*rand())
                cv2.imwrite(out(im,i,".png"), rz, [cv2.IMWRITE_PNG_COMPRESSION,9]) # clean
                SaveIm(rz[:,:,::-1], out(im,i,".jpg"), (tp[k],fs,no), qt, wa=wa) # mesh
                #SaveIm(im, out(im,i), (tp[k],fs,no), qt) # without crop/resize
    os.chdir(os.path.join(DIR,"..")) # parent directory


#####################################################################
if __name__ == "__main__":
    DIR = "E:/PyCharm/Clean"
    tp = [range(4), 0, 0.4] # Types,Fixed,Noise
    BatchSave(DIR, tp, qt=20, num=8, pg=0.3)
