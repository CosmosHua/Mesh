# coding:utf-8
# !/usr/bin/python3

import numpy as np
import shutil as sh
import os, cv2, random
#import mtcnn.mtcnn as mt
#from motion_blur import mBlur


'''
# Transform_Ref: incubator-mxnet/blob/master/python/mxnet/image/image.py
# Note: default image format: MXNet=RGB, OpenCV=BGR. So, use BGR->RGB.
# MTCNN install: pip3 install mtcnn # using backend=TensorFlow
'''
# Note: src should be RGB format.
################################################################################
def TypeCast(src, typ="float32"): return src.astype(typ)


def HorizontalFlip(src, prob=0): # prob: [0,1]
    if random.random()<prob: src = np.flip(src, axis=1)
    return src # random horizontal flip


def BrightnessJitter(src, jitter=0): # jitter: [0,1]
    if type(jitter)==float or type(jitter)==int:
        jitter = (-jitter, jitter) # symmetric interval
    alpha = random.uniform(jitter[0], jitter[1])
    src *= (1.0 + alpha); return src


def ContrastJitter(src, jitter=0): # jitter: [0,1]
    if type(jitter)==float or type(jitter)==int:
        jitter = (-jitter, jitter) # symmetric interval
    alpha = random.uniform(jitter[0], jitter[1])

    coef = np.array([[[0.299, 0.587, 0.114]]])
    gray = src * coef
    gray = 3 * alpha * np.mean(gray)
    src *= (1.0 + alpha); src += gray; return src


def SaturationJitter(src, jitter=0): # jitter: [0,1]
    if type(jitter)==float or type(jitter)==int:
        jitter = (-jitter, jitter) # symmetric interval
    alpha = random.uniform(jitter[0], jitter[1])

    coef = np.array([[[0.299, 0.587, 0.114]]])
    gray = src * coef
    gray = alpha * np.sum(gray, axis=2, keepdims=True)
    src *= (1.0 + alpha); src += gray; return src


def ColorJitter(src, jitter=0): # jitter: [0,1]
    jitter = random.uniform(0, jitter) * 256
    jitter *= 2 * np.random.random(src.shape) - 1
    src += jitter; return src


def HueJitter(src, jitter=0): # jitter: [0,1]
# Using approximate linear transfomation, Ref:
# https://beesbuzz.biz/code/hsv_color_transforms.php

    if type(jitter)==float or type(jitter)==int:
        jitter = (-jitter, jitter) # symmetric interval
    alpha = random.uniform(jitter[0], jitter[1])

    u, w = np.cos(alpha * np.pi), np.sin(alpha * np.pi)
    bt = np.array([[1.0, 0.0, 0.0],
                   [0.0, u, -w],
                   [0.0, w, u]])
    
    tyiq = np.array([[0.299, 0.587, 0.114],
                     [0.596, -0.274, -0.321],
                     [0.211, -0.523, 0.311]])
    ityiq = np.array([[1.0, 0.956, 0.621],
                      [1.0, -0.272, -0.647],
                      [1.0, -1.107, 1.705]])

    t = np.dot(np.dot(ityiq, bt), tyiq).T
    src = np.dot(src, t); return src


def Lighting(src, jitter=0): # jitter: [0,1]
    if type(jitter)==float or type(jitter)==int:
        jitter = (0, jitter) # default interval
    alpha = np.random.normal(jitter[0], jitter[1], size=(3,))
    
    eigval = np.array([55.46, 4.794, 1.148])
    eigvec = np.array([[-0.5675, 0.7192, 0.4009],
                       [-0.5808, -0.0045, -0.8140],
                       [-0.5836, -0.6948, 0.4203]])

    rgb = np.dot(eigvec * alpha, eigval)
    src += rgb; return src # PCA-based noise


def ColorNormalize(src, mean, std=None): # src(RGB)
    if mean==True: mean = np.array([123.68, 116.28, 103.53])
    if std==True:  std = np.array([58.395, 57.12, 57.375])
    if not mean: src -= mean
    if not std:  src /= std
    return src # in-place


def RandomGray(src, prob=0): # prob: [0,1]
    mat = np.array([[0.21, 0.21, 0.21],
                    [0.72, 0.72, 0.72],
                    [0.07, 0.07, 0.07]])

    if random.random()<prob: src = np.dot(src, mat)
    return src # randomly convert to gray


################################################################################
def Polar2XY(R, the=2, mod="norm"): # R>0, the=[0,2]
    if type(the)==float or type(the)==int: the = (0,the)

    if mod in "normalgauss": # =np.random.normal(0,1)
        gen = np.random.randn; r = abs(gen()*R/2)
    elif mod in "uniform": # =np.random.uniform(0,1)
        gen = np.random.rand; r = gen()*R
    
    t, pi = (np.random.rand()*2)%2, np.pi # angle
    t = np.clip(t, min(the), max(the)) # same as below
    #t = min(the) if t<min(the) else (max(the) if t>max(the) else t)
    return np.array([np.cos(t*pi)*r, np.sin(t*pi)*r], dtype=int)


def RandomSalt(src, jitter=0, mod="norm"): # jitter: [0,1]
    if type(jitter)==float or type(jitter)==int:
        jitter = (0, jitter) # default interval
    jitter = random.uniform(jitter[0], jitter[1])
    if random.uniform(0,jitter)<jitter/2: return src
    
    h,w = src.shape[0:2]; # N = int(w*h*jitter)
    ct = np.array([np.random.randint(i) for i in [w,h]])
    R = 0.5 * (w**2+h**2)**0.5 * np.random.uniform()
    
    a,b = (0,256) if mod[-1].isdigit() else (128,256)
    for i in range(round(R*R*jitter)):
        x,y = ct + Polar2XY(R, mod=mod[:-1])
        p1,p2 = np.random.uniform(size=2) # probs
        dx,dy = np.random.randint(1,6, size=2) # length
        if dx>3 and dy>3: dx,dy = (1,dy) if p1<0.5 else (dx,1)
        #v = np.random.choice(val)
        src[y:y+dy, x:x+dx] = np.random.randint(a,b)
        if dx!=dy and p2<0.5: # p2=whether draw cross
            x,y = (x+1,y-1) if dx>dy else (x-1,y+1)
            src[y:y+dx, x:x+dy] = np.random.randint(a,b)
    return src # in-place


def GaussianBlur(src, jitter=0): # jitter: [0,1]
    if type(jitter)==float or type(jitter)==int:
        jitter = (0, jitter) # default interval
    jitter = random.uniform(jitter[0], jitter[1])
    ks = [round(i*jitter)*2+1 for i in src.shape[:2]]
    return cv2.GaussianBlur(src, tuple(ks), 0)


def MotionBlur(src, xlen=0): # xlen = max_len > 0
    return mBlur(src, xlen=xlen).blur_image()[0]


def ResizeQt(src, mx=1, mod="lin"):
    sz = src.shape[1::-1] # original size
    if type(mx)==float: rs = mx
    elif "l" in mod:    rs = 1/random.randint(1,mx)
    elif "e" in mod:    rs = 1/2**random.randint(1,mx)
    src = cv2.resize(src, None, fx=rs, fy=rs) # shrink
    src = cv2.resize(src, sz) # resize back to original
    return src


def RandomQuality(src, out, qt=95, rs=1): # qt: [0,100]
    if type(src)==str: src = cv2.imread(src) # BGR
    if type(qt)!=int: qt = random.randint(qt[0], qt[1])
    #src = cv2.resize(src, None, fx=rs, fy=rs) # resize
    cv2.imwrite(out, src, [cv2.IMWRITE_JPEG_QUALITY, qt])


def Transform(src, flip=0.5, mean=None, std=None, hue=0, brightness=0, contrast=0,
              saturation=0, pca_noise=0, color=0, shrink=0, blur=0, salt=0, gray=0):

    # src: should be an image_path or a RGB image(numpy).
    if type(src)==str: src = cv2.imread(src)[:,:,::-1] # BGR->RGB
    if flip: src = HorizontalFlip(src, flip) # not-in-place
    
    src = TypeCast(src) # following need, not-in-place
    if hue:         src = HueJitter(src, hue) # not-in-place
    if brightness:  BrightnessJitter(src, brightness) # in-place
    if contrast:    ContrastJitter(src, contrast) # in-place
    if saturation:  SaturationJitter(src, saturation) # in-place
    if pca_noise:   Lighting(src, pca_noise) # in-place
    if color:       ColorJitter(src, color) # in-place
    if shrink:      src = ResizeQt(src, shrink) # not-in-place
    if salt:        RandomSalt(src, salt, "uni0") # in-place
    if blur:        src = GaussianBlur(src, blur) # not-in-place
    if salt:        RandomSalt(src, salt, "norm") # in-place
    
    if mean or std: ColorNormalize(src, mean, std) # in-place
    if gray: src = RandomGray(src, gray) # not-in-place
    else: src = src[:,:,::-1] # RGB->BGR format
    return src # not-in-place


################################################################################
def CenterScale(faces, cp=1):
    x, y, wd, ht = faces["box"]
    cx, cy = x+wd/2, y+ht/2 # center
    #cx, cy = x+(wd-1)/2, y+(ht-1)/2 # center
    wd, ht = [round(i*cp) for i in (wd,ht)] # new
    x, y = [round(max(i,0)) for i in (cx-wd/2,cy-ht/2)]
    return (x, y, wd, ht) # new rect


def FaceDetect(im, df=None, sh=0, cp=0):
    if type(im)==str: im = cv2.imread(im) # image
    if type(df)!=mt.MTCNN: df = mt.MTCNN() # detector
    
    faces = df.detect_faces(im) # detect faces
    items = ("box", "confidence", "keypoints")
    points = ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")
    
    if sh > 0: # show faces (time=sh)
        for k,fc in enumerate(faces): # would alter image
            bx, score, pt = [fc[i] for i in items]
            pt = [pt[i] for i in points]; info = ("F%d: %.3f" % (k,score))
            print("\nFace_%d:" % k, fc) # show info of the k-th face
            for p in pt: cv2.circle(im, center=p, radius=1, color=(255,0,0), thickness=2)
            cv2.rectangle(im, (bx[0], bx[1]), (bx[0]+bx[2], bx[1]+bx[3]), color=(0,255,0))
            cv2.putText(im, info, (bx[0], bx[1]), fontFace=1, fontScale=1, color=(0,255,0))
        cv2.imshow("FaceShow", im)
    
    cp_faces = []
    if cp > 0: # crop faces (ratio=cp)
        for k,fc in enumerate(faces):
            x, y, wd, ht = CenterScale(fc, cp)
            cp_faces.append(im[y:y+ht+1, x:x+wd+1])
            if sh>0: cv2.imshow("%d_Face"%k, cp_faces[k])
    
    if sh > 0: cv2.waitKey(max(sh*len(faces),1)*1000)
    
    if cp > 0: return cp_faces
    return faces


################################################################################
if __name__ == "__main__":
    path = "IDFace1W1/"; N = 4
    out = "IDFace1W"+str(N)+"/"
    if not os.path.exists(out): os.mkdir(out)
    #detector = mt.MTCNN(min_face_size=50)
    for i in os.listdir(path):
        src = cv2.imread(path+i) # BGR
        '''# for Face_Crop:
        im = FaceDetect(src, detector, cp=1.2)
        if len(im)<1: continue # if no faces detected
        else: RandomQuality(im[0], out+i, qt=100) # label
        src = im[0] # NOT for MotionBlur'''
        
        for k in range(N): # corrupted copys
            '''# for MotionBlur:
            im = MotionBlur(src, xlen=50) # motion blur
            im = FaceDetect(im, detector, cp=1.2) # avoid edge
            if len(im)<1: continue # if no faces detected
            else: im = im[0][:,:,::-1] # only use 1st-face->RGB'''
            
            kk = i[:-4]+"_"+str(k)+".jpg"; im = src[:,:,::-1] # RGB
            #if os.path.exists(out+kk): continue # whether overwrite
            im = Transform(im, flip=0, hue=0.2, brightness=(-0.4,0.3), contrast=0.05, saturation=0.05,
                           pca_noise=0.05, color=0.1, shrink=8, blur=(0.01,0.12), salt=(0,0.03)) # BGR
            RandomQuality(im, out+kk, qt=(10,60), rs=1) # save
            
            '''# for TF-pix2pix:
            im = cv2.imread(out+kk); # sh.copy(path+i, out+kk)
            im = np.concatenate((im,src), axis=1) # combine: AtoB
            cv2.imwrite(out+kk, im, [cv2.IMWRITE_JPEG_QUALITY, 100])'''
        print("Processed:", path+i)
    cv2.destroyAllWindows()
    

################################################################################
# Note: check the face data! 
# sz $(ls -S|tail); rm $(ls -S|tail) # delete non-faces(<1kB)
# sz $(ls *_*_*.jpg); rename _0.jpg .jpg *_*_0.jpg; rm $(ls *_*_*.jpg)
