# coding:utf-8
#!/usr/bin/python3

import os, cv2
import numpy as np
from glob import glob


################################################################################
#Ref: https://github.com/Newmu/dcgan_code
# import pprint; pp = pprint.PrettyPrinter()
# get_stddev = lambda x,h,w: (w*h*x.get_shape()[-1])**-0.5

globs = lambda DIR,ff: sorted(glob(os.path.join(DIR,ff)))
rsz = lambda im,wh: cv2.resize(im, tuple(wh), interpolation=cv2.INTER_LANCZOS4)
rsf = lambda im,rt: cv2.resize(im, None, fx=rt, fy=rt, interpolation=cv2.INTER_LANCZOS4)
################################################################################
# Load single Clean-Mesh image pair: resize->concatenate in channels
# params: A = Mesh image(input x); B = Clean image(label y)
# Note: Mesh & Clean images are in SAME folder for training!
def load_image(src, size=(256,256), isTrain=True, mod="_"):
    A = src if type(src)!=str else cv2.imread(src)
    sz = A.shape[1::-1]; B = A # Test: B is arbitrary
    if isTrain: # For Training
        if type(src)==str and "_" in mod: # mesh naming with "_"
            clean = src[:-4]+".png" # OR: src[:src.rfind("_")]+".jpg"
            # OR: sp = src.split("_"); clean = "_".join(sp[:-1])+".jpg"
            if os.path.isfile(clean): B = cv2.imread(clean) # label
            else: print("Invalid->Skip:", clean); return False
        else: # src: (Clean & Mesh) along axis=1(width)
            sz = (A.shape[1]//2, A.shape[0])
            B, A = np.split(A, 2, axis=1) # A[:,:w], A[:,w:]
        # flipCode: Vertical=0, Horizontal>0, Both/Central<0
        if np.random.rand()<(isTrain/2):
            A = cv2.flip(A,1); B = cv2.flip(B,1)
    
    #A = cv2.cvtColor(A, cv2.COLOR_BGR2RGB) # =A[:,:,::-1]
    #B = cv2.cvtColor(B, cv2.COLOR_BGR2RGB) # =B[:,:,::-1]
    A = rsz(A, size)/127.5-1 # resize->center normalize
    B = rsz(B, size)/127.5-1 # resize->center normalize
    BA = np.concatenate((B,A), axis=2) # along channels
    return BA, sz # BA=(size, B_channel+A_channel)


# Save images to path: resize or merge
def save_images(images, path, size): # to size
    im = (images+1.0)*127.5 # restore to [0,255]
    #im = im[:,:,:,::-1] # restore channels: RGB2BGR
    
    #mx = max(size); rt = 220/mx # for resize too small
    #if mx<220: size = tuple((np.array(size)*rt).astype(int))
    if len(im)<2: im = rsz(im[0], size) # resize single image
    else: im = Merge(im, (178,220)) # resize then merge images
    
    if path[-3:]=="png": cv2.imwrite(path, im, [cv2.IMWRITE_PNG_COMPRESSION,3])
    else: cv2.imwrite(path[:-4]+"r.jpg", im, [cv2.IMWRITE_JPEG_QUALITY,100])


# Join images after resize to uniform size
def Merge(IMs, size): # resize + merge
    N = len(IMs); w,h = size # size=(width,height)
    R = int(N**0.5); C = int(np.ceil(N/R)) # layout=(Row,Col)
    
    # Method 3: SUCCESS!
    pd = np.zeros((h, w, IMs.shape[-1])) # padding
    IMs = [rsz(im,size) for im in IMs] + [pd]*(R*C-N) # resize + pad
    IMs = [np.concatenate(IMs[i*C:i*C+C],axis=1) for i in range(R)] # widen
    return np.concatenate(IMs,axis=0) # heighten: join rows
    
    '''# Method 1: SUCCESS!
    img = np.zeros((R*h, C*w, IMs.shape[-1]))
    for id,im in enumerate(IMs): # (i,j)=(row,col)
        i,j = id//C, id%C;  img[i*h:i*h+h, j*w:j*w+w, :] = rsz(im,size)
    return img
    
    # Method 2: FAIL!
    pd = np.zeros((h, w, IMs.shape[-1])) # padding
    IMs = [rsz(im,size) for im in IMs] + [pd]*(R*C-N) # resize
    #IMs[N:] = [pd.copy() for i in range(R*C-N)] # unnecessary
    return np.array(IMs).reshape([R*h, C*w, -1]) # FAIL!'''


################################################################################
# Peak Signal to Noise Ratio
def PSNR(I, K, ch=1, L=255):
    if type(I)==str: I = cv2.imread(I)
    if type(K)==str: K = cv2.imread(K)
    # assert(I.shape == K.shape) # assert if False
    if I.shape!=K.shape: K = rsz(K, I.shape[1::-1])

    IK = (I-K*1.0)**2; MAX = L**2; ee = MAX*1E-10
    if ch<2: MSE = np.mean(IK) # combine/average channels
    else: MSE = np.mean(IK,axis=(0,1)) # separate channels
    return 10 * np.log10(MAX/(MSE+ee)) # PSNR


# Structural Similarity (Index Metric)
def SSIM(I, K, ch=1, k1=0.01, k2=0.03, L=255):
    if type(I)==str: I = cv2.imread(I)
    if type(K)==str: K = cv2.imread(K)
    # assert(I.shape == K.shape) # assert if False
    if I.shape!=K.shape: K = rsz(K, I.shape[1::-1])

    if ch<2: # combine/average channels->float
        mx, sx = np.mean(I), np.var(I,ddof=1)
        my, sy = np.mean(K), np.var(K,ddof=1)
        cov = np.sum((I-mx)*(K-my))/(I.size-1) # unbiased
        # cov = np.mean((I-mx)*(K-my)) # biased covariance
    else: # separate/individual/independent channels->np.array
        mx, sx = np.mean(I,axis=(0,1)), np.var(I,axis=(0,1),ddof=1)
        my, sy = np.mean(K,axis=(0,1)), np.var(K,axis=(0,1),ddof=1)
        cov = np.sum((I-mx)*(K-my),axis=(0,1))/(I.size/I.shape[-1]-1) # unbiased
        # cov = np.mean((I-mx)*(K-my),axis=(0,1)) # biased covariance
    
    c1, c2 = (k1*L)**2, (k2*L)**2 # stabilizer, avoid divisor=0
    SSIM = (2*mx*my+c1)/(mx**2+my**2+c1) * (2*cov+c2)/(sx+sy+c2)
    return SSIM # SSIM: separate or average channels


def BatchPS(val_dir, naming="_"): # Ref: load_image
    if "_" in naming: # Mesh images naming with "_"
        #mesh = globs(val_dir, naming)
        recov = globs(val_dir, naming[:-4]+"_.png")
        clean = [i[:i.rfind('_')]+'.png' for i in recov] # png/jpg
        psnr = np.mean([PSNR(i,k) for i,k in zip(clean,recov)])
        ssim = np.mean([SSIM(i,k) for i,k in zip(clean,recov)])
    else: # Clean & Mesh join in BA order, along axis=1(width)
        BA = globs(val_dir, naming) # Clean+Mesh
        psnr = ssim = 0; num = len(BA)
        for i in BA:
            B,A = np.split(cv2.imread(i), 2, axis=1)
            A = cv2.imread(i[:-4]+"_.png") # recov
            psnr += PSNR(B,A); ssim += SSIM(B,A)
        psnr /= num; ssim /= num # average
    return np.array([psnr, ssim, psnr*ssim])


################################################################################
def combine(folderA, out, folderB=None, mod="A2B"):
    ren = lambda im: im[:im.rfind("_")]+".jpg"
    if not os.path.exists(out): os.mkdir(out)
    for i in os.listdir(folderA):
        A = cv2.imread(os.path.join(folderA,i))
        if not folderB: B = np.zeros(A.shape)
        else:
            k = os.path.join(folderB,i); kk = ren(k)
            if os.path.exists(k): B = cv2.imread(k)
            elif os.path.exists(kk): B = cv2.imread(kk)
            else: continue # skip if no pairs
            if B.shape!=A.shape: B = rsz(B, A.shape[1::-1])
        if mod.find("A")>mod.find("B"): A,B = B,A
        B = np.concatenate((A,B), axis=1)
        cv2.imwrite(os.path.join(out,i), B)

#combine("FaceA/", "test/", "Face1K", mod="A-B")

