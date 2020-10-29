# coding:utf-8
#!/usr/bin/python3

import os, cv2
import numpy as np
from sys import argv
from glob import glob


res = lambda im,ss: cv2.resize(im, ss, interpolation=cv2.INTER_LANCZOS4)
##################################################################
def RSIP(Is): # Test: Resize InterPolation
    Is = Is + "/"*(Is[-1]!="/"); Ks = Is[:-1] + "_/"
    if not os.path.exists(Ks): os.mkdir(Ks) # mkdir
    ip = {cv2.INTER_NEAREST: "INTER_NEAREST-0",
          cv2.INTER_LINEAR: "INTER_LINEAR-1",
          cv2.INTER_CUBIC: "INTER_CUBIC-2",
          cv2.INTER_AREA: "INTER_AREA-3",
          cv2.INTER_LANCZOS4: "INTER_LANCZOS4-4"}
    for md in ip:
        for file in os.listdir(Is):
            im = cv2.imread(Is + file); sz = im.shape[1::-1]
            im = cv2.resize(im, (250,250), interpolation=md)
            im = cv2.resize(im, sz, interpolation=md)
            cv2.imwrite(Ks + file, im)
        print(ip[md], ":") # show InterPolation Mode
        Batch(Is, Ks, fun=PSNR); Batch(Is, Ks, fun=SSIM)


##################################################################
# Peak Signal to Noise Ratio
def PSNR(I, K, ch=1, L=255):
    if type(I)==str: I = cv2.imread(I)
    if type(K)==str: K = cv2.imread(K)
    # assert(I.shape == K.shape) # assert if False
    if I.shape!=K.shape: K = res(K, I.shape[1::-1])
    
    #assert( (I-K == I-K*1.0).all() ) # test overflow
    #if I.dtype=="uint8" or K.dtype=="uint8": # avoid overflow
    #    IK = (I.astype(int)-K)**2 # IK=(np.array(I,int)-K)**2
    IK = (I-K*1.0)**2 # to avoid "uint8" overflow
    
    MAX = L**2; ee = MAX*1E-10 # normalize PSNR to 100
    if ch<2: MSE = np.mean(IK) # combine/average channels
    else: MSE = np.mean(IK,axis=(0,1)) # separate channels
    return 10 * np.log10(MAX/(MSE+ee)) # PSNR


# Structural Similarity (Index Metric)
def SSIM(I, K, ch=1, k1=0.01, k2=0.03, L=255):
    if type(I)==str: I = cv2.imread(I)
    if type(K)==str: K = cv2.imread(K)
    # assert(I.shape == K.shape) # assert if False
    if I.shape!=K.shape: K = res(K, I.shape[1::-1])
    
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


##################################################################
def Batch(*args, fun=SSIM, ch=1):
    if type(args[0])!=str: args = args[0] # parse
    ff = lambda I,K: np.mean(fun(I,K,ch)) # 1-value
    
    MD = {SSIM: "SSIM", PSNR: "PSNR"} # assess mode
    if len(args)>2: # I=Origin, K=Mesh, R=Recover
        Is, Ks, Rs = args[0], args[1], args[2]
        I = [os.path.join(Is, i) for i in os.listdir(Is)]
        K = [os.path.join(Ks, i) for i in os.listdir(Ks)]
        R = [os.path.join(Rs, i) for i in os.listdir(Rs)]
        # res = [ff(i,r)-ff(i,k) for i,k,r in zip(I,K,R)]
        Ks = np.array([ff(i,k) for i,k in zip(I,K)])
        Rs = np.array([ff(i,r) for i,r in zip(I,R)])
        res = Ks, Rs, (Rs-Ks) #/abs(Ks) # Gain/Gain_Ratio
        avg = tuple([np.mean(i) for i in res])
        print(MD[fun], ":\tMesh=%f\tRecover=%f\tGain=%f" % avg)
    return avg, res


def BatchPS(test, clean):
    psnr = ssim = num = 0
    for i in os.listdir(test): # notice naming rule
        #k = i[:i.find("_")]+".jpg" if "_" in i else i # left
        k = i[:i.rfind("_")]+".jpg" if "_" in i else i # right
        i = os.path.join(test, i); k = os.path.join(clean, k)
        psnr += PSNR(i,k); ssim += SSIM(i,k); num += 1
    psnr /= num; ssim /= num; return [psnr, ssim, psnr*ssim]


##################################################################
if __name__ == "__main__":
    '''
    Is, Ks = "E:/Hua/PyCharm/code/Clean", "E:/Hua/PyCharm/code/Mesh"
    Rs = "E:/Hua/PyCharm/code/RCVer 22002_4"; # a = Batch(argv[1:])
    a = Batch(Is, Ks, Rs, fun=PSNR); b = Batch(Is, Ks, Rs, fun=SSIM)
    '''
    recov_dir, clean_dir = argv[1], argv[2]
    print(BatchPS(recov_dir, clean_dir))


##################################################################
# python3 PSNR.py Tools/AI_CV_Test_1 > out.txt &
 