# coding:utf-8
# !/usr/bin/python3

import os, cv2
import numpy as np
from glob import glob


################################################################################
def PNG_JPG(fname, dcd=0):
    from RGBDFace import ViewImage
    png = cv2.imread(fname+".png", -1)
    jpg = cv2.imread(fname+".jpg", -1)
    assert png.shape==jpg.shape; w = png.shape[1]//2
    depth = np.concatenate((png[:,:w],jpg[:,:w]), axis=1)
    color = np.concatenate((png[:,w:],jpg[:,w:]), axis=1)
    ViewImage(depth, dcd, color) # cv2.flip


def SwapVideo(tid, dim=1):
    assert dim in [0,1], "Wrong dim!"
    if type(tid)!=str: tid = str(tid)
    vid = cv2.VideoCapture(tid+".mkv")
    
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #codec = int(vid.get(cv2.CAP_PROP_FOURCC))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(tid+"_.mkv", codec, fps, (w,h))
    
    while True:
        rt, im = vid.read()
        if not rt: break # EOF
        if dim: im = (im[:,w//2:], im[:,:w//2])
        else:   im = (im[h//2:,:], im[:h//2,:])
        out.write(np.concatenate(im, axis=dim))
        # horizontal>0, vertical=0, both<0
        #im = cv2.flip(im, flipCode=dim) # not equate
    vid.release(); out.release()


def CropVideo(file, cp=0.7):
    vid = cv2.VideoCapture(file)
    dst = file[:-4]+"_c"+file[-4:]; b = (1-cp)/2
    w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)*cp)
    h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)*cp)
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(dst, codec, 20, (w,h))
    
    while True:
        rt, im = vid.read()
        if not rt: break # EOF
        y0 = int(h*b/cp); im = im[y0:y0+h, :w]
        out.write(im)
    vid.release(); out.release()


def ReName(DIR):
    if type(DIR)!=str: DIR = str(DIR)
    assert os.path.isdir(DIR)
    for dir, subs, files in os.walk(DIR):
        for i in files: # "_?"->"_0?"
            sp = i[:-4].split("_")
            if len(sp[1])<2: sp[1]="0"+sp[1]
            sp = dir+"/"+("_").join(sp)+i[-4:]
            if not os.path.isfile(sp):
                os.rename(dir+"/"+i, sp)
                print(dir+"/"+i, "==>", sp)
            else: print(sp, "Already Exists.")


################################################################################
if __name__ == "__main__":
    #PNG_JPG("1557312010_0", dcd=3)
    #SwapVideo(1556262733)
    #for i in glob("*.mkv"): SwapVideo(i[:10])
    #ReName("data_fc1")
    CropVideo("1556262733val_R.mkv")


################################################################################
