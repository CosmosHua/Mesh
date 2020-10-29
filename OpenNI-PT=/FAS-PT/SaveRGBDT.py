# coding:utf-8
# !/usr/bin/python3

import os, cv2
from time import time


rsz = lambda im,rt: cv2.resize(im, None, fx=rt, fy=rt)
################################################################################
def SaveRGBDT(dst=None):
    from CamLib.Cam import *
    if type(dst)!=str: dst = "%d"%time()
    if not os.path.isdir(dst): os.mkdir(dst)
    print("Data will be saved to <%s>."%dst)
    # create a RGBD cam, init stream
    A = Cam(); A.Init(CamD); A.Init(CamRGB)
    thm = cv2.VideoCapture(1); sv = False
    while True:
        im = A.GetRGBD() # get RGBD frame
        rt, tm = thm.read() # Thermal frame
        if rt: # add thermal horizontally
            if tm.shape[:2]==(292,384): # Xtherm: valid=[:288,:]
                tm = cv2.applyColorMap(tm[:288], cv2.COLORMAP_JET)
            tm = rsz(cv2.flip(tm,1), im.shape[0]/tm.shape[0])
            im = cv2.hconcat([im,tm]) # add to the right
        cv2.imshow("RGBD", rsz(im,0.5)); k = cv2.waitKey(10)
        if k==27: break # 27=ESC, 13=Enter
        if k==32: sv = not sv # 32=Space
        if sv: # save stream as images
            id = ("%.3f"%time()).replace(".","_")
            cv2.imwrite(dst+"/"+id+".png", im)
    A.Close(CamD); A.Close(CamRGB) # end stream
    cv2.destroyAllWindows(); thm.release()
    if os.listdir(dst)==[]: os.rmdir(dst)


RIT = lambda s: tuple(int(round(i)) for i in s)
map = lambda X: dict(d for d in enumerate(sorted(os.listdir(X))))
################################################################################
def LabelRGBDT(src, dst="./train"):
    # pip3 install mtcnn-pytorch
    from mtcnn import detect_faces
    d = int(input("[1=RGB,2=RGBD,3=RGBDT] for <%s>: "%src))
    BC = (0,255,0); cls = map(dst); print(cls); assert d in (1,2,3)
    for i in os.listdir(src): # loop images
        im = cv2.imread(src+"/"+i); w = im.shape[1]//d
        dp = im[:,:w]; co = im[:,w:w*2]; tm = im[:,w*2:w*3]
        bbx, pts = detect_faces(co, 40) # detect faces
        for j,bx in enumerate(bbx): # loop faces
            bx = RIT(bx); dd = dp.copy(); cc = co.copy()
            cv2.rectangle(dd, bx[:2], bx[2:4], BC, thickness=2)
            cv2.rectangle(cc, bx[:2], bx[2:4], BC, thickness=2)
            cv2.imshow("RGBD", cv2.hconcat([dd,cc])) # show
            dd = dp[bx[1]:bx[3], bx[0]:bx[2]] # depth_face
            cc = co[bx[1]:bx[3], bx[0]:bx[2]] # color_face
            tt = tm[bx[1]:bx[3], bx[0]:bx[2]] # thermal_face
            fc = cv2.hconcat([dd,cc,tt]); k = cv2.waitKey()
            while not 0<k<48+len(cls): k = cv2.waitKey()
            if k==27: cv2.destroyAllWindows(); return
            if k==13: continue # skip this face
            id = dst + "/%s/%s_%d.png"%(cls[k-48],i[:-4],j)
            cv2.imwrite(id, fc); print("Save => %s"%id)
    cv2.destroyAllWindows() # 27=ESC, 13=Enter, 48='0'


################################################################################
if __name__ == "__main__":
    while True:
        tid = "%d"%time(); SaveRGBDT(tid)
        if input("Again?[y/n]: ")!='y': break
    for tid in os.listdir("."):
        if tid.isdigit(): LabelRGBDT(tid)


################################################################################
