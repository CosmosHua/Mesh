# coding:utf-8
# !/usr/bin/python3

import os, cv2
import numpy as np
from time import time


cid = 0,1,3 # cam_id: (rgb,th1,th2)
rsz = lambda im,rt: cv2.resize(im, None, fx=rt, fy=rt)
################################################################################
def SaveRGBT():
    RGB, th1, th2 = [cv2.VideoCapture(i) for i in cid]
    while True: # as init-RGB is too slow
        dst = "%d"%time(); sv = False
        if not os.path.isdir(dst): os.mkdir(dst)
        print("Data will be saved to <%s>."%dst)
        while True: # Xtherm: valid=t1[:288,:]
            rt, co = RGB.read(); co = cv2.flip(co,1)
            rt, t1 = th1.read(); t1 = cv2.flip(t1,1)
            #rt, t2 = th2.read(); t2 = cv2.transpose(t2)
            #rt = co.shape[1]/t2.shape[0] # <=transpose
            #cv2.imshow("Th2", rsz(t2,rt)) # True=1
            cv2.imshow("RGB", co); k = cv2.waitKey(10)
            if k==27: break # 27=ESC, 13=Enter
            elif k==32: sv = not sv # 32=Space
            if sv: # save stream as images
                id = ("%.3f"%time()).replace(".","_")
                cv2.imwrite(dst+"/"+id+"_rgb.png", co)
                cv2.imwrite(dst+"/"+id+"_th1.png", t1)
                #cv2.imwrite(dst+"/"+id+"_th2.png", t2)
                cv2.imshow("Th1", rsz(t1,480/288))
            else: cv2.imshow("Th1", t1) # for hint
        cv2.destroyAllWindows() # end stream
        if os.listdir(dst)==[]: os.rmdir(dst)
        if input("Again?[y/n]:")=="n": break
    th1.release(); th2.release(); RGB.release()


################################################################################
def Align(co, t1, xy, k, sh=None):
    x, y = xy; rt = co # 27=ESC, 32=Space
    if   k==119: y -= 1 # 119=w, upward
    elif k==115: y += 1 # 115=s, downward
    elif k==100: x += 1 # 100=d, rightward
    elif k== 97: x -= 1 #  97=a, leftward
    if k!=32: # superpose->RGB
        rt = np.zeros(co.shape).astype("uint8")
        rt[y:y+t1.shape[0], x:x+t1.shape[1]] = t1
        cv2.putText(rt,"(%d,%d)"%(x,y),(x,y),1,1,(0,255,0))
        rt = cv2.addWeighted(co,0.6, rt,0.6, 0)
    else: # 32=Space: superpose->RGBA
        rt = (np.ones(co.shape[:2])*255).astype("uint8")
        rt[y:y+t1.shape[0], x:x+t1.shape[1]] = t1[:,:,0]
        #rt = np.concatenate((co,rt[:,:,None]), axis=2)
        rt = cv2.merge((co,rt)) # =(*cv2.split(co),rt)
    if type(sh)==str: cv2.imshow(sh,rt)
    return rt, (x,y)


def View(xy=(220,166)):
    dst = "%d"%time(); sv = False
    if not os.path.isdir(dst): os.mkdir(dst)
    RGB, th1 = [cv2.VideoCapture(i) for i in cid[:2]]
    while True: # Xtherm: valid=t1[:288,:]
        rt, co = RGB.read(); co = cv2.flip(co,1)
        rt, t1 = th1.read(); t1 = cv2.flip(t1,1)
        cv2.imshow("Th1", t1); k = cv2.waitKey(10)
        if k==27: cv2.destroyAllWindows(); break
        elif k==32: sv = not sv # 27=ESC, 32=Space
        rt, xy = Align(co, t1, xy, k, "RGB")
        if sv: cv2.imwrite(dst+"/%.3f.png"%time(), rt)
    if os.listdir(dst)==[]: os.rmdir(dst)
    th1.release(); RGB.release()


fKB = lambda f: os.path.getsize(f)/1024
RIT = lambda s: tuple(int(round(i)) for i in s)
map = lambda X: dict(d for d in enumerate(sorted(os.listdir(X))))
################################################################################
def LabelRGBT(src, dst="./train"):
    # pip3 install mtcnn-pytorch
    from mtcnn import detect_faces
    cls = map(dst); print(cls); xy = (184,96)
    k = input("[%s]: "%src); au = k.isdigit()
    k = int(k)+48 if k.isdigit() else -1
    for i in sorted(os.listdir(src)):
        if "_th" in i: continue
        co = cv2.imread(src+"/"+i) # load RGB
        t1 = cv2.imread(src+"/"+i.replace("rgb","th1"))
        while not au: # align RGB and Xtherm -> xy
            cv2.imshow("Th1", t1); k = cv2.waitKey(10)
            if k==27: cv2.destroyAllWindows(); return
            elif k==13: break # Enter: obtain xy
            rt, xy = Align(co, t1, xy, k, "RGB")
        bbx, pts = detect_faces(co, 40) # detect faces
        for j,bx in enumerate(bbx): # loop faces
            tx = RIT(bx[:4]-np.array(xy*2)); bx = RIT(bx)
            if not au: # manually label face -> k
                cv2.rectangle(rt, bx[:2], bx[2:4], (0,255,0), 1)
                cv2.imshow("RGB", rt); k = cv2.waitKey()
                while not 0<k<48+len(cls): k = cv2.waitKey()
                if k==27: cv2.destroyAllWindows(); return
                elif k==13: continue # Enter: skip this face
            cc = co[bx[1]:bx[3], bx[0]:bx[2]] # color_face
            rt = t1[tx[1]:tx[3], tx[0]:tx[2]] # therm_face
            ic, it = cc.shape[0]*cc.shape[1], rt.shape[0]*rt.shape[1]
            if ic<4860 or it<4860: continue # filter tiny by pixels
            ic = dst+"/%s/%s_%d.png"%(cls[k-48],i[:-4],j)
            cv2.imwrite(ic, cc); it = ic.replace("rgb","th1")
            cv2.imwrite(it, rt) # filter tiny by file_size
            if fKB(it)>10 and fKB(ic)>10: print("Save:", ic)
            else: os.remove(ic); os.remove(it)
    cv2.destroyAllWindows() # 27=ESC, 13=Enter, 48='0'


def Label_File(src, ext=".txt"):
    from glob import glob; import shutil
    for ff in glob("*"+ext):
        dst = ff[:-len(ext)]
        if not os.path.isdir(dst): os.mkdir(dst)
        with open(ff, "r") as f:
            for i in f.readlines():
                i = os.path.join(src, i.strip())
                if os.path.isfile(i): shutil.move(i,dst)


################################################################################
if __name__ == "__main__":
    View() # SaveRGBT()
    #for tid in os.listdir("."):
    #    if tid.isdigit(): LabelRGBT(tid)


################################################################################
