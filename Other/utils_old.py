# coding:utf-8
#!/usr/bin/python3

import os, cv2
from sys import argv


import numpy as np
# resize images in batch
#####################################################################
def Resize(src, sz=(178,220)):
    for path,sub,file in os.walk(src):
        for im in file:
            img = cv2.imread(src+"/"+im)
            ss = np.array(img.shape[1::-1])
            if (np.array(sz)!=ss).any():
                cv2.imwrite(src+"/"+im, cv2.resize(img, sz))
#ReBatch("E:/PyCharm/Clean")


# Delete useless images
#####################################################################
def Delete(path, list_file):
    with open(list_file, mode="r") as ff:
        ll = ff.readline().split(" ")
        for id,i in enumerate(ll):
            ff = path+i+".jpg"
            if os.path.exists(ff): os.remove(ff)
            print(id, "remove:", ff)
#Delete("id_face/", "id_face.txt")


# resize images in batch
#####################################################################
def Rename(src):
    for path,sub,file in os.walk(src):
        for i,s in enumerate(sub):
            os.rename(s+"/F.jpg", src+"/F"+str(i)+".jpg")
#Rename("E:/FacePic/facex_20181120")


# copy and rename
import shutil as sh
#####################################################################
def Copy(path, out, cp):
    if not os.path.exists(out): os.mkdir(out)
    re = lambda na,k: i[:-4]+"_"+str(k)+".jpg"
    for i in os.listdir(path):
        for k in range(cp):
            sh.copy(path+i, out+re(i,k))
#Copy("IDFace1W1/", "trainB/", 4)


# get booklet pages, for fun
#####################################################################
def Booklet(N, st=1):
    N += (4-N%4) if N%4!=0 else 0
    p = list(range(st,N+st))
    pg = []; M = N-1
    for i in range(N//2):
        if i>=N//4: break
        j = 2*i; pg.append((p[j],p[M-j],p[j+1],p[M-j-1]))
    return pg
#Booklet(int(argv[1]))


import random as rd
#####################################################################
def rdkey(N=20, s=0):
    x0 = "1234567890"
    x1 = "!@#$%^&()_+|=."
    x2 = "abcdefghijklmnopqrstuvwxyz"
    xs = x0 + x1 + x2.lower() + x2.upper()
    key = [rd.choice(xs) for i in range(N)]
    for i in range(s): rd.shuffle(key)
    return "".join(key)
