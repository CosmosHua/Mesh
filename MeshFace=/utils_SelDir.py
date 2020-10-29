# coding:utf-8
#!/usr/bin/python3
import os, shutil
from numpy import random
from scipy.special import comb, perm


##################################################################
# Rename files in dir to sequence
def renm(dir, wd=2): # rename files to sequence
    import os; os.chdir(dir); ff = sorted(os.listdir())
    for i,f in enumerate(ff): os.rename(f, str(i).zfill(wd)+f[-4:])


# Select num files from src to dst
# Para: it = (start,len) or list of subdirs
def SelDir(src, dst, it, num=2):
    assert(len(it)>=2 and type(it[1]==int))
    if not os.path.exists(dst): os.mkdir(dst)
    if len(it)==2:
        sub = sorted(os.listdir(src))
        if type(it[0])==str: it[0] = sub.index(it[0])
        it = sub[it[0]: it[0]+it[1]]
    for sub in it: # it = list of subdirs
        os.chdir(os.path.join(src,sub)); im = os.listdir()
        im = random.choice(im, min(num,len(im)), replace=False)
        for i in im: shutil.copy(i, os.path.join(dst,sub+i))


def Poss(Max, num, c=4, md=0):
# Max = The total number of choice questions;
# num = The number of questions with correct choice;
# c = The number of choices of those questions.
# md: 0=got num right exactly; 1=got num right at lease.
    pos = lambda ri,R,wr,W: (ri**R)*(wr**W)
    ri, wr, poss = 1/c, 1-1/c, 0
    N = md*(Max-num) + num+1
    for n in range(num, N):
        poss += comb(Max,n)*pos(ri,n,wr,Max-n)
    return poss


##################################################################
if __name__ == "__main__":
    #perm(5,3)==5*4*3, comb(5,3)==perm(5,3)/(3*2*1); #Poss(7,4,md=1)
    src = "E:/FacePic/WebFace"
    #dst = "E:/FacePic/CS_Train"; SelDir(src, dst, (171,1001), num=10)
    dst = "E:/FacePic/CS_Test2"; SelDir(src, dst, (1851, 2000), num=2)
