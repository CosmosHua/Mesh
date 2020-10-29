# coding:utf-8
# !/usr/bin/python3

import os, csv
from glob import glob


Play = lambda x: os.system('PotPlayerMini.lnk "%s"'%x)
################################################################################
def PlayCSV(DIR, sub="audio"):
    DIR = os.path.abspath(DIR+"/"+sub)+"/"
    CSVs = glob(DIR+"../*.csv")
    for k in CSVs:
        with open(k) as rf:
            for i,row in enumerate(csv.reader(rf),1):
                for d in os.listdir(DIR):
                    vd = glob(DIR+d+"/"+row[0]+"*")
                    if vd: vd=vd[0]; break
                if vd: print(i,vd); Play(vd)


def PlayDIR(DIR, sub="audio"):
    DIR = os.path.abspath(DIR+"/"+sub)+"/"
    CSVs = glob(DIR+"../*.csv")
    for dir,sub,ff in os.walk(DIR):
        for i,vid in enumerate(ff,1):
            for k in CSVs: # find row & k
                with open(k) as rf: #rf.seek(0)
                    for row in csv.reader(rf):
                        if vid[:-4] in row: break
                if vid[:-4] in row: break
            kk= os.path.basename(k)[:-4]+":"
            print(i,kk,row); Play(dir+"/"+vid)
            with open(k[:-4]+"_.csv", "a+", newline='') as wf:
                csv.writer(wf).writerow(row+[input()])


################################################################################
if __name__ == "__main__":
    from sys import argv
    PlayCSV(argv[1])
