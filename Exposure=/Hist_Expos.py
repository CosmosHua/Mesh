# coding:utf-8
#!/usr/bin/python3
import os, cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import MTCNN as df


#####################################################################
class FaceDetect(object):
    def __init__(self, minsize=50, threshold=(0.6, 0.7, 0.7), scale=0.709):
        self.minSize = minsize
        self.thresh = threshold
        self.factor = scale
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        with self.sess.as_default():
            self.pnet, self.rnet, self.onet = df.create_mtcnn(self.sess, None)

    def face(self, image):
        if type(image)==str: image = cv2.imread(image)
        faces, points = df.detect_face(image, self.minSize,
                                        self.pnet, self.rnet, self.onet,
                                        self.thresh, self.factor)
        faces = faces.astype(int) # float->int
        ROIs = []; im = image.copy() # backup
        for fc in faces: # ROI of face[x1,y1,x2,y2]
            ROIs.append(image[fc[1]:fc[3], fc[0]:fc[2]])
            cv2.rectangle(im, (fc[0],fc[1]), (fc[2],fc[3]), (0,255,0))
        cv2.imshow("image", im); cv2.waitKey(1000)
        return ROIs, faces


#####################################################################
# model = FaceDetect() # load model
def FaceCompare(im, std, ch=2, mod="HSV"):
    if type(std)==list: std = std[0]
    else: std = Hist(std, [ch], mod)[0]
    face, _ = model.face(im); score = []
    for f in face:
        hf = Hist(f, [ch], mod)[0]
        sim = cv2.compareHist(std, hf, cv2.HISTCMP_CORREL)
        score.append(sim)
    return score


#####################################################################
def Hist(im, ch, mod="HSV", wd=1, show=False):
    if type(im)==str: im = cv2.imread(im)
    if "HSV" in mod: im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    mod = [mod[i] for i in ch] # rearrange
    wh = im.shape[0]*im.shape[1] # normalize
    for i,c in enumerate(ch): # channels
        # Hue=[0,180), Saturation/Value/R/G/B=[0,256)
        Max = 180 if mod[i] in "Hh" else 256 # range
        ch[i] = cv2.calcHist([im], [c], None, [Max//wd], [0,Max])/wh
        if show: plt.plot(ch[i]); #plt.ylim([0,1])
    if show: plt.legend([i for i in mod]); plt.show()
    return ch # store hist, a list


fm = lambda v,d=1: sum([(i+(d-1)/2)*v[i] for i in range(len(v)-1)])

def Criteria(hist, wd, show=False):
    lo, hi, hi2 = 30, 254, 250
    h1 = hist[hi//wd:].sum()
    h2 = hist[hi2//wd:].sum()
    w1 = hist[:lo//wd].sum()
    w2 = hist[:2*lo//wd].sum()
    w3 = hist[:3*lo//wd].sum()
    mu = fm(hist, wd) # mean value

    res = "" # record abnorm types
    if h1>0.18 or h2>0.24:      res += "High="+2*"%0.3f_"%(h1,h2)
    if h2>0.10 and w1<0.01:     res += "High1="+2*"%0.3f_"%(h2,w1)
    #if h1>0.07 and h2-h1<0.01:  res += "High2="+2*"%0.3f_"%(h1,h2)
    if h2>0.13 and w2>0.15:     res += "Back="+2*"%0.3f_"%(h2,w2)
    if mu<40 and w3>0.40:       res += "Low1="+2*"%0.3f_"%(mu,w3)
    if w1>0.3:                  res += "Low2="+"%0.3f"%w1

    if show: return res # if show==True
    elif "High" in res: return "High", h2
    elif "Back" in res: return "High", h2
    elif "Low1" in res: return "Low", mu
    elif "Low2" in res: return "Low", w1
    else: return "Norm", h1


#####################################################################
def Test(DIR, wd=2):
    os.chdir(DIR); sub = ["ab/","norm/"]
    #for i in sub: os.mkdir(i) # sub_dirs
    for i in os.listdir("."):
        v,s = Hist(i, [2,1], "HSV", wd) # Value
        rs = Criteria(v, wd, True) # exposure types

        plt.plot(v); plt.plot(s); plt.ylim([0,wd/10])
        plt.legend([rs,i]); plt.show()

        im = cv2.imread(i); h,w = im.shape[:2]
        cv2.imshow("im", cv2.resize(im, (w//2,h//2)) )
        rs = 10+10*rs.count("="); cv2.waitKey(rs*1000)

        #if rs.count("=")>0: cv2.imwrite(sub[0]+i,im)
        #else: cv2.imwrite(sub[1]+i,im)
    cv2.destroyAllWindows()


#####################################################################
if __name__ == "__main__":
    #std = ("std0.jpg", "std1.jpg")
    #std = [Hist(i,[2],"HSV") for i in std]
    Test("=extra/")
