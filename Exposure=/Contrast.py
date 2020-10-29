# --coding:utf-8--
#!/usr/bin/python3
import os,cv2
import numpy as np


#####################################################################
# The exponential value of Gamma Function is 2.5.
# If gamma>1: decrease exposure; if gamma<1, increase exposure.
# Using Equalization Principle of lightness/color, design your Adaptive Algorithms.
def gamma_trans(img, gamma): # gamma transform
    gamma_table = [255 * np.power(i/255.0, gamma) for i in range(256)] # mapping table
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8) # color value is uint8
    return cv2.LUT(img, gamma_table) # look up color mapping table


def Contrast(src, f=2):
    def nothing(x): pass
    cv2.namedWindow("Demo")
    cv2.createTrackbar('Gamma', 'Demo', 20, 100, nothing)
    for path in os.listdir(src):
        im = cv2.imread(src + path); h,w = im.shape[:2]
        im = cv2.resize(im, (w//f,h//f)) # resize
        while(True):
            gamma = cv2.getTrackbarPos('Gamma', 'Demo') # get gamma
            gamma /= 20 # scale gamma value for fine adjustment
            cv2.imshow("Demo", gamma_trans(im, gamma))
            k = cv2.waitKey(2) # 13=Enter,27=ESC
            if k==27: cv2.destroyAllWindows(); return
            elif k==13: break


#####################################################################
if __name__ == "__main__":
    Contrast("../=abnorm/")