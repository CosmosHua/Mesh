# coding:utf-8
# !/usr/bin/python3
# Ref: https://github.com/KupynOrest/DeblurGAN/tree/master/motion_blur

import os, cv2
import numpy as np

import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt

from scipy import signal
from motion_blur_PSF import PSF
from motion_blur_Trajectory import Trajectory


################################################################################
class BlurImage(object):
    def __init__(self, image_path, PSFs=None, part=None, folder=None):
        """
        :param image_path: path to RGB/BGR image.
        :param PSFs: array of Kernels.
        :param part: int number of kernel to use.
        :param folder: folder to save results.
        """
        if type(image_path)==str:
            self.image_path = image_path
            self.original = cv2.imread(self.image_path)
        elif type(image_path)==np.ndarray:
            self.image_path = "blur.jpg"
            self.original = image_path
        else: raise Exception('Not an image!')

        self.shape = self.original.shape
        if len(self.shape) < 3:
            raise Exception('Only support RGB/BGR images yet.')
        
        self.folder = "." if folder==None else folder
        if PSFs is None:
            if self.folder is None:
                self.PSFs = PSF(canvas=self.shape[0]).fit()
            else:
                self.PSFs = PSF(canvas=self.shape[0], out=os.path.join(self.folder,'PSFs.png')).fit(save=True)
        else:
            self.PSFs = PSFs

        self.part = part
        self.result = []


    def blur_image(self, save=False, show=False):
        if self.part is None: psf = self.PSFs
        else: psf = [self.PSFs[self.part]]
        
        yN, xN, channel = self.shape
        key, kex = self.PSFs[0].shape
        delta = yN - key; ss = min(self.shape[:2])
        assert delta >= 0, "Require: Image Resolution > Kernel's."
        
        self.result = [0] * len(psf)
        square = cv2.resize(self.original, (ss,ss)) # square
        for i, p in enumerate(psf):
            tmp = np.pad(p, delta // 2, 'constant')
            cv2.normalize(tmp, tmp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            blured = cv2.normalize(square, square, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            for c in range(channel): blured[:,:,c] = np.array(signal.fftconvolve(blured[:,:,c], tmp, 'same'))
            blured = cv2.normalize(blured, blured, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            blured = cv2.resize(np.abs(blured)*255, self.shape[1::-1]) # abs, scale, resize square->original
            self.result[i] = blured
        
        if show or save: self.__plot_canvas(show, save)
        return self.result # BGR format


    def __plot_canvas(self, show, save):
        N = len(self.result) # number of result
        if N < 1: raise Exception('Please run blur_image() method first.')
        
        if show: # show result
            plt.close(); plt.axis('off')
            fig, axes = plt.subplots(1, N, figsize=(10, 10))
            if N > 1:
                for i in range(N): axes[i].imshow(self.result[i])
            else: plt.axis('off'); plt.imshow(self.result[0])
            plt.show()

        if save: # save result
            name = os.path.basename(self.image_path)
            for i in range(N): 
                out = name[:-4]+"_"+str(i)+".jpg"
                cv2.imwrite(os.path.join(self.folder, out), self.result[i])


################################################################################
def mBlur(src, canvas=64, xlen=30, part=None, expl=None, folder=None):
    params = [0.01, 0.009, 0.008, 0.007, 0.005, 0.003]
    
    if part==None: part = np.random.choice([1,2,3])
    if expl==None: expl = np.random.choice(params)
    elif type(expl)==list: expl = np.random.choice(expl)
    
    trajectory = Trajectory(canvas, max_len=xlen, expl=expl).fit()
    psf = PSF(canvas, trajectory=trajectory).fit() # Point Spread Function
    return BlurImage(src, PSFs=psf, part=part, folder=folder) # call blur_image


################################################################################
if __name__ == '__main__':
    folder = 'test/'
    for im in os.listdir(folder):
        im = os.path.join(folder, im)
        mBlur(im, folder=folder).blur_image(save=True)
