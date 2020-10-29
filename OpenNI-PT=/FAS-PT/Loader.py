# coding:utf-8
# !/usr/bin/python3

import os, cv2
import numpy as np
from PIL import Image
from DataRGBD import *


from torchvision import transforms
cmb = 13; MX = 3E3; jt = 0.4; hw = 224
inc = 4 if (cmb in [4,13,31]) else (1 if cmb in [1,2] else 3)
################################################################################
class Jitter(object):
    """Split the given PIL Image into A(=depth) and B(=color).
    1.Randomly Jitter brightness, contrast and saturation of B(color).
    2.Randomly Translate the depth of A(depth) in range of [200,max],
    while keeping the pixels in B and whose depth=0 in A unchanged.
    Args:
        sp (1,0): the dimension to split the image, default=1.
            sp=0: split along height. A,B = im[:h,:],im[h:2*h,:]
            sp=1: split along width.  A,B = im[:,:w],im[:,w:2*w]
        color (tuple): see torchvision.transforms.ColorJitter().
            default: brightness=0, contrast=0, saturation=0, hue=0.
        mx (int): the maximum translation distance, default=5E3.
        mod (int): refer to Encode|Decode in DataRGBD.py, default=1."""
    def __init__(self, color=(0,0,0,0), mx=0, mod=1, sp=1):
        self.sp = 1 if sp else 0; self.mx = mx; self.mod = mod
        self.jitter = transforms.ColorJitter(*color) # only for B


    def __call__(self, img): # A=depth, B=color
        """Args: img (PIL Image), consists of depth and color.
        Returns: the Color_Jittered and Translated image (PIL Image)."""
        assert isinstance(img, (str, Image.Image)) # require: RGB->BGR
        im = cv2.imread(img,-1) if type(img)==str else np.array(img)[:,:,::-1]
        
        sp, mod = self.sp, self.mod; d = im.shape[sp]//2 # split
        A, B = (im[:,:d],im[:,d:2*d]) if sp else (im[:d,:],im[d:2*d,:])
        
        dp = Decode(A,mod); mi, mx = dp.min(), dp.max()
        if 0==mi<mx: mi = (dp+(dp<1)*mx).min() # 2nd_min>0
        dx = np.random.randint(200-mi, max(1, 201-mi, self.mx-mx))
        A = Encode(dp+(dp>0)*dx, mod); self.dx = dx # translate->encode
        
        B = self.jitter(Image.fromarray(B[:,:,::-1])) # color_jitter
        im = np.concatenate((A, np.array(B)[:,:,::-1]), axis=sp)
        return Image.fromarray(im[:,:,::-1]) # BGR->RGB


    def __repr__(self):
        sp = {0: "height", 1: "width"}; sp = sp[self.sp]
        info = "(Split(%s), Translate(%d), %s)" % (sp, self.dx, str(self.color))
        return self.__class__.__name__ + info


################################################################################
class Superpose(object):
    """Split the given PIL Image into A(=depth) and B(=color),
    Superpose one to another. Ref: DataRGBD.Superpose, but RGB.
    Args:
        sp (1,0): the dimension to split the image, default=1.
            sp=0: split along height. A,B = im[:h,:],im[h:2*h,:]
            sp=1: split along width.  A,B = im[:,:w],im[:,w:2*w]
        cmb (int): the mode to superpose A and B, default=1.
         cmb=0/-1: join A and B along height/width dimension.
            cmb<0: superpose by cv2.addWeighted(A,1, B,1/cmb, 0).
            cmb=1: only use depth(A->1): A(1), independent of origin.
            cmb=3: just use depth(A->3): A(3), dependent on origin.
           cmb=13: join A(->Alpha) and B along channel: A(a)+B(3).
           cmb=31: join A and B(->Alpha) along channel: A(3)+B(a).
           others: just use the original image.
        mod (int): refer to Encode|Decode in DataRGBD.py, default=1."""
    def __init__(self, cmb=1, mod=1, sp=1):
        self.sp = 1 if sp else 0; self.cmb = cmb; self.mod = mod


    def __call__(self, img):
        """Args: img (PIL Image), consists of depth and color.
        Returns: the Split and Superposed image (PIL Image)."""
        assert isinstance(img, (str, Image.Image)) # require: RGB
        im = Image.open(img) if type(img)==str else np.array(img) # RGB
        
        sp, cmb, mod = self.sp, self.cmb, self.mod; d = im.shape[sp]//2
        A, B = (im[:,:d],im[:,d:2*d]) if sp else (im[:d,:],im[d:2*d,:])
        
        if cmb in [0,-1]: # join: (depth+color)(3)
            im = np.concatenate((A,B), axis=-cmb)
        # -2: converge slowly, acc=0.97-0.98+; acc_v=0.83-0.88
        elif cmb < 0: # superpose: (depth+color)(3)
            im = cv2.addWeighted(A, 1, B, -1/cmb, 0)
        # 1: converge stable, acc=0.95-0.96; acc_v=0.97-0.99
        elif cmb==1: # depth(1): clip origin|large
            im, _ = DepthClip(A[:,:,::-1], mod) # BGR
        # 2: converge sluggish, acc=0.87-0.88; acc_v=0.77-0.81
        elif cmb==2: # depth(1): not clip, normalize
            im = Decode(A[:,:,::-1], mod) # BGR
            im = (255/im.max()*im).astype("uint8") # [0,255]
        # 3: converge slowly, acc=0.95-0.96; acc_v=0.94-0.95
        elif cmb==3: im = A.copy() # depth(3): not clip
        # 4: converge slowly, acc=0.94-0.95; acc_v=0.97-0.98
        elif cmb==4: # combine: depth(3)+depth(a)
            im, _ = DepthClip(A[:,:,::-1], mod) # BGR->Alpha
            im = np.concatenate((A, im[:,:,None]), axis=2)
        # 13: converge fast, acc=0.98-0.99; acc_v=0.84-0.88
        elif cmb==13: # combine: color(3)+depth(a)
            A, x = DepthClip(A[:,:,::-1], mod) # BGR
            B[0,0] = Encode(x, mod) # backup origin
            im = np.concatenate((B, A[:,:,None]), axis=2)
        # 31: converge fast, acc=0.98-0.99+; acc_v=0.95-0.96+
        elif cmb==31: # combine: depth(3)+color(a)
            B = cv2.cvtColor(B, cv2.COLOR_RGB2GRAY)
            im = np.concatenate((A, B[:,:,None]), axis=2)
        #elif cmb==33: # invalid: depth(3)+color(3)
        #    im = np.concatenate((A,B), axis=2)
        return Image.fromarray(im) # restore to PIL.Image


    def __repr__(self):
        sp = {0: "height", 1: "width"}; sp = sp[self.sp]
        info = "(Split(%s), Superpose_Mode(%d))" % (sp, self.cmb)
        return self.__class__.__name__ + info


################################################################################
def Norm(cmb=cmb):
    avg, std = [0.485,0.458,0.407], [0.229,0.224,0.225] # color(RGB)
    avg2, std2 = [0.330,0.353,0.047], [0.25,0.25,0.047] # depth(RGB)
    if cmb==1:   avg, std = [0.15], [0.12] # cliped depth(1)
    elif cmb==2: avg, std = [0.5], [0.25]  # original depth(1)
    elif cmb==3: avg, std = avg2, std2    # original depth(3)
    elif cmb==4: avg, std = avg2+[0.15], std2+[0.12]  # depth(3)+depth(a)
    elif cmb==13: avg, std = avg+[0.15], std+[0.12]   # color(3)+depth(a)
    elif cmb==31: avg, std = avg2+[0.45], std2+[0.226] # depth(3)+color(a)
    elif cmb==33: avg, std = avg2+avg, std2+std  # depth(3)+color(3)
    return avg, std


avg, std = Norm(cmb) # OR: transforms.Normalize(*Norm(cmb))
################################################################################
TSFM_Train = transforms.Compose([Jitter(color=(jt,)*4, mx=MX, mod=1, sp=1),
                                Superpose(cmb=cmb, mod=1, sp=1),
                                transforms.RandomHorizontalFlip(0.5),
                                transforms.RandomAffine(45, (0.2,0.2)),
                                transforms.RandomResizedCrop(hw, (0.5,1.0)),
                                transforms.ToTensor(),
                                transforms.Normalize(avg,std) ])
TSFM_Test = transforms.Compose([Superpose(cmb=cmb, mod=1, sp=1),
                                transforms.Resize(256), #[hw,hw]
                                transforms.CenterCrop(hw),
                                transforms.ToTensor(),
                                transforms.Normalize(avg,std) ])
#TSFM_Test = TSFM_Train.transforms[-6:] # make a copy
#TSFM_Test[-5:-2] = [transforms.Resize(256), transforms.CenterCrop(hw)] # alter
#TSFM_Test = transforms.Compose(TSFM_Test) # compose


from torch.utils.data import DataLoader
from torchvision import datasets #, models
i_cls = lambda X: {k:v for k,v in enumerate(sorted(os.listdir(X)))}
i_cls = lambda X: dict(d for d in enumerate(sorted(os.listdir(X))))
################################################################################
def LoadData(dir, train=True, bs=64):
    TSFM = TSFM_Train if train else TSFM_Test; cls = i_cls(dir)
    Data = datasets.ImageFolder(dir, transform=TSFM); j = round(bs**0.25)
    assert cls=={v:k for k,v in Data.class_to_idx.items()}, "Classes discord!"
    return DataLoader(Data, batch_size=bs, shuffle=train, num_workers=j), cls


def LoadFace(im): # single PIL.Image->(1,C,H,W)
    if type(im)==str: im = Image.open(im) # require: ->RGB
    elif type(im)==np.ndarray: im = Image.fromarray(im[:,:,::-1])
    assert isinstance(im, Image.Image) # type(im)!=Image.Image
    x = TSFM_Test(im); return x.reshape(-1,*x.shape)


################################################################################
