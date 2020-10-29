# coding:utf-8
# !/usr/bin/python3


import os, cv2
import numpy as np
from PIL import Image
inc = 1 # default mode: RGBA
################################################################################
def valid_RGBT(img, key="_th1"):
    return img.lower().endswith(".png") and key in img


def load_RGBT(thm, inc=inc):
    if inc in [(3,1), 4]: # (rgb,th1)->RGBA
        assert valid_RGBT(thm); rgb = thm.replace("th1","rgb")
        # 1=cv2.IMREAD_COLOR: BGR->RGB, 0=cv2.IMREAD_GRAYSCALE
        rgb = cv2.imread(rgb,1)[:,:,::-1]; h,w = rgb.shape[:2]
        thm = cv2.resize(cv2.imread(thm,0),(w,h))[:,:,None]
        img = np.concatenate((rgb,thm), axis=2) # RGB->RGBA
        return Image.fromarray(img) # np.array->PIL.Image
    elif inc==3: return Image.open(thm).convert("RGB") # RGB
    elif inc==1: return Image.open(thm).convert("L") # GRAY


def norm_RGBT(inc=inc): # default: for RGBA
    avg, std = [0.485,0.458,0.407,0.5], [0.229,0.224,0.225,0.25]
    if inc==1: del(avg[:3], std[:3]) # for GRAY
    if inc==3: del(avg[3:], std[3:]) # for RGB
    return avg, std


i_cls = lambda X: {k:v for k,v in enumerate(sorted(os.listdir(X)))}
i_cls = lambda X: dict(d for d in enumerate(sorted(os.listdir(X))))
################################################################################
def LoadData(dir, train=True, bs=64):
    from torch.utils.data import DataLoader; cls = i_cls(dir)
    TSFM = TSFM_Train if train else TSFM_Test; j = round(bs**0.25)
    try:
        from Folder import FromFolder
        Data = FromFolder(dir, load_RGBT, transform=TSFM, is_valid=valid_RGBT)
    except:
        from torchvision.datasets import DatasetFolder as FromFolder
        Data = FromFolder(dir, load_RGBT, transform=TSFM, is_valid_file=valid_RGBT)
    assert cls=={v:k for k,v in Data.class_to_idx.items()}, "Classes discord!"
    return DataLoader(Data, batch_size=bs, shuffle=train, num_workers=j), cls


def LoadFace(img, inc=inc): # single Image->(1,C,H,W)
    if type(img)==str: img = load_RGBT(img, inc)
    elif type(img) in (tuple, list):
        assert all(type(x)==np.ndarray for x in img)
        if inc in [(3,1), 4]: # ref: load_RGBT
            rgb = img[0][:,:,::-1]; h,w = rgb.shape[:2]
            thm = cv2.resize(img[1][:,:,:1],(w,h)) # GRAY
            img = np.concatenate((rgb,thm), axis=2)
        elif inc==3: img = img[1][:,:,::-1] # thm->RGB
        elif inc==1: img = img[1][:,:,0] # thm->GRAY
        img = Image.fromarray(img) # np.array->PIL.Image
    x = TSFM_Test(img); return x.reshape(-1,*x.shape)


from torchvision import transforms; avg,std = norm_RGBT(inc)
################################################################################
TSFM_Train = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                transforms.RandomAffine(30, (0.2,0.2)),
                                transforms.RandomResizedCrop(224, (0.8,1.0)),
                                transforms.ToTensor(),
                                transforms.Normalize(avg,std) ])
TSFM_Test = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(avg,std) ])


################################################################################
