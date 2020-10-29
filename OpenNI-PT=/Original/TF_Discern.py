# coding:utf-8
# !/usr/bin/python3

import os, cv2
import numpy as np
from PIL import Image
from TF_DataRGBD import *
#pip install mtcnn # Tensorflow


from torchvision import transforms
cmb = 1; LK = 5; MX=3E3; jt=0.4; SQ = {}
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
        elif cmb==1: # depth(1): clip origin & large
            im, _ = DepthClip(A[:,:,::-1], mod) # BGR
        # 2: converge sluggish, acc=0.87-0.88; acc_v=0.77-0.81
        elif cmb==2: # depth(1): uint16, keep origin
            im = Decode(A[:,:,::-1], mod)/1E3 # BGR
        # 3: converge slowly, acc=0.95-0.96; acc_v=0.94-0.95
        elif cmb==3: im = A # depth(3): keep origin
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


mean, std = [0.485,0.458,0.407], [0.229,0.224,0.225] # color(RGB)
mean2, std2 = [0.330,0.353,0.047], [0.25,0.25,0.047] # depth(RGB)
if cmb==1:   mean, std = [0.15], [0.12] # cliped depth(1)
elif cmb==2: mean, std = [0.5], [0.25] # original depth(1)
elif cmb==3: mean, std = mean2, std2 # original depth(3)
elif cmb==4: mean, std = mean2+[0.15], std2+[0.12] # depth(3)+depth(a)
elif cmb==13: mean, std = mean+[0.15], std+[0.12] # color(3)+depth(a)
elif cmb==31: mean, std = mean2+[0.45], std2+[0.226] # depth(3)+color(a)
elif cmb==33: mean, std = mean2+mean, std2+std # depth(3)+color(3)
################################################################################
TSFM_Train = transforms.Compose([Jitter(color=(jt,)*4, mx=MX, mod=1, sp=1),
                                Superpose(cmb=cmb, mod=1, sp=1),
                                transforms.RandomHorizontalFlip(0.5),
                                transforms.RandomAffine(45, (0.2,0.2)),
                                transforms.RandomResizedCrop(224, (0.5,1.0)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean,std)] )
TSFM_Test = transforms.Compose([Superpose(cmb=cmb, mod=1, sp=1),
                                transforms.Resize(256), #[224,224]
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean,std)] )
#TSFM_Test = TSFM_Train.transforms[-6:] # make a copy
#TSFM_Test[-5:-2] = [transforms.Resize(256), transforms.CenterCrop(224)] # alter
#TSFM_Test = transforms.Compose(TSFM_Test) # compose


from torch.utils.data import DataLoader
from torchvision import datasets #, models
# get (index: class) pairs from DIR: sort(essential)->list->dict(map)
i_cls = lambda X: dict([(k,v) for k,v in enumerate(sorted(os.listdir(X)))])
################################################################################
def LoadData(DIR, train=True, bs=64):
    TSFM = TSFM_Train if train else TSFM_Test
    Data = datasets.ImageFolder(DIR, transform=TSFM); j = round(bs**0.25)
    Loader = DataLoader(Data, batch_size=bs, shuffle=train, num_workers=j)
    cls = Data.class_to_idx; cls = dict([(v,k) for k,v in cls.items()]) # swap
    assert i_cls(DIR)==cls, "class_ids discord!"; return Loader


def LoadImg(im): # load single PIL.Image->(1,C,H,W)
    if type(im)==str: im = Image.open(im) # require: RGB
    elif type(im)==np.ndarray: im = Image.fromarray(im[:,:,::-1])
    assert isinstance(im, Image.Image) # type(im)!=Image.Image
    x = TSFM_Test(im); x = x.reshape(-1,*x.shape); return x


import torch
import torch.nn as nn # (negative_slope, inplace)
ReLU = nn.LeakyReLU(1/LK, True) if LK else nn.ReLU(True)
################################################################################
class Fire(nn.Module): # inc->mid->(expand1+expand3)
    def __init__(self, inc, mid, expand1, expand3):
        '''
        self.squeeze = nn.Sequential(nn.Conv2d(inc, oup, 1), ReLU)
        self.expand1 = nn.Sequential(nn.Conv2d(oup, expand1, 1), ReLU)
        self.expand3 = nn.Sequential(nn.Conv2d(oup, expand3, 3, padding=1), ReLU)
        '''
        super(Fire, self).__init__() # BatchNorm accelerate convergence
        self.squeeze = nn.Sequential(nn.Conv2d(inc, mid, kernel_size=1),
                                     nn.BatchNorm2d(mid), ReLU)
        self.branch1 = nn.Sequential(nn.Conv2d(mid, expand1, kernel_size=1),
                                     nn.BatchNorm2d(expand1), ReLU)
        self.branch3 = nn.Sequential(nn.Conv2d(mid, expand3, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(expand3), ReLU)


    def forward(self, x): # spatial dimension unchanged
        '''
        x = self.squeeze(x)
        return torch.cat([self.expand1(x), self.expand3(x)], 1)
        '''
        x = self.squeeze(x) # x=(N,C,H,W), squeeze: C->mid
        return torch.cat([self.branch1(x), self.branch3(x)], dim=1)


inc = 4 if (cmb in [4,13,31]) else (1 if cmb in [1,2] else 3)
################################################################################
class SqueezeNet(nn.Module): # ver=1.1
    def __init__(self, inc=3, cls=1000, hw=224):
        super(SqueezeNet, self).__init__()
        cls = cls if type(cls)==int else len(cls); assert hw%32==0
        
        # HW = [1+(HW+2*pad-dilation*(kernel-1)-1)/stride]
        conv_init = nn.Conv2d(inc, 64, kernel_size=3, stride=2)
        self.features = nn.Sequential( # 224*224*inc->13*13*512
            conv_init, nn.BatchNorm2d(64), ReLU,                   # ->111*111*64
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), # ->55*55*64
            Fire(64, 16, 64, 64), Fire(64*2, 16, 64, 64),          # ->55*55*(64+64)
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), # ->27*27*128
            Fire(64*2, 32, 128, 128), Fire(128*2, 32, 128, 128),   # ->27*27*(128+128)
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), # ->13*13*256
            Fire(128*2, 48, 192, 192), Fire(192*2, 48, 192, 192),  # ->13*13*(192+192)
            Fire(192*2, 64, 256, 256), Fire(256*2, 64, 256, 256) ) # ->13*13*(256+256)
        
        conv_last = nn.Conv2d(256*2, cls, kernel_size=1) # ->13*13*cls
        self.classifier = nn.Sequential(nn.Dropout(p=0.5), conv_last,
            nn.BatchNorm2d(cls), ReLU, nn.AdaptiveAvgPool2d(1) )
        
        for m in self.modules(): # initialization
            if isinstance(m, nn.Conv2d): # initialize convs
                if m is not conv_last: nn.init.kaiming_uniform_(m.weight)
                else: nn.init.xavier_uniform_(m.weight) # xavier_normal_
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear): # initialize linear
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d): # initialize norm
                m.weight.data.fill_(1); m.bias.data.zero_()


    def forward(self, x): # x=(N,C,H->1,W->1)
        x = self.classifier(self.features(x))
        return x.view(x.size(0), -1)


#from torch.autograd import Variable
#from torch.nn import functional as F
################################################################################
class FocalLoss(nn.Module):
    """Loss(X,k) = -weight(k) * (1-softmax(X,k))^gamma * log(softmax(X,k)).
    Default: The losses are averaged across observations for each minibatch.
    # Ref: https://arxiv.org/abs/1708.02002
    Args:
        weight(1D tensor|class_num): the weighting factor for class imbalance.
        gamma(float>0): reduces the relative loss for well-classiﬁed examples(p>0.5),
            in order to put more focus on hard, misclassiﬁed examples. Default: 2.
        reduction(string): the reduction to apply to the output: 'none'|'mean'|'sum'.
            'none': no reduction will be applied; 'sum': the output will be summed;
            'mean': the mean value of the outputs. Default: 'mean'."""
    def __init__(self, weight, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma; self.rd = reduction
        assert isinstance(weight, (int, torch.Tensor))
        self.wt = torch.ones(weight) if type(weight)==int else weight


    def forward(self, input, target):
        """Args: input (2D Tensor): (N=batch_size, C=class_num), probs.
                 target (1D Tensor): (N=batch_size), labels.
        Returns: the focal loss of input/inference and taget/label."""
        wt = self.wt.to(input.device)
        p = torch.softmax(input, dim=1)
        loss = -wt * (1-p)**self.gamma * p.log()
        
        # Method_1:
        mask = torch.zeros_like(loss)
        mask.scatter_(1, target.view(-1,1), 1.0) # dim,index,src
        loss = (loss*mask).sum(dim=1)
        # Method_2:
        #loss = [loss[i,k] for i,k in enumerate(target)]
        #loss = torch.tensor(loss, device=wt.device)
        
        if self.rd=="mean": loss = loss.mean()
        elif self.rd=="sum": loss = loss.sum()
        return loss # also for "none"


from time import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
################################################################################
def SaveParam(dst, net, ep, opt=None): # save ckpt
    ckpt = {"model": net.state_dict(), "epoch": ep}
    #ckpt = {"model": net.cpu().state_dict(), "epoch": ep}
    if dst[-4:]==".tar" and opt: ckpt["optim"] = opt.state_dict()
    dst = dst.replace(".pth", "_%d.pth"%ep); torch.save(ckpt, dst)
    print("=>Save Ckpt: %s\n"%dst); return ckpt # *.pth/.pth.tar


def SavePolicy(dst, net, ep, acc, opt=None, Num=6):
    mi = min(SQ.values()) if len(SQ)>0 else 0.9
    if acc<mi or ep in SQ: return # acc>=min>=0.9
    
    out = "=>[ep=%03d]:%.5f " % (ep, acc)
    if len(SQ)==Num: # replace first lowest
        k = [k for k in SQ if SQ[k]==mi][0] # first
        out += "=>[%03d]:%.5f" % (k, mi); SQ.pop(k)
        os.remove(dst.replace(".pth", "_%d.pth"%k))
    with open("sq.log", "a+") as f: f.write(out+"\n")
    SQ[ep] = acc; return SaveParam(dst, net, ep, opt)


def LoadParam(dst, net, opt=None): # load ckpt
    dt = time(); assert os.path.isfile(dst) # check file
    ckpt = torch.load(dst, map_location=device) # *.pth/.pth.tar
    if dst[-4:]==".tar" and opt: opt.load_state_dict(ckpt["optim"])
    net.load_state_dict(ckpt["model"]); dt = time()-dt # load param
    print("=>Load Ckpt: %s\t%.3fs."%(dst,dt)); return ckpt["epoch"]


from torch import optim
# Ref: https://arxiv.org/abs/1812.01187
################################################################################
def WarmUp(optmz, i, N):
    optmz.param_groups[0]["lr"] = optmz.defaults["lr"]*(i/N)


def TrainModel(DIR, dst, val, Epoch=100, bs=64, wp=5):
    Loader = LoadData(DIR, train=True, bs=bs) # get cls{id: name}
    net = SqueezeNet(inc,cls).to(device); net.train() # train_mode
    
    optmz = optim.SGD(net.parameters(), lr=0.1*(bs/256), momentum=0.9,
            weight_decay=1E-4, dampening=0, nesterov=True) # better>adam
    
    # (factor,patience,cooldown): (0.8,8,3), (0.5,8,3), (0.5,3,3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optmz, mode='min',
                factor=0.8, patience=8, verbose=True, cooldown=3) # loss
    
    sk = LoadParam(dst, net, optmz) if os.path.isfile(dst) else 0
    if sk>0: dst = dst.replace("_%d"%sk, "") # rename for saving
    
    N = min(len(Loader)//2, max(100,Epoch)); K = len(cls)
    try: # in case: out of memeory
        for ep in range(sk+1, Epoch + sk+1):
            loss, acc = 0, 0
            for i, batch in enumerate(Loader, 1):
                x, y = [b.to(device) for b in batch] # inputs
                #x,y = [Variable(b).to(device) for b in batch]
                ym = net(x) # forward->(batch,cls)
                
                wt = torch.Tensor([sum(y==i) for i in range(K)])
                wt = (1-wt/sum(wt)).to(device) # batch_weight
                #LF = nn.CrossEntropyLoss(wt) # balanced CEL
                LF = FocalLoss(wt); L = LF(ym,y) # Focal_Loss

                # zero_grad() must before backward()
                optmz.zero_grad() # clear gradients
                L.backward() # backward->get gradients
                optmz.step() # update parameters
                
                loss += L.item() # tensor->number: float(L)
                v, k = torch.max(ym, dim=1) # (value,index)
                acc += (y==k).double().mean().item() # mean->number
                #if i%N==0: print("[Ep:%03d, Ba:%03d] Loss: %.3f, Acc: %.3f"
                #    % (ep,i,loss/i,acc/i)) # N-batch accumulated statistics
            
            loss /= i; acc /= i # average per-batch
            print("[Ep:%03d] Loss: %.3f, Acc: %.3f" % (ep,loss,acc))
            loss_v, acc_v = EvalModel(val, net.state_dict(), bs)
            
            SavePolicy(dst, net, ep, acc*acc_v, optmz) # save param
            if acc>0.99 and acc_v>0.98: break # return # early stop
            
            if scheduler: scheduler.step(loss) # min->loss, max->acc
            
    finally: # in case: max_epoch, early_stop, any training error
        return SavePolicy(dst, net, ep, acc_v, optmz) # save


def EvalModel(DIR, param, bs=64): # with labels
    Loader = LoadData(DIR, train=False, bs=bs) # update cls
    net = SqueezeNet(inc,cls).to(device); net.eval() # eval_mode
    
    wt = [len(os.listdir(DIR+"/"+i)) for i in cls.values()]
    wt = (1-torch.Tensor(wt)/sum(wt)).to(device) # global_weight
    LF = nn.CrossEntropyLoss(weight=wt) # mean balanced CEL
    
    if type(param)!=str: net.load_state_dict(param) # state_dict
    elif os.path.isfile(param): LoadParam(param, net) # file
    
    loss, acc = 0, 0; t0 = time()
    N = len(datasets.ImageFolder(DIR).imgs)
    with torch.no_grad(): # inference
        if bs<25: print(cls) # show {index: class}
        for i, (x,y) in enumerate(Loader, 1):
            x,y = [b.to(device) for b in (x,y)] # inputs
            #x,y = [Variable(b).to(device) for b in (x,y)]
            ym = net(x) # forward->(batch,cls)
            loss += LF(ym, y).item() # Tensor->number
            v, k = torch.max(ym, dim=1) # (value,index)
            acc += (y==k).double().sum().item() # sum->number
            if bs<25: print("Guess:",k.cpu(), "\nLabel:",y.cpu())
    loss /= i; acc /= N; t0 = (time()-t0)/i # per-batch
    print("[%s] Loss: %.3f, Acc: %.3f, Time/Ba: %.3fs." % (DIR,loss,acc,t0))
    return loss, acc


################################################################################
def InferImage(im, net): # infer single unlabled image
    t0 = time(); sh = (type(im)==str) # initial time
    with torch.no_grad(): y = net(LoadImg(im).to(device))
    y = torch.softmax(y, dim=1) # softmax->prob: (B=1,cls)
    v, k = [i.item() for i in torch.max(y,dim=1)] # tensor->number
    if sh: print(im, "-> %s(%.5f), t=%.3fs."%(cls[k],v,time()-t0))
    return v, k


def TestModel(DIR, param): # infer without labels
    net = SqueezeNet(inc,cls).to(device); net.eval()
    if type(param)!=str: net.load_state_dict(param) # state_dict
    elif os.path.isfile(param): LoadParam(param, net) # file
    
    if os.path.isfile(DIR): InferImage(DIR, net)
    elif os.path.isdir(DIR): # loop DIR
        for dir, subs, files in os.walk(DIR):
            for i in files: InferImage(dir+"/"+i, net)


################################################################################
def MarkInfer(im, net, dect, cp=1.1): # BGR
    wd = im.shape[1]//2; dt = time()
    im_d, im_c = im[:,:wd], im[:,wd:]
    faces = dect.detect_faces(im_c)
    crops = CropFaces(im_c, faces, cp=cp)
    for i,fc in enumerate(crops):
        x,y,w,h = fc["box"]; ROI = im_d[y:y+h, x:x+w]
        if DepthClip(ROI,tp=0)>3500: info = "Far"
        elif min(ROI.shape[:2])<30: info = "Small"
        elif DepthFrac(ROI)<0.6: info = "Out/Block"
        else: # classify using CNN
            ROI = np.concatenate((ROI,fc["ROI"]), axis=1)
            v, k = InferImage(ROI, net) # inference
            info = "%s: %.3f" % (cls[k], v) # softmax
            if v<0.75: info = "X" + info # uncertain
        MarkFace_(im_c, faces[i]["box"], info) # show
        faces[i]["info"] = info # add info/attribute
    dt = time()-dt; dt = "FPS=%d (%dms)"%(round(1/dt),dt*1000)
    cv2.putText(im_c, dt, (0,im.shape[0]-4), 5, 1.0, (0,255,99))
    return im_c, faces # alter im; (fontFace=5, fontScale=1.0)


def Infer2Video(tid, param):
    if type(tid)==int: tid = str(tid)
    elif tid[-4:]==".mkv": tid = tid[:-4]
    dst = os.path.basename(tid)+"_"+param[:param.find(".pth")]+".mkv"
    
    import mtcnn.mtcnn as mt; t0 = time()
    dect = mt.MTCNN(min_face_size=50) # test_mode
    net = SqueezeNet(inc,cls).to(device); net.eval()
    if type(param)!=str: net.load_state_dict(param) # state_dict
    elif os.path.isfile(param): LoadParam(param, net) # file
    
    if os.path.isfile(tid+".mkv"): # infer video
        vid = cv2.VideoCapture(tid+".mkv")
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'XVID')
        #codec = int(vid.get(cv2.CAP_PROP_FOURCC))
        ht = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        wd = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))//2
        out = cv2.VideoWriter(dst, codec, fps, (wd,ht))
        while True: # alter im
            rt, im = vid.read()
            if not rt: break; # EOF
            im, faces = MarkInfer(im, net, dect)
            out.write(im) # im->frame
        out.release(); vid.release()
    elif os.path.isdir(tid): # infer DIR
        images = sorted(os.listdir(tid))
        im = cv2.imread(tid+"/"+images[0], -1)
        ht, wd = im.shape[:2]; wd //= 2; fps = 12
        codec = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(dst, codec, fps, (wd,ht))
        for im in images: # alter im
            im = cv2.imread(tid+"/"+im, -1)
            im, faces = MarkInfer(im, net, dect)
            out.write(im) # im->frame
        out.release()
    print("=>%s: %.3fs.\n" % (dst,time()-t0))


################################################################################
if __name__ == "__main__":
    train = "../train"; val = "../val"; cls = i_cls(train)
    pre = "RGBD_c%dk%d" % (cmb, LK); tp = ".pth.tar"
    ep = 500; para = pre + "_%d"%ep + tp
    
    TrainModel(train, pre+tp, val, ep) # from init
    #TrainModel(train, para, val, ep) # from ckpt
    #EvalModel(train, para) # on training set
    #EvalModel(val, para, 20) # on validation
    #TestModel(val, para) # for non-label set
    #Infer2Video("../1558595220", para) # infer->mkv


################################################################################
