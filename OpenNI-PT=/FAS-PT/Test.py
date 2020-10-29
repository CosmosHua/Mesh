# coding:utf-8
# !/usr/bin/python3


from Train import *
################################################################################
def InferFace(im, net): # infer single unlabled image
    with torch.no_grad(): # softmax->prob: (B=1,cls)
        t0 = time(); y = net(LoadFace(im).to(device))
        y = torch.softmax(y, dim=1); t1 = (time()-t0)*1000
    v,k = [i.item() for i in torch.max(y,dim=1)] # ->number
    if type(im)!=str: return cls[k],v # swap
    print(im, "-> %s(%.5f), T=%.1fms."%(cls[k],v,t1) )


def TestModel(im, param): # infer without labels
    net = model(inc,cls).to(device); net.eval()
    if type(param)!=str: net.load_state_dict(param) # state_dict
    elif os.path.isfile(param): LoadParam(param, net) # file
    
    if os.path.isfile(im): InferFace(im, net)
    elif os.path.isdir(im): # loop dir
        for dir, subs, files in os.walk(im):
            for i in files: InferFace(dir+"/"+i, net)


################################################################################
def MarkInfer(im, net, cp=1.1): # BGR
    BC = {"R":(0,255,0), "P":(0,0,255), "N":(222,)*3, "X":(255,0,255),
          "O":(99,222,99), "F":(99,222,99), "S":(99,222,99) }
    wd = im.shape[1]//2; im_d, im_c = im[:,:wd], im[:,wd:]
    t0 = time(); faces = Detect(im_c); t1 = (time()-t0)*1000
    crops = CropFaces(im_c, faces, cp=cp); dp = im_d.copy()
    for i,fc in enumerate(crops):
        x,y,w,h = fc["box"]; ROI = dp[y:y+h, x:x+w]
        if min(ROI.shape[:2])<30: info = "Small"
        elif DepthFrac(ROI)<0.5: info = "Out/Block"
        elif not 350<DepthClip(ROI,tp=0)<2000: info = "Far/Near"
        else: # classify using CNN
            ROI = np.concatenate((ROI,fc["ROI"]), axis=1)
            k,v = InferFace(ROI, net) # softmax inference
            info = ("X" if v<0.75 else "") + "%s: %.3f"%(k,v)
        MarkFace_(im_c, faces[i]["box"], info, BC[info[0]])
        MarkFace_(im_d, faces[i]["box"], info, BC[info[0]])
        faces[i]["info"] = info # add info/attribute
    t2 = time()-t0; TM = "FPS=%.1f (%dms|%dms)"%(1/t2,t2*1000,t1)
    cv2.putText(im_c, TM, (0,im.shape[0]-4), 5, 1.0, (0,255,99))
    return im_c, im_d, faces # alter im; (fontFace=5, fontScale=1.0)


################################################################################
def Infer2Video(tid, param):
    if type(tid)==int: tid = str(tid)
    elif tid[-4:]==".mkv": tid = tid[:-4]
    
    net = model(inc,cls).to(device); net.eval(); t0 = time()
    if type(param)!=str: net.load_state_dict(param) # state_dict
    elif os.path.isfile(param): LoadParam(param, net) # file
    dst = os.path.basename(tid)+"_"+param[:param.find(".pth")]+".mkv"
    
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
            im, _, faces = MarkInfer(im, net)
            out.write(im) # im->frame
        out.release(); vid.release()
    elif os.path.isdir(tid): # infer dir
        images = sorted(os.listdir(tid))
        im = cv2.imread(tid+"/"+images[0], -1)
        ht, wd = im.shape[:2]; wd //= 2; fps = 12
        codec = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(dst, codec, fps, (wd,ht))
        for im in images: # alter im
            im = cv2.imread(tid+"/"+im, -1)
            im, _, faces = MarkInfer(im, net)
            out.write(im) # im->frame
        out.release()
    print("=>%s: %.3fs.\n" % (dst,time()-t0))


# pip3 install onnx-simplifier
# Ref: https://github.com/daquexian/onnx-simplifier
SOnnx = lambda x: os.system("python3 -m onnxsim %s %s"%(x,x)) # simplify
O2ncnn = lambda x,o: os.system('"%s" %s.onnx %s.param %s.bin'%(x,*(o,)*3))
N2nmem = lambda x,o: os.system('"%s" %s.param %s.bin %s_id.h %s_mem.h'%(x,*(o,)*4))
################################################################################
def pth2onnx(C="../train", ncnn="."):
    nmem = glob(ncnn + "/ncnn2mem*")
    ncnn = glob(ncnn + "/onnx2ncnn*")
    x = torch.randn(7, inc, 224, 224)
    if type(C) != int: C = len(i_cls(C))
    net = model(inc,C).to(device); net.eval()
    for i in glob("*.pth*"):
        out = i[:i.find(".pth")]+".onnx"
        if not os.path.isfile(out):
            LoadParam(i, net) # load pth->onnx
            torch.onnx.export(net, x, out, verbose=True)
        os.system("python3 -m onnxsim %s %s"%(out,out)) # simplify
        if ncnn: O2ncnn(ncnn[0], out[:-5]); print(out,"->ncnn!")
        if nmem: N2nmem(nmem[0], out[:-5]); print(out,"->nmem!")


from CamLib.Cam import *
################################################################################
def CamDemo(param, sh=0):
    net = model(inc,cls).to(device); net.eval() # eval_mode
    if type(param)!=str: net.load_state_dict(param) # state_dict
    elif os.path.isfile(param): LoadParam(param, net) # file
    
    A = Cam(); A.Init(CamD); A.Init(CamRGB) # Init Cam
    while True:
        im = A.GetRGBD() # get D+RGB frame
        im, dp, faces = MarkInfer(im, net) # infer->mark
        if sh==1: im = cv2.addWeighted(im,1, dp,0.6, 0)
        elif sh==2: im = np.concatenate([dp,im], axis=1)
        cv2.imshow("RGBD", im); k = cv2.waitKey(10)
        if k==27: cv2.destroyAllWindows(); break
    A.Close(CamD); A.Close(CamRGB) # Close Cam


################################################################################
if __name__ == "__main__":
    from glob import glob; para = glob("*.pth*")
    x = input("%s\nSelect [1-N]: "%para)
    x = int(x)-1 if x.isdigit() else 0
    
    cls = i_cls("../train")
    #TestModel("../val", para[x]) # for non-label set
    #Infer2Video("../1558595220", para[x]) # ->mkv
    CamDemo(para[x], 2) # Demo using Cam


################################################################################
