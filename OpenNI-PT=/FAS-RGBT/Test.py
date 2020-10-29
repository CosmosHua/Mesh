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


# pip3 install mtcnn-pytorch
from mtcnn import detect_faces
RIT = lambda s: tuple(int(round(i)) for i in s)
################################################################################
def MarkFace_(im, bx, info=None, BC=(0,255,0), pt=[]): # alter im
    x, y, w, h = bx; tc = (0,0,0); pc = (255,0,0) # text/point color
    cv2.rectangle(im, (x,y), (x+w,y+h), color=BC, thickness=1) # face_box
    if type(info)==str: # info: label, prob, and so on
        cv2.rectangle(im, (x,y), (x+len(info)*7, y+12), BC, thickness=-1)
        cv2.putText(im, info, (x,y+11), fontFace=1, fontScale=0.8, color=tc)
    if type(pt)==dict: pt = pt.values() # iterable like list/tuple
    for p in pt: cv2.circle(im, center=p, radius=1, color=pc, thickness=2)
    return im # im has been altered


def MarkInfer(im, net): # BGR: im=(co,th)
    assert type(im) in (tuple,list); t0 = time()
    co, th = [i.copy() for i in im]; xy = (184,96)
    bbx, pts = detect_faces(co, 40); t1 = (time()-t0)*1000
    BC = {"R":(0,255,0), "F":(0,0,255), "O":(222,)*3}
    for i,bx in enumerate(bbx): # loop faces
        tx = RIT(bx[:4]-np.array(xy*2)); bx = RIT(bx)
        cc = im[0][bx[1]:bx[3], bx[0]:bx[2]] # color_face
        tt = im[1][tx[1]:tx[3], tx[0]:tx[2]] # therm_face
        ic = np.array(cc.shape[:2]+tt.shape[:2])
        if ic[2]*ic[3]<ic[0]*ic[1]*0.5: info = "Out_FOV"
        else: info = "%s: %.3f"%(*InferFace((cc,tt),net))
        MarkFace_(co, bx, info, BC[info[0]]) # color_face
    t2 = time()-t0; TM = "FPS=%.1f (%dms|%dms)"%(1/t2,t2*1000,t1)
    cv2.putText(co, TM, (0,co.shape[0]-4), 5, 1.0, (0,255,99))
    return co, th # alter im; (fontFace=5, fontScale=1.0)


cid = 2,1 # cam_id: (rgb,th1)
################################################################################
def CamDemo(param, sh=0):
    net = model(inc,cls).to(device); net.eval() # eval_mode
    if type(param)!=str: net.load_state_dict(param) # state_dict
    elif os.path.isfile(param): LoadParam(param, net) # file
    
    RGB, th1 = [cv2.VideoCapture(i) for i in cid]
    while True:
        rt, co = RGB.read(); co = cv2.flip(co,1)
        rt, th = th1.read(); th = cv2.flip(th,1)
        co, th = MarkInfer((co,th), net) # infer->mark
        cv2.imshow("RGB",co); cv2.imshow("Th1",th)
        if cv2.waitKey(10)==27: break
    cv2.destroyAllWindows(); th1.release(); RGB.release()


################################################################################
if __name__ == "__main__":
    from glob import glob; para = glob("*.pth*")
    x = input("%s\nSelect [1-N]: "%para)
    x = int(x)-1 if x.isdigit() else 0
    
    cls = i_cls("../train")
    #TestModel("../val", para[x]) # for non-label set
    CamDemo(para[x], 2) # Demo using Cam


################################################################################
