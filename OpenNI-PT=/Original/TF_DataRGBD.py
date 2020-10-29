# coding:utf-8
# !/usr/bin/python3

import os, cv2
import numpy as np
#pip install mtcnn # Tensorflow
#from matplotlib import pyplot as plt


# fc.keys(): {"box","confidence","keypoints"}
# fc["keypoints"].keys(): {"left_eye","right_eye","nose","mouth_left","mouth_right"}
################################################################################
def MarkFace_(im, bx, info=None, BC=(0,255,0), pt=[]): # alter im
    x, y, w, h = bx; tc = (0,0,0); pc = (255,0,0) # text/point color
    cv2.rectangle(im, (x,y), (x+w,y+h), color=BC, thickness=1) # face_box
    if type(info)==str: # info: prob, and so on
        cv2.rectangle(im, (x,y), (x+len(info)*7, y+12), BC, thickness=-1)
        cv2.putText(im, info, (x,y+11), fontFace=1, fontScale=0.8, color=tc)
    if type(pt)==dict: pt = pt.values() # iterable like list/tuple
    for p in pt: cv2.circle(im, center=p, radius=1, color=pc, thickness=2)
    return im # im has been altered


def TestMarker(im, faces, sh=0):
    im = im.copy() # backup im
    for k,fc in enumerate(faces):
        bx, prob, pt = fc.values()
        info = ("F%d: %.3f" % (k,prob))
        MarkFace_(im, bx, info, pt=pt) # alter im
        if sh: print("Face_%d:"%k, fc) # show info
    if sh: cv2.imshow("Face",im); cv2.waitKey(sh*1000)
    return im # different from the input


def NewKeyPt(org, face): # for single face
    key = face["keypoints"]; org = np.array(org)
    for k,v in key.items(): key[k] = tuple(np.array(v)-org)
    return key # translate as origin=org


def CropFaces(im, faces, cp=1, sh=0):
    cp_faces = [] # cp = ratio
    for k,fc in enumerate(faces):
        x, y, wd, ht = fc["box"]
        cx, cy = x+wd/2, y+ht/2 # center
        #cx, cy = x+(wd-1)/2, y+(ht-1)/2 # center
        wd, ht = [round(i*cp)+1 for i in (wd,ht)] # new
        x, y = [round(max(i,0)) for i in (cx-wd/2,cy-ht/2)]
        crop = im[y:y+ht, x:x+wd] # crop_face ROI
        cp_faces.append({"ROI":crop, "box":(x,y,wd,ht)})
        if sh: cv2.imshow("%d"%k, crop); cv2.waitKey(sh*1000)
    return cp_faces


################################################################################
def Encode(depth, mod=1): # Ref: MSCam.cpp
    if type(depth)==int: dp = np.array([[depth]])
    else: assert depth.ndim<3; dp = depth.copy()
    if mod==1:
        cy = 22; r = 255//(cy-1)
        R = r*(dp % cy); dp //= cy
        G = r*(dp % cy); dp //= cy
        B = r*(dp % cy); #dp //= cy
        dp = np.array([B,G,R], dtype="uint8")
        return dp.transpose((1,2,0)) # (h,w,c)
    elif mod==2:
        cy = 22; r = 255//(cy-1); h = 180//(cy-1)
        H = h*(dp % cy); dp //= cy
        S = r*(dp % cy); dp //= cy
        V = r*(dp % cy); #dp //= cy
        dp = np.array([H,S,V], dtype="uint8")
        dp = dp.transpose((1,2,0)) # (c,h,w)->(h,w,c)
        return cv2.cvtColor(dp, cv2.COLOR_HSV2BGR)
    elif mod==3:
        c1, c2 = 250, 40; r1, r2 = 255//c1, 255//c2
        H = 66 # any fixed value
        S = r1*(dp % c1); dp //= c1
        V = r2*(dp % c2); #dp //= c2
        dp = np.array([H,S,V], dtype="uint8")
        dp = dp.transpose((1,2,0)) # (c,h,w)->(h,w,c)
        return cv2.cvtColor(dp, cv2.COLOR_HSV2BGR)


def Decode(color, mod=1): # Ref: MSCam.cpp
    if color.ndim==1: color = np.array([[color]])
    assert color.ndim>2 and color.shape[2]>2
    if mod==1:
        cy = 22; r = 255//(cy-1)
        dp = color.astype("uint16")//r # overflow
        B,G,R = [dp[:,:,i] for i in range(3)]
        return ((cy**2)*B + cy*G + R).astype("uint16")
    elif mod==2:
        cy = 22; r = 255//(cy-1); h = 180//(cy-1)
        dp = cv2.cvtColor(color, cv2.COLOR_BGR2HSV).astype(float)
        H = dp[:,:,0]/h; S = dp[:,:,1]/r; V = dp[:,:,2]/r
        return ((cy**2)*V + cy*S + H).astype("uint16")
    elif mod==3:
        c1, c2 = 40, 250; r1, r2 = 255//c1, 255//c2
        dp = cv2.cvtColor(color, cv2.COLOR_BGR2HSV).astype(float)
        H = 66; S = dp[:,:,1]/r1; V = dp[:,:,2]/r2
        return (c1*V + S).astype("uint16")


def DepthFrac(depth, mod=1): # nonzero fraction
    dp = Decode(depth, mod) if depth.ndim>2 else depth
    return np.count_nonzero(dp)/dp.size


def DepthClip(depth, mod=1, tp="8"): # {0,[200,1E4]}
    dp = Decode(depth, mod) if depth.ndim>2 else depth.copy()
    mi, mx = dp.min(), dp.max() # mx=mi|mx>mi(=0|>0)
    if 0==mi<mx: mi = (dp+(dp<1)*mx).min() # 2nd_min
    if type(tp)!=str: return int(mi) # get nonzero min
    
    if mx>0: mx = np.average(dp, weights=(dp>0)) # nonzeros
    # clip origin & large: mi->(mi>0); (x>=M)->0; !=np.clip
    dp -= (dp>0)*(mi-(mi>0)); dp *= dp<min(2**int(tp),mx+512)
    return dp.astype("uint"+tp), int(mi) # backup mi


################################################################################
def ShowPix_(event, x, y, flags, param):
    im, image, dcd, dh = param; h,w = dcd.shape[:2]
    if event==cv2.EVENT_MOUSEMOVE and x<w and y<h:
        im[:] = image; # restore the original image first
        xy = str((x,y))+": "; info = str(dcd[y,x]); s = len(info)*8
        cv2.rectangle(im, (x+1,y-14), (x+1+s,y-2), (0,255,0), -1) # underlay
        cv2.putText(im, info, (x+1,y-4), 1, 0.8, (0,0,0)) # mouse pixel_info
        cv2.putText(im, xy+info, (0,h+dh-2), 1, 0.8, (0,0,0)) # bar_info
        #cv2.putText(im, info, (x,y), x%8, 0.8, (255,255,255)) # test font
        #print("[x=%d,y=%d,font=%d]:"%(x,y,x%8), info) # show font_id


def ViewImage(image, dcd=0, color=None):
    if type(image)==str: # image name
        print(image); image = cv2.imread(image,-1)
    if image.ndim<3: # gray image(alpha)
        dcd = dcd if type(dcd)==np.ndarray else image
        if image.dtype!="uint8": image = Encode(image, 1)
        else: image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2]>3: # with alpha channel
        mi, depth = image[0,0], image[:,:,3].astype("uint16")
        if type(dcd)!=np.ndarray: # no decode_map
            if dcd: # make decode_map
                dcd = depth + (depth>0)*(Decode(mi,dcd)-1)
                dcd = np.concatenate((dcd, depth), axis=1)
            else: dcd = image # just show pixel
        # Note: cv2.imshow() omits alpha channel
        depth = cv2.cvtColor(image[:,:,3], cv2.COLOR_GRAY2BGR)
        image = np.concatenate((image[:,:,:3], depth), axis=1)
    else: # without alpha channel
        if type(dcd)!=np.ndarray: # no decode_map
            dcd = Decode(image,dcd) if dcd else image
        if type(color)==np.ndarray: # superpose
            image = cv2.addWeighted(image,1.5, color,0.5, 0)
            #image = (0.6*image+0.4*color).clip(0,255).astype("uint8")
    
    mx, id, w = dcd.max(), dcd.argmax(), dcd.shape[1] # max
    mn = np.average(dcd, weights=(dcd>0)) # nonzeros mean
    print("max=%d at (%d,%d), mean=%d" % (mx,id%w,id//w,mn) )
    
    dh = 13; bar = image[:dh].copy(); bar[:] = 222
    im = np.concatenate((image, bar), axis=0) # add bar
    cv2.imshow("View",im); param = [im, im.copy(), dcd, dh]
    cv2.setMouseCallback("View", ShowPix_, param) # show pixel
    while(cv2.waitKey(10)!=27): cv2.imshow("View", im)
    cv2.destroyAllWindows()


def TestViewer(im, dcd=0, cp=1.1):
    import mtcnn.mtcnn as mt
    if type(im)==str: im = cv2.imread(im,-1)
    ViewImage(im, dcd); w = im.shape[1]//2
    depth, color = im[:,:w], im[:,w:]
    
    dect = mt.MTCNN(min_face_size=50)
    faces = dect.detect_faces(color); #print(faces)
    mark_c = TestMarker(color, faces)
    mark_d = TestMarker(depth, faces)
    ViewImage(mark_d, Decode(depth,dcd), mark_c)
    
    crop_c = CropFaces(mark_c, faces, cp=cp)
    crop_d = CropFaces(mark_d, faces, cp=cp)
    for c,d in zip(crop_c, crop_d):
        ViewImage(d["ROI"], dcd, c["ROI"])


################################################################################
def IOU(A, B): # rec = (x,y,w,h)
    it_width  = min(A[0]+A[2],B[0]+B[2]) - max(A[0],B[0])
    it_height = min(A[1]+A[3],B[1]+B[3]) - max(A[1],B[1])
    Intersect = max(it_width, 0) * max(it_height, 0)
    Union = (A[2]*A[3]+B[2]*B[3]) - Intersect
    return Intersect/Union


def RDCrop(im, dect, N=10, iu=0.1, mi=30, mod=1):
    if type(im)==str: im = cv2.imread(im,-1)
    H, W = im.shape[0], im.shape[1]//2 # width
    depth, color = im[:,:W], im[:,W:] # split
    faces = dect.detect_faces(color); roi = []
    randi = np.random.randint
    while(len(roi)<N):
        try:
            x = randi(0,W-mi); w = randi(mi,min(W-x,5*mi))
            y = randi(0,H-w);  h = randi(w,min(H-y,1.5*w))
        except: continue
        ts = [IOU((x,y,w,h),i["box"]) for i in faces+roi]
        if (np.array(ts)<iu).all():
            crop = [i[y:y+h, x:x+w] for i in (depth,color)]
            if DepthFrac(crop[0], mod)>0.7:
                crop = np.concatenate(crop, axis=1)
                roi.append({"ROI":crop, "box":(x,y,w,h)})
    return roi


def TestRDCrop(tid, N=10, dst=None):
    import mtcnn.mtcnn as mt
    dect = mt.MTCNN(min_face_size=50)
    if type(tid)==int: tid = str(tid)+"/"
    if type(dst)!=str: dst = tid[:-1]+"_n/"
    if not os.path.isdir(dst): os.mkdir(dst)
    
    for im in os.listdir(tid):
        img = cv2.imread(tid+im); W = img.shape[1]//2
        for i,b in enumerate(RDCrop(tid+im,dect,N)):
            A = MarkFace_(img[:,:W].copy(), b["box"])
            B = MarkFace_(img[:,W:].copy(), b["box"])
            cv2.imshow(im, np.concatenate([A,B],axis=1))
            cv2.imshow("X", b["ROI"]); k = cv2.waitKey()
            if k==32: cv2.imwrite(dst+im[:-4]+"_%02d.png"%i, b["ROI"])
            elif k==27: cv2.destroyAllWindows(); return
        cv2.destroyWindow(im)


################################################################################
def Superpose(im, dst, dect, cmb, cp=1.1):
    if type(im)==str: im = cv2.imread(im,-1)
    wd = im.shape[1]//2 # width/2
    im_d, im_c = im[:,:wd], im[:,wd:]
    if cp > 0: # detect->crop faces
        faces = dect.detect_faces(im_c)
        crops = CropFaces(im_c, faces, cp=cp)
        for k,fc in enumerate(crops):
            x,y,w,h = fc["box"]; ROI_c = fc["ROI"]
            ROI_d = im_d[y:y+h, x:x+w] # depth ROI
            if cmb in [0,-1]: # join: (depth+color)(3)
                im = np.concatenate((ROI_d,ROI_c), axis=-cmb)
            elif cmb<0: # superpose: (depth+color)(3)
                im = cv2.addWeighted(ROI_d, 1, ROI_c, -1/cmb, 0)
            elif cmb==1: # depth(1): clip origin & large
                im, _ = DepthClip(ROI_d) # discard origin
            elif cmb==2: im = Decode(ROI_d) # depth(1): uint16
            elif cmb==3: im = ROI_d # depth(3): keep origin
            elif cmb==4: # combine: depth(3)+depth(a)
                im, _ = DepthClip(ROI_d) # keep & clip origin
                im = np.concatenate((ROI_d, im[:,:,None]), axis=2)
            elif cmb==13: # combine: color(3)+depth(a)
                ROI_d, x = DepthClip(ROI_d); ROI_c[0,0] = Encode(x)
                im = np.concatenate((ROI_c, ROI_d[:,:,None]), axis=2)
            elif cmb==31: # combine: depth(3)+color(a)
                ROI_c = cv2.cvtColor(ROI_c, cv2.COLOR_BGR2GRAY)
                im = np.concatenate((ROI_d, ROI_c[:,:,None]), axis=2)
            #elif cmb==33: # invalid: depth(3)+color(3)
            #    im = np.concatenate((ROI_d,ROI_c), axis=2)
            #print(im.mean(axis=(0,1)),"\t",im.std(axis=(0,1)))
            cv2.imwrite(dst+"_%d.png"%k, im)
    else: # extract whole frames
        if cmb in [0,-1]: # join: (depth+color)(3)
            im = np.concatenate((im_d,im_c), axis=-cmb)
        elif cmb<0: # superpose: (depth+color)(3)
            im = cv2.addWeighted(im_d, 1, im_c, -1/cmb, 0)
        elif cmb==1: # depth(1): clip origin & large
            im, _ = DepthClip(im_d) # discard origin
        elif cmb==2: im = Decode(im_d) # depth(1): uint16
        elif cmb==3: im = im_d # depth(3): keep origin
        elif cmb==4: # combine: depth(3)+depth(a)
            im, _ = DepthClip(im_d) # keep & clip origin
            im = np.concatenate((im_d, im[:,:,None]), axis=2)
        elif cmb==13: # combine: color(3)+depth(a)
            im_d, x = DepthClip(im_d); im_c[0,0] = Encode(x)
            im = np.concatenate((im_c, im_d[:,:,None]), axis=2)
        elif cmb==31: # combine: depth(3)+color(a)
            im_c = cv2.cvtColor(im_c, cv2.COLOR_BGR2GRAY)
            im = np.concatenate((im_d, im_c[:,:,None]), axis=2)
        #elif cmb==33: # invalid: depth(3)+color(3)
        #    im = np.concatenate((im_d,im_c), axis=2)
        cv2.imwrite(dst+".png", im)


def GenData(tid, cmb=1, cp=1.1):
    import mtcnn.mtcnn as mt
    dect = mt.MTCNN(min_face_size=50) if cp>0 else 0
    
    if type(tid)==int: tid = str(tid)
    elif tid[-4:]==".mkv": tid = tid[:-4]
    dir = tid + "_fc%d/"%cmb # dst_dir
    if not os.path.isdir(dir): os.mkdir(dir)
    
    if os.path.isfile(tid+".mkv"):
        vid = cv2.VideoCapture(tid+".mkv")
        while True:
            i = int(vid.get(cv2.CAP_PROP_POS_FRAMES))
            rt, im = vid.read(); i = tid + "_%04d"%i
            if not rt: break # EOF
            Superpose(im, dir+i, dect, cmb, cp)
        vid.release()
    elif os.path.isdir(tid):
        for im in os.listdir(tid):
            Superpose(tid+"/"+im, dir+im[:-4], dect, cmb, cp)


################################################################################
if __name__ == "__main__":
    from glob import glob
    TestViewer("1557906718_09.png", dcd=1)
    #ViewImage("1557906729_13_0.png", dcd=1)
    #for i in glob("*.png"): ViewImage(i, dcd=1)
    
    #TestRDCrop("origin_png/1557906613/", N=10)
    #GenData(1557478895, cmb=2, cp=1.1)
    #for i in glob("*.mkv"):
    #    if i[:-4].isdigit(): GenData(i)
    #for i in os.listdir():
    #    if i.isdigit(): GenData(i, cmb=1)


################################################################################
