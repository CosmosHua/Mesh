#!/usr/bin/env python3

import os
os.environ['GLOG_minloglevel'] = '2'
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)
import cv2
import numpy as np
from skimage import transform as tf
from PIL import Image as im
from io import BytesIO
import scipy,scipy.spatial
import sklearn.metrics.pairwise as pw
import warnings
warnings.filterwarnings("ignore")
import copy


class Faceapp:
    def __init__(self, caffe_model_path):
        self.faceCascade = cv2.CascadeClassifier( caffe_model_path + '/haarcascade_frontalface_default.xml')
        self.PNet = caffe.Net(caffe_model_path + "/det1.prototxt", caffe_model_path + "/det1.caffemodel", caffe.TEST)
        self.RNet = caffe.Net(caffe_model_path + "/det2.prototxt", caffe_model_path + "/det2.caffemodel", caffe.TEST)
        self.ONet = caffe.Net(caffe_model_path + "/det3.prototxt", caffe_model_path + "/det3.caffemodel", caffe.TEST)
        self.Net = caffe.Net(caffe_model_path + "/VGG_FACE_deploy.prototxt", caffe_model_path + "/VGG_FACE.caffemodel", caffe.TEST)
    
    def bbreg(self, boundingbox, reg):
        reg = reg.T

        # calibrate bouding boxes
        if reg.shape[1] == 1:
            pass # reshape of reg
        w = boundingbox[:,2] - boundingbox[:,0] + 1
        h = boundingbox[:,3] - boundingbox[:,1] + 1

        bb0 = boundingbox[:,0] + reg[:,0]*w
        bb1 = boundingbox[:,1] + reg[:,1]*h
        bb2 = boundingbox[:,2] + reg[:,2]*w
        bb3 = boundingbox[:,3] + reg[:,3]*h

        boundingbox[:,0:4] = np.array([bb0, bb1, bb2, bb3]).T
        return boundingbox


    def pad(self, boxesA, w, h):
        boxes = boxesA.copy() # shit, value parameter!!!

        tmph = boxes[:,3] - boxes[:,1] + 1
        tmpw = boxes[:,2] - boxes[:,0] + 1
        numbox = boxes.shape[0]

        dx = np.ones(numbox)
        dy = np.ones(numbox)
        edx = tmpw
        edy = tmph

        x = boxes[:,0:1][:,0]
        y = boxes[:,1:2][:,0]
        ex = boxes[:,2:3][:,0]
        ey = boxes[:,3:4][:,0]


        tmp = np.where(ex > w)[0]
        if tmp.shape[0] != 0:
            edx[tmp] = -ex[tmp] + w-1 + tmpw[tmp]
            ex[tmp] = w-1

        tmp = np.where(ey > h)[0]
        if tmp.shape[0] != 0:
            edy[tmp] = -ey[tmp] + h-1 + tmph[tmp]
            ey[tmp] = h-1

        tmp = np.where(x < 1)[0]
        if tmp.shape[0] != 0:
            dx[tmp] = 2 - x[tmp]
            x[tmp] = np.ones_like(x[tmp])

        tmp = np.where(y < 1)[0]
        if tmp.shape[0] != 0:
            dy[tmp] = 2 - y[tmp]
            y[tmp] = np.ones_like(y[tmp])

        # for python index from 0, while matlab from 1
        dy = np.maximum(0, dy-1)
        dx = np.maximum(0, dx-1)
        y = np.maximum(0, y-1)
        x = np.maximum(0, x-1)
        edy = np.maximum(0, edy-1)
        edx = np.maximum(0, edx-1)
        ey = np.maximum(0, ey-1)
        ex = np.maximum(0, ex-1)

        return [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]

    def rerec(self, bboxA):
        # convert bboxA to square
        w = bboxA[:,2] - bboxA[:,0]
        h = bboxA[:,3] - bboxA[:,1]
        l = np.maximum(w,h).T

        bboxA[:,0] = bboxA[:,0] + w*0.5 - l*0.5
        bboxA[:,1] = bboxA[:,1] + h*0.5 - l*0.5
        bboxA[:,2:4] = bboxA[:,0:2] + np.repeat([l], 2, axis = 0).T
        return bboxA


    def nms(self, boxes, threshold, type):
        """nms
        :boxes: [:,0:5]
        :threshold: 0.5 like
        :type: 'Min' or others
        :returns: TODO
        """
        if boxes.shape[0] == 0:
            return np.array([])
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        s = boxes[:,4]
        area = np.multiply(x2-x1+1, y2-y1+1)
        I = np.array(s.argsort()) # read s using I

        pick = [];
        while len(I) > 0:
            xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])
            yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
            xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
            yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            if type == 'Min':
                o = inter / np.minimum(area[I[-1]], area[I[0:-1]])
            else:
                o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
            pick.append(I[-1])
            I = I[np.where( o <= threshold)[0]]
        return pick


    def generateBoundingBox(self, map, reg, scale, t):
        stride = 2
        cellsize = 12
        map = map.T
        dx1 = reg[0,:,:].T
        dy1 = reg[1,:,:].T
        dx2 = reg[2,:,:].T
        dy2 = reg[3,:,:].T
        (x, y) = np.where(map > t)

        yy = y
        xx = x


        score = map[x,y]
        reg = np.array([dx1[x,y], dy1[x,y], dx2[x,y], dy2[x,y]])

        if reg.shape[0] == 0:
            pass
        boundingbox = np.array([yy, xx]).T

        bb1 = np.fix((stride * (boundingbox) + 1) / scale).T # matlab index from 1, so with "boundingbox-1"
        bb2 = np.fix((stride * (boundingbox) + cellsize - 1 + 1) / scale).T # while python don't have to
        score = np.array([score])

        boundingbox_out = np.concatenate((bb1, bb2, score, reg), axis=0)


        return boundingbox_out.T


    def __detect_face(self, img, minsize, PNet, RNet, ONet, threshold, fastresize, factor):

        img2 = img.copy()
        factor_count = 0
        total_boxes = np.zeros((0,9), np.float)
        points = []
        h = img.shape[0]
        w = img.shape[1]
        minl = min(h, w)
        img = img.astype(float)
        m = 12.0/minsize
        minl = minl*m


        # create scale pyramid
        scales = []
        while minl >= 12:
            scales.append(m * pow(factor, factor_count))
            minl *= factor
            factor_count += 1

        # first stage
        for scale in scales:
            hs = int(np.ceil(h*scale))
            ws = int(np.ceil(w*scale))

            if fastresize:
                im_data = (img-127.5)*0.0078125 # [0,255] -> [-1,1]
                im_data = cv2.resize(im_data, (ws,hs)) # default is bilinear
            else:
                im_data = cv2.resize(img, (ws,hs)) # default is bilinear
                im_data = (im_data-127.5)*0.0078125 # [0,255] -> [-1,1]


            im_data = np.swapaxes(im_data, 0, 2)
            im_data = np.array([im_data], dtype = np.float)
            PNet.blobs['data'].reshape(1, 3, ws, hs)
            PNet.blobs['data'].data[...] = im_data
            out = PNet.forward()
            boxes = self.generateBoundingBox(out['prob1'][0,1,:,:], out['conv4-2'][0], scale, threshold[0])
            if boxes.shape[0] != 0:

                pick = self.nms(boxes, 0.5, 'Union')

                if len(pick) > 0 :
                    boxes = boxes[pick, :]

            if boxes.shape[0] != 0:
                total_boxes = np.concatenate((total_boxes, boxes), axis=0)

        numbox = total_boxes.shape[0]
        if numbox > 0:
            # nms
            pick = self.nms(total_boxes, 0.7, 'Union')
            total_boxes = total_boxes[pick, :]


            # revise and convert to square
            regh = total_boxes[:,3] - total_boxes[:,1]
            regw = total_boxes[:,2] - total_boxes[:,0]
            t1 = total_boxes[:,0] + total_boxes[:,5]*regw
            t2 = total_boxes[:,1] + total_boxes[:,6]*regh
            t3 = total_boxes[:,2] + total_boxes[:,7]*regw
            t4 = total_boxes[:,3] + total_boxes[:,8]*regh
            t5 = total_boxes[:,4]
            total_boxes = np.array([t1,t2,t3,t4,t5]).T


            total_boxes = self.rerec(total_boxes) # convert box to square


            total_boxes[:,0:4] = np.fix(total_boxes[:,0:4])

            [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(total_boxes, w, h)



        numbox = total_boxes.shape[0]
        if numbox > 0:
            # second stage


            # construct input for RNet
            tempimg = np.zeros((numbox, 24, 24, 3)) # (24, 24, 3, numbox)
            for k in range(numbox):
                tmp = np.zeros((int(tmph[k]) +1, int(tmpw[k]) + 1,3))



                tmp[int(dy[k]):int(edy[k])+1, int(dx[k]):int(edx[k])+1] = img[int(y[k]):int(ey[k])+1, int(x[k]):int(ex[k])+1]


                tempimg[k,:,:,:] = cv2.resize(tmp, (24, 24))

            tempimg = (tempimg-127.5)*0.0078125 # done in imResample function wrapped by python


            tempimg = np.swapaxes(tempimg, 1, 3)


            RNet.blobs['data'].reshape(numbox, 3, 24, 24)
            RNet.blobs['data'].data[...] = tempimg
            out = RNet.forward()


            score = out['prob1'][:,1]

            pass_t = np.where(score>threshold[1])[0]


            score =  np.array([score[pass_t]]).T
            total_boxes = np.concatenate( (total_boxes[pass_t, 0:4], score), axis = 1)



            mv = out['conv5-2'][pass_t, :].T

            if total_boxes.shape[0] > 0:
                pick = self.nms(total_boxes, 0.7, 'Union')

                if len(pick) > 0 :
                    total_boxes = total_boxes[pick, :]

                    total_boxes = self.bbreg(total_boxes, mv[:, pick])

                    total_boxes = self.rerec(total_boxes)


            numbox = total_boxes.shape[0]
            if numbox > 0:
                # third stage

                total_boxes = np.fix(total_boxes)
                [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(total_boxes, w, h)



                tempimg = np.zeros((numbox, 48, 48, 3))
                for k in range(numbox):
                    tmp = np.zeros((int(tmph[k]), int(tmpw[k]),3))
                    tmp[int(dy[k]):int(edy[k])+1, int(dx[k]):int(edx[k])+1] = img[int(y[k]):int(ey[k])+1, int(x[k]):int(ex[k])+1]
                    tempimg[k,:,:,:] = cv2.resize(tmp, (48, 48))
                tempimg = (tempimg-127.5)*0.0078125 # [0,255] -> [-1,1]

                # ONet
                tempimg = np.swapaxes(tempimg, 1, 3)
                ONet.blobs['data'].reshape(numbox, 3, 48, 48)
                ONet.blobs['data'].data[...] = tempimg
                out = ONet.forward()

                score = out['prob1'][:,1]
                points = out['conv6-3']
                pass_t = np.where(score>threshold[2])[0]
                points = points[pass_t, :]
                score = np.array([score[pass_t]]).T
                total_boxes = np.concatenate( (total_boxes[pass_t, 0:4], score), axis=1)


                mv = out['conv6-2'][pass_t, :].T
                w = total_boxes[:,3] - total_boxes[:,1] + 1
                h = total_boxes[:,2] - total_boxes[:,0] + 1

                points[:, 0:5] = np.tile(w, (5,1)).T * points[:, 0:5] + np.tile(total_boxes[:,0], (5,1)).T - 1
                points[:, 5:10] = np.tile(h, (5,1)).T * points[:, 5:10] + np.tile(total_boxes[:,1], (5,1)).T -1

                if total_boxes.shape[0] > 0:
                    total_boxes = self.bbreg(total_boxes, mv[:,:])

                    pick = self.nms(total_boxes, 0.7, 'Min')


                    if len(pick) > 0 :
                        total_boxes = total_boxes[pick, :]

                        points = points[pick, :]

        return total_boxes, points



    def detectFace(self,img,minsize = 20,threshold = [0.6, 0.7, 0.7],factor = 0.709):
        img_matlab = img.copy()
        tmp = img_matlab[:,:,2].copy()
        img_matlab[:,:,2] = img_matlab[:,:,0]
        img_matlab[:,:,0] = tmp
        return self.__detect_face(img_matlab, minsize, self.PNet, self.RNet, self.ONet, threshold, False, factor)

    def alignment(self,image,points,ref_points=np.array([30.2946, 51.6963, 65.5318, 51.5014, 48.0252, 71.7366, 33.5493, 92.3655, 62.7299, 92.2041]),ref_shape = (112, 112, 3)):
        assert(len(points)==len(ref_points))
        rows,cols,ch = ref_shape
        num_point=len(ref_points)//2
        dst=np.empty((num_point,2),dtype=np.int)
        k=0
        for i in range(num_point):
            for j in range(2):
                dst[i][j]=ref_points[k]
                k=k+1
        
        src=np.empty((num_point,2),dtype=np.int)
        k=0
        for i in range(num_point):
            for j in range(2):
                src[i][j]=points[k]
                k=k+1
        
        tfrom = tf.estimate_transform('similarity',dst,src)
        warpimage = tf.warp(image,inverse_map=tfrom)
        i = warpimage*255
        return i[0:rows,0:cols,:]

    def run(self,image,th = [0.6, 0.7, 0.7],Ensure = True):
        b, p = self.detectFace(image,threshold =th)
        if Ensure:
            i=0
            while not len(p):
                th[i%3] -= 0.1
                i +=1
                b, p = self.detectFace(image,threshold =th)
        points = np.empty((10,1),dtype=np.float)
        for j in range(5):
            points[2*j] = p[0][j]
            points[2*j+1] = p[0][j+5]
        warpimage = self.alignment(image,points)
        return  warpimage

    def getRep(self,im1):
        Net = self.Net
        transformer = caffe.io.Transformer({'data': Net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        Net.blobs['data'].reshape(2,3,224,224)
        Net.blobs['data'].data[...] = transformer.preprocess('data', im1)
        out = Net.forward_all(data=np.asarray([transformer.preprocess('data', im1)]))
        feature = copy.copy(Net.blobs['fc7'].data[0])
        return feature

    def cos_distance(self,feature1,feature2):
        cos_distance = 1 - scipy.spatial.distance.cosine(feature1,feature2)
        return cos_distance

    def compare(self,path1,path2):
        try:
            img2 = im.open(path2)
            try:
                img1 = im.open(path1)
            except Exception as err:
                with open(path1,"rb") as f:
                    hexstr = f.read()
                if len(hexstr) > 32:
                    hexstr = hexstr[:32]+ bytes().fromhex('C7') + hexstr[33:]    
                    bs = BytesIO(hexstr)
                    img1 = im.open(bs)
            img1 = img1.convert("RGB")
            img2 = img2.convert("RGB")
            #img1 = img1.resize((224,224), im.BILINEAR)
            #img2 = img2.resize((224,224), im.BILINEAR)
            get_image1 = np.array(img1)
            get_image2 = np.array(img2)
            get_image1 = get_image1[:, :, ::-1]
            gray1 = cv2.cvtColor(get_image1, cv2.COLOR_BGR2GRAY)
            faces = self.faceCascade.detectMultiScale(gray1,scaleFactor=1.1,minNeighbors=5,minSize=(50, 50))
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    roiImage = get_image1[y:y+h,x:x+w]
            else:
                roiImage = get_image1
            #cv2.imwrite('2.jpg',roiImage)
            roiImage_mirr = roiImage.copy()
            for i in range(roiImage.shape[0]):
                for j in range(roiImage.shape[1]):
                    roiImage_mirr[i,roiImage.shape[1]-1-j] = roiImage[i,j]
            get_image2 = get_image2[:, :, ::-1]
            boxes2,p2 = self.detectFace(get_image2)
            #print(len(boxes2))
            #print(len(p2))
            if len(boxes2) >= 1:
                feature1_1 = self.getRep(roiImage)
                feature1_2 = self.getRep(roiImage_mirr)
                feature1 = np.concatenate([feature1_1,feature1_2],axis=0)
                #feature1 = ( feature1_1 + feature1_2)/2.0
                #tmp_boxes = []
                #for num_boxes in boxes2:
                #    tmp_boxes.append((num_boxes[3]-num_boxes[1]+1)*(num_boxes[2]-num_boxes[0]+1))
                #boxes2 = boxes2[tmp_boxes.index((np.max(tmp_boxes)))]
                #get_image2 = get_image2[int(boxes2[1]):int(boxes2[3])+1,int(boxes2[0]):int(boxes2[2])+1]
                wrapimage2 = self.run(get_image2)
                #print(len(wrapimage2))
                #print(wrapimage2.shape)
                #cv2.imwrite('1.jpg',wrapimage2)
                wrapimage2_mirr = wrapimage2.copy()
                #print(wrapimage2_mirr.shape)
                for i in range(wrapimage2.shape[0]):
                    for j in range(wrapimage2.shape[1]):
                        wrapimage2_mirr[i,wrapimage2.shape[1]-1-j] = wrapimage2[i,j]
                #cv2.imwrite('2.jpg',wrapimage2_mirr)
                feature2_1 = self.getRep(wrapimage2)
                #print(feature2_1)
                #print(len(feature2_1))
                #print(type(feature2_1))
                feature2_2 = self.getRep(wrapimage2_mirr)
                #print(feature2_2)
                feature2 = np.concatenate([feature2_1,feature2_2],axis=0)
                #print(len(feature2))
                #print(feature2)
                similar = self.cos_distance(feature1,feature2)
                similar = float("%.3f" %similar)
            else:
                similar = 0.000

            jurgeanswer_cos = " "
            similar_threshold = 0.7
            if similar >= similar_threshold:
                reuslt = "True"
                jurgeanswer_cos = str(similar)
            else:
                reuslt = "False"
                jurgeanswer_cos = str(similar)
            return reuslt, jurgeanswer_cos

        except Exception as e:
            #print(e)
            reuslt = "False"
            jurgeanswer_cos = -1
            return reuslt, jurgeanswer_cos
if __name__ == "__main__":
    fc=Faceapp(caffe_model_path='/home/finance/Domains/faceselfmodel2.msxf.lotest/wyc/model')
    ishimself,confidence   = fc.compare('./tmp_api_m2_new/20170822170125.jpg','./tmp_api_m2_new/20180314120319.jpg')
    print((ishimself,confidence))
