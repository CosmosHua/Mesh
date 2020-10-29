# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import re
import os
import math
import argparse
import facenet
import align.detect_face

#  import other libraries
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

class face_app:
    def __init__(self,model):
        self.gpu_memory_fraction = 1.0
        self.minsize = 50  # minimum size of face
        self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        self.factor = 0.709  # scale factor
        self.margin = 44
        self.image_size = 160
#        model = r'../model/20170512-110547'

        with tf.Graph().as_default():
            # Load the model
            config = tf.ConfigProto()  
#            config.gpu_options.allow_growth=True  
            config.gpu_options.per_process_gpu_memory_fraction = 0.4
            self.sess = tf.Session(config=config) 
#            with tf.device("/cpu:0"):
            model_exp = os.path.expanduser(model)
            meta_file, ckpt_file = self.get_model_filenames(model_exp)
            saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
            saver.restore(self.sess, os.path.join(model_exp, ckpt_file))

            # Get input and output tensors
            self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(self.sess, None)

    def compare(self, image1, image2):
        # 人脸特征提取
        image1 = misc.imread(os.path.expanduser(image1))
        image2 = misc.imread(os.path.expanduser(image2))
        image1 = self.salt(image1, 500)
        try:
            image_files = self.detect(image1, image2)
            images = self.load_and_align_data(image_files)
        except Exception as e:
            print(e)
            distance = 1.574
            result = 'False'
            return result,distance

        if len(image_files[1]) ==0:
            distance = 1.574
            result = 'False'
            return result,distance

        feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
        emb = self.sess.run(self.embeddings, feed_dict=feed_dict)
        dist = np.sqrt(np.sum(np.square(np.subtract(emb[0, :], emb[1, :]))))

        if dist > 1.1:
            result = 'False'
            distance = round(dist / 1.1,3)
        else:
            result = 'True'
            distance = round(dist / 1.1,3)
        return result, distance

    def get_model_filenames(self,model_dir):
        files = os.listdir(model_dir)
        meta_files = [s for s in files if s.endswith('.meta')]
        if len(meta_files)==0:
            raise ValueError('No meta file found in the model directory (%s)' % model_dir)
        elif len(meta_files)>1:
            raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
        meta_file = meta_files[0]
        meta_files = [s for s in files if '.ckpt' in s]
        max_step = -1
        for f in files:
            step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
            if step_str is not None and len(step_str.groups())>=2:
                step = int(step_str.groups()[1])
                if step > max_step:
                    max_step = step
                    ckpt_file = step_str.groups()[0]
        return meta_file, ckpt_file


    def rotate(self, img, angle):
        height = img.shape[0]
        width = img.shape[1]
        if angle % 180 == 0:
            scale = 1
        elif angle % 90 == 0:
            scale = float(max(height, width)) / min(height, width)
        else:
            scale = math.sqrt(pow(height, 2) + pow(width, 2)) / min(height, width)
            # print 'scale %f\n' %scale
        rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
        rotateImg = cv2.warpAffine(img, rotateMat, (width, height))
        # cv2.imshow('rotateImg',rotateImg)
        # cv2.waitKey(0)
        return rotateImg

    def load_and_align_data(self, image_paths):
        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        image_size = 160
        margin = 44
        gpu_memory_fraction = 1.0

        nrof_samples = len(image_paths)
        img_list = [None] * nrof_samples
        for i in range(nrof_samples):
            #        print(image_paths[i])
            aligned = misc.imresize(image_paths[i], (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            img_list[i] = prewhitened
            cv2.imwrite("align"+str(i)+".jpg",aligned) 
        images = np.stack(img_list)
        return images

    def salt(self, img, n):
        for k in range(n):
            i = int(np.random.random() * img.shape[1]);
            j = int(np.random.random() * img.shape[0]);
            if img.ndim == 2:
                img[j, i] = 255
            elif img.ndim == 3:
                img[j, i, 0] = 255
                img[j, i, 1] = 255
                img[j, i, 2] = 255
        image = cv2.medianBlur(img, 5)
        return image

    def hint(self, image):
        lut = np.zeros(256, dtype=image.dtype)  # 创建空的查找表
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()  # 计算累积直方图
        cdf_m = np.ma.masked_equal(cdf, 0)  # 除去直方图中的0值
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (
            cdf_m.max() - cdf_m.min())  # 等同于前面介绍的lut[i] = int(255.0 *p[i])公式
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')  # 将掩模处理掉的元素补为0
        image = cv2.LUT(image, cdf)
        return image

    def detect(self, image1, image2):
        # create a list of your images
        images = []
        images.append(image1)
        images.append(image2)
        minsize = 20
        pnet = self.pnet
        rnet = self.rnet
        onet = self.onet
        threshold = self.threshold
        factor = self.factor

        # Start code from facenet/src/compare.py
        head_images = []
        for i in images:
#            img = misc.imread(os.path.expanduser(i))
            img = i  
            if  img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            bounding_boxes, _ = align.detect_face.detect_face(
                img, minsize, pnet,
                rnet, onet, threshold, factor)
            if len(bounding_boxes) == 0:
                result = '%s 未检测出人脸' % i
                head_images.append(img)
            else:
                headimage = self.get_head(bounding_boxes, img)
                if len(headimage) == 0:
                    head_images.append(img)
                else:
                    head_images.append(headimage)
#                head_images.append(headimage)
        return head_images

    def get_head(self, bounding_boxes, img):
        head_imgs = []
        for (x1, y1, x2, y2, acc) in bounding_boxes:
            w = x2 - x1
            h = y2 - y1
            out_img = cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w),
                                                              int(y1 + h)), (255, 0, 0), 2)
            head_img = out_img[int(y1):int(y1 + h), int(x1):int(x1 + w)]
            head_imgs.append(head_img)
            if len(head_imgs) == 2:
                if len(head_imgs[0]) >= len(head_imgs[1]):
                    return head_imgs[0]
                else:
                    return head_imgs[0]
            else:
                return head_img

    def app_close(self):
        self.sess.close()
