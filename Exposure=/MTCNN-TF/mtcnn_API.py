# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 15:05:41 2018 @author: wei.li09
Modified on Tue Sep 27 17:05:41 2018 @author: Joshua
"""

import cv2
import mtcnn as df
import tensorflow as tf


class FaceDetect(object):
    def __init__(self, minsize = 50, threshold = (0.6, 0.7, 0.7), scale_factor = 0.709):
        self.minSize = minsize
        self.thresh = threshold
        self.factor = scale_factor
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        with self.sess.as_default():
            self.pnet, self.rnet, self.onet = df.create_mtcnn(self.sess, None)
            
    def test(self, image):
        if type(image)==str: image = cv2.imread(image)
        f_scale = min(1.0, 1000/min(image.shape[:2]))
        if f_scale < 1.0:
            size = [int(i*f_scale) for i in image.shape[::-1]]
            image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        bounding_boxes, points = df.detect_face(image, self.minSize, 
                                                self.pnet, self.rnet, self.onet, 
                                                self.thresh, self.factor)
        bounding_boxes /= f_scale
        return bounding_boxes
        
    def show(self, image, out='result.jpg'):
        bounding_boxes = self.test(image)
        for face_position in bounding_boxes:
            face_position = face_position.astype(int)  
            cv2.rectangle(image,
                          (face_position[0], face_position[1]), 
                          (face_position[2], face_position[3]), 
                          (0, 255, 0), 2)
        cv2.imwrite(out, image)

    
if __name__ == '__main__':
    image_path = 'liwei.jpg'
    image = cv2.imread(image_path)
    faceDet_model = FaceDetect() # initialize the detection model
    bboxes = faceDet_model.show(image) # detect the face

