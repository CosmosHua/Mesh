# coding:utf-8
# !/usr/bin/python3
import os, cv2
import numpy as np
import mxnet as mx

from mxnet import nd, image
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.model_zoo.vision import get_model


################################################################################
def LoadModel(out, ctx, params="Expos"):
    Net = get_model("ResNet50_v2", pretrained=False)
    with Net.name_scope(): Net.output = nn.Dense(out)
    Net.load_parameters(params+".params", ctx=ctx)
    return Net


def Transform():
    mn, std = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]
    transformer = transforms.Compose([
        transforms.Resize(224), transforms.ToTensor(),
        transforms.Normalize(mn,std) ])
    return transformer


def Classify(im, Net, cls):
    if type(im)==str: im = cv2.imread(im)
    if type(im)==np.ndarray: im = nd.array(im[:,:,::-1])

    y = Net(Transform()(im).expand_dims(axis=0))
    id = y.argmax(axis=1).astype('int').asscalar()
    res, prob = cls[id], y.softmax()[0][id].asscalar()
    return res, prob


def Show(src, Net, cls, s=10):
    for root,dir,file in os.walk(src):
        for ff in file:
            im = cv2.imread(os.path.join(root,ff))
            res, prob = Classify(im, Net, cls) # get class+prob
            print("[%s] is [%s], with probability [%.3f]." % (ff,res,prob) )
            im = cv2.resize(im, (0,0), fx=0.5, fy=0.5)
            cv2.imshow("im", im); cv2.waitKey(s*1000)
    cv2.destroyAllWindows()


################################################################################
if __name__ == "__main__":
    cls = ("abnorm", "norm")
    ctx = [mx.cpu()] # [mx.gpu()]
    model = LoadModel(len(cls), ctx)
    Show("./root", model, cls)


################################################################################
# execute ". activate mx" before "python Train.py"
