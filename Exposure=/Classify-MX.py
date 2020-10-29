# coding:utf-8
# !/usr/bin/python3
import os, cv2, time
import numpy as np
import mxnet as mx

from mxnet import autograd as ag
from mxnet import gluon, init, nd, image
from mxnet.gluon import nn

from mxnet.gluon.utils import split_and_load
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.data.vision import ImageFolderDataset
from mxnet.gluon.model_zoo.vision import get_model
# from mxnet.gluon.model_zoo import vision
# from gluoncv.model_zoo import get_model
# from gluoncv.utils import TrainingHistory


################################################################################
# Hyperparameters:
gpu, threads = (1, 4)
batch_size_per_device = 8
batch_size = batch_size_per_device * max(gpu, 1)
CTX = lambda N: [mx.gpu(i) for i in range(N)] if N>0 else [mx.cpu()]

lr_factor = 0.75
lr_steps = (30, 60, 100, 150, np.inf)
optimizer = 'nag'  # Nesterov accelerated gradient descent
optimizer_params = {'learning_rate': 0.001, 'wd': 0.0001, 'momentum': 0.9}


################################################################################
# Load Model:
def LoadModel(out, ctx, params="", name="ResNet50_v2"):
    if os.path.isfile(params): # for test
        Net = get_model(name, pretrained=False)
        #Net = vision.resnet50_v2(pretrained=False)
        with Net.name_scope(): Net.output = nn.Dense(out)
        Net.load_parameters(params, ctx=ctx)
    else: # for train/finetune
        Net = get_model(name, pretrained=True)
        #Net = vision.resnet50_v2(pretrained=True)
        with Net.name_scope(): Net.output = nn.Dense(out)
        Net.output.initialize(init.Xavier(), ctx=ctx)
        Net.collect_params().reset_ctx(ctx)
        Net.hybridize()
    return Net


################################################################################
# Data Loader->Batch:
def LoadData(path, mod="train", batch=batch_size):
    shuf = "train" in mod # True for train, else False
    batch_data = gluon.data.DataLoader(
        ImageFolderDataset(path).transform_first(Transform(mod)),
        batch_size=batch, shuffle=shuf, last_batch='rollover', num_workers=threads)
    return batch_data # an iterable object, len(.)=num of batches


# Data Augmentation:
# Ref: incubator-mxnet/python/mxnet/image/image.py
def Transform(mod="test", jitter=0.4, lighting=0.1):
    mn, std = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]
    if "train" in mod: # for train
        transformer = transforms.Compose([
            transforms.RandomFlipLeftRight(),
            transforms.RandomResizedCrop(224), # Randomly Crop, then Resize
            transforms.RandomColorJitter(contrast=jitter, saturation=jitter),
            #transforms.RandomColorJitter(brightness=jitter, contrast=jitter, saturation=jitter),
            transforms.RandomLighting(lighting), # Randomly Add PCA_based Noise
            transforms.ToTensor(), # uint8(H x W x C) -> float(C x H x W)
            transforms.Normalize(mn,std) ]) # Normalize with mean & std
    else: # for test/val
        transformer = transforms.Compose([
            transforms.Resize(224),
            #transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(), # uint8(H x W x C) -> float(C x H x W)
            transforms.Normalize(mn,std) ]) # Normalize with mean & std
    return transformer


def Transform_test(im): # Transform("test")
    if type(im)==str: im = image.imread(im)
    if type(im)==np.ndarray: im = nd.array(im[:,:,::-1])
    assert(type(im)==nd.ndarray.NDArray)

    im = image.resize_short(im, 256) # =transforms.Resize(256)
    im, _ = image.center_crop(im, (224, 224)) # =transforms.CenterCrop(224)
    im = im.transpose((2,0,1)).astype('float32')/255 # =transforms.ToTensor()

    mn = nd.array([0.485, 0.456, 0.406]).reshape((3,1,1)) # rgb_mean
    std = nd.array([0.229, 0.224, 0.225]).reshape((3,1,1)) # rgb_std
    im = (im - mn) / std # =transforms.Normalize(mn,std)
    return im.expand_dims(axis=0)


# Split Batch -> data+label:
def Split(batch, ctx): # data/label is a 1-element list of <mxnet::NDArray>
    data = split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
    label = split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
    return data[0], label[0] # <NDArray>: (batch, channel, height, width), (batch)


################################################################################
# Test Model:
def Train(Data, Net, ctx, epochs):
    if type(Data)==str: # default batch_size
        Data = [LoadData(os.path.join(Data,i), i) for i in ("train","val")]
    train_data, val_data = Data; num_batch = len(train_data) # iterable

    metric = mx.metric.Accuracy()
    L = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(Net.collect_params(), optimizer, optimizer_params)
    # train_history = TrainingHistory(['training-error', 'validation-error'])

    for k in range(epochs):
        tic = time.time()
        train_loss = 0; metric.reset()
        if k in lr_steps: # lr update policy
            trainer.set_learning_rate(trainer.learning_rate * lr_factor)

        for i, batch in enumerate(train_data):
            X, Y = Split(batch, ctx) # batch=[data,label]
            with ag.record(): X = Net(X); loss = L(X,Y) # batch forward
            loss.backward() # batch back-propagation-> get gradients
            trainer.step(batch_size=len(Y)) # optimize/update weights
            train_loss += loss.mean().asscalar() # loss.shape=(batch)
            metric.update(Y, X) # X.shape=(batch,out), Y.shape=(batch)
        train_loss /= num_batch # average loss over batches

        _, train_acc = metric.get()
        _, val_acc = Test(val_data, Net, ctx)
        # train_history.update([1-train_acc, 1-val_acc])

        mark = ""; sh = train_acc*val_acc # save weights policy
        if sh>0.8: Net.save_parameters("ResNet50_%d.params" % k); mark = str(sh)

        print("[Epoch %d] Train=%.3f Val=%.3f Loss=%.3f Time=%.2f" %
              (k, train_acc, val_acc, train_loss, time.time()-tic), mark)
    return Net


# Test Model:
def Test(Data, Net, ctx):
    if type(Data)==str: Data = LoadData(Data,"test")
    metric = mx.metric.Accuracy()
    for i, batch in enumerate(Data):
        X, Y = Split(batch, ctx) # data and labels
        metric.update(Y, Net(X)) # batch forward
        # X = X.transpose((0,2,3,1)).asnumpy()
    return metric.get()


# Classify/Forward:
def Classify(im, Net, cls):
    if type(im)==str: im = cv2.imread(im) #image.imread(im)
    if type(im)==np.ndarray: im = nd.array(im[:,:,::-1])

    x = Transform()(im) # <NDArray> x.shape=(C,H,W)
    y = Net(x.expand_dims(axis=0)) # y.shape=(batch=1,out)
    #im = x.transpose((1,2,0)).asnumpy() # transformed im
    id = y.argmax(axis=1).astype('int').asscalar()
    res, prob = cls[id], y.softmax()[0][id].asscalar()
    return res, prob


# Visualize result:
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
    ctx = [mx.cpu()] # CTX(gpu)
    #model = LoadModel(len(cls), ctx); print(model)
    #model = Train("root", model, ctx, 250) # Train model

    model = LoadModel(len(cls), ctx, "Expos.params")
    res = Test("root/test", model, ctx); print(res)
    #Show("root", model, cls)


################################################################################
# execute ". activate mx" before "python Train.py"
