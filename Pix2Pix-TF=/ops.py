# coding:utf-8
#!/usr/bin/python3

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops


TType = tf.float32; # TType = tf.float16
################################################################################
def lrelu(x, leak=0.2, name="lrelu"):   return tf.maximum(x, leak*x)
def Lrelu(x, leak=0.1, name="Lrelu"):   return tf.maximum(x, leak*x)


# batch normalization: => instance for batch=1
# deals with poor initialization, enhances gradient flow
'''batch_normalization(inputs, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, 
beta_initializer=<tensorflow.python.ops.init_ops.Zeros object at 0x7f9c79c1a048>, 
gamma_initializer=<tensorflow.python.ops.init_ops.Ones object at 0x7f9c79c1a080>, 
moving_mean_initializer=<tensorflow.python.ops.init_ops.Zeros object at 0x7f9c79c1a0b8>, 
moving_variance_initializer=<tensorflow.python.ops.init_ops.Ones object at 0x7f9c79c1a0f0>, 
beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None, 
training=False, trainable=True, name=None, reuse=None, renorm=False, renorm_clipping=None, 
renorm_momentum=0.99, fused=None, virtual_batch_size=None, adjustment=None)'''
def batch_norm(x, isTrain=True, name="batch_norm"): # instance_norm for batch=1
    return tf.layers.batch_normalization(x, axis=-1, momentum=0.9, epsilon=1E-5, training=isTrain, name=name)
    #return tf.contrib.layers.batch_norm(x, decay=0.9, epsilon=1E-5, scale=True, is_training=isTrain, 
    #                                    updates_collections=None, scope=name) # early version


def binary_cross_entropy(preds, targets, name=None):
    """
    Computes binary cross entropy given `preds`.
    For brevity, let `x = `, `z = targets`.  The logistic loss is
        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))
    Args:
        preds: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `preds`.
    """
    eps = 1e-12
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds+eps) + (1-targets) * tf.log(1-preds+eps)))


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape(); y_shapes = y.get_shape()
    return tf.concat([x, y*tf.ones([x_shapes[0],x_shapes[1],x_shapes[2],y_shapes[3]])], 3)


def conv2d(input, output_dim, k_h=4, k_w=4, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        shape = [k_h, k_w, input.get_shape()[-1], output_dim]
        w = tf.get_variable('w', shape, TType, initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input, w, strides=[1, d_h, d_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], TType, initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        return conv


def deconv2d(input, output_shape, k_h=4, k_w=4, d_h=2, d_w=2, stddev=0.02, name="deconv2d", with_w=True):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        shape = [k_h, k_w, output_shape[-1], input.get_shape()[-1]]
        w = tf.get_variable('w', shape, TType, initializer=tf.random_normal_initializer(stddev=stddev))
        try:
            deconv = tf.nn.conv2d_transpose(input, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
        except AttributeError: # Support for verisons of TensorFlow before 0.7.0
            deconv = tf.nn.deconv2d(input, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], TType, initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        return (deconv, w, biases) if with_w else deconv


def linear(input, output, name=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input.get_shape().as_list()
    with tf.variable_scope(name or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output], TType, tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output], TType, initializer=tf.constant_initializer(bias_start))
        if with_w:  return tf.matmul(input, matrix) + bias, matrix, bias
        else:       return tf.matmul(input, matrix) + bias


################################################################################
# Peak Signal to Noise Ratio
def PSNR(I, K, md=1, ch=1, L=255):
    assert(I.shape==K.shape) # Tensor[height,width,channel]
    if md==1: I,K = (I+1)*127.5,(K+1)*127.5 # restore to 255
    IK = (I-K*1.0)**2 # avoid "uint8" overflow
    MAX = L**2; ee = MAX*1E-10; gg = np.log(10) # for tf.log()
    if ch<2: MSE = tf.reduce_mean(IK) # combine/average channels
    else: MSE = tf.reduce_mean(IK,axis=(0,1)) # separate channels
    return 10 * tf.log(MAX/(MSE+ee)) / gg # PSNR


# Structural Similarity (Index Metric)
def SSIM(I, K, md=1, ch=1, k1=0.01, k2=0.03, L=255):
    assert(I.shape==K.shape); h,w,c = I.shape.as_list()
    if md==1: I,K = (I+1)*127.5,(K+1)*127.5 # restore to 255
    if ch<2: # combine/average channels->float
        mx, my = tf.reduce_mean(I), tf.reduce_mean(K)
        sx, sy = tf.reduce_mean((I-mx)**2), tf.reduce_mean((K-mx)**2)
        # cov = tf.reduce_sum((I-mx)*(K-my))/(h*w*c-1) # unbiased
        cov = tf.reduce_mean((I-mx)*(K-my)) # biased covariance
    else: # separate/individual/independent channels->np.array
        mx, my = tf.reduce_mean(I,axis=(0,1)), tf.reduce_mean(K,axis=(0,1))
        sx, sy = tf.reduce_mean((I-mx)**2,axis=(0,1)), tf.reduce_mean((K-mx)**2,axis=(0,1))
        # cov = tf.reduce_sum((I-mx)*(K-my),axis=(0,1))/(h*w-1) # unbiased
        cov = tf.reduce_mean((I-mx)*(K-my),axis=(0,1)) # biased covariance
    c1, c2 = (k1*L)**2, (k2*L)**2 # stabilizer, avoid divisor=0
    SSIM = (2*mx*my+c1)/(mx**2+my**2+c1) * (2*cov+c2)/(sx+sy+c2)
    return SSIM # SSIM: separate or average channels


# SSIM Loss Function: Differentiable
def SSLF(I, K, md=1, b=1): # class C^1 of differentiability
    """
    # Fx = b * tf.cond(x>1.0, lambda:(x+3)**2, lambda:16*x**0.5)
    # f(x)=g(x):a*x**0.5=b*(x+c)**2, f'(x)=g'(x):(a/2)/x**0.5=2*b*(x+c). x=1 => c=3,a=16*b
    # Fx = b * tf.cond(x>1.0, lambda:(x+7)**2, lambda:64*x**0.25)
    # f(x)=g(x):a*x**0.25=b*(x+c)**2, f'(x)=g'(x):(a/4)/x**0.75=2*b*(x+c). x=1 => c=7,a=64*b
    """
    assert(I.shape==K.shape) # Tensor[batch,ht,wd,ch]
    N = I.shape.as_list()[0]; lf = [0]*N # batch
    for i in range(N):
        x = 1 - SSIM(I[i], K[i], md) # convert x in [0,2]
        lf[i] = tf.cond(x>1.0, lambda:(x+3)**2, lambda:16*x**0.5)
    return b * tf.reduce_mean(lf)


def TestSP(I, K, md=1, b=1): # Test SSIM/PSNR
    sym = lambda x: x/127.5-1
    if type(I)==str: I = cv2.imread(I)
    if type(K)==str: K = cv2.imread(K)
    if md==1: I, K = sym(I), sym(K)
    I1, K1 = tf.convert_to_tensor(I*1.0), tf.convert_to_tensor(K*1.0)
    with tf.Session() as ss: # eval: tf.Tensor->np.array
        print("SSIM:", ss.run(SSIM(I1,K1,md)))
        print("SSLF:", ss.run(SSLF(I1,K1,md,b)))
        print("PSNR:", PSNR(I1,K1,md).eval(session=ss))

