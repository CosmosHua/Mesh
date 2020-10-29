# coding:utf-8
# !/usr/bin/python3

import numpy as np
import os, cv2, random
import mxnet.ndarray as nd


'''
Ref: incubator-mxnet/blob/master/python/mxnet/image/image.py
Note: default image format: MXNet=RGB, OpenCV=BGR. So, use BGR->RGB.
'''
################################################################################
def color_normalize(src, mean, std=None):
    """Normalize src with mean and std.

    Parameters
    ----------
    src : NDArray
        Input image
    mean : NDArray
        RGB mean to be subtracted
    std : NDArray
        RGB standard deviation to be divided

    Returns
    -------
    NDArray
        An `NDArray` containing the normalized image.
    """
    if mean is not None: src -= mean
    if std is not None: src /= std
    return src


class Augmenter(object):
    """Image Augmenter base class"""
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        for k, v in self._kwargs.items():
            if isinstance(v, nd.NDArray):
                v = v.asnumpy()
            if isinstance(v, np.ndarray):
                v = v.tolist()
                self._kwargs[k] = v

    def __call__(self, src):
        """Abstract implementation body"""
        raise NotImplementedError("Must override implementation.")


class SequentialAug(Augmenter):
    """Composing a sequential augmenter list.

    Parameters
    ----------
    ts : list of augmenters
        A series of augmenters to be applied in sequential order.
    """
    def __init__(self, ts):
        super(SequentialAug, self).__init__()
        self.ts = ts

    def __call__(self, src):
        """Augmenter body"""
        for aug in self.ts: src = aug(src)
        return src


class RandomOrderAug(Augmenter):
    """Apply list of augmenters in random order

    Parameters
    ----------
    ts : list of augmenters
        A series of augmenters to be applied in random order
    """
    def __init__(self, ts):
        super(RandomOrderAug, self).__init__()
        self.ts = ts

    def __call__(self, src):
        """Augmenter body"""
        random.shuffle(self.ts)
        for t in self.ts: src = t(src)
        return src


class BrightnessJitterAug(Augmenter):
    """Random brightness jitter augmentation.

    Parameters
    ----------
    brightness : float
        The brightness jitter ratio range, [0, 1]
    """
    def __init__(self, brightness):
        super(BrightnessJitterAug, self).__init__(brightness=brightness)
        self.brightness = brightness

    def __call__(self, src):
        """Augmenter body"""
        alpha = 1.0 + random.uniform(-self.brightness, self.brightness)
        src *= alpha
        return src


class ContrastJitterAug(Augmenter):
    """Random contrast jitter augmentation.

    Parameters
    ----------
    contrast : float
        The contrast jitter ratio range, [0, 1]
    """
    def __init__(self, contrast):
        super(ContrastJitterAug, self).__init__(contrast=contrast)
        self.contrast = contrast
        self.coef = nd.array([[[0.299, 0.587, 0.114]]])

    def __call__(self, src):
        """Augmenter body"""
        alpha = 1.0 + random.uniform(-self.contrast, self.contrast)
        gray = src * self.coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * nd.sum(gray)
        src *= alpha
        src += gray
        return src


class SaturationJitterAug(Augmenter):
    """Random saturation jitter augmentation.

    Parameters
    ----------
    saturation : float
        The saturation jitter ratio range, [0, 1]
    """
    def __init__(self, saturation):
        super(SaturationJitterAug, self).__init__(saturation=saturation)
        self.saturation = saturation
        self.coef = nd.array([[[0.299, 0.587, 0.114]]])

    def __call__(self, src):
        """Augmenter body"""
        alpha = 1.0 + random.uniform(-self.saturation, self.saturation)
        gray = src * self.coef
        gray = nd.sum(gray, axis=2, keepdims=True)
        gray *= (1.0 - alpha)
        src *= alpha
        src += gray
        return src


class HueJitterAug(Augmenter):
    """Random hue jitter augmentation.

    Parameters
    ----------
    hue : float
        The hue jitter ratio range, [0, 1]
    """
    def __init__(self, hue):
        super(HueJitterAug, self).__init__(hue=hue)
        self.hue = hue
        self.tyiq = np.array([[0.299, 0.587, 0.114],
                              [0.596, -0.274, -0.321],
                              [0.211, -0.523, 0.311]])
        self.ityiq = np.array([[1.0, 0.956, 0.621],
                               [1.0, -0.272, -0.647],
                               [1.0, -1.107, 1.705]])

    def __call__(self, src):
        """Augmenter body.
        Using approximate linear transfomation described in:
        https://beesbuzz.biz/code/hsv_color_transforms.php
        """
        alpha = random.uniform(-self.hue, self.hue)
        u = np.cos(alpha * np.pi)
        w = np.sin(alpha * np.pi)
        bt = np.array([[1.0, 0.0, 0.0],
                       [0.0, u, -w],
                       [0.0, w, u]])
        t = np.dot(np.dot(self.ityiq, bt), self.tyiq).T
        src = nd.dot(src, nd.array(t))
        return src


class ColorJitterAug(RandomOrderAug):
    """Apply random brightness, contrast and saturation jitter in random order.

    Parameters
    ----------
    brightness : float
        The brightness jitter ratio range, [0, 1]
    contrast : float
        The contrast jitter ratio range, [0, 1]
    saturation : float
        The saturation jitter ratio range, [0, 1]
    """
    def __init__(self, brightness, contrast, saturation):
        ts = []
        if brightness > 0:
            ts.append(BrightnessJitterAug(brightness))
        if contrast > 0:
            ts.append(ContrastJitterAug(contrast))
        if saturation > 0:
            ts.append(SaturationJitterAug(saturation))
        super(ColorJitterAug, self).__init__(ts)


class LightingAug(Augmenter):
    """Add PCA based noise.

    Parameters
    ----------
    alphastd : float
        Noise level
    eigval : 3x1 np.array
        Eigen values
    eigvec : 3x3 np.array
        Eigen vectors
    """
    def __init__(self, alphastd, eigval, eigvec):
        super(LightingAug, self).__init__(alphastd=alphastd, eigval=eigval, eigvec=eigvec)
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, src):
        """Augmenter body"""
        alpha = np.random.normal(0, self.alphastd, size=(3,))
        rgb = np.dot(self.eigvec * alpha, self.eigval)
        src += nd.array(rgb)
        return src


class ColorNormalizeAug(Augmenter):
    """Mean and std normalization.

    Parameters
    ----------
    mean : NDArray
        RGB mean to be subtracted
    std : NDArray
        RGB standard deviation to be divided
    """
    def __init__(self, mean, std):
        super(ColorNormalizeAug, self).__init__(mean=mean, std=std)
        self.mean = mean if mean is None or isinstance(mean, nd.NDArray) else nd.array(mean)
        self.std = std if std is None or isinstance(std, nd.NDArray) else nd.array(std)

    def __call__(self, src):
        """Augmenter body"""
        return color_normalize(src, self.mean, self.std)


class RandomGrayAug(Augmenter):
    """Randomly convert to gray image.

    Parameters
    ----------
    p : float
        Probability to convert to grayscale
    """
    def __init__(self, p):
        super(RandomGrayAug, self).__init__(p=p)
        self.p = p
        self.mat = nd.array([[0.21, 0.21, 0.21],
                             [0.72, 0.72, 0.72],
                             [0.07, 0.07, 0.07]])

    def __call__(self, src):
        """Augmenter body"""
        if random.random() < self.p:
            src = nd.dot(src, self.mat)
        return src


class HorizontalFlipAug(Augmenter):
    """Random horizontal flip.

    Parameters
    ----------
    p : float
        Probability to flip image horizontally
    """
    def __init__(self, p):
        super(HorizontalFlipAug, self).__init__(p=p)
        self.p = p

    def __call__(self, src):
        """Augmenter body"""
        if random.random() < self.p:
            src = nd.flip(src, axis=1)
        return src


class CastAug(Augmenter):
    """Cast to float32"""
    def __init__(self, typ='float32'):
        super(CastAug, self).__init__(type=typ)
        self.typ = typ

    def __call__(self, src):
        """Augmenter body"""
        src = src.astype(self.typ)
        return src


def CreateAugmenter(data_shape, rand_mirror=False, mean=None, std=None, brightness=0, contrast=0,
                    saturation=0, hue=0, pca_noise=0, rand_gray=0, inter_method=2):
    """Creates an augmenter list.

    Parameters
    ----------
    data_shape : tuple of int
        Shape for output data
    rand_gray : float
        [0, 1], probability to convert to grayscale for all channels, the number
        of channels will not be reduced to 1
    rand_mirror : bool
        Whether to apply horizontal flip to image with probability 0.5
    mean : np.ndarray or None
        Mean pixel values for [r, g, b]
    std : np.ndarray or None
        Standard deviations for [r, g, b]
    brightness : float
        Brightness jittering range (percent)
    contrast : float
        Contrast jittering range (percent)
    saturation : float
        Saturation jittering range (percent)
    hue : float
        Hue jittering range (percent)
    pca_noise : float
        Pca noise level (percent)

    Examples
    --------
    >>> # An example of creating multiple augmenters
    >>> augs = mx.image.CreateAugmenter(data_shape=(3, 300, 300), rand_mirror=True,
    ...    mean=True, brightness=0.125, contrast=0.125, rand_gray=0.05,
    ...    saturation=0.125, pca_noise=0.05, inter_method=10)
    """
    auglist = []

    if rand_mirror:
        auglist.append(HorizontalFlipAug(0.5))

    auglist.append(CastAug())

    if brightness or contrast or saturation:
        auglist.append(ColorJitterAug(brightness, contrast, saturation))

    if hue:
        auglist.append(HueJitterAug(hue))

    if pca_noise > 0:
        eigval = np.array([55.46, 4.794, 1.148])
        eigvec = np.array([[-0.5675, 0.7192, 0.4009],
                           [-0.5808, -0.0045, -0.8140],
                           [-0.5836, -0.6948, 0.4203]])
        auglist.append(LightingAug(pca_noise, eigval, eigvec))

    if rand_gray > 0:
        auglist.append(RandomGrayAug(rand_gray))

    if mean is True:
        mean = nd.array([123.68, 116.28, 103.53])
    elif mean is not None:
        assert isinstance(mean, (np.ndarray, nd.NDArray)) and mean.shape[0] in [1, 3]

    if std is True:
        std = nd.array([58.395, 57.12, 57.375])
    elif std is not None:
        assert isinstance(std, (np.ndarray, nd.NDArray)) and std.shape[0] in [1, 3]

    if mean is not None or std is not None:
        auglist.append(ColorNormalizeAug(mean, std))

    return auglist


################################################################################
