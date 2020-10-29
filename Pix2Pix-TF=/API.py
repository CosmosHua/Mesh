# coding:utf-8
#!/usr/bin/python3
from model import *


################################################################################
class P2P(object):
    def __init__(self, model="./checkpoint", sess=None):
        assert os.path.isdir(model); self.model_dir = model
        self.batch_size = 1; self.phase = ""; self.test_dir = ""
        self.train_dir = ""; self.val_dir = ""; self.mesh_name = "*_*.jpg"
        self.L1_lambda = 200; self.SS_lambda = 3; self.lr = 2E-4; self.beta1 = 0.5
        self.epoch = 0; self.train_num = 0; self.model_num = 0
        
        # device_count: limit valid GPU number; soft_placement: auto-assign GPU
        config = tf.ConfigProto(device_count={"GPU":1}, allow_soft_placement=True)
        #config.gpu_options.per_process_gpu_memory_fraction = 0.1 # gpu_memory ratio
        config.gpu_options.allow_growth = True # dynamicly apply gpu_memory
        self.sess = sess if type(sess)==tf.Session else tf.Session(config=config)


    def infer(self, images):
        self.test_dir = images; # self.phase = "infer"
        p2p = pix2pix(self.sess, self); return p2p.infer(self)


    def infer2(self, images):
        if os.path.isdir(images): data = globs(images, "*.jpg")
        else: data = [images] # images = np.array or image_path
        p2p = pix2pix(self.sess, self) # initialize model
        if not p2p.load_model(self.model_dir): return # load model
        for i, image in enumerate(data): # batch_size=1, p2p.sz[0]
            im = p2p.load_data([image], isTrain=False); sz = (178,220)
            im = p2p.sess.run(p2p.fake_B, feed_dict={p2p.real_data: im})
            if type(image)==str: save_images(im, image[:-4]+"_.png", sz)
            else: return rsz(127.5*(im[0]+1.0), sz).astype("uint8")


#os.environ["CUDA_VISIBLE_DEVICES"]="" # assign/disable GPUs
################################################################################
if __name__ == '__main__':
    from sys import argv; api = P2P()
    for im in argv[1:]: api.infer2(im)

