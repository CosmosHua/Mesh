# coding:utf-8
#!/usr/bin/python3

import os, cv2
import numpy as np
import tensorflow as tf

from ops import *
from utils import *
from time import time


################################################################################
class pix2pix(object):
    def __init__(self, sess, args):
        """ Args: sess: TensorFlow session."""
        self.sess = sess; self.args = args
        
        self.gf_dim = 64; self.df_dim = 64
        self.input_nc = 3; self.output_nc = 3
        self.input_size = 256; self.output_size = 256
        
        self.batch_size = args.batch_size # default=1
        self.L1_lambda = args.L1_lambda # default=200
        self.SS_lambda = args.SS_lambda # accelerate training
        self.drop_kp = 0.8 if args.phase=="train" else 1 # 0.5
        self.isTrain = True #args.phase=="train" # bug of bn?
        
        self.naming = args.mesh_name # default="*_*.jpg"
        self.kp = args.model_num # max number of models
        self.ps = [] # store [PNSR, SSIM, PNSR*SSIM]
        
        self.build_model() # initialize model
        if  "train" in args.phase:  self.train(args) # Train
        elif "test" in args.phase:  self.test(args)  # Test
        elif "infer" in args.phase: self.infer(args) # Infer


    def build_model(self):
        shape = [self.batch_size, self.input_size, self.input_size, self.input_nc+self.output_nc]
        self.real_data = tf.placeholder(dtype=TType, shape=shape, name='real_A_and_B_images')
        
        self.real_A = self.real_data[:, :, :, self.input_nc:] # mesh image
        self.real_B = self.real_data[:, :, :, :self.input_nc] # clean image
        self.fake_B = self.generator(self.real_A) # recover image
        
        #if self.input_size==250: # help(tf.pad): [batch_size,height,width,channels]
        #   paddings = tf.constant([[0, 0], [3, 3], [3, 3], [0, 0]])
        #   self.real_A = tf.pad(self.real_A, paddings=paddings, mode='REFLECT')
        #   self.real_B = tf.pad(self.real_B, paddings=paddings, mode='REFLECT')
        
        self.real_AB = tf.concat([self.real_A, self.real_B], 3)
        self.fake_AB = tf.concat([self.real_A, self.fake_B], 3)

        self.D, self.D_logits = self.discriminator(self.real_AB) # reuse=False
        self.D_, self.D_logits_ = self.discriminator(self.fake_AB) # reuse=True
        
        # self.d_sum = tf.summary.histogram("d", self.D) # unnecessary
        # self.d__sum = tf.summary.histogram("d_", self.D_) # unnecessary
        # self.fake_B_sum = tf.summary.image("fake_B", self.fake_B) # unnecessary
        
        SCE_L = tf.nn.sigmoid_cross_entropy_with_logits
        self.d_loss_real = tf.reduce_mean(SCE_L(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(SCE_L(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.d_loss = self.d_loss_real + self.d_loss_fake # L1_lambda=100 if (d_loss/=2) else 200
        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
       
        self.g_loss_gan = tf.reduce_mean(SCE_L(logits=self.D_logits_, labels=tf.ones_like(self.D_)))
        self.g_loss_l1  = tf.reduce_mean(tf.abs(self.real_B - self.fake_B)) * self.L1_lambda
        self.g_loss_ss  = SSLF(self.real_B, self.fake_B) * self.SS_lambda # SSIM loss
        self.g_loss = self.g_loss_gan + self.g_loss_l1 + self.g_loss_ss
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        
        var_list = tf.trainable_variables()
        self.g_vars = [x for x in var_list if 'g_' in x.name] # generator
        self.d_vars = [x for x in var_list if 'd_' in x.name] # discriminator
        
        var_list = tf.global_variables() # for batch_norm
        bn_moving_vars =  [x for x in var_list if 'moving_mean' in x.name]
        bn_moving_vars += [x for x in var_list if 'moving_variance' in x.name]
        self.g_vars += [x for x in bn_moving_vars if 'g_bn' in x.name] # generator
        self.d_vars += [x for x in bn_moving_vars if 'd_bn' in x.name] # discriminator
        
        self.saver = tf.train.Saver(max_to_keep=self.kp) # var_list=self.d_vars+self.g_vars


    def discriminator(self, image, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("discriminator", reuse=reuse) as scope:
            marker = "fake_" if reuse else "real_"
            
            bn_train = self.isTrain; ReLU = lrelu # lrelu
            # image is (256 x 256 x (input_nc+output_nc))
            # h1 is (128 x 128 x self.df_dim)
            h1 = ReLU(conv2d(image, self.df_dim, name='d_h1_conv'))
            
            # h2 is (64 x 64 x self.df_dim*2)
            h2 = conv2d(h1, self.df_dim*2, name='d_h2_conv')
            h2 = ReLU(batch_norm(h2, bn_train, name='d_bn2'))
            
            # h3 is (32x 32 x self.df_dim*4)
            h3 = conv2d(h2, self.df_dim*4, name='d_h3_conv')
            h3 = ReLU(batch_norm(h3, bn_train, name='d_bn3'))
            
            # h4 is (16 x 16 x self.df_dim*8)
            h4 = conv2d(h3, self.df_dim*8, name='d_h4_conv')
            h4 = ReLU(batch_norm(h4, bn_train, name='d_bn4'))
            
            # Method 1:
            #h5 = linear(tf.reshape(h4, [self.batch_size,-1]), output=1, name='d_h5_lin')
            # Method 2:
            #h5 = conv2d(h4, 1, k_h=16, k_w=16, d_h=16, d_w=16, name='d_h5_conv')
            
            # h5 is (16 x 16 x self.df_dim*4)
            h5 = conv2d(h4, self.df_dim*4, k_h=3, k_w=3, d_h=1, d_w=1, name='d_h5_conv')
            h5 = ReLU(tf.nn.dropout(batch_norm(h5, bn_train, name='d_bn5'), self.drop_kp))
            
            # h5 is (8 x 8 x 1)->patch
            h6 = conv2d(h5, 1, k_h=2, k_w=2, d_h=2, d_w=2, name='d_h6_conv')
            h6 = tf.reduce_min(h6, axis=(1,2,3)) # axis=0: batch_size
            
            #print("\n" + (marker+"discriminator_")*3)
            #for i in (h1, h2, h3, h4, h5, h6): print(i)
            return tf.nn.sigmoid(h6), h6


    def generator(self, image, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("generator", reuse=reuse) as scope:
            s = self.output_size; bn_train = self.isTrain
            s2, s4, s8, s16, s32, s64, s128 = [int(s/2**i) for i in range(1,8)]
            # scope.reuse_variables() # for sampler

            ReLU = lrelu # default=lrelu
            # image is (256 x 256 x input_nc)
            # e1 is (128 x 128 x self.gf_dim)
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            # e2 is (64 x 64 x self.gf_dim*2)
            e2 = batch_norm(conv2d(ReLU(e1), self.gf_dim*2, name='g_e2_conv'), bn_train, name="g_bn_e2")
            # e3 is (32 x 32 x self.gf_dim*4)
            e3 = batch_norm(conv2d(ReLU(e2), self.gf_dim*4, name='g_e3_conv'), bn_train, name="g_bn_e3")
            # e4 is (16 x 16 x self.gf_dim*8)
            e4 = batch_norm(conv2d(ReLU(e3), self.gf_dim*8, name='g_e4_conv'), bn_train, name="g_bn_e4")
            # e5 is (8 x 8 x self.gf_dim*8)
            e5 = batch_norm(conv2d(ReLU(e4), self.gf_dim*8, name='g_e5_conv'), bn_train, name="g_bn_e5")
            # e6 is (4 x 4 x self.gf_dim*8)
            e6 = batch_norm(conv2d(ReLU(e5), self.gf_dim*8, name='g_e6_conv'), bn_train, name="g_bn_e6")
            # e7 is (2 x 2 x self.gf_dim*8)
            e7 = batch_norm(conv2d(ReLU(e6), self.gf_dim*8, name='g_e7_conv'), bn_train, name="g_bn_e7")
            # e8 is (1 x 1 x self.gf_dim*8)
            e8 = batch_norm(conv2d(ReLU(e7), self.gf_dim*8, name='g_e8_conv'), bn_train, name="g_bn_e8")
            
            ReLU = Lrelu # default=tf.nn.relu
            # d1 is (2 x 2 x self.gf_dim*8*2)
            shape = [self.batch_size, s128, s128, self.gf_dim*8]
            d1, d1_w, d1_b = deconv2d(ReLU(e8), shape, name='g_d1')
            d1 = tf.nn.dropout(batch_norm(d1, bn_train, name="g_bn_d1"), self.drop_kp)
            d1 = tf.concat([d1, e7], 3)

            # d2 is (4 x 4 x self.gf_dim*8*2)
            shape = [self.batch_size, s64, s64, self.gf_dim*8]
            d2, d2_w, d2_b = deconv2d(ReLU(d1), shape, name='g_d2')
            d2 = tf.nn.dropout(batch_norm(d2, bn_train, name="g_bn_d2"), self.drop_kp)
            d2 = tf.concat([d2, e6], 3)

            # d3 is (8 x 8 x self.gf_dim*8*2)
            shape = [self.batch_size, s32, s32, self.gf_dim*8]
            d3, d3_w, d3_b = deconv2d(ReLU(d2), shape, name='g_d3')
            d3 = tf.nn.dropout(batch_norm(d3, bn_train, name="g_bn_d3"), self.drop_kp)
            d3 = tf.concat([d3, e5], 3)

            # d4 is (16 x 16 x self.gf_dim*8*2)
            shape = [self.batch_size, s16, s16, self.gf_dim*8]
            d4, d4_w, d4_b = deconv2d(ReLU(d3), shape, name='g_d4')
            d4 = tf.nn.dropout(batch_norm(d4, bn_train, name="g_bn_d4"), self.drop_kp)
            d4 = tf.concat([d4, e4], 3)

            # d5 is (32 x 32 x self.gf_dim*4*2)
            shape = [self.batch_size, s8, s8, self.gf_dim*4]
            d5, d5_w, d5_b = deconv2d(ReLU(d4), shape, name='g_d5')
            d5 = tf.concat([batch_norm(d5, bn_train, name="g_bn_d5"), e3], 3)

            # d6 is (64 x 64 x self.gf_dim*2*2)
            shape = [self.batch_size, s4, s4, self.gf_dim*2]
            d6, d6_w, d6_b = deconv2d(ReLU(d5), shape, name='g_d6')
            d6 = tf.concat([batch_norm(d6, bn_train, name="g_bn_d6"), e2], 3)

            # d7 is (128 x 128 x self.gf_dim*1*2)
            shape = [self.batch_size, s2, s2, self.gf_dim]
            d7, d7_w, d7_b = deconv2d(ReLU(d6), shape, name='g_d7')
            d7 = tf.concat([batch_norm(d7, bn_train, name="g_bn_d7"), e1], 3)

            # d8 is (256 x 256 x output_nc)
            shape = [self.batch_size, s, s, self.output_nc]
            d8, d8_w, d8_b = deconv2d(ReLU(d7), shape, name='g_d8')
            
            #print("\n" + "generator_"*6)
            #for i in (e1, e2, e3, e4, e5, e6, e7, e8): print(i)
            #for i in (d1, d2, d3, d4, d5, d6, d7, d8): print(i)
            return tf.nn.tanh(d8)


    def train(self, args): # Train pix2pix
        counter = 0; start_time = time()

        self.g_sum = tf.summary.merge([self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.d_loss_real_sum, self.d_loss_sum])
        # self.g_sum = tf.summary.merge([self.d__sum, self.fake_B_sum, self.d_loss_fake_sum, self.g_loss_sum])
        # self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        # self.writer = tf.summary.FileWriter("./logs", self.sess.graph)
        
        d_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1)
        g_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1)
        #d_optim = tf.train.MomentumOptimizer(1E-2, momentum=0.9, use_nesterov=True)
        #g_optim = tf.train.MomentumOptimizer(1E-2, momentum=0.9, use_nesterov=True)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for batch_norm
        with tf.control_dependencies(update_ops):
            d_optim = d_optim.minimize(self.d_loss, var_list=self.d_vars)
            g_optim = g_optim.minimize(self.g_loss, var_list=self.g_vars)

        self.sess.run(tf.global_variables_initializer())
        self.load_model(args.model_dir) # restore model
        #self.validate_model(args, counter) # eval initial model
        for epoch in range(args.epoch):
            # load data: can be changed in different epoch
            data = globs(args.train_dir, self.naming)
            batch_num = min(len(data), args.train_num) // self.batch_size
            print("\n", "#"*50, "\nThe Number of Training =", len(data))
            #iter_num = batch_num*args.epoch # max of counter
            np.random.shuffle(data) # shuffle data
            
            for id in range(batch_num):
                batch = data[id*self.batch_size:(id+1)*self.batch_size]
                images = self.load_data(batch, isTrain=True) # [B,H,W,C]
                if type(images)==bool: continue # discard this batch

                # Update D network:
                _, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict={self.real_data: images})
                # self.writer.add_summary(summary_str, counter)

                # Update G network:
                _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={self.real_data: images})
                # self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={self.real_data: images})
                # self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.real_data: images})
                errD_real = self.d_loss_real.eval({self.real_data: images})
                errG = self.g_loss.eval({self.real_data: images})
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" % \
                      (epoch, id, batch_num, time()-start_time, errD_fake+errD_real, errG))

                counter += 1
                if counter%500 == 0: self.validate_model(args, counter) # save model
                #if counter % 500 == 0: self.sample_model(args, counter) # monitor
                #if counter % 1000 == 0: self.save_model(args.model_dir, counter)
        if counter%500!=0: self.validate_model(args, counter) # test/save final model


    def test(self, args): # Test pix2pix
        data = globs(args.test_dir, "*.jpg")
        print("\n"+"#"*50+"\nTesting = %d images" % len(data))
        
        start_time = time(); batch = self.batch_size
        #self.sess.run(tf.global_variables_initializer())
        if not self.load_model(args.model_dir): return # load model
        print("Load time: %f s" % (time()-start_time)); load_time = time()
        
        data = [data[i:i+batch] for i in range(0,len(data),batch)]
        for i,image in enumerate(data): # More I/O time cost
            im = self.load_data(image, isTrain=False) # [B,H,W,C]
            im = self.sess.run(self.fake_B, feed_dict={self.real_data: im})
            save_images(im, image[0][:-4]+"_.png", self.sz[0]) # save png
        print("Test time: %f s" % (time()-load_time))
        #if batch<2: print("[PSNR SSIM] =", BatchPS(args.test_dir))


    def infer(self, args): # batch_size=1
        data = args.test_dir # input images: path/np.array
        if type(data)!=str or os.path.isfile(data): data = [data]
        elif os.path.isdir(data): data = globs(data, "*.jpg")
        else: data = []; print("No Images!\n"); return
        if not self.load_model(args.model_dir): return # load model
        for i,image in enumerate(data): # batch_size=1
            im = self.load_data([image], isTrain=False); sz = self.sz[0]
            #im, sz = load_image(image, isTrain=False) # load [H,W,C]
            #im = im[None,:].astype(np.float32) # insert dim->[1,H,W,C]
            im = self.sess.run(self.fake_B, feed_dict={self.real_data: im})
            if type(image)==str: save_images(im, image[:-4]+"_.png", sz)
            else: im = rsz(127.5*(im[0]+1.0), sz); return im.astype("uint8")


    # similiar to test: save model, without load_model
    def validate_model(self, args, counter, id=2): # validation
        if self.batch_size>1: # BatchPS require batch=1
            self.save_model(args.model_dir, counter); return
        
        mesh = globs(args.val_dir, self.naming)
        # insert batch_dim=1 after axis=0 (where None->1)
        images = self.load_data(mesh, isTrain=False)[:,None] # [N,B,H,W,C]
        for i,image in enumerate(images): # output *.png to args.val_dir
            im = self.sess.run(self.fake_B, feed_dict={self.real_data: image})
            save_images(im, mesh[i][:-4]+"_.png", self.sz[i]) # batch_size=1
        
        ps = self.ps; out = "" # ps is reference of self.ps
        pp = BatchPS(args.val_dir, self.naming) # [PNSR, SSIM, PNSR*SSIM]
        
        # Other Strategy: {mean/max/min/...} + {[id]/all/any/...}
        if len(ps)<2: out = "@" # enqueue: initialize ps[:2]
        elif (pp>np.mean(ps,axis=0))[id]: # mean/max/min, [id]/all/any
            if len(ps)<self.kp: out = "@" # enqueue if underfill
            elif (pp>ps[0])[id]: ps.pop(0); out = "@" # dequeue oldest
        '''
        if len(ps)<2: out = "@" # enqueue: initialize ps[:2]
        elif len(ps)<self.kp: # enqueue: when exceed mean/max/min/...
            if (pp>np.mean(ps,axis=0))[id]: out = "@" # [id]/all()/any()
        else: # dequeue: when exceed the oldest
            if (pp>ps[0])[id]: ps.pop(0); out = "@" # [id]/all()/any()
        '''
        if counter<5: out = "" # ignore initial models
        if out!="": ps.append(pp); self.save_model(args.model_dir,counter)
        
        ff = os.path.basename(args.train_dir)+".ps"
        out += "num=%d\t[PSNR SSIM] = %s\n" % (counter, str(pp))
        with open(ff,"a+") as f: f.write(out) # append+read-write
        print(out); return pp


    def save_model(self, model_dir, step):
        model_name = os.path.join(model_dir, "pix2pix")
        self.saver.save(self.sess, model_name, global_step=step)


    def load_model(self, model_dir):
        print(" [*] Loading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            model = os.path.join(model_dir, ckpt_name)
            self.saver.restore(self.sess, model)
            print(" [*] Load SUCCESS: %s" % model); return True
        else: print(" [!] Load FAIL!"); return False


    def sample_model(self, args, counter): # record train
        data = globs(args.val_dir, self.naming)
        data = np.random.choice(data, self.batch_size)
        images = self.load_data(data, isTrain=True) # [B,H,W,C]
        if type(images)==bool: return # discard this batch
        im, d_loss, g_loss = self.sess.run([self.fake_B, self.d_loss, self.g_loss], feed_dict={self.real_data: images})
        save_images(im, './{}/train_{:08d}.png'.format(args.val_dir, counter), self.sz[0])
        print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))


    def load_data(self, data, isTrain): # ->[B,H,W,C]
        image, self.sz = [],[]; size = (self.input_size,)*2
        for i,im in enumerate(data): # load images to batch
            img = load_image(im, size, isTrain, self.naming)
            if type(img)!=tuple: return False # discard the batch
            else: image.append(img[0]); self.sz.append(img[1])
        return np.array(image).astype(np.float32)

