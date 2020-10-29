# coding:utf-8
# !/usr/bin/python3

import ops, utils
import tensorflow as tf
from reader import Reader
from discriminator import Discriminator
from generator_G import Generator_G
from generator_F import Generator_F

REAL_LABEL = 0.95  # ��ǩʵ��ֵ, ����������ֵΪ=1


class DFGAN:
    def __init__(self, X_train_file='', Y_train_file='', Z_train_file='',
                 batch_size=1, image_size=[220, 178], use_lsgan=True,
                 norm='instance', lambda1=10.0, lambda2=10.0,
                 learning_rate=2e-4, beta1=0.5, ngf=16 ):
        """
        Args:
          X_train_file: string, X tfrecords file for training
          Y_train_file: string Y tfrecords file for training
          Z_train_file: string Z tfrecords file for training
          batch_size: integer, batch size
          image_size: list with two integer, image size
          lambda1: integer, weight for forward cycle loss (X->Y->X)
          lambda2: integer, weight for backward cycle loss (Y->X->Y)
          use_lsgan: boolean
          norm: 'instance' or 'batch'
          learning_rate: float, initial learning rate for Adam
          beta1: float, momentum term of Adam
          ngf: number of gen filters in first conv layer
        """

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.use_lsgan = use_lsgan
        use_sigmoid = not use_lsgan
        self.batch_size = batch_size
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.X_train_file = X_train_file
        self.Y_train_file = Y_train_file
        self.Z_train_file = Z_train_file
        self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')

        self.G = Generator_G('G', self.is_training, ngf=ngf, norm=norm, image_size=image_size)  # ��������G
        self.F = Generator_F('F', self.is_training, ngf=ngf, norm=norm, image_size=image_size)  # �ϳ�����F

        self.D_X = Discriminator('D_X',  # X���б���
                                 self.is_training, norm=norm, use_sigmoid=use_sigmoid)
        self.D_Y = Discriminator('D_Y',  # Y���б���
                                 self.is_training, norm=norm, use_sigmoid=use_sigmoid)
        self.D_Z = Discriminator('D_Z',  # Z���б���
                                 self.is_training, norm=norm, use_sigmoid=use_sigmoid)

        self.fake_x = tf.placeholder(tf.float32, shape=[batch_size, image_size[0], image_size[1], 3])
        self.fake_y = tf.placeholder(tf.float32, shape=[batch_size, image_size[0], image_size[1], 3])
        self.fake_z = tf.placeholder(tf.float32, shape=[batch_size, image_size[0], image_size[1], 3])

    def model(self):

        # ��ȡ����õ�tfrecords���ݣ���ת����tensor
        X_reader = Reader(self.X_train_file, name='X',
                          image_size=self.image_size, batch_size=self.batch_size)
        Y_reader = Reader(self.Y_train_file, name='Y',
                          image_size=self.image_size, batch_size=self.batch_size)
        Z_reader = Reader(self.Z_train_file, name='Z',
                          image_size=self.image_size, batch_size=self.batch_size)

        x = X_reader.feed()
        y = Y_reader.feed()
        z = Z_reader.feed()

        cycle_loss = self.cycle_consistency_loss(self.G, self.F, x, y, z)  # ѭ��һ����ʧ

        # **************************************************************************************************************
        # X -> Y,Z
        fake_y, fake_z = self.G(x)  # ��������G���ɵ�ȥ����ͼ���Լ�����ͼ��

        G_gan_loss_y = self.generator_loss(self.D_Y, fake_y)  # Y��������ʧ����¼��tensorboard�в鿴��
        G_gan_loss_z = self.generator_loss(self.D_Z, fake_z)  # Z��������ʧ����¼��tensorboard�в鿴��
        G_loss_y = G_gan_loss_y + cycle_loss  # ʵ��ѵ���Ż���Y��������ʧ
        G_loss_z = G_gan_loss_z + cycle_loss  # ʵ��ѵ���Ż���Z��������ʧ
        D_Y_loss, D_Z_loss = self.discriminator_y_z_loss(self.D_Y, self.D_Z, y, self.fake_y, z, self.fake_z)  # Y��Z���б���ʧ

        # Y, Z -> X
        fake_x = self.F(y, z)  # �ϳ�����F�ϳɵĴ�����ͼ��

        F_gan_loss_x = self.generator_loss(self.D_X, fake_x)  # X��������ʧ����¼��tensorboard�в鿴��
        F_loss_x = F_gan_loss_x + cycle_loss  # ʵ��ѵ���Ż���X��������ʧ
        D_X_loss = self.discriminator_x_loss(self.D_X, x, self.fake_x)  # X���б���ʧ

        # summary: �ܽᵽtensorboard�в鿴
        tf.summary.histogram('D_Y/true', self.D_Y(y))
        tf.summary.histogram('D_Y/fake', self.D_Y(fake_y))
        tf.summary.histogram('D_Z/true', self.D_Z(z))
        tf.summary.histogram('D_Z/fake', self.D_Y(fake_z))
        tf.summary.histogram('D_X/true', self.D_X(x))
        tf.summary.histogram('D_X/fake', self.D_X(fake_x))

        tf.summary.scalar('loss/G_Y', G_gan_loss_y)
        tf.summary.scalar('loss/D_Y', D_Y_loss)
        tf.summary.scalar('loss/G_Z', G_gan_loss_z)
        tf.summary.scalar('loss/D_Z', D_Z_loss)
        tf.summary.scalar('loss/F_X', F_gan_loss_x)
        tf.summary.scalar('loss/D_X', D_X_loss)
        tf.summary.scalar('loss/cycle', cycle_loss)

        tf.summary.image('X/generatedY', utils.batch_convert2int(fake_y))
        tf.summary.image('X/generatedZ', utils.batch_convert2int(fake_z))
        y_hat, z_hat = self.G(x)
        tf.summary.image('X/reconstruction', utils.batch_convert2int(self.F(y_hat, z_hat)))

        tf.summary.image('Y_Z/generated', utils.batch_convert2int(self.F(y, z)))
        generatey, generatez = self.G(self.F(y, z))
        tf.summary.image('Y_Z/reconstructionY', utils.batch_convert2int(generatey))
        tf.summary.image('Y_Z/reconstructionZ', utils.batch_convert2int(generatez))

        return G_loss_y, G_loss_z, D_Y_loss, D_Z_loss, F_loss_x, D_X_loss, fake_y, fake_z, fake_x


    def optimize(self, G_loss_y, G_loss_z, D_Y_loss, D_Z_loss, F_loss_x, D_X_loss):
        def make_optimizer(loss, variables, name='Adam'):
            """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
                  and a linearly decaying rate that goes to zero over the next 100k steps
            """
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = self.learning_rate
            end_learning_rate = 0.0
            start_decay_step = 100000   # default: 50 epochs
            decay_steps = 100000    # default: 50 epochs
            beta1 = self.beta1
            learning_rate = (
                tf.where(
                    tf.greater_equal(global_step, start_decay_step),
                    tf.train.polynomial_decay(starter_learning_rate, global_step-start_decay_step,
                                              decay_steps, end_learning_rate,
                                              power=1.0),
                    starter_learning_rate
                )
            )
            tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

            learning_step = (
                tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
                        .minimize(loss, global_step=global_step, var_list=variables)
            )
            return learning_step

        G_optimizer_y = make_optimizer(G_loss_y, self.G.variables, name='Adam_G_Y')
        G_optimizer_z = make_optimizer(G_loss_z, self.G.variables, name='Adam_G_Z')
        D_Y_optimizer = make_optimizer(D_Y_loss, self.D_Y.variables, name='Adam_D_Y')
        D_Z_optimizer = make_optimizer(D_Z_loss, self.D_Z.variables, name='Adam_D_Z')
        F_optimizer = make_optimizer(F_loss_x, self.F.variables, name='Adam_F_X')
        D_X_optimizer = make_optimizer(D_X_loss, self.D_X.variables, name='Adam_D_X')

        # �Կ�ѵ�����ԣ�˳�򲻿����׽���
        with tf.control_dependencies([G_optimizer_y, D_Y_optimizer, G_optimizer_z, D_Z_optimizer, F_optimizer, D_X_optimizer]):
            return tf.no_op(name='optimizers')

    def discriminator_x_loss(self, D, x, fake_x):
        #x���б���ʧ
        #:param D: �б�����x
        #:param x: x������
        #:param fake_x: �ϳ�����F�ϳɵ�����
        #:return: �б���ʧ
        
        error_real = tf.reduce_mean(tf.squared_difference(D(x), REAL_LABEL))
        error_fake = tf.reduce_mean(tf.square(D(fake_x)))
        loss = (error_real + error_fake) / 2
        return loss

    def discriminator_y_z_loss(self, D_y, D_z, y, fake_y, z, fake_z):
        #y��z���б���ʧ
        #:param D_y: �б�����y
        #:param D_z: �б�����z
        #:param y: y������
        #:param z: z������
        #:param fake_y: ��������G�������y������
        #:param fake_z: ��������G�������z������
        #:return: �б���ʧ

        error_real_y = tf.reduce_mean(tf.squared_difference(D_y(y), REAL_LABEL))
        error_real_z = tf.reduce_mean(tf.squared_difference(D_z(z), REAL_LABEL))
        error_fake_y = tf.reduce_mean(tf.square(D_y(fake_y)))
        error_fake_z = tf.reduce_mean(tf.square(D_z(fake_z)))
        loss_y = (error_real_y + error_fake_y) / 2
        loss_z = (error_real_z + error_fake_z) / 2
        return loss_y, loss_z

    def generator_loss(self, D, fake):
        #������ʧ
        #:param D: �б���
        #:param fake: ���������ɵ�ͼ��
        #:return: ������ʧ

        loss = tf.reduce_mean(tf.squared_difference(D(fake), REAL_LABEL))
        return loss

    def cycle_consistency_loss(self, G, F, x, y, z):
        #ѭ��һ����ʧ(���õ� L1 ����)
        #differce_loss���¼������ʧ��ͼ�����ز���ʧ��

        y_hat, z_hat = G(x)
        forward_loss = tf.reduce_mean(tf.abs(F(y_hat, z_hat) - x))
        y_, z_ = G(F(y, z))
        backward_loss = tf.reduce_mean(tf.abs(y_ - y) + tf.abs(z_ - z))
        differce_loss = tf.reduce_mean(tf.abs(y_hat - (1-z_hat) - x))  # ������ʧ����
        loss = self.lambda1*forward_loss + self.lambda2*backward_loss + 10.0*differce_loss
        return loss

