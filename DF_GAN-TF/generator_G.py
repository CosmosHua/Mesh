import ops, utils
import tensorflow as tf


class Generator_G:
    def __init__(self, name, is_training, ngf=32, norm='instance', image_size=[220, 178]):
        self.name = name
        self.reuse = False
        self.ngf = ngf
        self.norm = norm
        self.is_training = is_training
        self.image_size = image_size

    def __call__(self, input_x):
        """
        Args:
            ngf: number of gen filters in first conv layer
            input_x: batch_size x height x width x 3 (domain x)
        Returns:
            output: two picture of same size (domain y and domain z)
        """
        with tf.variable_scope(self.name):
            # 卷积网络
            c7s1_32_x = ops.c7s1_k(input_x, self.ngf, is_training=self.is_training, norm=self.norm,
                reuse=self.reuse, name='c7s1_32_x')                             # (?, h, w, 32)   [220,178,32]
            d64_x = ops.dk_pad(c7s1_32_x, 2*self.ngf, is_training=self.is_training, norm=self.norm,
                reuse=self.reuse, name='d64_x')                                 # (?, h/2, w/2, 64)  [110,90,64]
            d128_x = ops.dk(d64_x, 4*self.ngf, is_training=self.is_training, norm=self.norm,
                reuse=self.reuse, name='d128_x')                                # (?, h/4, w/4, 128)  [55,45,128]
            s256_x = ops.sk(d128_x, 8*self.ngf, is_training=self.is_training, norm=self.norm,
                          reuse=self.reuse, name='s256_x')                      # (?, h/4, w/4, 256)  [55,45,256]

            # 分离结构：[batch, 55, 45, 256] --> [batch, 55, 45, 255], [batch, 55, 45, 1]
            split_y = tf.slice(s256_x, [0, 0, 0, 0],  # tf.slice(inputs, begin, size, name)
                               [-1, s256_x.get_shape().as_list()[1],
                                s256_x.get_shape().as_list()[2],
                                                      s256_x.get_shape().as_list()[3]-5])   # [?, 55, 45, 255]

            split_z = tf.slice(s256_x, [0, 0, 0, s256_x.get_shape().as_list()[3]-5],
                               [-1, s256_x.get_shape().as_list()[1], s256_x.get_shape().as_list()[2], 5])

            # **********************************************************************************************************
            # 使用5个残差块
            res_output_y = ops.n_res_blocks(split_y, reuse=self.reuse, n=5)      # (?, h/4, w/4, 255)(?, 55, 45, 255)

            u128_y = ops.sk(res_output_y, 4 * self.ngf, is_training=self.is_training, norm=self.norm,
                          reuse=self.reuse, name='u128_y')                       # (?, h/4, w/4, 128)(?, 55, 45, 128)

            # 反卷积（解码）
            u64_y = ops.uk(u128_y, 2*self.ngf, is_training=self.is_training, norm=self.norm,
                reuse=self.reuse, name='u64_y')                                  # (?, h/2, w/2, 64)(?, 110, 90, 64)

            u32_y = ops.uk(u64_y, self.ngf, is_training=self.is_training, norm=self.norm,
                reuse=self.reuse, name='u32_y')                                  # (?, h, w, 32)    (?, 220, 180, 64)
            # 还原图像大小
            clipping_y = tf.slice(u32_y, [0, 0, 1, 0], [u32_y.get_shape().as_list()[0],
                                                      u32_y.get_shape().as_list()[1], u32_y.get_shape().as_list()[2]-2,
                                                      u32_y.get_shape().as_list()[3]])

            output_y = ops.c7s1_k(clipping_y, 3, norm=None,
                activation='tanh', reuse=self.reuse, name='output_y')           # (?, h, w, 3)

            # **********************************************************************************************************
            # 网纹使用1个残差块
            res_output_z = ops.n_res_blocks(split_z, reuse=self.reuse, n=1)         # (?, h/4, w/4, 255)(4, 55, 45, 1)

            u128_z = ops.sk(res_output_z, 4 * self.ngf, is_training=self.is_training, norm=self.norm,
                          reuse=self.reuse, name='u128_z')                          # (?, h/4, w/4, 128)(4, 55, 45, 128)

            # 反卷积（解码）
            u64_z = ops.uk(u128_z, 2*self.ngf, is_training=self.is_training, norm=self.norm,
                reuse=self.reuse, name='u64_z')                                 # (?, h/2, w/2, 64)(?, 110, 90, 64)
            u32_z = ops.uk(u64_z, self.ngf, is_training=self.is_training, norm=self.norm,
                reuse=self.reuse, name='u32_z')  # output_size=self.image_size)      # (?, h, w, 32)(?, 220, 180, 32)
            # 还原网纹图像大小
            clipping_z = tf.slice(u32_z, [0, 0, 1, 0], [u32_z.get_shape().as_list()[0],
                                                      u32_z.get_shape().as_list()[1], u32_z.get_shape().as_list()[2]-2,
                                                      u32_z.get_shape().as_list()[3]])
            output_z = ops.c7s1_k(clipping_z, 3, norm=None,
                activation='tanh', reuse=self.reuse, name='output_z')           # (?, w, h, 3)(?, 220, 180, 3)

        # set reuse=True for next call
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return output_y, output_z

    def sample(self, input_x):
        output_y, output_z = self.__call__(input_x)
        image_y = utils.batch_convert2int(output_y)
        image_z = utils.batch_convert2int(output_z)
        image_y = tf.image.encode_jpeg(tf.squeeze(image_y, [0]))
        image_z = tf.image.encode_jpeg(tf.squeeze(image_z, [0]))
        return image_y, image_z
