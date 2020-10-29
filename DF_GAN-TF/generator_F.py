import ops, utils
import tensorflow as tf
"""
合成网络F
"""


class Generator_F:
    def __init__(self, name, is_training, ngf=32, norm='instance', image_size=[220, 178]):
        self.name = name
        self.reuse = False
        self.ngf = ngf
        self.norm = norm
        self.is_training = is_training
        self.image_size = image_size

    def __call__(self, input_y, input_z):
        """
        Args:
          input_y: batch_size x height x width x 3(domain y)
          input_z: batch_size x height x width x 3(domain z)
        Returns:
          output: same size as input_y
        """
        with tf.variable_scope(self.name):

            # y域卷积网络
            c7s1_32_y = ops.c7s1_k(input_y, self.ngf, is_training=self.is_training, norm=self.norm,
                reuse=self.reuse, name='c7s1_32_y')                 # (?, h, w, 32)   [batch, 220, 178, 32]
            # 对宽度先加边再卷积(针对[220,178]的图像，若图像size变换，需对应修改ops.dk_pad())
            d64_y = ops.dk_pad(c7s1_32_y, 2*self.ngf, is_training=self.is_training, norm=self.norm,
                reuse=self.reuse, name='d64_y')                     # (?, h/2, w/2, 64)  [batch, 110, 90, 64]
            d128_y = ops.dk(d64_y, 4*self.ngf, is_training=self.is_training, norm=self.norm,
                reuse=self.reuse, name='d128_y')                    # (?, h/4, w/4, 128)  [batch, 55, 45, 128]
            s255_y = ops.sk(d128_y, 255, is_training=self.is_training, norm=self.norm,
                            reuse=self.reuse, name='s255_y')        # (?, h/4, w/4, 255)  [batch, 55, 45, 255]

            # **********************************************************************************************************
            # z域卷积网络
            c7s1_32_z = ops.c7s1_k(input_z, self.ngf, is_training=self.is_training, norm=self.norm,
                                   reuse=self.reuse, name='c7s1_32_z')          # (?, h, w, 32) [batch, 220, 178, 32]
            # 对宽度先加边再卷积(针对[220,178]的图像，若图像size变换，需对应修改ops.dk_pad())
            d64_z = ops.dk_pad(c7s1_32_z, 2 * self.ngf, is_training=self.is_training, norm=self.norm,
                           reuse=self.reuse, name='d64_z')                      # (?, h/2, w/2, 64)[batch, 90, 110, 64]
            d128_z = ops.dk(d64_z, 4 * self.ngf, is_training=self.is_training, norm=self.norm,
                            reuse=self.reuse, name='d128_z')                    # (?, h/4, w/4, 128)[batch, 55, 45, 128]
            s1_z = ops.sk(d128_z, 1, is_training=self.is_training, norm=self.norm,
                            reuse=self.reuse, name='s1_z')                      # (?, h/4, w/4, 1)   [batch, 55, 45, 1]

            # **********************************************************************************************************
            # 将图像在潜在层联合
            concat_x = tf.concat(axis=3, values=[s255_y, s1_z])  # [?, 55, 45, 256]

            # 使用5个残差快
            res_output_x = ops.n_res_blocks(concat_x, reuse=self.reuse, n=5)      # (batch, 55, 45, 256)

            u128_x = ops.sk(res_output_x, 4 * self.ngf, is_training=self.is_training, norm=self.norm,
                            reuse=self.reuse, name='u128_x')                      # (batch, 55, 45, 128)

            # 反卷积（解码）
            u64_x = ops.uk(u128_x, 2*self.ngf, is_training=self.is_training, norm=self.norm,
                reuse=self.reuse, name='u64_x')                                   # (batch, 110, 90, 64)

            u32_x = ops.uk(u64_x, self.ngf, is_training=self.is_training, norm=self.norm,
                reuse=self.reuse, name='u32_x')  # output_size=self.image_size)   # (batch, 220, 180, 32)

            # 还原图像size: (batch, 220, 180, 32) --> (batch, 220, 178, 32)
            clipping = tf.slice(u32_x, [0, 0, 1, 0], [u32_x.get_shape().as_list()[0],
                                                      u32_x.get_shape().as_list()[1],
                                                      u32_x.get_shape().as_list()[2]-2,
                                                      u32_x.get_shape().as_list()[3]])

            output_x = ops.c7s1_k(clipping, 3, norm=None,
                activation='tanh', reuse=self.reuse, name='output')           # (batch, 128, 128, 3)
        # 设置变量重复使用与更新
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return output_x

    def sample(self, input_y, inpyt_z):
        image_x = utils.batch_convert2int(self.__call__(input_y, inpyt_z))
        image_x = tf.image.encode_jpeg(tf.squeeze(image_x, [0]))
        return image_x
