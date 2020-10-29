import ops
import tensorflow as tf
"""
判别网络
"""


class Discriminator:
    def __init__(self, name, is_training, norm='instance', use_sigmoid=False):
        self.name = name
        self.is_training = is_training
        self.norm = norm
        self.reuse = False
        self.use_sigmoid = use_sigmoid

    def __call__(self, input):
        """
        Args:
          input: batch_size x image_size x image_size x 3
        Returns:
          output: 4D tensor batch_size x out_size x out_size x 1 (default 1x5x5x1)
                  filled with 0.9 if real, 0.0 if fake
        """
        with tf.variable_scope(self.name):
            # 判别器卷积网络结构
            C64 = ops.Ck(input, 64, reuse=self.reuse, norm=None,
                         is_training=self.is_training, name='C64')

            C128 = ops.Ck(C64, 128, reuse=self.reuse, norm=self.norm,
                          is_training=self.is_training, name='C128')

            C256 = ops.Ck(C128, 256, reuse=self.reuse, norm=self.norm,
                          is_training=self.is_training, name='C256')

            C512 = ops.Ck(C256, 512, reuse=self.reuse, norm=self.norm,
                          is_training=self.is_training, name='C512')

            output = ops.last_conv(C512, reuse=self.reuse,  # 这并不是全连接层，这一层的输出还是tensor,判别器最终对tensor的每一个像素值做了判别
                                    use_sigmoid=self.use_sigmoid, name='output')

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return output
