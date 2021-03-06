import utils as utils
import tensorflow as tf

class Reader():
    def __init__(self, tfrecords_file, image_size=[220, 178],
        min_queue_examples=1000, batch_size=8, num_threads=8, name=''):
        """
        Args:
          tfrecords_file: string, tfrecords file path
          min_queue_examples: integer, minimum number of samples to retain in the queue that provides of batches of examples
          batch_size: integer, number of images per batch
          num_threads: integer, number of preprocess threads
        """
        self.tfrecords_file = tfrecords_file
        self.image_size = image_size
        self.min_queue_examples = min_queue_examples
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.reader = tf.TFRecordReader()
        self.name = name

    def feed(self):
        """
        Returns:
          images: 4D tensor [batch_size, image_width, image_height, image_depth]
        """
        with tf.name_scope(self.name):
            # 文件名队列
            filename_queue = tf.train.string_input_producer([self.tfrecords_file])
            reader = tf.TFRecordReader()

            _, serialized_example = self.reader.read(filename_queue)
            # 解析包装器
            features = tf.parse_single_example(
                serialized_example,
                features={
                    'image/file_name': tf.FixedLenFeature([], tf.string),
                    'image/encoded_image': tf.FixedLenFeature([], tf.string),
                })

            image_buffer = features['image/encoded_image']
            image = tf.image.decode_jpeg(image_buffer, channels=3)
            image = self._preprocess(image)
            images = tf.train.batch(  # shuffle_batch
                [image], batch_size=self.batch_size, num_threads=self.num_threads,
                capacity=self.min_queue_examples + 3*self.batch_size  # ,min_after_dequeue=self.min_queue_examples
                )

            tf.summary.image('input', images)
        return images

    def _preprocess(self, image):
        # image = tf.image.resize_images(image, size=(220, 178))
        image = utils.convert2float(image)
        image.set_shape([self.image_size[0], self.image_size[1], 3])
        return image


def test_reader():  # 测试读取图像
    TRAIN_FILE_1 = 'data/tfrecords/x.tfrecords'
    TRAIN_FILE_2 = 'data/tfrecords/y.tfrecords'
    TRAIN_FILE_3 = 'data/tfrecords/z.tfrecords'

    with tf.Graph().as_default():
        reader1 = Reader(TRAIN_FILE_1, batch_size=2)
        reader2 = Reader(TRAIN_FILE_2, batch_size=2)
        reader3 = Reader(TRAIN_FILE_3, batch_size=2)

        images_op1 = reader1.feed()
        images_op2 = reader2.feed()
        images_op3 = reader3.feed()

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            while not coord.should_stop():
                batch_images1 = sess.run([images_op1])
                batch_images2 = sess.run([images_op2])
                batch_images3 = sess.run([images_op3])
                # print(batch_images1, '-----------------------')
                print("image shape: {}".format(batch_images1))
                print("image shape: {}".format(batch_images2))
                print("image shape: {}".format(batch_images3))
                print("="*10)
                step += 1
        except KeyboardInterrupt:
            print('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    test_reader()
