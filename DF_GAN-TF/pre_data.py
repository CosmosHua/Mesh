# coding:utf-8
# !/usr/bin/python3

import random, os, cv2
import tensorflow as tf


"""
功能：将文件夹下的jpg格式图像转换成tfrecords格式保存
输入：X,Y,Z域的图像路径
输出：对应的tfrecords文件
"""

im_size = [220, 178]
data_root_dir = "data/"
FLAGS = tf.flags.FLAGS

# the path of input and out
tf.flags.DEFINE_string('X_dir', data_root_dir+'trainX/', "Mesh_Face_Dir")
tf.flags.DEFINE_string('Y_dir', data_root_dir+'trainY/', "Clean_Face_Dir")
tf.flags.DEFINE_string('Z_dir', data_root_dir+'trainZ/', "Mesh_Apart_Dir")

tf.flags.DEFINE_string('X', data_root_dir+'x.tfrecords', "Mesh_Faces")
tf.flags.DEFINE_string('Y', data_root_dir+'y.tfrecords', "Clean_Faces")
tf.flags.DEFINE_string('Z', data_root_dir+'z.tfrecords', "Mesh_Nets")


def get_net(mesh, clean, out): # sever mesh
    h,w = im_size
    if not os.path.exists(out): os.makedirs(out)
    for im in os.listdir(mesh):
        cn = im[:im.rfind("_")]+".jpg"
        cn = os.path.join(clean, cn)
        cc = cv2.resize(cv2.imread(cn), (w,h))
        cv2.imwrite(cn, cc)
        
        mn = os.path.join(mesh, im)
        mm = cv2.resize(cv2.imread(mn), (w,h))
        cv2.imwrite(mn, mm)

        net = os.path.join(out, im)
        cc = cc.astype(int)-mm.astype(int)
        cv2.imwrite(net, 255-cc)


def data_reader(input_dir, shuffle=True):
    """
    Read images from input_dir then shuffle them
    Args:
        input_dir: string, path of input dir, e.g., /path/to/dir
    Returns:
        file_paths: list of strings
    """
    file_paths = []

    for img_file in os.scandir(input_dir):  # 递归遍历指定文件目录
        if img_file.name.endswith('.jpg') and img_file.is_file():
            file_paths.append(img_file.path)

    if shuffle:  # 是否随机打乱图片顺序
        shuffled_index = list(range(len(file_paths)))
        random.seed(12345)
        random.shuffle(shuffled_index)
        file_paths = [file_paths[i] for i in shuffled_index]

    return file_paths


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(file_path, image_buffer):
    """
    Build an Example proto for an example.
    Args:
        file_path: string, path to an image file, e.g., '/path/to/example.JPG'
        image_buffer: string, JPEG encoding of RGB image
    Returns:
        Example proto
    """
    file_name = file_path.split('/')[-1]
    file_name = file_name.encode()
    example = tf.train.Example(
            features = tf.train.Features(
                feature={
                    'image/file_name': _bytes_feature((os.path.basename(file_name))),
                    'image/encoded_image': _bytes_feature((image_buffer))}
            )
        )
    return example


def data_writer(input_dir, output_file):
    """Write data to tfrecords"""
    file_paths = data_reader(input_dir)
    images_num = len(file_paths)

    writer = tf.python_io.TFRecordWriter(output_file)

    for i in range(images_num):
        file_path = file_paths[i]

        with tf.gfile.FastGFile(file_path, 'rb') as f:  # 读取图片
            image_data = f.read()

        example = _convert_to_example(file_path, image_data)
        writer.write(example.SerializeToString())

    print("Processed {}/{}.".format(i+1, images_num))
    writer.close()


def make_data():
    # X = faceNetDir, Y = faceDir, Z = outDir
    print("seperate net from %s:" % FLAGS.X_dir)
    get_net(FLAGS.X_dir, FLAGS.Y_dir, FLAGS.Z_dir)
    
    print("Convert X data to tfrecords:")
    data_writer(FLAGS.X_dir, FLAGS.X)
    print("Convert Y data to tfrecords:")
    data_writer(FLAGS.Y_dir, FLAGS.Y)
    print("Convert Z data to tfrecords:")
    data_writer(FLAGS.Z_dir, FLAGS.Z)


if __name__ == '__main__':
    make_data()
