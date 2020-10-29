# coding:utf-8
# !/usr/bin/python3

import os, utils
import tensorflow as tf


image_size = [220, 178] # [height, width]
config = tf.ConfigProto(device_count={"GPU":1}, allow_soft_placement=True)
config.gpu_options.allow_growth = True # dynamicly apply gpu_memory

"""使用保存好的.pb模型进行图像去除网纹"""
def inference(pb_path, input_path, output_path, image_size):
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        with tf.gfile.FastGFile(input_path, 'rb') as f:
            image_data = f.read()
            input_image = tf.image.decode_jpeg(image_data, channels=3)
            input_image = tf.image.resize_images(input_image, size=tuple(image_size))
            input_image = utils.convert2float(input_image)
            input_image.set_shape(image_size+[3])
        with tf.gfile.FastGFile(pb_path, 'rb') as model_file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(model_file.read())
        [output_image] = tf.import_graph_def(graph_def, input_map={'input_image_x': input_image},
                            return_elements=['output_image:0'], name='output')

    with tf.Session(graph=graph, config=config) as sess:
        generated = output_image.eval()
        with open(output_path, 'wb') as f:
            f.write(generated)


def main(argv):
    pb_path = 'checkpoints/x2yz.pb' # 模型路径
    input_path = 'data/trainX/' # 带网纹图片文件夹
    output_path = 'data/test_X2Y/' # 去网纹结果保存路径
    if not os.path.exists(output_path): os.makedirs(output_path)
    for i in os.listdir(input_path):
        inference(pb_path, input_path+i, output_path+i[:-4]+'_.jpg', image_size)


if __name__ == '__main__':
    tf.app.run()
