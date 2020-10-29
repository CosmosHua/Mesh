# coding:utf-8
# !/usr/bin/python3

import os, logging
import tensorflow as tf
from datetime import datetime
from utils import ImagePool
from model import DFGAN
from pre_data import *

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
tf.flags.DEFINE_list('image_size', im_size, 'image size, default: [220, 178]')
tf.flags.DEFINE_bool('use_lsgan', True, 'use lsgan (mean squared error) or cross entropy loss')
tf.flags.DEFINE_string('norm', 'instance', '[instance, batch] use instance norm or batch norm')
tf.flags.DEFINE_float('lambda1', 10.0, 'weight for forward cycle loss (X->Y->X)')
tf.flags.DEFINE_float('lambda2', 10.0, 'weight for backward cycle loss (Y->X->Y)')
tf.flags.DEFINE_float('learning_rate', 2e-4, 'initial learning rate for Adam')
tf.flags.DEFINE_float('beta1', 0.5, 'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('pool_size', 50, 'size of image buffer that stores previously generated images')
tf.flags.DEFINE_integer('ngf', 64, 'number of gen filters in first conv layer, default: 32')
tf.flags.DEFINE_string('load_model', "_", 'folder of saved model to continue training')
tf.flags.DEFINE_string('checkpoint_dir', './checkpoints', 'checkpoints directory path')
# 定义保存模型的名字（分离网络模型）
tf.flags.DEFINE_string('XtoYZ_model', 'x2yz.pb', 'XtoYZ model name, default: x2yz.pb')
# 定义保存模型的名字（合成网络模型）
tf.flags.DEFINE_string('YZtoX_model', 'yz2x.pb', 'YZtoX model name, default: yz2x.pb')


config = tf.ConfigProto(device_count={"GPU":1}, allow_soft_placement=True)
config.gpu_options.allow_growth = True # dynamicly apply gpu_memory

# 训练整个模型
def train(max_step=200000):
    graph = tf.Graph()
    checkpoint_dir = FLAGS.checkpoint_dir

    with graph.as_default():
        df_gan = DFGAN(
            X_train_file=FLAGS.X, Y_train_file=FLAGS.Y, Z_train_file=FLAGS.Z,
            batch_size=FLAGS.batch_size, image_size=FLAGS.image_size, use_lsgan=FLAGS.use_lsgan,
            norm=FLAGS.norm, lambda1=FLAGS.lambda1, lambda2=FLAGS.lambda1,
            learning_rate=FLAGS.learning_rate, beta1=FLAGS.beta1, ngf=FLAGS.ngf )
        
        G_loss_y, G_loss_z, D_Y_loss, D_Z_loss, F_loss_x, D_X_loss, fake_y, fake_z, fake_x = df_gan.model()
        optimizers = df_gan.optimize(G_loss_y, G_loss_z, D_Y_loss, D_Z_loss, F_loss_x, D_X_loss)
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(checkpoint_dir, graph)
        saver = tf.train.Saver()

    with tf.Session(graph=graph, config=config) as sess:
        step = 0
        sess.run(tf.global_variables_initializer())
        if FLAGS.load_model is not None:
            checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
            meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
            restore = tf.train.import_meta_graph(meta_graph_path)
            restore.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
            step = int(meta_graph_path.split("-")[-1][:-5])

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            fake_Y_pool = ImagePool(FLAGS.pool_size)
            fake_Z_pool = ImagePool(FLAGS.pool_size)
            fake_X_pool = ImagePool(FLAGS.pool_size)

            while not coord.should_stop():
                step += 1
                # get previously generated images
                fake_y_val, fake_z_val, fake_x_val = sess.run([fake_y, fake_z, fake_x])

                # train
                _, G_loss_y_val, G_loss_z_val, D_Y_loss_val, D_Z_loss_val, F_loss_x_val, D_X_loss_val, summary = sess.run(
                    [optimizers, G_loss_y, G_loss_z, D_Y_loss, D_Z_loss, F_loss_x, D_X_loss, summary_op],
                    feed_dict={ df_gan.fake_y: fake_Y_pool.query(fake_y_val),
                                df_gan.fake_z: fake_Z_pool.query(fake_z_val),
                                df_gan.fake_x: fake_X_pool.query(fake_x_val) } )

                if step % 100 == 0:
                    train_writer.add_summary(summary, step)
                    train_writer.flush()

                if step % 100 == 0:
                    logging.info('-----------Step %d:-------------' % step)
                    logging.info('  G_loss_y : {}'.format(G_loss_y_val))
                    logging.info('  G_loss_z : {}'.format(G_loss_z_val))
                    logging.info('  F_loss_x : {}'.format(F_loss_x_val))
                    logging.info('  D_Y_loss : {}'.format(D_Y_loss_val))
                    logging.info('  D_Z_loss : {}'.format(D_Z_loss_val))
                    logging.info('  D_X_loss : {}'.format(D_X_loss_val))

                if step % 500 == 0:
                    save_path = saver.save(sess, checkpoint_dir + "/model.ckpt", global_step=step)
                    logging.info("Model saved in file: %s" % save_path)

                if step > max_step: # 自己设定的终止步数
                    logging.info('运行到指定' % step, '步，终止程序')
                    coord.request_stop()

        except KeyboardInterrupt:
            logging.info('\nKeyboard Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            save_path = saver.save(sess, checkpoint_dir + "/model.ckpt", global_step=step)
            logging.info("Model saved in file: %s" % save_path)
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)


"""
将训练好的模型导出保存以便使用
输入：训练时产生的checkpoints文件夹路径
输出：.pb模型文件
"""
def export_graph(model_name, XtoYZ=True):
    graph = tf.Graph()
    image_size = FLAGS.image_size
    model_dir = FLAGS.checkpoint_dir

    with graph.as_default():
        df_gan = DFGAN(ngf=FLAGS.ngf, norm=FLAGS.norm, image_size=image_size)
        df_gan.model()
        if XtoYZ:
            input_image_x = tf.placeholder(tf.float32, shape=image_size+[3], name='input_image_x')
            output_image_y, output_image_z = df_gan.G.sample(tf.expand_dims(input_image_x, 0))
            output_image = output_image_y
        else:
            input_image_y = tf.placeholder(tf.float32, shape=image_size+[3], name='input_image_y')
            input_image_z = tf.placeholder(tf.float32, shape=image_size+[3], name='input_image_z')
            output_image_x = df_gan.F.sample(tf.expand_dims(input_image_y, 0), tf.expand_dims(input_image_z, 0))
            # output_image_x = tf.identity(output_image_x, name='output_image_x')
            output_image = output_image_x
        output_image = tf.identity(output_image, name='output_image')
        restore_saver = tf.train.Saver()  # 保存和恢复变量

    with tf.Session(graph=graph, config=config) as sess:
        sess.run(tf.global_variables_initializer())
        latest_ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        restore_saver.restore(sess, latest_ckpt)
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                graph.as_graph_def(), [output_image.op.name])
        tf.train.write_graph(output_graph_def, model_dir, model_name, as_text=False)


def main(argv):
    #print("\nPreprocess Images:"); make_data()
    try:
        print("\nTraining Model:"); train()
    finally:
        print("\nExport Model:")
        export_graph(FLAGS.XtoYZ_model, XtoYZ=True)
        #export_graph(FLAGS.YZtoX_model, XtoYZ=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
