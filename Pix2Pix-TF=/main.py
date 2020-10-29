# coding:utf-8
#!/usr/bin/python3

import argparse
from model import *


################################################################################
parser = argparse.ArgumentParser(description='')
parser.add_argument('--phase', dest='phase', default='train', help='train, test, infer')
parser.add_argument('--epoch', dest='epoch', type=int, default=1, help='max number of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_num', dest='train_num', type=int, default=880000, help='# images used to train')
#parser.add_argument('--fine_size', dest='fine_size', type=int, default=256, help='then crop to this size')
#parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
#parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
#parser.add_argument('--direction', dest='direction', default='AtoB', help='AtoB or BtoA')
#parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
#parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--lr', dest='lr', type=float, default=2E-4, help='initial learning rate for optim')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
#parser.add_argument('--niter', dest='niter', type=int, default=200, help='# of iter at starting learning rate')
#parser.add_argument('--flip', dest='flip', type=float, default=0.5, help='prob of flipping images for argumentation')
parser.add_argument('--SS_lambda', dest='SS_lambda', type=float, default=3, help='SSIM_Loss weight, accelerate')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=200, help='L1_Loss weight, blurry')
parser.add_argument('--model_num', dest='model_num', type=int, default=10, help='max number of models to keep')
parser.add_argument('--model_dir', dest='model_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--val_dir', dest='val_dir', default='./Val', help='validation sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./Test', help='test data are saved here')
parser.add_argument('--train_dir', dest='train_dir', default='./Face1W32', help='training data are saved here')
parser.add_argument('--mesh_name', dest='mesh_name', default="*_*.jpg", help='mesh images naming format')
args = parser.parse_args()


#os.environ["CUDA_VISIBLE_DEVICES"]="" # assign/disable GPUs
# device_count: limit valid GPU number; soft_placement: auto-assign GPU
config = tf.ConfigProto(device_count={"GPU":1}, allow_soft_placement=True)
#config.gpu_options.per_process_gpu_memory_fraction = 0.1 # gpu_memory ratio
config.gpu_options.allow_growth = True # dynamicly apply gpu_memory
################################################################################
def main(dataset=args.train_dir):
    args.train_dir = dataset # dataset path/name
    if args.phase=="train": assert os.path.isdir(args.train_dir)
    args.model_dir += "/" + os.path.basename(args.train_dir)
    if not os.path.isdir(args.model_dir): os.makedirs(args.model_dir)
    print("\nParameters:\n", args, "\n") # Train/Test/Infer
    with tf.Session(config=config) as sess: pix2pix(sess, args)


def infer(images, model=args.model_dir): # test/infer
    args.model_dir = model; args.test_dir = images; args.phase = ""
    with tf.Session(config=config) as sess:
        p2p = pix2pix(sess, args); return p2p.infer(args)
        #args.phase = "infer"; pix2pix(sess, args)


def infer2(images, model=args.model_dir): # test/infer
    if os.path.isdir(images): data = globs(images, "*.jpg")
    else: data = [images] # images = np.array or image_path
    args.model_dir = model; args.phase = ""
    with tf.Session(config=config) as sess:
        p2p = pix2pix(sess, args) # initialize model
        if not p2p.load_model(args.model_dir): return # load model
        for i, image in enumerate(data): # batch_size=1
            im = p2p.load_data([image], isTrain=False); sz = p2p.sz[0]
            im = sess.run(p2p.fake_B, feed_dict={p2p.real_data: im})
            if type(image)==str: save_images(im, image[:-4]+"_.png", sz)
            else: return rsz(127.5*(im[0]+1.0), sz).astype("uint8")


################################################################################
if __name__ == '__main__': main()


################################################################################
# ln -s /data1/share/id_face HRFace # create links
# nohup python main.py --phase=train >log & # Train
# rm test/*.png; python main.py --phase=test --batch_size=5; sz test/*.png
