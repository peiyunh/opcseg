'''
    Multi-GPU training.
    Near linear scale acceleration for multi-gpus on a single machine.
    Will use H5 dataset in default. If using normal, will shift to the normal dataset.
'''

import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
from kitti_dataset import KittiDataset

parser = argparse.ArgumentParser()
parser.add_argument('--num_gpus', type=int, default=1, help='How many gpus to use [default: 1]')
parser.add_argument('--model', type=str, default='pointnet2_reg_msg', help='Model name [default: pointnet2_reg_msg]')
parser.add_argument('--loss', type=str, default='L1', choices=['L1', 'MSE'], help='Loss function we use')
parser.add_argument('--data_dir', type=str, required=True, help='Data dir')
parser.add_argument('--log_dir', type=str, required=True, help='Log dir')
parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 30]')
parser.add_argument('--resample', action='store_true', help='Whether resample point cloud during training')
parser.add_argument('--class-specific', action='store_true')
parser.add_argument('--augment', action='store_true', help='Data augmentation during training')
parser.add_argument('--min_pts', type=int, default=10, help='Minimum number of points per segment [default: 10]')
parser.add_argument('--max_pts', type=int, default=1024, help='Maximum number of points per segment [default: 1024]')
parser.add_argument('--batch_size', type=int, default=128, help='Batch Size during training [default: 256]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', type=str, default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

CLASSES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']

NUM_GPUS = FLAGS.num_gpus
BATCH_SIZE = FLAGS.batch_size
assert(BATCH_SIZE % NUM_GPUS == 0)
DEVICE_BATCH_SIZE = BATCH_SIZE / NUM_GPUS

CLASS_SPECIFIC = FLAGS.class_specific
OUTPUT_DIM = len(CLASSES) if CLASS_SPECIFIC else 1
RESAMPLE = FLAGS.resample
AUGMENT = FLAGS.augment
MIN_PTS = FLAGS.min_pts
MAX_PTS = FLAGS.max_pts
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

DATA_DIR = FLAGS.data_dir

MODEL = importlib.import_module(FLAGS.model) # import network module
LOSS = FLAGS.loss
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
if not os.path.exists(LOG_DIR): os.mkdir('%s/checkpoints')

os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

if AUGMENT:
    TRAIN_DATASET = KittiDataset(split='train', min_pts=MIN_PTS, max_pts=MAX_PTS, batch_size=BATCH_SIZE, data_dir=DATA_DIR, 
                                 class_specific=CLASS_SPECIFIC, resample=RESAMPLE, random_flip=True, random_jitter=True)
else:
    TRAIN_DATASET = KittiDataset(split='train', min_pts=MIN_PTS, max_pts=MAX_PTS, batch_size=BATCH_SIZE, data_dir=DATA_DIR, 
								 class_specific=CLASS_SPECIFIC, resample=RESAMPLE, random_flip=False, random_jitter=False)

TEST_DATASET = KittiDataset(split='val', min_pts=MIN_PTS, max_pts=MAX_PTS, batch_size=BATCH_SIZE, data_dir=DATA_DIR, 
                            class_specific=CLASS_SPECIFIC, resample=RESAMPLE, random_flip=False, random_jitter=False)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  From tensorflow tutorial: cifar10/cifar10_multi_gpu_train.py
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    #for g, _ in grad_and_vars:
    for g, v in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            # pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, OUTPUT_DIM)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Set learning rate and optimizer
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)

            # -------------------------------------------
            # Get model and loss on multiple GPU devices
            # -------------------------------------------
            # Allocating variables on CPU first will greatly accelerate multi-gpu training.
            # Ref: https://github.com/kuza55/keras-extras/issues/21
            MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay, output_dim=OUTPUT_DIM)

            tower_grads = []
            pred_gpu = []
            total_loss_gpu = []
            for i in range(NUM_GPUS):
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    with tf.device('/gpu:%d'%(i)), tf.name_scope('gpu_%d'%(i)) as scope:
                        # Evenly split input data to each GPU
                        pc_batch = tf.slice(pointclouds_pl,
                            [i*DEVICE_BATCH_SIZE,0,0], [DEVICE_BATCH_SIZE,-1,-1])
                        if CLASS_SPECIFIC: 
                            label_batch = tf.slice(labels_pl,
                                [i*DEVICE_BATCH_SIZE,0], [DEVICE_BATCH_SIZE,-1])
                        else: 
                            label_batch = tf.slice(labels_pl,
                                [i*DEVICE_BATCH_SIZE], [DEVICE_BATCH_SIZE])

                        pred, end_points = MODEL.get_model(pc_batch,
                            is_training=is_training_pl, bn_decay=bn_decay, output_dim=OUTPUT_DIM)

                        MODEL.get_loss(pred, label_batch, LOSS)
                        losses = tf.get_collection('losses', scope)
                        total_loss = tf.add_n(losses, name='total_loss')
                        for l in losses + [total_loss]:
                            tf.summary.scalar(l.op.name, l)

                        grads = optimizer.compute_gradients(total_loss)
                        tower_grads.append(grads)

                        pred_gpu.append(pred)
                        total_loss_gpu.append(total_loss)

            # Merge pred and losses from multiple GPUs
            pred = tf.concat(pred_gpu, 0)
            total_loss = tf.reduce_mean(total_loss_gpu)

            # Get training operator
            grads = average_gradients(tower_grads)
            train_op = optimizer.apply_gradients(grads, global_step=batch)

            # correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            # accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            # tf.summary.scalar('accuracy', accuracy)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(max_to_keep=MAX_EPOCH)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': total_loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        best_acc = -1
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)

            # # Save the variables to disk.
            # if epoch % 10 == 0:
            #     save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
            #     log_string("Model saved in file: %s" % save_path)

            # save variables to disk every epoch (ahahahaha)
            save_path = saver.save(sess, os.path.join(LOG_DIR, "checkpoints", "model_epoch%02d" % epoch))
            log_string("Model saved in file: %s" % save_path)

def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    log_string(str(datetime.now()))

    # Make sure batch data is of same size
    # cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,TRAIN_DATASET.num_channel()))
    # cur_batch_label = np.zeros((BATCH_SIZE,2), dtype=np.float32)

    # total_correct = 0
    # total_seen = 0
    loss_sum = 0
    batch_idx = 0
    while TRAIN_DATASET.has_next_batch():
        # batch_data, batch_label = TRAIN_DATASET.next_batch(augment=True)
        batch_data, batch_label = TRAIN_DATASET.next_batch()

        #batch_data = provider.random_point_dropout(batch_data)

        num_points = len(batch_data[0])
        cur_batch_data = np.zeros((BATCH_SIZE,num_points,TRAIN_DATASET.num_channel()))
        if CLASS_SPECIFIC: 
            cur_batch_label = np.zeros((BATCH_SIZE, len(CLASSES)), dtype=np.float32)
        else:
            cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.float32)

        bsize = batch_data.shape[0]
        cur_batch_data[0:bsize,...] = batch_data
        cur_batch_label[0:bsize] = batch_label

        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['labels_pl']: cur_batch_label,
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        # pred_val = np.argmax(pred_val, 1)
        # correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        # total_correct += correct
        # total_seen += bsize
        loss_sum += loss_val
        if (batch_idx+1)%50 == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
            log_string('mean loss: %f' % (loss_sum / 50))
            # log_string('accuracy: %f' % (total_correct / float(total_seen)))
            # total_correct = 0
            # total_seen = 0
            loss_sum = 0
        batch_idx += 1

    TRAIN_DATASET.reset()

def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False

    # Make sure batch data is of same size
    # cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,TEST_DATASET.num_channel()))
    # cur_batch_label = np.zeros((BATCH_SIZE,2), dtype=np.float32)

    # total_correct = 0
    # total_seen = 0
    loss_sum = 0
    batch_idx = 0
    # shape_ious = []
    # total_seen_class = [0 for _ in range(NUM_CLASSES)]
    # total_correct_class = [0 for _ in range(NUM_CLASSES)]

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))

    while TEST_DATASET.has_next_batch():
        # batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
        batch_data, batch_label = TEST_DATASET.next_batch()
        bsize = batch_data.shape[0]

        #
        num_points = len(batch_data[0])
        cur_batch_data = np.zeros((BATCH_SIZE,num_points,TRAIN_DATASET.num_channel()))
        # cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.float32)
        if CLASS_SPECIFIC: 
            cur_batch_label = np.zeros((BATCH_SIZE, len(CLASSES)), dtype=np.float32)
        else:
            cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.float32)

        # for the last batch in the epoch, the bsize:end are from last batch
        cur_batch_data[0:bsize,...] = batch_data
        cur_batch_label[0:bsize] = batch_label

        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['labels_pl']: cur_batch_label,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        # pred_val = np.argmax(pred_val, 1)
        # correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        # total_correct += correct
        # total_seen += bsize
        loss_sum += loss_val
        batch_idx += 1
        # for i in range(0, bsize):
        #     l = batch_label[i]
        #     total_seen_class[l] += 1
        #     total_correct_class[l] += (pred_val[i] == l)

    log_string('eval mean loss: %f' % (loss_sum / float(batch_idx)))
    # log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    # log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    EPOCH_CNT += 1

    TEST_DATASET.reset()
    # return total_correct/float(total_seen)


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
