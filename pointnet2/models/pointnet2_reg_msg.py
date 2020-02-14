import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_sa_module_msg

# def placeholder_inputs(batch_size, num_point):
def placeholder_inputs(batch_size, output_dim=1):
    # pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, None, 3))
    if output_dim == 1: 
        labels_pl = tf.placeholder(tf.float32, shape=(batch_size))
    elif output_dim > 1: 
        labels_pl = tf.placeholder(tf.float32, shape=(batch_size, output_dim))
    else: 
        raise ValueError('Invalid output dimension: %d' % output_dim)
    return pointclouds_pl, labels_pl

def get_model(point_cloud, is_training, bn_decay=None, output_dim=1):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    # num_point = point_cloud.get_shape()[1].value
    end_points = {}

    l0_xyz = point_cloud
    l0_points = None

    # Set abstraction layers
    l1_xyz, l1_points = pointnet_sa_module_msg(l0_xyz, l0_points, 512, [0.1,0.2,0.4], [16,32,128], [[32,32,64], [64,64,128], [64,96,128]], is_training, bn_decay, scope='layer1', use_nchw=True)
    l2_xyz, l2_points = pointnet_sa_module_msg(l1_xyz, l1_points, 128, [0.2,0.4,0.8], [32,64,128], [[64,64,128], [128,128,256], [128,128,256]], is_training, bn_decay, scope='layer2')
    l3_xyz, l3_points, _ = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3')

    # Fully connected layers
    net = tf.reshape(l3_points, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.4, is_training=is_training, scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.4, is_training=is_training, scope='dp2')
    # net = tf_util.fully_connected(net, 1, activation_fn=None, scope='fc3')
    net = tf_util.fully_connected(net, output_dim, activation_fn=None, scope='fc3')
    net = tf.squeeze(tf.sigmoid(net))

    return net, end_points

def get_loss(pred, label, name):
    if name == 'MSE':
        loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=label, predictions=pred))
    elif name == 'L1':
        loss = tf.reduce_mean(tf.abs(label - pred))
    else:
        raise ValueError('Unknown loss function: %s' % name)
    tf.summary.scalar(name, loss)
    tf.add_to_collection('losses', loss)
    return loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        net, _ = get_model(inputs, tf.constant(True))
        print(net)
