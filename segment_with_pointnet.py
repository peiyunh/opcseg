from sklearn.cluster import DBSCAN
import tensorflow as tf
import numpy as np
import argparse
import importlib
import pickle
import ipdb
import time
import os
from tree_utils import flatten_scores, flatten_indices
import sys
pointnet_dir = './pointnet2'
sys.path.append(pointnet_dir)
sys.path.append('%s/models' % pointnet_dir)
sys.path.append('%s/utils' % pointnet_dir)

#
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='kitti_object', help='Which dataset are we processing')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--split', type=str, default='val', choices=['train', 'val'], help='Data split')
parser.add_argument('--seq', type=int, default=0, help='Which sequence are we looking for')
parser.add_argument('--model', default='pointnet2_reg_msg', help='Model name. [default: pointnet2_reg_msg]')
parser.add_argument('--aggr-func', default='min', choices=['min', 'avg', 'wavg', 'd2wavg', 'sum'], help='aggregation function')
parser.add_argument('--model-path', default='logs/pointnet2_reg_msg_log_min10_max1024/model.ckpt', help='Model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--min-pts', type=int, default=10, help='Minimum number of points per segment [default: 10]')
parser.add_argument('--max-pts', type=int, default=1024, help='Maximum number of points per segment [default: 1024]')
parser.add_argument('--eps-list', nargs='*', type=float, default=[2.0, 1.0, 0.5, 0.25])
parser.add_argument('--num-votes', type=int, default=1, help='Number of votes per segment')
parser.add_argument('--res-dir', type=str, default='pointnet_res', help='Path to store segmentation results')
parser.add_argument('--tag', type=str, default='')
args = parser.parse_args()

# infer model name from log directory
model_name = args.model_path.split('/')[1]

# load tensorflow model
pointnet = importlib.import_module(args.model)
with tf.device('/gpu:'+str(args.gpu)):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(args.num_votes, None, 3))
    is_training_pl = tf.placeholder(tf.bool, shape=())
    score, _ = pointnet.get_model(pointclouds_pl, is_training_pl)
    saver = tf.train.Saver()

# create a session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False
sess = tf.Session(config=config)

# restore model from disk
saver.restore(sess, '%s/%s' % (pointnet_dir, args.model_path))

# define ops
ops = {'pointclouds_pl': pointclouds_pl,
       'is_training_pl': is_training_pl,
       'score': score}

def evaluate(points):
    # rotate to center
    _mean = points.mean(axis=0)
    _theta = np.arctan2(_mean[1], _mean[0])
    _rot = np.array([[ np.cos(_theta), np.sin(_theta)],
                     [-np.sin(_theta), np.cos(_theta)]])
    points[:,:2] = _rot.dot((points[:,:2] - _mean[None,:2]).T).T
    # re-sample
    samples = np.stack([points[np.random.choice(len(points), args.max_pts, replace=True)] for i in range(args.num_votes)])
    feed_dict = {ops['pointclouds_pl']: samples, ops['is_training_pl']: False}
    score = sess.run(ops['score'], feed_dict=feed_dict)
    if args.num_votes == 1: return score.item()
    else: return np.mean(score).item()

def segment(id_, eps_list, cloud, original_indices=None):
    if not all(eps_list[i] > eps_list[i+1] for i in range(len(eps_list)-1)):
        raise ValueError('eps_list is not sorted in descending order')
    # pick the first threshold from the list
    max_eps = eps_list[0]
    #
    if original_indices is None: original_indices = np.arange(cloud.shape[0])
    if isinstance(original_indices, list): original_indices = np.array(original_indices)
    # spatial segmentation
    dbscan = DBSCAN(max_eps, min_samples=1).fit(cloud[original_indices,:])
    labels = dbscan.labels_
    # evaluate every segment
    indices, scores = [], []
    for unique_label in np.unique(labels):
        inds = original_indices[np.flatnonzero(labels == unique_label)]
        indices.append(inds.tolist())
        scores.append(evaluate(cloud[inds,:]))
    # return if we are done
    if len(eps_list) == 1: return indices, scores
    # expand recursively
    final_indices, final_scores = [], []
    for i, (inds, score) in enumerate(zip(indices, scores)):
        # focus on this segment
        fine_indices, fine_scores = segment(id_, eps_list[1:], cloud, inds)
        # flatten scores to get the minimum (keep structure)
        flat_fine_scores = flatten_scores(fine_scores)
        if args.aggr_func == 'min':
            aggr_score = np.min(flat_fine_scores)
        elif args.aggr_func == 'avg':
            aggr_score = np.mean(flat_fine_scores)
        elif args.aggr_func == 'sum':
            aggr_score = np.sum(flat_fine_scores)
        elif args.aggr_func == 'wavg':
            # compute a weighted average (each score is weighted by the number of points)
            flat_fine_indices = flatten_indices(fine_indices)
            sum_count, sum_score = 0, 0.0
            for indices, score in zip(flat_fine_indices, flat_fine_scores):
                sum_count += len(indices)
                sum_score += len(indices)*score
            aggr_score = float(sum_score)/sum_count
        elif args.aggr_func == 'd2wavg':
            # compute a weighted average (each score is weighted by the number of points)
            flat_fine_indices = flatten_indices(fine_indices)
            sum_count, sum_score = 0, 0.0
            for indices, score in zip(flat_fine_indices, flat_fine_scores):
                squared_dists = np.sum(cloud[inds,:]**2, axis=1)
                sum_count += np.sum(squared_dists)
                sum_score += np.sum(squared_dists * score)
            aggr_score = float(sum_score)/sum_count

        # COMMENTING THIS OUT BECAUSE OF ADDING SUM AS AN AGGR FUNC
        # assert(aggr_score <= 1 and aggr_score >= 0)

        # if splitting is better
        if score < aggr_score:
            final_indices.append(fine_indices)
            final_scores.append(fine_scores)
        else: # otherwise
            final_indices.append(inds)
            final_scores.append(score)
    return final_indices, final_scores


aggr_func = args.aggr_func + '_' + '_'.join(['%.1f' % x for x in args.eps_list])

if args.dataset == 'kitti_object':
    with open('./kitti/object/%s.txt' % args.split, 'r') as f:
        ids = [int(l.rstrip()) for l in f]
    res_dir = 'results/%s/%s/%s/%s' % (args.dataset, args.res_dir, args.split, aggr_func + '_' + model_name)
elif args.dataset == 'kitti_tracking':
    assert(args.split == 'train')  # no clean data for test split
    with open('./kitti/tracking/devkit/python/data/tracking/evaluate_tracking.seqmap.training', 'r') as f:
        lines = f.readlines()
    seq, _, first_frame, end_frame = lines[args.seq].split()
    ids = np.arange(int(first_frame), int(end_frame))
    res_dir = 'results/%s/%s/%s/%04d/%s' % (args.dataset, args.res_dir, args.split, args.seq, aggr_func + '_' + model_name)
else:
    raise ValueError('Unknown dataset: %s' % args.dataset)


res_dir = '%s_%dvotes' % (res_dir, args.num_votes)
if not os.path.exists(res_dir): os.makedirs(res_dir)

stats = []
for id_ in ids:
    res_file = '%s/%06d.pkl' % (res_dir, id_)
    # if os.path.exists(res_file): continue

    if args.dataset == 'kitti_object':
        velo_file = './kitti/object/training_clean/velodyne/%06d.bin' % id_
        if not os.path.exists(velo_file): continue
        pts_velo_cs = np.fromfile(velo_file, np.float32).reshape((-1,4))

    elif args.dataset == 'kitti_tracking':
        velo_file = './kitti/tracking/training/filtered_velodyne/%04d/%06d.npy' % (args.seq, id_)
        if not os.path.exists(velo_file): continue
        pts_velo_cs = np.load(velo_file)

    if len(pts_velo_cs) == 0: continue

    # segmentation with point-net
    indices, scores = segment(id_, args.eps_list, pts_velo_cs[:,:3], None)

    # flatten list(list(...(indices))) into list(indices)
    flat_indices = flatten_indices(indices)
    flat_scores = flatten_scores(scores)

    # save results
    with open(res_file, 'wb') as f:
        pickle.dump(flat_scores, f)
        pickle.dump(flat_indices, f)
