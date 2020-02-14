import os
import scipy
import pickle
import argparse
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from functools import reduce
import matplotlib.pyplot as plt
import sys
sys.path.append('./pointnet2/kitti')
from utils import *

kitti_dir = './kitti/object'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='kitti_object')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--method', type=str, choices=['segmenter', 'detector'], required=True)
    parser.add_argument('--res-dirs', type=str, nargs='+', required=True)
    parser.add_argument('--out-dir', type=str, default='segmentation/underseg_overseg/segmentation_output')
    return parser.parse_args()

def read_results(method, res_dirs, id_):
    segs, scores = [], []
    if method == 'segmenter':
        assert(len(res_dirs) == 1)
        res_file = '%s/%06d.pkl' % (res_dirs[0], id_)
        if os.path.exists(res_file):
            with open(res_file, 'rb') as f:
                scores = pickle.load(f)
                segs = pickle.load(f)
    elif method == 'detector':
        dets = []
        for res_dir in res_dirs:
            res_file = '%s/%06d.txt' % (res_dir, id_)
            if not os.path.exists(res_file):
                print('warning: cannot find results at %s' % res_file)
                continue
            dets.extend(read_detection(res_file))
        segs, scores, _ = convert_dets_to_segs(pts_velo_cs, velo_to_cam_tf, dets)
    return segs, scores

args = parse_args()

#
with open('./kitti/object/%s.txt' % args.split) as f:
    ids = [int(l.rstrip()) for l in f]

eval_dir = os.path.join(args.out_dir, args.split, args.name, 'obj_eval')
if not os.path.exists(eval_dir): os.makedirs(eval_dir)

for id_ in ids:
    # load calibration
    calib_file = '%s/training/calib/%06d.txt' % (kitti_dir, id_)
    velo_to_cam_tf = read_calibration(calib_file)

    # load raw velodyne scan
    velo_file = '%s/training_clean/velodyne/%06d.bin' % (kitti_dir, id_)
    pts_velo_cs = np.fromfile(velo_file, np.float32).reshape((-1,4))[:,:3]

    # filter point cloud with ground truth bounding boxes
    gt_file = '%s/training/label_2/%06d.txt' % (kitti_dir, id_)
    gtdets = read_detection(gt_file)
    gtsegs, gtclasses = convert_gtdets_to_gtsegs(pts_velo_cs, velo_to_cam_tf, gtdets)

    # read predictions (segmentation or detection)
    # res_dirs = [os.path.join('results', args.dataset, res_dir) for res_dir in args.res_dirs]
    # NOTE: results/kitti are only for some of the methods
    segs, scores = read_results(args.method, args.res_dirs, id_)

    # figure out left overs
    leftovers = np.arange(len(pts_velo_cs))
    if len(segs) > 0 and len(reduce(np.union1d, segs)) > 0:
        leftovers = np.setdiff1d(leftovers, reduce(np.union1d, segs))

    if args.method == 'segmenter':
        # assert(len(leftovers) == 0)
        if len(leftovers) > 0:
            print('Warning: non-empty leftover set (%d)' % (len(leftovers)))

    elif args.method == 'detector':
        # process leftovers
        if len(leftovers) > 0:
            #
            leftover_eps = 1.0
            # first, segment leftovers
            model = DBSCAN(leftover_eps, min_samples=1).fit(pts_velo_cs[leftovers,:3])
            labels = model.labels_
            lo_segs = []
            for unique_label in np.unique(labels):
                inds = np.flatnonzero(labels == unique_label)
                lo_segs.append(leftovers[inds])
            # second, try connecting leftover segs to existing segs
            # if everything is leftover, nothing to connect to
            if len(leftovers) == len(pts_velo_cs):
                segs = lo_segs
                scores = [float('nan') for s in lo_segs]
            # if there are existing segments
            else:
                # linearize points in the existing segments
                all_pts_list = []    # actual points
                all_seg_ids = []     # segment id for each point
                for i in range(len(segs)):
                    all_pts_list.append(pts_velo_cs[segs[i]])
                    all_seg_ids.extend([i for _ in segs[i]])
                all_pts = np.concatenate(all_pts_list, axis=0)
                assert(len(all_pts) == len(all_seg_ids))  # make sure length match

                # build a nearest neighbor index
                nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(all_pts)

                # try connecting every leftover segment to an existing one
                for lo_seg in lo_segs:
                    # extract actual points for the leftover segment
                    lo_seg_points = pts_velo_cs[lo_seg,:]
                    # retrieve every point's nearest neighbor from existing segments
                    dists, inds = nbrs.kneighbors(lo_seg_points)
                    # if the nearest neighbor of all has a larger distance than eps
                    # we simply add the leftover segment as a standalone segment
                    if np.min(dists) > leftover_eps:
                        segs.append(lo_seg)
                        scores.append(float('nan'))
                    # if this leftover is connected to an existing segment (will choose the closest)
                    else:
                        # get which point from linearized existing segments is closest to leftover
                        idx = inds[np.argmin(dists), 0]  # 0 for 1-nearest neighbor
                        seg_id = all_seg_ids[idx]
                        # merge the leftover segment into that existing segment
                        segs[seg_id] = np.append(segs[seg_id], lo_seg, axis=0)
                        # no need to change the score

    # # make sure this is a valid segment
    # if len(segs) > 0:
    #     assert(len(reduce(np.union1d, segs)) == len(pts_velo_cs))
    #     for i in range(len(segs)):
    #         for j in range(i+1, len(segs)):
    #             assert(len(np.intersect1d(segs[i], segs[j])) == 0)

    # find out which ground truth overlaps with each other
    gt_has_ovlp = np.full((len(gtsegs), len(gtsegs)), False, bool)
    for i in range(len(gtsegs)):
        for j in range(i+1, len(gtsegs)):
            test = len(np.intersect1d(gtsegs[i], gtsegs[j])) > 0
            gt_has_ovlp[j][i] = gt_has_ovlp[i][j] = test

    # NOTE:: this might actually be a fucking bug...
    # gt_has_ovlp = np.all(gt_has_ovlp, axis=1)
    gt_has_ovlp = np.any(gt_has_ovlp, axis=1)

    #
    eval_file = os.path.join(eval_dir, '%06d.txt' % id_)
    eval_fd = open(eval_file, 'w')
    for i in range(len(gtsegs)):
        # David's evaluation script will ignore such ground truth segments anyways
        if len(gtsegs[i]) == 0: continue

        # extract info about detection (if it is available)
        pos_points_all = np.array([len(np.intersect1d(gtsegs[i], segs[j])) for j in range(len(segs))])
        max_pos_points = np.max(pos_points_all)
        J = np.flatnonzero(pos_points_all == max_pos_points)

        # if there are multiple segments that have the same size of intersection
        # pick the one with least points (something we can optimize)
        if len(J) > 1:
            jmax = J[np.argmin([len(segs[j]) for j in J])]
        else:
            jmax = J[0]

        #
        pos_points = pos_points_all[jmax]
        blob_points = len(segs[jmax])
        other_pos_points = len(gtsegs[i]) - pos_points

        if len(pts_velo_cs) > 0 and len(gtsegs[i]) > 0:
            gt_dist = np.sqrt(np.sum(pts_velo_cs[gtsegs[i],:].mean(axis=0)**2))
        else:
            gt_dist = 80

        eval_fd.write('%d %s -1 %f %f -1 %f -1 -1 %f -1 -1 %f -1 %f -1 -1 %f\n' % (
            id_, gtclasses[i], pos_points, blob_points, other_pos_points, gt_dist, 0, -1, gt_has_ovlp[i]
        ))

    eval_fd.close()
