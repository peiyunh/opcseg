# I TRY TO UNIFY OVERALL AND PERCLASS INSTANCE SEGMENTATION EVALUATION IN ONE SCRIPT 
import sys
if sys.version_info[0] < 3:
    raise Exception("Must be reading with Python 3")

import os

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
CLASSES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--min-cluster-size', type=int, default=1)   # THIS WAS BROUGHT IN TO HELP HDBSCAN
    parser.add_argument('--iou-thrs', type=float, nargs='+', default=np.linspace(0.5, 0.95, 10))
    parser.add_argument('--res-dir', type=str, required=True)
    parser.add_argument('--out-dir', type=str, default='segmentation/instanceseg')
    return parser.parse_args()

def read_segs(res_dir, id_): 
    segs, scores = [], []
    res_file = '%s/%06d.pkl' % (res_dir, id_)
    if os.path.exists(res_file):
        try: 
            with open(res_file, 'rb') as f:
                scores = pickle.load(f)
                segs = pickle.load(f)
        except UnicodeDecodeError:
            with open(res_file, 'rb') as f:
                scores = pickle.load(f, encoding='latin1')
                segs = pickle.load(f, encoding='latin1')
    else:
        print('Warning: cannot find result at %s' % (res_file))
    return segs, scores
            
args = parse_args()

#
with open('./kitti/object/%s.txt' % args.split) as f:
    ids = [int(l.rstrip()) for l in f]

#
confidences = []
perclass_npos = np.zeros(len(CLASSES),int)
perclass_fps = []
perclass_tps = []

# max_id = max(ids) # for printing only
for id_ in ids:
    print('processing:', id_)

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
    segs, scores = read_segs(args.res_dir, id_)

    # figure out left overs (mostly for detection methods)
    leftovers = np.arange(len(pts_velo_cs))
    if len(segs) > 0 and len(reduce(np.union1d, segs)) > 0:
        leftovers = np.setdiff1d(leftovers, reduce(np.union1d, segs))

    # we assume that constraints have been enforced 
    # assert(len(leftovers) == 0) 

    # EVALUATE POINT CLOUD INSTANCE SEGMENTATION

    # sort segments according to the scores
    argsort = np.argsort(scores)[::-1]

    # decide whether each detection is a false positive, true positive, or dontcare
    perclass_fp = np.full((len(CLASSES), len(segs), len(args.iou_thrs)), False)
    perclass_tp = np.full((len(CLASSES), len(segs), len(args.iou_thrs)), False)
    perclass_claimed = np.full((len(CLASSES), len(gtsegs), len(args.iou_thrs)), False)

    # go through each class 
    for c in range(len(CLASSES)): 
        for i in argsort:
            ovmax = -float('inf')
            jmax = -1
            if len(gtsegs) > 0:
                inters = np.array([len(np.intersect1d(segs[i], gtseg)) for gtseg in gtsegs])
                unions = np.array([len(np.union1d(segs[i], gtseg)) for gtseg in gtsegs])
                ious = [float(inter) / union for (inter, union) in zip(inters, unions)]
                jmax = np.argmax(ious)
                ovmax = ious[jmax]

            for k in range(len(args.iou_thrs)):
                if ovmax >= args.iou_thrs[k] and jmax >= 0:
                    cmax = CLASSES.index(gtclasses[jmax])
                    if cmax == c and len(gtsegs[jmax]) >= args.min_cluster_size: 
                        if not perclass_claimed[c, jmax, k]: # true positive
                            perclass_tp[c,i,k] = True
                            perclass_claimed[c,jmax,k] = True
                        else: # false positive (duplicate)
                            perclass_fp[c,i,k] = True
                else: # false positive (insufficient overlap)
                    perclass_fp[c,i,k] = True

        #
        for gtseg, gtclass in zip(gtsegs, gtclasses):
            perclass_npos[c] += (1 if (len(gtseg) >= args.min_cluster_size) and (gtclass==CLASSES[c]) else 0)

    #
    confidences.append(scores)
    perclass_fps.append(perclass_fp)
    perclass_tps.append(perclass_tp)

confidences = np.concatenate(confidences)
order = np.argsort(confidences)[::-1]

# perclass statistics 
perclass_tps = np.concatenate(perclass_tps, axis=1)
perclass_fps = np.concatenate(perclass_fps, axis=1)

perclass_tp = perclass_tps[:,order,:]
perclass_fp = perclass_fps[:,order,:]

perclass_ctp = np.cumsum(perclass_tp, axis=1)
perclass_cfp = np.cumsum(perclass_fp, axis=1)

perclass_rec = perclass_ctp / perclass_npos[:,None,None]
# perclass_prec = perclass_ctp / (perclass_ctp + perclass_cfp)
# perclass_prec[np.isnan(perclass_prec)] = 1.  # because there are ignored segments, ctp+cfp could be 0 to start with

perclass_divisor = perclass_ctp + perclass_cfp
perclass_divisor[perclass_divisor==0] = 1
perclass_prec = perclass_ctp / perclass_divisor 

# overall statistics
# THIS SHOULD BE EQUIVALENT TO OUR PREVIOUS IMPLEMNTATION
# A DETECTION IS TRUE POSITIVE IF IT IS TRUE POSITIVE ON ANY CLASS
# A DETECTION IS FALSE POSITIVE IF IT IS FALSE POSITIVE ON ANY CLASS
# AND A DETECTION WOULD NOT GO FROM FP TO TP OR FROM TP TO FP
overall_npos = perclass_npos.sum()

overall_tps = perclass_tps.max(0)
overall_fps = perclass_fps.max(0)

overall_tp = overall_tps[order,:]
overall_fp = overall_fps[order,:] 

overall_ctp = np.cumsum(overall_tp, axis=0)
overall_cfp = np.cumsum(overall_fp, axis=0)

overall_rec = overall_ctp / overall_npos
# overall_prec = overall_ctp / (overall_ctp + overall_cfp)
# overall_prec[np.isnan(overall_prec)] = 1.  # because there are ignored segments, ctp+cfp could be 0 to start with

overall_divisor = overall_ctp+overall_cfp
overall_divisor[overall_divisor==0] = 1
overall_prec = overall_ctp / overall_divisor 

eval_dir = os.path.join(args.out_dir, args.split)
if not os.path.exists(eval_dir): os.makedirs(eval_dir)

eval_file = os.path.join(eval_dir, '%s.pkl' % (args.name))
with open(eval_file, 'wb') as f:
    pickle.dump(args.iou_thrs, f)
    pickle.dump(perclass_prec, f)
    pickle.dump(perclass_rec,  f)
    pickle.dump(perclass_npos, f)
    pickle.dump(overall_prec,  f)
    pickle.dump(overall_rec,   f)
    pickle.dump(overall_npos,  f)
