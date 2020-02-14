'''
   1. filter point cloud with ground truth bounding boxes
   2. extract ground truth segments based on ground truth bounding boxes
   3. collect dbscan segments
   4. save all of them to one pickle file

   I forgot to save one element for every segment
   I ended up saving one for every image
   but this is not how random sampling works

   to process data more efficiently, maybe we only save coarse segments
   and we generate fine segments on the fly by rolling a dice
'''
import os
import sys
import pickle
import argparse
from utils import *
from functools import reduce
from sklearn.cluster import DBSCAN

kitti_dir = '/home/peiyunh/Kitti/object'
eps_list = [0.25, 0.5, 1.0, 2.0]
ground_plane_thresh = 0.3

CLASSES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']

def process(split, reweight, class_specific, remove_ground):

    with open('%s/%s.txt' % (kitti_dir, split)) as f :
        ids = [int(l.rstrip()) for l in f.readlines()]
  
    # now we do dbscan on filtered point cloud
    if reweight == 'squared_distance':
        data_dir = '../data/kitti_eps%s_sqr_d_w_iou' % ('_'.join(['%.2f' % eps for eps in eps_list]))
    elif reweight == 'distance':
        data_dir = '../data/kitti_eps%s_d_w_iou' % ('_'.join(['%.2f' % eps for eps in eps_list]))
    elif reweight == 'none':
        data_dir = '../data/kitti_eps%s' % ('_'.join(['%.2f' % eps for eps in eps_list]))
    else:
        raise ValueError('Unknown reweighting scheme: %s' % reweight)

    if class_specific: 
        data_dir += '_class_specific' 

    if remove_ground:
        data_dir += '_ground_removed_%.1f' % (ground_plane_thresh)

    if not os.path.exists(data_dir): os.makedirs(data_dir)

    points_list = []
    target_list = []
    for id_ in ids:
        print('processing %s: %d/%d' % (split, id_, ids[-1]))

        # load calibration
        calib_file = '%s/training/calib/%06d.txt' % (kitti_dir, id_)
        velo_to_cam_tf = read_calibration(calib_file)

        # load raw velodyne scan
        velo_file = '%s/training/velodyne/%06d.bin' % (kitti_dir, id_)
        pts_velo_cs = np.fromfile(velo_file, np.float32).reshape((-1,4))[:,:3]

        # remove ground
        if remove_ground:
            plane_file = '%s/training/planes/%06d.txt' % (kitti_dir, id_)
            a, b, c, d = np.loadtxt(plane_file, skiprows=3)
            pts_cam_cs = transform_se3(pts_velo_cs[:,:3], velo_to_cam_tf)
            x, y, z = pts_cam_cs.T
            I = np.flatnonzero(a*x+b*y+c*z+d>=ground_plane_thresh)
            pts_velo_cs = pts_velo_cs[I,:]

        # filter point cloud with ground truth bounding boxes
        gt_file = '%s/training/label_2/%06d.txt' % (kitti_dir, id_)
        gtdets = read_detection(gt_file)
        gtsegs, gtclasses = convert_gtdets_to_gtsegs(pts_velo_cs, velo_to_cam_tf, gtdets)

        # if there are no points within ground truth bounding boxes, we skip
        if len(gtsegs) == 0: continue

        gt_pt_inds = reduce(np.union1d, gtsegs)
        if len(gt_pt_inds) == 0: continue

        # filter out points that are outside ground truth bounding boxes
        gt_pts_velo_cs = pts_velo_cs[gt_pt_inds, :]

        # redo conversion to reindex (easier to code... XD)
        gtsegs, gtclasses = convert_gtdets_to_gtsegs(gt_pts_velo_cs, velo_to_cam_tf, gtdets)
        print(gtclasses)

        # make sure we did not lose any point after reindexing
        assert(len(reduce(np.union1d, gtsegs)) == len(gt_pts_velo_cs))

        # now we do dbscan on filtered point cloud
        if reweight == 'squared_distance':
            weights = np.sum(gt_pts_velo_cs**2, axis=1)
        elif reweight == 'distance':
            weights = np.sqrt(np.sum(gt_pts_velo_cs**2, axis=1))
        elif reweight == 'none':
            weights = np.ones(len(gt_pts_velo_cs))
        else:
            raise ValueError('Unknown reweighting scheme: %s' % reweight)

        for eps in eps_list:
            model = DBSCAN(eps, min_samples=1).fit(gt_pts_velo_cs)
            labels = model.labels_
            for unique_label in np.unique(labels):
                seg = np.flatnonzero(labels == unique_label)
                points = gt_pts_velo_cs[seg,:]

                # find the best ground truth segment
                # intersections = [len(np.intersect1d(seg, gtseg)) for gtseg in gtsegs]
                # imax = np.argmax(intersections)
                # under_seg_ratio = float(intersections[imax]) / len(seg)
                # over_seg_ratio = float(intersections[imax]) / len(gtsegs[imax])
                # targets = (under_seg_ratio, over_seg_ratio)
                # NOTE: too complicated (let's just do IOU)
                # target = float(intersections[imax]) / len(np.union1d(seg, gtsegs[imax]))

                # find the best ground truth segment according to iou (instead of intersection)
                if class_specific:   # target_iou is a vector
                    target_iou = np.zeros(len(CLASSES))
                    for i, CLASS in enumerate(CLASSES):
                        all_iou = [float(np.sum(weights[np.intersect1d(seg,gtseg)])) / np.sum(weights[np.union1d(seg,gtseg)]) for gtseg, gtclass in zip(gtsegs, gtclasses) if gtclass == CLASS]
                        if len(all_iou) > 0: 
                            target_iou[i] = np.max(all_iou)
                else:  # target_iou is a scalar
                    all_iou = [float(np.sum(weights[np.intersect1d(seg,gtseg)])) / np.sum(weights[np.union1d(seg,gtseg)]) for gtseg in gtsegs]
                    target_iou = np.max(all_iou)

                # add data into our training pool
                points_list.append(points)
                # target_list.append(np.max(ious))
                target_list.append(target_iou)

    with open('%s/%s.pkl' % (data_dir, split), 'wb') as f:
        pickle.dump(points_list, f, protocol=2)
        pickle.dump(target_list, f, protocol=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--reweight', type=str, required=True, choices=['squared_distance', 'distance', 'none'])
    parser.add_argument('--class-specific', action='store_true')
    parser.add_argument('--remove-ground', action='store_true')
    args = parser.parse_args()

    process(args.split, args.reweight, args.class_specific, args.remove_ground)
