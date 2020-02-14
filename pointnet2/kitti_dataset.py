import os
import os.path
import json
import numpy as np
import sys
import pickle

class KittiDataset(object):
    def __init__(self, split='train', min_pts=10, max_pts=1024, batch_size=32, data_dir='',
                 class_specific=False, resample=False, random_flip=False, random_jitter=False,
                 rotate_to_center=True, shuffle=True):
        if not resample:
            raise ValueError('Temporariliy disable not resampling')
        self.CLASSES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
        self.min_pts = min_pts
        self.max_pts = max_pts
        self.class_specific = class_specific
        self.resample = resample
        self.random_flip = random_flip
        self.random_jitter = random_jitter
        self.batch_size = batch_size
        self.rotate_to_center = rotate_to_center
        assert(self.rotate_to_center)

        with open('%s/%s.pkl' % (data_dir, split), 'rb') as f:
            raw_points_list = pickle.load(f)
            raw_target_list = pickle.load(f)

        # ignore segments with less than 10 points
        self.points_list = []
        self.target_list = []
        for (points, target) in zip(raw_points_list, raw_target_list):
            if len(points) >= self.min_pts:
                self.points_list.append(points)
                self.target_list.append(target)

        self.shuffle = shuffle
        self.reset()

    def _get_item(self, index):
        points = self.points_list[index]
        target = self.target_list[index]

        # standardization: zero-mean
        if self.rotate_to_center:
            _mean = points.mean(axis=0)
            _theta = np.arctan2(_mean[1], _mean[0])
            _rot = np.array([[ np.cos(_theta), np.sin(_theta)],
                             [-np.sin(_theta), np.cos(_theta)]])
            points[:,:2] = _rot.dot((points[:,:2] - _mean[None,:2]).T).T

        # resampling to a fixed number of points
        # even though uniform resampling should not affect the mean
        # we do it after standardization because it is faster
        # NOTE: resampling uniformly preserves the target
        if self.resample:
            _choice = np.random.choice(len(points), self.max_pts, replace=True)
            points = points[_choice]

        # augmentation: flip the frustum point cloud
        if self.random_flip:
            # NOTE: randomly flip point cloud by y-axis (velo cs)
            # the intuition is that if we drive in the middle of a road
            # we might see the same thing on either side of the road
            # such phenomenon can be simulated using flipping
            if np.random.rand() > 0.5:
                points[:,1] = -points[:,1]

        # # augmentation: shift the entire frustum pointcloud along depth
        # if self.random_shift:
        #     # NOTE: this is inspired by Frustum PointNet
        #     # I feel they wanted to compensate noise in monocular depth prediction
        #     # I don't think it would be super helpful here...

        # augmentation: jitter
        if self.random_jitter:
            # parameters taken from modelnet_dataset.py of the PointNet++ project
            jitter = np.clip(0.01 * np.random.randn(points.shape[0], points.shape[1]), -0.05, 0.05)
            points += jitter

        return points, target

    def __getitem__(self, index):
        return self._get_item(index)

    def __len__(self):
        return len(self.points_list)

    def num_channel(self):
        return 3

    def reset(self):
        self.idxs = np.arange(0, len(self.points_list))
        if self.shuffle:
            np.random.shuffle(self.idxs)
        self.num_batches = (len(self.points_list)+self.batch_size-1) // self.batch_size
        self.batch_idx = 0

    def has_next_batch(self):
        return self.batch_idx < self.num_batches

    def next_batch(self, augment=False):
        ''' returned dimension may be smaller than self.batch_size '''
        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx+1) * self.batch_size, len(self.points_list))
        bsize = end_idx - start_idx
        batch_data = np.zeros((bsize, self.max_pts, self.num_channel()))
        if self.class_specific:
            batch_label = np.zeros((bsize, len(self.CLASSES)), dtype=np.float32)
        else: 
            batch_label = np.zeros((bsize), dtype=np.float32)
        for i in range(bsize):
            points, target = self._get_item(self.idxs[i+start_idx])
            batch_data[i] = points
            batch_label[i] = target
        self.batch_idx += 1
        return batch_data, batch_label

    # def next_batch(self):
    #     ''' returned dimension may be smaller than self.batch_size '''
    #     start_idx = self.batch_idx * self.batch_size
    #     end_idx = min((self.batch_idx+1) * self.batch_size, len(self.points_list))
    #     bsize = end_idx - start_idx
    #     # batch_data = np.zeros((bsize, self.npoints, self.num_channel()))
    #     # batch_label = np.zeros((bsize, 2), dtype=np.float32)
    #     raw_batch_points = []
    #     raw_batch_targets = []
    #     for i in range(bsize):
    #         points, target = self._get_item(self.idxs[i+start_idx])
    #         raw_batch_points.append(points)
    #         raw_batch_targets.append(target)
    #         # batch_data[i] = ps
    #         # batch_label[i] = cls
    #     # figure out the maximum segment and pad everything to match that
    #     max_size = max(len(points) for points in raw_batch_points)
    #     max_size = self.max_pts if max_size > self.max_pts else max_size
    #     for i in range(bsize):
    #         size = len(raw_batch_points[i])
    #         if size < max_size:
    #             choice = np.random.choice(size, max_size-size, replace=True)
    #             raw_batch_points[i] = np.concatenate((raw_batch_points[i], raw_batch_points[i][choice]), axis=0)
    #         if size > max_size:
    #             choice = np.random.choice(size, max_size, replace=False)
    #             raw_batch_points[i] = raw_batch_points[i][choice]
    #     assert(all(len(points)<=self.max_pts for points in raw_batch_points))
    #     batch_points = np.array(raw_batch_points)
    #     batch_targets = np.array(raw_batch_targets)
    #     self.batch_idx += 1
    #     # if augment: batch_data = self._augment_batch_data(batch_data)
    #     # return batch_data, batch_label
    #     return batch_points, batch_targets


if __name__ == '__main__':
    # d = ModelNetDataset(root = '../data/modelnet40_normal_resampled', split='test')
    eps_list=[0.25, 0.5, 1.0, 2.0]
    data_dir = 'data/kitti_eps%s' % ('_'.join('%.2f' % eps for eps in eps_list))
    d = KittiDataset(root='/home/peiyunh/Kitti/object/', data_dir=data_dir, split='train')
    print(d.shuffle)
    print(len(d))
    import time
    tic = time.time()
    for i in range(10):
        ps, cls = d[i]
    print(time.time() - tic)
    print(ps.shape, type(ps), cls)

    print(d.has_next_batch())
    ps_batch, cls_batch = d.next_batch(True)
    print(ps_batch.shape)
    print(cls_batch.shape)
