![Demo result](https://raw.githubusercontent.com/peiyunh/3dseg/master/demo.png)

# Learning to Optimally Segment Point Clouds
By Peiyun Hu, David Held, and Deva Ramanan at Carnegie Mellon University.

## Introduction
For segmenting LiDAR point clouds, if we score a segmentation by the worst objectness score among its individual segments, there is an algorithm that efficiently finds the **optimal** worst-case segmentation among an exponentially large number of candidate segmentations. The proposed algorithm takes a pre-processed LIDAR point cloud (top - with background removed) and produces a class-agnostic instance-level segmentation over all foreground points (bottom). We use a different color for each segment and plot an extruded polygon to show the spatial extent.

You can read our paper (open-access) here: https://ieeexplore.ieee.org/abstract/document/8954778. 

In this repo, we provide our implementation of this work.

### Citing us
If you find our work useful in your research, please consider citing:
```latex
@article{hu2020learning,
  title={Learning to Optimally Segment Point Clouds},
  author={Hu, Peiyun and Held, David and Ramanan, Deva},
  journal={IEEE Robotics and Automation Letters},
  year={2020},
  publisher={IEEE}
}
```

## Roadmap
Currently, code release is a work in progress. Below are what I plan to work on next:
- Update README to describe
  - How to train the objectness model (PointNets) (`pointnet2/`)
  - How to run segmentation (`segment_with_pointnet.py`)
  - How to evaluate under-segmentation and over-segmentation (`evaluate_under_over*.py`)
  - How to evaluate instance-segmentation (`evaluate_instance_all.py`)
  - How to evaluate existing detectors (`evaluate_instance_all.py`)
- Merge `evaluate_under_over.py` and `evaluate_under_over_ovlp_part_ignored.py`
  - When evaluating under-segmentation and over-segmentation, we either
    - Ignore objects with overlapping bounding boxes
    - Or ignore points that fall into the overlapping regions
  - Right now, they are highly redundant. I plan to merge them together.
- Release all pre-trained models. 

## Installation


## Demo


## Training


## FAQ
