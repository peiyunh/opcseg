
![Demo result](https://raw.githubusercontent.com/peiyunh/3dseg/master/demo.png)

# Learning to Optimally Segment Point Clouds
By Peiyun Hu, David Held, and Deva Ramanan at Carnegie Mellon University and Argo AI. 

## Introduction
For segmenting LiDAR point clouds, if we score a segmentation by the worst objectness score among its individual segments, there is an algorithm that efficiently finds the **optimal** worst-case segmentation among an exponentially large number of candidate segmentations. The proposed algorithm takes a pre-processed LIDAR point cloud (top - with background removed) and produces a class-agnostic instance-level segmentation over all foreground points (bottom). We use a different color for each segment and plot an extruded polygon to show the spatial extent. 

This work was initially described in an [arXiv tech report](https://arxiv.org/abs/1912.04976). 

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

## Installation 


## Demo


## Training 


## FAQ
