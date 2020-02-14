python train_multi_gpu.py --num_gpus 4 --loss MSE --data_dir data/kitti_eps0.25_0.50_1.00_2.00 --log_dir logs/pointnet2_reg_msg_mse_min10_max1024_resampled --resample

#### BELOW ARE EXPERIMENTS WITH AUGMENTATION (TURNS OUT THEY ARE NOT USEFUL)
# python train_multi_gpu.py --num_gpus 4 --loss MSE --data_dir data/kitti_eps0.25_0.50_1.00_2.00_sqr_d_w_iou --log_dir logs/pointnet2_reg_msg_mse_min10_max1024_sqr_d_w_iou_resampled_augmented --resample --augment
# python train_multi_gpu.py --num_gpus 4 --loss MSE --data_dir data/kitti_eps0.25_0.50_1.00_2.00 --log_dir logs/pointnet2_reg_msg_mse_min10_max1024_resampled_augmented --resample --augment
# python train_multi_gpu.py --num_gpus 4 --loss L1 --data_dir data/kitti_eps0.25_0.50_1.00_2.00_sqr_d_w_iou --log_dir logs/pointnet2_reg_msg_l1_min10_max1024_sqr_d_w_iou_resampled_augmented --resample --augment
# python train_multi_gpu.py --num_gpus 4 --loss L1 --data_dir data/kitti_eps0.25_0.50_1.00_2.00 --log_dir logs/pointnet2_reg_msg_l1_min10_max1024_resampled_augmented --resample --augment
