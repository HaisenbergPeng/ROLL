clc;
clear;
close all
% % extrensics
% 
% t_lidar2imu = [0.010524; 1.613485 ; 1.228857]
% quat_dxl = [-0.003936 0.007856 0.707845 0.706313]; % xyzw
% quat_dxl2 = [0.706313 -0.003936 0.007856 0.707845]; % wxyz
% R_lidar2imu = quat2rotm(quat_dxl2)
% R_imu2lidar = R_lidar2imu'
% t_imu2lidar = -R_imu2lidar*t_lidar2imu

% pcshow

pc = pcread("/mnt/sdb/Datasets/NCLT/datasets/fastlio_loc2/2012-02-02-gt/map_pcd/errorMap.pcd");
pcshow(pc)