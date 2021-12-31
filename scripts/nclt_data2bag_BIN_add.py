from re import S
import os
import tf
import cv2
import rospy
import rosbag
import progressbar
from tf2_msgs.msg import TFMessage
from datetime import datetime
from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo, Imu, NavSatFix
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped, TwistStamped, Transform
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import numpy as np
import argparse
import sys
import struct
import scipy.interpolate

import rosbag, rospy
from std_msgs.msg import Float64, UInt16
from sensor_msgs.msg import NavSatStatus, NavSatFix

from geometry_msgs.msg import Vector3, Quaternion
from math import sin,cos

from tqdm import tqdm
# import scipy

record_time = '2012-01-15'
root_dir = '/media/binpeng/BIGLUCK/Datasets/NCLT/datasets'
save_dir = '/media/binpeng/BIGLUCK/Datasets/NCLT/datasets'
gt_filename = os.path.join(root_dir,record_time,'groundtruth_'+record_time+'.csv')
gt_cov_filename = os.path.join(root_dir,record_time,'cov_'+record_time+'.csv')
velo_frame_id = '/lidar'
imu_frame_id = '/imu'
gt_frame_id = '/body'
gps_frame_id = '/gps'
lidar_topic = '/velodyne_points'
gt_topic = '/ground_truth_cov'
gps_rtk_topic = '/fix'
# bag_name = os.path.join(save_dir, record_time, record_time+'_test.bag')

bag_name = '/media/binpeng/BIGLUCK/Datasets/NCLT/datasets/2012-01-15/2012-01-15_bin.bag'

# print(iterable)
bar = progressbar.ProgressBar()
bag = rosbag.Bag(bag_name, 'a')

def main():
    write_gt_cov()     
    bag.close()

def toROScov(inputArr):
    if len(inputArr) != 21:
        print("wrong length of inputArr!")
        return
    # outArr = [0 for _ in range(36)] # or 0*[36]
    # for i in range(6):
    #     for j in range(6):
    #         outArr[i*6+j] = inputArr[6*i+j-(i-1)*i/2]
    #         outArr[6*j+i] = inputArr[6*i+j-(i-1)*i/2]
    outArr = (inputArr[0],inputArr[1], inputArr[2], inputArr[3], inputArr[4], inputArr[5],\
    inputArr[1] ,inputArr[6] ,inputArr[7] ,inputArr[8] ,inputArr[9], inputArr[10],\
    inputArr[2] ,inputArr[7] ,inputArr[11] ,inputArr[12] ,inputArr[13] ,inputArr[14],\
    inputArr[3] ,inputArr[8], inputArr[12] ,inputArr[15] ,inputArr[16], inputArr[17],\
    inputArr[4] ,inputArr[9] ,inputArr[13], inputArr[16], inputArr[18] ,inputArr[19],\
    inputArr[5] ,inputArr[10] ,inputArr[14] ,inputArr[17] ,inputArr[19], inputArr[20])
    return outArr

def write_gt_cov():
    # ground truth
    print('Converting ground truth with covariance......')
    # add covariance
    gt = np.loadtxt(gt_filename, delimiter = ",")
    cov = np.loadtxt(gt_cov_filename, delimiter = ",")

    t_cov = cov[5:, 0]
    # Note: Interpolation is not needed, this is done as a convenience
    interp = scipy.interpolate.interp1d(gt[:, 0], gt[:, 1:], kind='nearest', axis=0)
    pose_gt = interp(t_cov)

    # NED (North, East Down)
    x = pose_gt[:, 0]
    y = pose_gt[:, 1]
    z = pose_gt[:, 2]

    r = pose_gt[:, 3]
    p = pose_gt[:, 4]
    h = pose_gt[:, 5]

    for i, utime in enumerate(t_cov):
        odom = Odometry()
        odom.header.frame_id = gt_frame_id
        timestamp = rospy.rostime.Time.from_sec(utime/1e+6)
        odom.header.stamp = timestamp
        odom.pose.pose.position.x = x[i]
        odom.pose.pose.position.y = y[i]
        odom.pose.pose.position.z = z[i]
        # sxyz by default, RzRyRx, same with NCLT
        q = tf.transformations.quaternion_from_euler(r[i],p[i],h[i],'sxyz') 
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        f64 = toROScov(cov[i,1:])
        odom.pose.covariance = f64
        bag.write(gt_topic,odom,t=timestamp)

if __name__ == "__main__":
    main()


