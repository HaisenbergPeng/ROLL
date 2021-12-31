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

import rosbag, rospy
from std_msgs.msg import Float64, UInt16
from sensor_msgs.msg import NavSatStatus, NavSatFix

from geometry_msgs.msg import Vector3, Quaternion
from math import sin,cos
# import scipy

record_time = '2013-02-23'
root_dir = '/media/binpeng/BIGLUCK/Datasets/NCLT/datasets'
save_dir = '/media/binpeng/BIGLUCK/Datasets/NCLT/datasets'
ms25_filename = os.path.join(root_dir, record_time,record_time,'ms25.csv')
ms25E_filename = os.path.join(root_dir, record_time,record_time,'ms25_euler.csv')
gps_filename = os.path.join(root_dir, record_time,record_time,'gps_rtk.csv')
velo_data_dir = os.path.join(root_dir, record_time,record_time,'velodyne_sync')
gt_filename = os.path.join(root_dir,'groundtruth_'+record_time+'.csv')
gt_cov_filename = os.path.join(root_dir,'cov_'+record_time+'.csv')
velo_frame_id = '/lidar'
imu_frame_id = '/imu'
gt_frame_id = '/body'
lidar_topic = '/velodyne_points'
gt_topic = '/ground_truth'
gps_rtk_topic = '/fix'
write_lidar_data = True
if write_lidar_data:
    bag_name = os.path.join(save_dir, record_time, record_time+'.bag')
else:
    bag_name = os.path.join(save_dir, record_time, record_time+'._no_vel.bag')
# print(iterable)
bar = progressbar.ProgressBar()
bag = rosbag.Bag(bag_name, 'w')

def main():
    write_gt()
    write_gps()
    write_imu()

    frameFewPoints = 0
    frameTooManyPoints = 0
    velo_filenames = sorted(os.listdir(velo_data_dir))
    num_files = len(velo_filenames)
    print("In total %d "%num_files+" files transformed")

    if write_lidar_data:
        try:
            print("Exporting velodyne data")
            for filename in bar(velo_filenames):
                velo_filename = os.path.join(velo_data_dir, filename)
                # read binary data
                f_bin = open(velo_filename, "r")
                hits = []
                cnt = 0
                while True:
                    x_str = f_bin.read(2)
                    if x_str == '': # eof
                        break
                    x = struct.unpack('<H', x_str)[0]
                    y = struct.unpack('<H', f_bin.read(2))[0]
                    z = struct.unpack('<H', f_bin.read(2))[0]
                    i = struct.unpack('B', f_bin.read(1))[0] #intensity
                    l = struct.unpack('B', f_bin.read(1))[0] # scan ID or ring
                    x, y, z = convert(x, y, z)
                    hits += [x, y, z,float(i),float(l)]
                    cnt += 1

                f_bin.close()
                if cnt < 1000:
                    # print("points too few!")
                    frameFewPoints += 1
                    continue
                elif cnt > 2*1800*32:
                    frameTooManyPoints += 1
                    # print("points too many!")
                #     continue
                # pcl_msg = pcl2.create_cloud(header, fields, scan)
                # bag.write(topic, pcl_msg, t=pcl_msg.header.stamp)
                msg= PointCloud2()
                utime = float(filename[:-4]) # in micro seconds
                timestamp = rospy.Time.from_sec(utime/1e+6) 
                msg.header.stamp=  timestamp
                msg.header.frame_id= velo_frame_id
                msg.header.seq = 1
                msg.height = 1 # unordered pc
                msg.width = cnt # pc number
                msg.fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('intensity', 12, PointField.FLOAT32, 1),
                PointField('ring', 16, PointField.FLOAT32, 1)  ]
                msg.is_bigendian = False
                msg.is_dense=False
                msg.point_step = 20 # byte length of one point
                msg.row_step = msg.point_step * msg.width
                msg.data = np.array(hits,dtype='float32').tostring()        
                bag.write(lidar_topic, msg, timestamp)
        finally:
            bag.close()
    print("frames fewer than 1000 points: ",frameFewPoints)
    print("frames more than 2*1800*3200 points: ",frameTooManyPoints)

def toROScov(inputArr):
    print(len(inputArr))
    outArr = np.empty(64)
    for i in range(6):
        for j in range(6):
            outArr[i*6+j] = inputArr[6*i+j-(i+1)*i/2]
            outArr[6*j+i] = inputArr[6*i+j-(i+1)*i/2]
    return outArr


def toQuaternion(roll, pitch, yaw): # checked, perfectly right
    cy = cos(yaw * 0.5)
    sy = sin(yaw * 0.5)
    cp = cos(pitch * 0.5)
    sp = sin(pitch * 0.5)
    cr = cos(roll * 0.5)
    sr = sin(roll * 0.5) 
    q = Quaternion(cy * cp * sr - sy * sp * cr,
                sy * cp * sr + cy * sp * cr,
                sy * cp * cr - cy * sp * sr,
                cy * cp * cr + sy * sp * sr)
    return q
# test it with ALOAM liosam lego-loam
def convert(x_s, y_s, z_s):
    scaling = 0.005 # 5 mm
    offset = -100.0
    x = x_s * scaling + offset
    y = y_s * scaling + offset
    z = z_s * scaling + offset
    return x, y, z

def write_gt():
    # ground truth
    print('Converting ground truth......')
    pose_gt = np.loadtxt(gt_filename, delimiter = ",")
    # cov = np.loadtxt(gt_cov_filename, delimiter = ",")

    # t_cov = cov[:, 0]
    # # Note: Interpolation is not needed, this is done as a convience
    # interp = scipy.interpolate.interp1d(gt[:, 0], gt[:, 1:], kind='nearest', axis=0)
    # pose_gt = interp(t_cov)

    # NED (North, East Down)
    t_pose = pose_gt[:,0]
    x = pose_gt[:, 1]
    y = pose_gt[:, 2]
    z = pose_gt[:, 3]

    r = pose_gt[:, 4]
    p = pose_gt[:, 5]
    h = pose_gt[:, 6]
    # if len(t_pose) != len(t_cov):
    #     print('NOT EQUAL LENGTH! ','pose no.: ',len(t_pose),'cov no. : ',len(t_cov))  # pose no. : 835469, 'cov no. : ', 1297
    #     return
    for i, utime in enumerate(t_pose):
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
        # f64 = toROScov(cov[i,1:])
        # odom.pose.covariance = f64
        bag.write(gt_topic,odom,t=timestamp)

def write_gps():
    # gps
    print('bag name: ',bag_name)
    print('Converting gps data to bag ......')
    gps = np.loadtxt(gps_filename, delimiter = ",")
    utimes = gps[:, 0]
    modes = gps[:, 1]
    num_satss = gps[:, 2]
    lats = gps[:, 3]
    lngs = gps[:, 4]
    alts = gps[:, 5]
    tracks = gps[:, 6]
    speeds = gps[:, 7]

    for i, utime in enumerate(utimes):
        # print(utime)
        timestamp = rospy.Time.from_sec(utime/1e6)

        status = NavSatStatus()

        if modes[i]==0 or modes[i]==1:
            status.status = NavSatStatus.STATUS_NO_FIX # -1
        else:
            status.status = NavSatStatus.STATUS_FIX # 0

        status.service = NavSatStatus.SERVICE_GPS

        num_sats = UInt16()
        num_sats.data = num_satss[i]

        fix = NavSatFix()
        fix.header.stamp = timestamp # necessary
        fix.status = status

        fix.latitude = np.rad2deg(lats[i])
        fix.longitude = np.rad2deg(lngs[i])
        fix.altitude = alts[i]
        bag.write(gps_rtk_topic, fix, timestamp)

def write_imu():
    # imu
    print('Converting 6 axis imu data to bag......')
    ms25 = np.loadtxt(ms25_filename, delimiter = ",")
    # print('see some samples: ',ms25[1,:])
    t = ms25[:, 0]
    mag_x = ms25[:, 1] #magnetic field strength, in GAUSS
    mag_y = ms25[:, 2]
    mag_z = ms25[:, 3]
    accel_x = ms25[:, 4]
    accel_y = ms25[:, 5]
    accel_z = ms25[:, 6]
    rot_r = ms25[:, 7]
    rot_p = ms25[:, 8]
    rot_h = ms25[:, 9]

    ms25E = np.loadtxt(ms25E_filename, delimiter = ",")
    tE = ms25E[:, 0]
    rE = ms25E[:, 1]
    pE = ms25E[:, 2]
    hE = ms25E[:, 3]
    # not exactly the same, a few short
    # besides, it it too slow to merge using 'associate.py'
    print('6 axis data: ',len(t),'euler angle data: ',len(tE)) 

    for i in range(len(t)):
        imu = Imu()
        angular_v = Vector3()
        linear_a = Vector3()
        angular_v.x = rot_r[i]
        angular_v.y = rot_p[i]
        angular_v.z = rot_h[i]
        linear_a.x = accel_x[i]
        linear_a.y = accel_y[i]
        linear_a.z = accel_z[i]
        imuStamp = rospy.rostime.Time.from_sec(t[i]/ 1000000)  # from us to sec
        imu.header.stamp=imuStamp
        imu.header.frame_id = imu_frame_id
        imu.angular_velocity = angular_v
        imu.linear_acceleration = linear_a
        bag.write("imu_6",imu,imuStamp)

    for i in range(len(tE)):
        imu = Imu()
        euler = toQuaternion(rE[i],pE[i],hE[i])
        imuStamp = rospy.rostime.Time.from_sec(tE[i]/ 1000000)  # from us to sec
        imu.header.stamp=imuStamp
        imu.header.frame_id = imu_frame_id
        imu.orientation = euler
        bag.write("imu_euler",imu,imuStamp)
if __name__ == "__main__":
    main()


