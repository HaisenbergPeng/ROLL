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

record_time = '2012-01-08'
root_dir = '/mnt/sdb/Datasets/NCLT/datasets/MISC/2012-01-08/2012-01-08'
save_dir = root_dir
ms25_filename = os.path.join(root_dir,'ms25.csv')
ms25E_filename = os.path.join(root_dir,'ms25_euler.csv')
gps_filename = os.path.join(root_dir,'gps_rtk.csv')
velo_bin_path = os.path.join(root_dir,"velodyne_hits.bin")
gt_filename = os.path.join(root_dir,'groundtruth_'+record_time+'.csv')
gt_cov_filename = os.path.join(root_dir,record_time,'cov_'+record_time+'.csv')
velo_frame_id = '/lidar'
imu_frame_id = '/imu'
gt_frame_id = '/body'
gps_frame_id = '/gps'
lidar_topic = '/velodyne_points'
gt_topic = '/ground_truth'
gps_rtk_topic = '/fix'
write_lidar_data = True
if write_lidar_data:
    bag_name = os.path.join(save_dir, record_time+'_bin.bag')
else:
    bag_name = os.path.join(save_dir, record_time+'._no_vel.bag')
# print(iterable)
bar = progressbar.ProgressBar()
bag = rosbag.Bag(bag_name, 'w')

def main():
    # write_gt_cov() 
    
    write_gt() 
    write_gps()
    write_imu()

    f_vel = open(velo_bin_path, "rb")
    try:
        write_vel(f_vel, bag)
    finally:
        f_vel.close()
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
# sth wrong
def write_gt_cov():
    # ground truth
    print('Converting ground truth with covariance......')
    # add covariance
    gt = np.loadtxt(gt_filename, delimiter = ",")
    cov = np.loadtxt(gt_cov_filename, delimiter = ",")

    t_cov = cov[100:, 0] # increase it if A value in x_new is above the interpolation range 
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

def verify_magic(s):

    magic = 44444

    m = struct.unpack('<HHHH', s)

    return len(m)>=3 and m[0] == magic and m[1] == magic and m[2] == magic and m[3] == magic

def read_first_vel_packet(f_vel, bag):
    magic = f_vel.read(8)

    num_hits = struct.unpack('<I', f_vel.read(4))[0]
    
    utime = struct.unpack('<Q', f_vel.read(8))[0]

    f_vel.read(4) # padding
    for i in range(num_hits):
        x = struct.unpack('<H', f_vel.read(2))[0]
        y = struct.unpack('<H', f_vel.read(2))[0]
        z = struct.unpack('<H', f_vel.read(2))[0]
        i = struct.unpack('B', f_vel.read(1))[0]
        l = struct.unpack('B', f_vel.read(1))[0]
    return utime
    
def write_vel(f_vel,bag):
    size = os.path.getsize(velo_bin_path)
    print(size/28/32)
    pbar = tqdm(total=size)
    num_hits = 384
    is_first = True
    last_time = 0
    last_packend_time = 0

    if is_first:
        is_first = False
        magic = f_vel.read(8)
        num_hits = struct.unpack('<I', f_vel.read(4))[0]
        last_packend_time = last_time = struct.unpack('<Q', f_vel.read(8))[0]
        f_vel.read(4) # padding
        for i in range(num_hits):
            x = struct.unpack('<H', f_vel.read(2))[0]
            y = struct.unpack('<H', f_vel.read(2))[0]
            z = struct.unpack('<H', f_vel.read(2))[0]
            i = struct.unpack('B', f_vel.read(1))[0]
            l = struct.unpack('B', f_vel.read(1))[0]
    data=[]
    while True:
        # a = f_vel.read(size-3)
        magic = f_vel.read(8)
        if len(magic) < 8:
            return

        if magic == '': # eof
            print("NO MAGIC")
            return

        if not verify_magic(magic):
            print("Could not verify magic")
            return 

        num_hits = struct.unpack('<I', f_vel.read(4))[0]
        utime = struct.unpack('<Q', f_vel.read(8))[0]
        f_vel.read(4) # padding
        pbar.update(24)
        
        # if utime > 1357847302646637:
        #     return
        
        layer_point_num = np.zeros( 32 ,dtype=np.int16)
        yaw_ind = np.zeros( (32,12) ,dtype=np.float32)
        offset_time_ind = np.zeros( (32,12) ,dtype=np.float32) 
        offset_time_base = last_packend_time - last_time # what is for /???
        dt = float(utime - last_packend_time) / 12.0
        l_last = 0
        N = 1

        # print(utime, num_hits, offset_time_base, dt)

        for i in range(num_hits):
            x = struct.unpack('<H', f_vel.read(2))[0]
            y = struct.unpack('<H', f_vel.read(2))[0]
            z = struct.unpack('<H', f_vel.read(2))[0]
            i = struct.unpack('B', f_vel.read(1))[0]
            l = struct.unpack('B', f_vel.read(1))[0]

            if l <= l_last:
                N += 1
            
            if N>12:
                N = 12

            l_last = l
            
            # layer_point_num[l] += 1
            # offset_time_ind[l][layer_point_num[l]] = offset_time_base + dt * N
            # if layer_point_num[l] >= 12:
            #     print(l, yaw_ind[l], offset_time_ind[l])
            x, y, z = convert(x, y, z)
            offset_time = int(offset_time_base + dt * N)
            if offset_time + last_time >= utime:
                offset_time = utime - last_time
            off_t = float(offset_time)
            data.append([x, y, z, i, offset_time, l])
            # if l == 31:
            #     print(l,offset_time_base + dt * N, int(offset_time_base + dt * N))
            # print(float(offset_time))
            # print(offset_time)
            pbar.update(8)
        
        last_packend_time = utime

        # fill pcl msg
        if utime - last_time > 1e5:
            # print(last_time / 1e6)
            # print(utime)
            header = Header()
            header.frame_id = velo_frame_id
            header.stamp = rospy.Time.from_sec(last_time/1e6)
            fields = [PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1),
                    PointField('intensity', 12, PointField.FLOAT32, 1),
                    PointField('time', 16, PointField.FLOAT32, 1),
                    PointField('ring', 20, PointField.UINT16, 1)]
            pcl_msg = pcl2.create_cloud(header, fields, data)
            pcl_msg.is_bigendian = False
            pcl_msg.is_dense = False
            bag.write("velodyne_points", pcl_msg, t=pcl_msg.header.stamp)
            last_time = utime
            data=[]

def write_gt():
    # ground truth
    print('Converting ground truth......')
    # add covariance
    pose_gt = np.loadtxt(gt_filename, delimiter = ",")
    # cov = np.loadtxt(gt_cov_filename, delimiter = ",")

    # t_cov = cov[:, 0]
    # # Note: Interpolation is not needed, this is done as a convenience
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

        if modes[i]==0 or modes[i]==1 or num_satss[i] <= 6:
            status.status = NavSatStatus.STATUS_NO_FIX # -1
        else:
            status.status = NavSatStatus.STATUS_FIX # 0

        status.service = NavSatStatus.SERVICE_GPS

        num_sats = UInt16()
        num_sats.data = num_satss[i]

        fix = NavSatFix()
        fix.header.stamp = timestamp # necessary
        fix.header.frame_id = gps_frame_id
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


