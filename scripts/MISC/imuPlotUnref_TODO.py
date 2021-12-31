#!/usr/bin/env python
import rospy
import tf
from tf import *
import sys, struct, time,threading
# Messages
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion, Vector3
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt

# usage: rosrun kloam imu.py
saveFolder = "/home/binpeng/Documents/KLOAM/imu_test/"
pubIMU = rospy.Publisher("/imuF", Imu, queue_size = 100)
xAcc = []
yAcc = []
zAcc = []
xAV = []
AVbuf = Vector3()
initFinish = False
alpha = 0.1

def imu_callback(msg):
	xAcc.append(msg.linear_acceleration.x)
	yAcc.append(msg.linear_acceleration.y)
	zAcc.append(msg.linear_acceleration.z)
	xAV.append(msg.angular_velocity.x)

def imu_filter_callback(msg):
	if initFinish:
		imuOut = msg
		AV = Vector3()
		AV.x = alpha*msg.linear_acceleration.x + (1-alpha)*AVbuf.x
		AV.y = alpha*msg.linear_acceleration.y + (1-alpha)*AVbuf.y
		AV.z = alpha*msg.linear_acceleration.z + (1-alpha)*AVbuf.z
		imuOut.linear_acceleration = AV
		# imuOut.angular_velocity = msg.angular_velocity
		# imuOut.orientation = msg.orientation
		pubIMU.publish(imuOut)
	else:
		initFinish = True
	AVbuf = msg.linear_acceleration
		
		


def odomListener():
	rospy.init_node('acc_plot_node', anonymous = True)
	rospy.loginfo("IMU plotting node initialized!")
	subIMUfilter = rospy.Subscriber("/imu", Imu, imu_filter_callback)
	subIMUplot = rospy.Subscriber("/imuF", Imu, imu_callback)
	rospy.spin()
def plot(folder):
	rospy.loginfo("Start plotting imu")
	plt.title('IMU')
	plt.subplot(1,2,1)
	plt.plot(xAcc,color='green', label='acc x')
	plt.xlabel('Acc x')
	plt.subplot(1,2,2)
	plt.plot(xAV,color='red', label='angular vel x')
	plt.xlabel('Angular velocity x')
	plt.legend()
	# plt.savefig(folder+"imu.jpg")
	plt.show()
	rospy.loginfo("Done plotting and saving")
# def save(folder):
# 	with open(folder+"odomW.txt",'w') as f1:
# 		for i in range(len(xWheel)):
# 			f1.write("%8.3f %8.3f \n"%(xWheel[i],yWheel[i]))
# 	with open(folder+"odomL.txt",'w') as f2:
# 		for i in range(len(xSLAM)):
# 			f2.write("%8.3f %8.3f \n"%(xSLAM[i],ySLAM[i]))
# 	rospy.loginfo("Done saving odometry as txts")
if __name__ == '__main__':
	# try: 
	# 	odomListener()
	# except rospy.ROSInterruptException: // this won't catch the "ctrl+C"
	# 	print("wtf")
	# 	plot()
	odomListener()
	if rospy.is_shutdown():
		plot(saveFolder)
		# save(saveFolder)



 

