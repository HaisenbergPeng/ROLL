#!/usr/bin/env python
import rospy
import tf
from tf import *
import sys, struct, time,threading
# Messages
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt

# usage: rosrun kloam topicSub.py
saveFolder = "/home/binpeng/Documents/KLOAM/test/"

xWheel = []
yWheel = []

xSLAM = []
ySLAM = []


def odom_callback(odomW):
	xWheel.append(odomW.pose.pose.position.x)
	yWheel.append(odomW.pose.pose.position.y)
	# tf.createQuaternionFrom
def lidar_callback(odomL):
	xSLAM.append(odomL.pose.pose.position.x)
	ySLAM.append(odomL.pose.pose.position.y)
def odomListener():
	rospy.init_node('odom_plot_node', anonymous = True)
	rospy.loginfo("Odom plotting node initialized!")
	subWheel = rospy.Subscriber("/odom", Odometry, odom_callback)
	subLidar = rospy.Subscriber("/kloam/mapping/odometry", Odometry, lidar_callback)
	rospy.spin()
def plot(folder):
	rospy.loginfo("Start plotting odometry")
	plt.title('Odometry comparison')
	plt.plot(xWheel,yWheel,color='green', label='Wheel encoder')
	plt.plot(xSLAM,ySLAM,color='red', label='SLAM odometry')
	plt.legend()
	plt.xlabel('X (m)')
	plt.ylabel('Y (m)')
	plt.savefig(folder+"odom.jpg")
	plt.show()
	rospy.loginfo("Done plotting and saving")
def save(folder):
	with open(folder+"odomW.txt",'w') as f1:
		for i in range(len(xWheel)):
			f1.write("%8.3f %8.3f \n"%(xWheel[i],yWheel[i]))
	with open(folder+"odomL.txt",'w') as f2:
		for i in range(len(xSLAM)):
			f2.write("%8.3f %8.3f \n"%(xSLAM[i],ySLAM[i]))
	rospy.loginfo("Done saving odometry as txts")
if __name__ == '__main__':
	# try: 
	# 	odomListener()
	# except rospy.ROSInterruptException: // this won't catch the "ctrl+C"
	# 	print("wtf")
	# 	plot()
	odomListener()
	if rospy.is_shutdown():
		if xWheel:
			x0 = xWheel[0]
			y0 = yWheel[0]
			for i in range(len(xWheel)):
				xWheel[i] = xWheel[i]-x0
				yWheel[i] = yWheel[i]-y0
		if xSLAM:		
			x0 = xSLAM[0]
			y0 = ySLAM[0]
			for i in range(len(xSLAM)):
				xSLAM[i] = xSLAM[i]-x0
				ySLAM[i] = ySLAM[i]-y0
		plot(saveFolder)
		save(saveFolder)



 

