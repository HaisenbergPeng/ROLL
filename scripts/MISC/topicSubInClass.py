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

class odomPlot():
	def __init__(self):
		rospy.init_node('odom_plot_node', anonymous = True)
		self.xWheel = []
		self.yWheel = []
		self.wWheel = []
		self.xSLAM = []
		self.ySLAM = []
		self.subWheel = rospy.Subscriber("/odom", Odometry, self.odom_callback)
		self.subLidar = rospy.Subscriber("/kloam/mapping/odometry", Odometry, self.lidar_callback)
		self.start() # cannot put rospy.spin() here, otherwise you will get "no init_node" error, don't know why
	def odom_callback(self,odomW):
		self.xWheel.append(odomW.pose.pose.position.x)
		self.yWheel.append(odomW.pose.pose.position.y)
		# tf.createQuaternionFrom
	def lidar_callback(self,odomL):
		self.xSLAM.append(odomL.pose.pose.position.x)
		self.ySLAM.append(odomL.pose.pose.position.y)
	def plot(self):
		plt.title('Odometry comparison')
		plt.plot(self.xWheel,self.yWheel,color='green', label='Wheel encoder')
		plt.plot(self.xSLAM,self.ySLAM,color='red', label='SLAM odometry')
		plt.legend()
		plt.xlabel('X (m)')
		plt.ylabel('Y (m)')
		plt.show()
	def start(self):
		rospy.spin()
if __name__ == '__main__':
	odom = odomPlot()
	if rospy.is_shutdown():
		odom.plot()



 

