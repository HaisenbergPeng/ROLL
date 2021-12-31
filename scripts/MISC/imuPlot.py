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

## PROBLEM:
# 1. # get rid of abrupt change: a period of constant value, why???
# 2. imu and imuF are both "filtered"?
class imuPlot():
	def __init__(self):
		rospy.init_node('acc_plot_node', anonymous = True)
		rospy.loginfo("IMU plotting node initialized!")
		self.subIMUfilter = rospy.Subscriber("/imu_6", Imu, self.imu_filter_callback)
		self.subIMU = rospy.Subscriber("/imu_6", Imu, self.imu_callback)
		self.subIMUfiltered = rospy.Subscriber("/imuF", Imu, self.imu_callbackF)
		self.pubIMU = rospy.Publisher("/imuF", Imu, queue_size = 100)
		self.saveFolder = "/home/binpeng/Documents/KLOAM-LOC/imu_test/"
		self.xAcc = []
		self.yAcc = []
		self.zAcc = []
		self.xAV = []		
		self.yAV = []	
		self.zAV = []	
		self.xAccF = []
		self.yAccF = []
		self.zAccF = []
		self.xAVF = []		
		self.yAVF = []	
		self.zAVF = []
		self.initFinish = False
		self.stdArray = []
		self.alpha = 0.99
		self.AVbuf = Vector3()
		self.threD = 2
		rospy.spin()
	def imu_callback(self, msg):
		self.xAcc.append(msg.linear_acceleration.x)
		self.yAcc.append(msg.linear_acceleration.y)
		self.zAcc.append(msg.linear_acceleration.z)
		self.xAV.append(msg.angular_velocity.x)
		self.yAV.append(msg.angular_velocity.y)
		self.zAV.append(msg.angular_velocity.z)
	
	def imu_callbackF(self, msg):
		self.xAccF.append(msg.linear_acceleration.x)
		self.yAccF.append(msg.linear_acceleration.y)
		self.zAccF.append(msg.linear_acceleration.z)
		self.xAVF.append(msg.angular_velocity.x)
		self.yAVF.append(msg.angular_velocity.y)
		self.zAVF.append(msg.angular_velocity.z)

	def imu_filter_callback(self, msg):
		if self.initFinish:
			imuOut = Imu()
			# AV = Vector3()
			# seems useless!!! just a time delay
			# AV.x = self.alpha*msg.linear_acceleration.x + (1-self.alpha)*self.AVbuf.x
			# AV.y = self.alpha*msg.linear_acceleration.y + (1-self.alpha)*self.AVbuf.y
			# AV.z = self.alpha*msg.linear_acceleration.z + (1-self.alpha)*self.AVbuf.z
			# imuOut.linear_acceleration = AV
			imuOut.angular_velocity = msg.angular_velocity
			imuOut.orientation = msg.orientation
			imuOut.linear_acceleration = msg.linear_acceleration
			imuOut.header = msg.header
			# get rid of abrupt change: a period of constant value, why???
			if abs(msg.linear_acceleration.x - self.AVbuf.x) > self.threD:
				imuOut.linear_acceleration.x = self.AVbuf.x
			if abs(msg.linear_acceleration.y - self.AVbuf.y) > self.threD:
				imuOut.linear_acceleration.y = self.AVbuf.y
			if abs(msg.linear_acceleration.z - self.AVbuf.z) > self.threD:
				imuOut.linear_acceleration.z = self.AVbuf.z
			
			
			if not rospy.is_shutdown():
				self.pubIMU.publish(imuOut)
		else:
			self.initFinish = True
		self.AVbuf = msg.linear_acceleration
			
	def plot(self):
		rospy.loginfo("Start plotting imu")
		plt.title('IMU')
		plt.subplot(3,2,1)
		plt.plot(self.xAcc,color='green',label = 'xAcc')
		# plt.plot(self.xAccF,color='red',label = 'xAcc filtered')
		plt.ylabel('Acc x m/s^2')
		# plt.legend()
		plt.subplot(3,2,3)
		plt.plot(self.yAcc,color='green',label = 'yAcc')
		# plt.plot(self.yAccF,color='red',label = 'yAcc filtered')
		plt.ylabel('Acc y m/s^2')
		# plt.legend()
		plt.subplot(3,2,5)
		plt.plot(self.zAcc,color='green',label = 'zAcc')
		# plt.plot(self.zAccF,color='red',label = 'zAcc filtered')
		plt.ylabel('Acc z m/s^2')
		# plt.legend()
		plt.subplot(3,2,2)
		plt.plot(self.xAV,color='red',label = 'xAV')
		plt.ylabel('Angular velocity x rad/s')
		plt.subplot(3,2,4)
		plt.plot(self.yAV,color='red',label = 'zAV')
		plt.ylabel('Angular velocity y rad/s')
		plt.subplot(3,2,6)
		plt.plot(self.zAV,color='red',label = 'zAV')
		plt.ylabel('Angular velocity z rad/s')
		# plt.legend()
		plt.savefig(self.saveFolder+"imu2.jpg")
		plt.show()
		rospy.loginfo("Done plotting and saving")
if __name__ == '__main__':
	imuP = imuPlot()
	if rospy.is_shutdown():
		imuP.plot()



 

