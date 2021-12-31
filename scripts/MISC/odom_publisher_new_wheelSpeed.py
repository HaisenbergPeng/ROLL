#!/usr/bin/env python
import rospy
import tf
from tf import *
import sys, serial, struct, time,threading
from joystick import *
# Messages
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
#from odom_publish import myOdom


BAUD_RATES = (  
    300,600,1200,2400,4800,9600,14400,19200,
    28800,38400,57600,115200)

class myOdometry(object):
    docking_state = 0
    emergency_stop_flag = 0    
    x = 0.0;y = 0.0;theta = 0.0; v_cur = 0.0; w_cur = 0.0; v_cmd = 0.0; w_cmd = 0.0;
    js = 0;
    controlstate=0;    
    START ='\xFF\x01'
    END ='\x00\xBB'

    def __init__(self, tty, baudrate):
        # serial
        self.ser = serial.Serial(tty, baudrate=baudrate, timeout=2)
        self.buffer = ''
        #self.ser.open()
       # self.ser.write('\xAF\x10\x01\x00\x00\x01\x00\x00\x12')
        self.lock = threading.RLock()

    def read(self):
        #print self.ser.inWaiting() != 0
        #find tou and weiba 
        #print "read"

        self.ser.write('\xAF\x01\x01') # the command to read in odometry info
    
        while self.buffer.find(self.START)<0:
            self.buffer += self.ser.read(1)

        self.buffer += self.ser.read(11)
    
            
        self.buffer = self.buffer.split(self.START,2)[1]
        #print self.buffer
        str = self.buffer[0:11]
    #    print len(str)
    #    self.buffer = self.buffer[12:]
        
        if(len(str)<11):
            return

        with self.lock:
            #crti:2016-06-22
            # '>' means 'big endian';
            # 'h' short, 2 bytes; 'B' unsigned char, 1byte
            self.x, self.y, self.theta, self.v_cur,self.w_cur,self.controlstate =  struct.unpack('>3h2h1B', str[0:11])
            #self.x, self.y, self.theta, self.docking_state ,self.w_cur =  struct.unpack('>3h2b', str[0:8])
            #print 5
            self.x = self.x/100.0 
            self.y = self.y/100.0
            self.theta = self.theta / 1000.0
            #self.theta = -self.theta
            #self.theta = self.theta/10.0
            self.v_cur /= 1000.0; self.w_cur /= 1000.0; 
            #print('curcmd=(%.2f,%.3f)'%(self.v_cur, self.w_cur))
            while self.theta > 3.1415926: # make it -pi to pi
                self.theta -= 2*3.1415926
            while self.theta < -3.1415926:
                self.theta += 2*3.1415926
            
            self.buffer = ''            
            #print self.x, self.y


    def write (self):
        with self.lock:
            if self.cmd_v > 0:
                vsign = 0x01
                vabs = self.cmd_v * 1000.00
            else:
                vsign = 0x02
                vabs = -self.cmd_v * 1000.00
            if self.cmd_w > 0:
                wsign = 0x01
                wabs = self.cmd_w  * 1000.0
                #wabs = self.cmd_w * 180 /3.141592653 * 1000.0
            else:
                wsign = 0x02
                wabs = - self.cmd_w * 1000.0
                #wabs = - self.cmd_w * 180.0 /3.141592653 *1000.0

            hvabs = (int(vabs)>>8) & 0xFF
            lvabs = int(vabs) & 0xFF
            hwabs = (int(wabs)>>8) & 0xFF
            lwabs = int(wabs) & 0xFF

            parity = (0x10 + vsign + hvabs + lvabs + wsign + hwabs + lwabs) & 0xFF

            #print  vsign, vabs, wsign, wabs
            #print('write=%x,%x,%x,%x\n'%(vsign, vabs, wsign, wabs)),

            data = struct.pack('>9B', 0xAF, 0x10, vsign, hvabs, lvabs, wsign, hwabs, lwabs, parity)
            self.ser.write(data);
            self.ser.read(3);
            
    def set_velocity (self, v, w):
        with self.lock:
            self.cmd_v = v
            self.cmd_w = w
            #print self.cmd_v
            #print self.cmd_w

    
    def close(self):
        self.ser.flush()
        self.ser.close()
    
    def update(self, v, w):
        #print "odom update"
        self.set_velocity(v,w)
        self.write()
        self.read()
        
    def steer(self):
        return 0
    def odometry(self):
        return self.x, self.y, self.theta,self.controlstate
        
    def velocity(self):
        return self.v_cur, self.w_cur

    #def emergency_stop(self):
    #    return self.emergency_stop_flag
    def get_docking_state(self):
        return self.docking_state

    def set_pose_zero(self):
        self.ser.write('\xEE\xAA\x01\x00\x00\x01\x00\x00\xFE\xDC\x00\xBB')
        print 'set_pose_zero'

    def open_projector(self):
        self.ser.write('\xEE\xAA\x01\x00\x00\x01\x00\x00\xFE\xA2\x00\xBB')
        print 'open_projector'

    def close_projector(self):
        self.ser.write('\xEE\xAA\x01\x00\x00\x01\x00\x00\xFE\xA1\x00\xBB')
        print 'close_projector'

import time, array, threading, copy,  math

def create_thread(task):
    if callable(task):
        thread = threading.Thread(target = task)
        thread.setDaemon(True)
        thread.start()
        return thread
    else:
        raise 'task must be callable'

class Driver:
    cmd_lock = threading.RLock()
    cmd_v = 0; cmd_w = 0; cmd_manual = False

    odom_lock = threading.RLock()
    x = 0; y = 0; theta = 0; v = 0; w = 0; steer = 0

    set_zero_flag = True
    open_projector_flag = True
    close_projector_flag = True
    controlstate = 0
   

    def __init__(self, tty, baudrate,joy_index = 0):
		self.is_running = True
		try:
			self.joy = Joystick( joy_index )
			self.joy_thread = create_thread(self._joystick)
		except:
			print 'no joystick found'
        
		self.odom = myOdometry(tty, baudrate)
	
    def __del__(self):

		self.is_running = False
		self.joy_thread.join(1000) # timeout=1000, the thread will be forced give control to the main thread after 1000s

    def update(self, v, w):
        #print "driver update"
        with self.cmd_lock:
            if not self.cmd_manual:
                self.cmd_v = v; self.cmd_w = w
            cmd_v = copy.copy(self.cmd_v)
            cmd_w = copy.copy(self.cmd_w)
        #print cmd_v
        #print cmd_w
        self.odom.update( cmd_v, cmd_w)
        with self.odom_lock:
            self.x, self.y, self.theta,self.controlstate = self.odom.odometry()
            self.v, self.w = self.odom.velocity()
            self.steer = self.odom.steer()
    def _joystick(self):
        while self.is_running:
            self.joy.update()
            with self.cmd_lock:
                if self.joy.button(5):self.cmd_manual = True
                elif self.joy.button(4):
                    self.cmd_manual = False
                    self.cmd_v = 0
                    self.cmd_w = 0
                if self.cmd_manual:
                    self.cmd_v = -0.7*self.joy.axis(1)
                    self.cmd_w = -0.7*self.joy.axis(0)
                    if abs(self.cmd_v) < 0.15:
                        self.cmd_v = 0.0
                    if abs(self.cmd_w) < 0.15:
                        self.cmd_w = 0.0
                    self.cmd_v *= 1.0
                    self.cmd_w *= 1.5

                    if self.joy.button(1):self.cmd_v = 0.1
                    if self.joy.button(3):self.cmd_v = -0.1 
                    if self.joy.button(0):self.cmd_w = -0.3
                    if self.joy.button(2):self.cmd_w = 0.3

                if self.joy.button(12) and self.set_zero_flag:
                    self.cmd_manual = True
                    self.cmd_v = 0
                    self.cmd_w = 0
                    self.odom.set_pose_zero()
                    self.set_zero_flag = False
                if not self.joy.button(12):
                    self.set_zero_flag = True

                if self.joy.button(5) and self.open_projector_flag:
                    self.odom.open_projector()
                    self.open_projector_flag = False
                if not self.joy.button(5):
                    self.open_projector_flag = True

                if self.joy.button(4) and self.close_projector_flag:
                    self.odom.close_projector()
                    self.close_projector_flag = False
                if not self.joy.button(4):
                    self.close_projector_flag = True
            time.sleep(0.05)
                             


class Odom_publisher():
    
    def __init__(self, tty, baudrate):
        
        
        self.flag=0
        self.v = 0
        self.w = 0
        self.vl=0
        self.wl=0
        self.controlstate=0
        rospy.init_node('odom_publisher', anonymous = True)        
        self.pub = rospy.Publisher('/odom', Odometry, queue_size=10)
        #self.sub = rospy.Subscriber("/smoother_cmd_vel", Twist, self.vel_callback)
        #self.sub1= rospy.Subscriber("/flag1", Twist, self.flag_callback)
        #self.sub = rospy.Subscriber("/cmd_vel1", Twist, self.vel_callback)
        #self.sub2= rospy.Subscriber("/cmd_vel1", Twist, self.vel_callback1)
        self.sub = rospy.Subscriber("/cmd_vel", Twist, self.vel_callback)
        self.rate = rospy.Rate(20)
        self.driver = Driver(tty,baudrate,0)
        self.current_time = rospy.get_rostime()
        self.receivCmd_time = self.current_time
        self.currentv=0
        self.previousv=0
        self.start()

    def vel_callback(self,data):
        self.receivCmd_time = rospy.Time.now()
        self.v = data.linear.x
        self.w  = data.angular.z
        #print self.v
        #print self.w

    def flag_callback(self,data):
        self.receivCmd_time = rospy.Time.now()
        self.flag = data.linear.x
        #print self.v
        #print self.w

    def vel_callback1(self,data):
        self.receivCmd_time = rospy.Time.now()
        self.vl = data.linear.x
        self.wl  = data.angular.z
        #print self.v
        #print self.w
        
    def start(self):
        
        while not rospy.is_shutdown():
            self.current_time = rospy.Time.now()
            x, y, t,controlste, v, w, steer, docking_state = self.update(self.v,self.w)
            rospy.loginfo('pos=(%.2f,%.2f,%.2f)\tvw=(%.2f,%.2f)\tsteer=%.2f'%(x, y, t, v, w, controlste)),
    	    print('cmd=(%.2f,%.3f)'%(self.v, self.w))
            odom = Odometry()   
            quaternion = Quaternion()
      	    odom.header.stamp = rospy.Time.now()
      	    odom.header.frame_id = "odom"
            odom.child_frame_id = "base_footprint"
            #quaternion = tf.createQuaternionMsgFromRollPitchYaw (0.0, 0.0, t)
                            
            quaternion = tf.transformations.quaternion_from_euler(0, 0, t)
            odom.pose.pose.orientation.x = quaternion[0]
            odom.pose.pose.orientation.y = quaternion[1]
            odom.pose.pose.orientation.z = quaternion[2]
            odom.pose.pose.orientation.w = quaternion[3]
            odom.pose.pose.position.x=x
            odom.pose.pose.position.y=y
            odom.pose.pose.position.z=controlste
            odom.twist.twist.linear.x = v
            odom.twist.twist.linear.y = 0
            odom.twist.twist.angular.z = w
            self.pub.publish(odom)
            self.rate.sleep()

    def update(self,v,w):
        #if (w<0.1)and(w>-0.1):
           # w = 0
        self.previousv=self.currentv
        self.currentv=v
        
        self.driver.update(v,w)
        x, y, t,self.controlstate = self.driver.odom.odometry()
        x = x
        y = y
        v, w = self.driver.odom.velocity()
        steer = self.driver.odom.steer()
        #emergency_stop_flag = driver.odom.emergency_stop()
        docking_state = self.driver.odom.get_docking_state()
        #the pose is in the frame of body
        t = t - steer
        return x, y, t,self.controlstate, v, w, steer, docking_state

if __name__ == '__main__':
	try:
		odom = Odom_publisher('/dev/ttyUSB0',19200)

	except rospy.ROSInterruptException:
		pass



 

