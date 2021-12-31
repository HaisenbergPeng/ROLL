#!/usr/bin/env python
from __future__ import print_function
import sys, serial, struct, time,threading,time,math
import matplotlib.pyplot as plt
from datetime import datetime
#from odom_publish import myOdom
CCW=-1 # CCW：counter-clockwise
vControl=0.2 # speed control m/s 
distance=2  # m , the size of the rectangular trajectory
wControl=0.1 # angular speed control ,rad/s
degControl=90 # degrees, turning angle

pi=math.pi
minLineSpeed=0.05 # m/s
minAnguSpeed=0.04 # rad/s
baudrate = 19200
tty='/dev/ttyUSB0'
#tty='COM1'
bufferLen=21

class SerialRW():
    controlstate=0;    
    START ='\xFF\x01'
    control_resp='\xFF 10 10'
    def __init__(self, tty, baudrate,fileName,interval):
        self.ser = serial.Serial(tty, baudrate=baudrate, timeout=2)
        self.buffer=''
        self.interval=interval
        self.lock = threading.RLock()
        self.fileName = fileName
        self.x=[]
        self.y=[]
        self.theta=[]
        self.v_cur=[]
        self.w_cur=[]
        self.goodRead=[]
    def readXYW(self): 
        goodRead=1
        self.ser.write('\xAF\x01\x01')
        while self.buffer.find(self.START)<0: # loop until it find the starting bytes
            self.buffer += self.ser.read(1)
        self.buffer += self.ser.read(bufferLen-2)    
        # self.buffer=self.ser.read(bufferLen)  
        self.buffer = self.buffer.split(self.START,2)[1] # cut the buffer with '\xFF \x01'
        #print self.buffer
        str = self.buffer[0:bufferLen-2]
        if len(str) is not bufferLen-2: # sometimes it is not 19 byte long
            goodRead=0
            self.appendXYW(0,0,0,0,0,0)
            return 0,0,0,goodRead
        with self.lock:  # the same as "try ... finally"
            #crti:2016-06-22
            #h 2bytes; i 4 bytes (most of the times);B unsigned char ; b signed char 1byte
            x, y, theta,v_cur,w_cur,Rpulse,Lpulse,controlstate =  struct.unpack('>2i5h1B', str[0:bufferLen-2])
            # print(v_cur,w_cur,'\n')
            x = x/100.0 # cm to m
            y = y/100.0
            theta = theta / 1000.0 # to rad
            self.appendXYW(x,y,theta,v_cur,w_cur,1)
            # print(x,y,theta,v_cur,w_cur,Rpulse,Lpulse,'\n')
            goodRead=1
        return x,y,theta,goodRead
    def appendXYW(self,x,y,theta,v_cur,w_cur,goodRead):
        self.x.append(x)
        self.y.append(y)
        self.theta.append(theta)
        self.v_cur.append(v_cur)
        self.w_cur.append(w_cur)
        self.goodRead.append(goodRead)
    def writeVW(self,cmd_v,cmd_w):
        with self.lock:
            if cmd_v > 0:
                vsign = 0x01
                vabs = cmd_v * 1000.00
            else:
                vsign = 0x02
                vabs = -cmd_v * 1000.00
            if cmd_w > 0:
                wsign = 0x01
                wabs = cmd_w  * 1000.0
                #wabs = self.cmd_w * 180 /3.141592653 * 1000.0
            else:
                wsign = 0x02
                wabs = - cmd_w * 1000.0
                #wabs = - self.cmd_w * 180.0 /3.141592653 *1000.0

            hvabs = (int(vabs)>>8) & 0xFF
            lvabs = int(vabs) & 0xFF
            hwabs = (int(wabs)>>8) & 0xFF
            lwabs = int(wabs) & 0xFF

            parity = (0x10 + vsign + hvabs + lvabs + wsign + hwabs + lwabs) & 0xFF

            #print  vsign, vabs, wsign, wabs
            #print('write=%x,%x,%x,%x\n'%(vsign, vabs, wsign, wabs)),
            data = struct.pack('>9B', 0xAF, 0x10, vsign, hvabs, lvabs, wsign, hwabs, lwabs, parity)
            self.ser.write(data)
            buff=self.ser.read(3); # read 3 characters
            #print(data,'\n')
            #print("Control response: ",buff) # control response

    def trajStraight(self,v,L):
        print('Start straight line...')
        self.writeVW(v,0)
        # x,y,w,_=self.readXYW()
        inited=0
        flag=1
        with self.lock:
            while True:
                x,y,theta,goodRead=self.readXYW()
                if not inited:
                    x0=x
                    y0=y
                if not goodRead:
                    continue
                dist=math.sqrt((x-x0)**2+(y-y0)**2)
                if math.fabs(dist-L)<0.1 and flag==1:
                    self.ser.write(minLineSpeed,0)
                    flag=0
                if dist>=L:
                    break
                print("x: %0.2f y: %0.2f w:%0.2f \n"%(x,y,theta))
                inited=1
                time.sleep(self.interval)
    def trajRotate(self,w,deg):# CCW deg>0
        print('Start pure rotation...')
        factor=pi/180
        thetaTarget=deg*factor
        self.writeVW(0,CCW*w)
        inited=0
        flag=1
        with self.lock:
            while True:
                x,y,theta,goodRead=self.readXYW()
                if not goodRead:
                    continue
                if not inited:
                    theta0=theta
                if thetaTarget>0 and theta0+thetaTarget>2*pi and theta<pi:
                    theta=theta+2*pi
                elif thetaTarget<0 and theta0+thetaTarget<-2*pi and theta>-pi:
                    theta=theta-2*pi
                if math.fabs(theta-theta0-thetaTarget)<0.2 and flag==1:
                    self.writeVW(0,minAnguSpeed*CCW)
                    flag=0
                if self.rotateFinished(theta,theta0,thetaTarget):
                    break
                print("x: %0.2f y: %0.2f theta:%0.2f \n"%(x,y,theta))
                inited=1    
                time.sleep(self.interval)
    def rotateFinished(self,theta,theta0,thetaTarget):
        if thetaTarget>0:
            return theta-theta0>=thetaTarget
        else:
            return theta-theta0<=thetaTarget
    def writeFile(self):
        with open(self.fileName,'w') as f:
            for i in range(len(self.x)):
                f.write(str(self.x[i])+' '+str(self.y[i])+' '+str(self.theta[i])+' ' \
                +str(self.v_cur[i])+' '+str(self.w_cur[i])+' '+str(self.goodRead[i])+'\n')
    def plot(self):
        timeA=[i*self.interval for i in range(len(self.x))]
        fig = plt.figure()
        ax1 = fig.add_subplot(2,2,1) 
        ax2 = fig.add_subplot(2,2,2) 
        ax3 = fig.add_subplot(2,2,3) 
        ax4 = fig.add_subplot(2,2,4)
        ax1.plot(self.x,self.y)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax2.plot(timeA,self.theta)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Theta (rad)')
        ax3.plot(timeA,self.v_cur)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Line velocity  (m/s)')
        ax4.plot(timeA,self.w_cur)
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Angular velocity (rad/s)')
        plt.savefig(self.fileName+'.png')
        plt.show()
    def clearOdometry(self):
        self.ser.write('\xAF\x02\x02')
        resp=self.ser.read(2)
        if resp is not '\x20\x20':
            print('Odometry clearing failed!\n')
        else:
            print(resp)
if __name__ == '__main__':
    fileName =str(datetime.now()).replace(':','')+'.txt'
    interval=0.05 # control interval: sec
    sleepTime=1
    serialC=SerialRW(tty,baudrate,fileName,interval)
    serialC.clearOdometry()
    for i in range(4):
        serialC.trajStraight(vControl,distance)
        time.sleep(sleepTime)
        serialC.trajRotate(wControl,degControl*CCW)
        time.sleep(sleepTime)
    serialC.writeFile()
    serialC.plot()