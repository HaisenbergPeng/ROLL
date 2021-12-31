#!/usr/bin/env python
from __future__ import print_function
import sys, serial, struct, time,threading,time,math
#from odom_publish import myOdom
baudrate = 19200
tty='/dev/ttyUSB0'
#tty='COM1'
bufferLen=21
CCW=1
vControl=0.2 # m/s 
distance=2  # m , it takes ~100sec to finish
wControl=0.1 # rad/s
degControl=90 # degrees, should be less than 180
pi=math.pi
class SerialRW():
    controlstate=0;    
    START ='\xFF\x01'
    control_resp='\xFF 10 10'
    def __init__(self, tty, baudrate):
        self.ser = serial.Serial(tty, baudrate=baudrate, timeout=2)
        self.buffer=''
        self.lock = threading.RLock()

    def readXYW(self): 
        goodRead=1
        self.ser.write('\xAF\x01\x01')
        # while self.buffer.find(self.START)<0: # loop until it find the starting bytes
        #     self.buffer += self.ser.read(1)
        # self.buffer += self.ser.read(bufferLen-2)    
        self.buffer=self.ser.read(bufferLen)  
        self.buffer = self.buffer.split(self.START,2)[1] # cut the buffer with '\xFF \x01'
        #print self.buffer
        str = self.buffer[0:bufferLen-2]
        if len(str) is not bufferLen-2: # sometimes it is not 19 byte long
            goodRead=0
            return 0,0,0,goodRead
        with self.lock:  # the same as "try ... finally"
            #crti:2016-06-22
            #h 2bytes; i 4 bytes (most of the times);B unsigned char ; b signed char 1byte
            x, y, theta,v_cur,w_cur,Rpulse,Lpulse,controlstate =  struct.unpack('>2i5h1B', str[0:bufferLen-2])
            # print(v_cur,w_cur,'\n')
            x = x/100.0 # cm to m
            y = y/100.0
            theta = theta / 1000.0 # to rad
            # print(x,y,theta,v_cur,w_cur,Rpulse,Lpulse,'\n')
            goodRead=1
        return x,y,theta,goodRead
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
        self.writeVW(v,0)
        x0,y0,w0,_=self.readXYW()
        x,y,w,_=self.readXYW()
        with self.lock:
            while math.sqrt((x-x0)**2+(y-y0)**2)<=L:
                x,y,_,goodRead=self.readXYW()
                if not goodRead:
                    continue
                print("x: %0.2f y: %0.2f w:%0.2f \n"%(x,y,w))
                time.sleep(0.05)
    def trajRotate(self,w,deg):# CCW deg>0
        factor=pi/180
        wTarget=deg*factor
        self.writeVW(0,w)
        x0,y0,w0,_=self.readXYW()
        x,y,w,_=self.readXYW()
        print("w: %0.2f w0: %0.2f \n"%(w,w0))
        with self.lock:
            while not self.rotateFinished(w,w0,wTarget):
                x,y,w,goodRead=self.readXYW()
                if not goodRead:
                    continue
                print("x: %0.2f y: %0.2f w:%0.2f \n"%(x,y,w))
                if wTarget>0 and w0+wTarget>2*pi and w<pi:
                    w=w+2*pi
                elif wTarget<0 and w0+wTarget<-2*pi and w>0:
                    w=w-2*pi
                time.sleep(0.05)    

    def rotateFinished(self,w,w0,wTarget):
        if wTarget>0:
            return w-w0>=wTarget
        else:
            return w-w0<=wTarget
if __name__ == '__main__':
    serialC=SerialRW(tty,baudrate)
    signRotate=1 if CCW==1 else -1
    for i in range(4):
        # serialC.trajStraight(vControl,distance)
        serialC.trajRotate(wControl,degControl*signRotate)
    for i in range(4):
        # serialC.trajStraight(vControl,distance)
        serialC.trajRotate(-wControl,-degControl*signRotate)

        
        

