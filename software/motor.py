
from serial import Serial
import random 
import math

BAUDRATE = 115200


def rotateLeft(ser, angle, port, toggle):
    if(toggle is True):
        print("True")
        ser.write(b'1')
    else:
        print("False")
        ser.write(b'0')

    return

def rotateRight(angle):

    return

def rotateUp(angle):

    return

def rotateDown(angle):

    return

