
from serial import Serial
import random 
import math

BAUDRATE = 115200

STEPperANGLE = 516/180

def rotateLeft(ser, angle):
    steps = int(angle * STEPperANGLE)
    string_to_send = "left, " + str(steps) + "\n"
    ser.write(string_to_send.encode())
    return

def rotateUp(ser, angle):
    steps = int(angle * STEPperANGLE)
    string_to_send = "up, " + str(steps) + "\n"
    ser.write(string_to_send.encode())
    return

