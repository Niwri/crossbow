/******************************************************************
 This is the library for the Adafruit Motor Shield V2 for Arduino.
 It supports DC motors & Stepper motors with microstepping as well
 as stacking-support. It is *not* compatible with the V1 library!

 It will only work with https://www.adafruit.com/products/1483

 Adafruit invests time and resources providing this open
 source code, please support Adafruit and open-source hardware
 by purchasing products from Adafruit!

 Written by Limor Fried/Ladyada for Adafruit Industries.
 BSD license, check license.txt for more information.
 All text above must be included in any redistribution.
 ******************************************************************/

#ifndef STEPPER_MOTOR_H
#define STEPPER_MOTOR_H

#include <inttypes.h>
#include "motor.h"
//#define MOTORDEBUG

#define MICROSTEPS 16 // 8 or 16

#define MOTOR1_A 2
#define MOTOR1_B 3
#define MOTOR2_A 1
#define MOTOR2_B 4
#define MOTOR4_A 0
#define MOTOR4_B 6
#define MOTOR3_A 5
#define MOTOR3_B 7

#define FORWARD 1
#define BACKWARD 2
#define BRAKE 3
#define RELEASE 4

#define SINGLE 1
#define DOUBLE 2
#define INTERLEAVE 3
#define MICROSTEP 4
#define LOW 0
#define HIGH 1

void step(uint16_t steps, uint8_t dir, uint8_t style, uint8_t num);
uint8_t onestep(uint8_t dir, uint8_t style, uint8_t num);
void release(uint8_t num);

struct StepperMotor {
	uint32_t usperstep;
	uint8_t PWMApin, AIN1pin, AIN2pin;
	uint8_t PWMBpin, BIN1pin, BIN2pin;
	uint16_t revsteps; // # steps per revolution
	uint8_t currentstep;
	uint8_t steppernum;
};

void setStepper(uint16_t steps, uint8_t num);
void begin(uint16_t freq);
void setSpeed(uint16_t rpm, uint8_t num);

void setPWM_stepper(uint8_t pin, uint16_t val);
void setPin(uint8_t pin, uint8_t val);

#endif