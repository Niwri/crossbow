#include "stepper_motor.h"
#include "config.h"
///! A sinusoial microstepping curve for the PWM output (8-bit range) with 17
/// points - last one is start of next step.
static uint8_t microstepcurve[] = {0,   25,  50,  74,  98,  120, 141, 162, 180,
                                   197, 212, 225, 236, 244, 250, 253, 255};
extern uint16_t _freq;
extern struct StepperMotor steppers[2];
																	 
void begin(uint16_t freq) {
  // init PWM w/_freq
  _freq = freq;
  setPWMFreq(_freq); // This is the maximum PWM frequency
  for (uint8_t i = 0; i < 16; i++)
    setPWM(i, 0, 0);
}

void setPWM_stepper(uint8_t pin, uint16_t value) {
  if (value > 4095) {
    setPWM(pin, 4096, 0);
  } else
    setPWM(pin, 0, value);
}

void setPin(uint8_t pin, uint8_t value) {
  if (value == LOW)
    setPWM(pin, 0, 0);
  else
    setPWM(pin, 4096, 0);
}

void setSpeed(uint16_t rpm, uint8_t num) {
  // Serial.println("steps per rev: "); Serial.println(revsteps);
  // Serial.println("RPM: "); Serial.println(rpm);
	if(num > 2)
		return;
	num--;
  steppers[num].usperstep = 60000000 / ((uint32_t)(steppers[num].revsteps) * (uint32_t)rpm);
}

void release(uint8_t num) {
	
	if(num > 2)
		return;
	num--;
	
	struct StepperMotor stepper = steppers[num];
	
  setPin(stepper.AIN1pin, LOW);
  setPin(stepper.AIN2pin, LOW);
  setPin(stepper.BIN1pin, LOW);
  setPin(stepper.BIN2pin, LOW);
  setPWM_stepper(stepper.PWMApin, 0);
  setPWM_stepper(stepper.PWMBpin, 0);
}

void setStepper(uint16_t steps, uint8_t num) {
	if(num > 2) 
		return;
	
  num--;

  if (steppers[num].steppernum == 0) {
    // not init'd yet!
    steppers[num].steppernum = num;
    steppers[num].revsteps = steps;
    uint8_t pwma, pwmb, ain1, ain2, bin1, bin2;
    if (num == 0) {
      pwma = 8;
      ain2 = 9;
      ain1 = 10;
      pwmb = 13;
      bin2 = 12;
      bin1 = 11;
    } else {
      pwma = 2;
      ain2 = 3;
      ain1 = 4;
      pwmb = 7;
      bin2 = 6;
      bin1 = 5;
    }
    steppers[num].PWMApin = pwma;
    steppers[num].PWMBpin = pwmb;
    steppers[num].AIN1pin = ain1;
    steppers[num].AIN2pin = ain2;
    steppers[num].BIN1pin = bin1;
    steppers[num].BIN2pin = bin2;
  }
}
																												
void step(uint16_t steps, uint8_t dir, uint8_t style, uint8_t num) {
	if(num > 2)
		return;
	
	num--;
	
  uint32_t uspers = steppers[num].usperstep;

  if (style == INTERLEAVE) {
    uspers /= 2;
  } else if (style == MICROSTEP) {
    uspers /= MICROSTEPS;
    steps *= MICROSTEPS;
#ifdef MOTORDEBUG
    Serial.print("steps = ");
    Serial.println(steps, DEC);
#endif
  }

  while (steps--) {
    // Serial.println("step!"); Serial.println(uspers);
    onestep(dir, style, num);
    HAL_Delay(uspers);
#ifdef ESP8266
    yield(); // required for ESP8266
#endif
  }
}

/**************************************************************************/
/*!
    @brief  Move the stepper motor one step only, with no delays
    @param  dir The direction to go, can be FORWARD or BACKWARD
    @param  style How to perform each step, can be SINGLE, DOUBLE, INTERLEAVE or
   MICROSTEP
    @returns The current step/microstep index, useful for
   Adafruit_StepperMotor.step to keep track of the current location, especially
   when microstepping
*/
/**************************************************************************/
uint8_t onestep(uint8_t dir, uint8_t style, uint8_t num) {
	if(num > 2)
		return NULL;
	num--;
	
  uint8_t ocrb, ocra;

  ocra = ocrb = 255;

  // next determine what sort of stepping procedure we're up to
  if (style == SINGLE) {
    if ((steppers[num].currentstep / (MICROSTEPS / 2)) % 2) { // we're at an odd step, weird
      if (dir == FORWARD) {
        steppers[num].currentstep += MICROSTEPS / 2;
      } else {
        steppers[num].currentstep -= MICROSTEPS / 2;
      }
    } else { // go to the next even step
      if (dir == FORWARD) {
        steppers[num].currentstep += MICROSTEPS;
      } else {
        steppers[num].currentstep -= MICROSTEPS;
      }
    }
  } else if (style == DOUBLE) {
    if (!(steppers[num].currentstep / (MICROSTEPS / 2) % 2)) { // we're at an even step, weird
      if (dir == FORWARD) {
        steppers[num].currentstep += MICROSTEPS / 2;
      } else {
        steppers[num].currentstep -= MICROSTEPS / 2;
      }
    } else { // go to the next odd step
      if (dir == FORWARD) {
        steppers[num].currentstep += MICROSTEPS;
      } else {
        steppers[num].currentstep -= MICROSTEPS;
      }
    }
  } else if (style == INTERLEAVE) {
    if (dir == FORWARD) {
      steppers[num].currentstep += MICROSTEPS / 2;
    } else {
      steppers[num].currentstep -= MICROSTEPS / 2;
    }
  }

  if (style == MICROSTEP) {
    if (dir == FORWARD) {
      steppers[num].currentstep++;
    } else {
      // BACKWARDS
      steppers[num].currentstep--;
    }

    steppers[num].currentstep += MICROSTEPS * 4;
    steppers[num].currentstep %= MICROSTEPS * 4;

    ocra = ocrb = 0;
    if (steppers[num].currentstep < MICROSTEPS) {
      ocra = microstepcurve[MICROSTEPS - steppers[num].currentstep];
      ocrb = microstepcurve[steppers[num].currentstep];
    } else if ((steppers[num].currentstep >= MICROSTEPS) && (steppers[num].currentstep < MICROSTEPS * 2)) {
      ocra = microstepcurve[steppers[num].currentstep - MICROSTEPS];
      ocrb = microstepcurve[MICROSTEPS * 2 - steppers[num].currentstep];
    } else if ((steppers[num].currentstep >= MICROSTEPS * 2) &&
               (steppers[num].currentstep < MICROSTEPS * 3)) {
      ocra = microstepcurve[MICROSTEPS * 3 - steppers[num].currentstep];
      ocrb = microstepcurve[steppers[num].currentstep - MICROSTEPS * 2];
    } else if ((steppers[num].currentstep >= MICROSTEPS * 3) &&
               (steppers[num].currentstep < MICROSTEPS * 4)) {
      ocra = microstepcurve[steppers[num].currentstep - MICROSTEPS * 3];
      ocrb = microstepcurve[MICROSTEPS * 4 - steppers[num].currentstep];
    }
  }

  steppers[num].currentstep += MICROSTEPS * 4;
  steppers[num].currentstep %= MICROSTEPS * 4;

#ifdef MOTORDEBUG
  Serial.print("current step: ");
  Serial.println(currentstep, DEC);
  Serial.print(" pwmA = ");
  Serial.print(ocra, DEC);
  Serial.print(" pwmB = ");
  Serial.println(ocrb, DEC);
#endif
  setPWM_stepper(steppers[num].PWMApin, ocra * 16);
  setPWM_stepper(steppers[num].PWMBpin, ocrb * 16);

  // release all
  uint8_t latch_state = 0; // all motor pins to 0

  // Serial.println(step, DEC);
  if (style == MICROSTEP) {
    if (steppers[num].currentstep < MICROSTEPS)
      latch_state |= 0x03;
    if ((steppers[num].currentstep >= MICROSTEPS) && (steppers[num].currentstep  < MICROSTEPS * 2))
      latch_state |= 0x06;
    if ((steppers[num].currentstep  >= MICROSTEPS * 2) && (steppers[num].currentstep  < MICROSTEPS * 3))
      latch_state |= 0x0C;
    if ((steppers[num].currentstep  >= MICROSTEPS * 3) && (steppers[num].currentstep  < MICROSTEPS * 4))
      latch_state |= 0x09;
  } else {
    switch (steppers[num].currentstep  / (MICROSTEPS / 2)) {
    case 0:
      latch_state |= 0x1; // energize coil 1 only
      break;
    case 1:
      latch_state |= 0x3; // energize coil 1+2
      break;
    case 2:
      latch_state |= 0x2; // energize coil 2 only
      break;
    case 3:
      latch_state |= 0x6; // energize coil 2+3
      break;
    case 4:
      latch_state |= 0x4; // energize coil 3 only
      break;
    case 5:
      latch_state |= 0xC; // energize coil 3+4
      break;
    case 6:
      latch_state |= 0x8; // energize coil 4 only
      break;
    case 7:
      latch_state |= 0x9; // energize coil 1+4
      break;
    }
  }
#ifdef MOTORDEBUG
  Serial.print("Latch: 0x");
  Serial.println(latch_state, HEX);
#endif

  if (latch_state & 0x1) {
    // Serial.println(AIN2pin);
    setPin(steppers[num].AIN2pin, HIGH);
  } else {
    setPin(steppers[num].AIN2pin, LOW);
  }
  if (latch_state & 0x2) {
    setPin(steppers[num].BIN1pin, HIGH);
    // Serial.println(BIN1pin);
  } else {
    setPin(steppers[num].BIN1pin, LOW);
  }
  if (latch_state & 0x4) {
    setPin(steppers[num].AIN1pin, HIGH);
    // Serial.println(AIN1pin);
  } else {
    setPin(steppers[num].AIN1pin, LOW);
  }
  if (latch_state & 0x8) {
    setPin(steppers[num].BIN2pin, HIGH);
    // Serial.println(BIN2pin);
  } else {
    setPin(steppers[num].BIN2pin, LOW);
  }

  return steppers[num].currentstep;
}