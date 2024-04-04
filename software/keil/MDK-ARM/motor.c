#include "motor.h"
#include <stdint.h>
#include "config.h"
#include <stdio.h>
extern uint8_t i2c_addr;

void reset_motor(void){ 
	char msg[100];
	sprintf(msg, "%d\n", i2c_addr);
	print_msg(msg);
	HAL_Delay(1000);
	write8(PCA9685_MODE1, 0x0); 
}

void setPWMFreq(float freq) {
  // Serial.print("Attempting to set freq ");
  // Serial.println(freq);

  freq *= 0.9; // Correct for overshoot in the frequency setting (see issue #11).

  float prescaleval = 25000000;
  prescaleval /= 4096;
  prescaleval /= freq;
  prescaleval -= 1;
  // Serial.print("Estimated pre-scale: "); Serial.println(prescaleval);
  uint8_t prescale = (uint8_t)(prescaleval + 0.5);
  // Serial.print("Final pre-scale: "); Serial.println(prescale);

  uint8_t oldmode = read8(PCA9685_MODE1);
  uint8_t newmode = (oldmode & 0x7F) | 0x10; // sleep
  write8(PCA9685_MODE1, newmode);            // go to sleep
  write8(PCA9685_PRESCALE, prescale);        // set the prescaler
  write8(PCA9685_MODE1, oldmode);
  HAL_Delay(5);
  write8(PCA9685_MODE1, oldmode | 0xa1); //  This sets the MODE1 register to turn on auto increment.
                    // This is why the beginTransmission below was not working.
  //  Serial.print("Mode now 0x"); Serial.println(read8(PCA9685_MODE1), HEX);
}

void setPWM(uint8_t num, uint16_t on, uint16_t off) {
  // Serial.print("Setting PWM "); Serial.print(num); Serial.print(": ");
  // Serial.print(on); Serial.print("->"); Serial.println(off);
  uint8_t buffer[5];
  buffer[0] = LED0_ON_L + 4 * num;
  buffer[1] = on;
  buffer[2] = on >> 8;
  buffer[3] = off;
  buffer[4] = off >> 8;
	
	while(HAL_I2C_Master_Transmit(&hi2c2, i2c_addr, buffer, 5, 100) != HAL_OK);
	HAL_Delay(0);
}

uint8_t read8(uint8_t addr) {
  uint8_t buffer[1] = {addr};
	
	HAL_Delay(1000);
	while(HAL_I2C_Master_Transmit(&hi2c2, i2c_addr, buffer, 1, 100) != HAL_OK);
	HAL_Delay(0);
	while(HAL_I2C_Master_Receive(&hi2c2, i2c_addr, buffer, 1, 100) != HAL_OK);
  return buffer[0];
}

void write8(uint8_t addr, uint8_t d) {
  uint8_t buffer[2] = {addr, d};
	char msg[100];
	sprintf(msg, "%d\n", i2c_addr);
	print_msg(msg);
	HAL_Delay(1000);
  while(HAL_I2C_Master_Transmit(&hi2c2, i2c_addr, buffer, 2, 100) != HAL_OK);
	HAL_Delay(0);
}