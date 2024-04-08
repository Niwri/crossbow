/*
This is a test sketch for the Adafruit assembled Motor Shield for Arduino v2
It won't work with v1.x motor shields! Only for the v2's with built in PWM
control

For use with the Adafruit Motor Shield v2
---->  http://www.adafruit.com/products/1438
*/

#include <Adafruit_MotorShield.h>

// Create the motor shield object with the default I2C address
Adafruit_MotorShield AFMS = Adafruit_MotorShield();
// Or, create it with a different I2C address (say for stacking)
// Adafruit_MotorShield AFMS = Adafruit_MotorShield(0x61);

// Connect a stepper motor with 200 steps per revolution (1.8 degree)
// to motor port #2 (M3 and M4)
Adafruit_StepperMotor *myMotor = AFMS.getStepper(200, 1);


void setup() {
  Serial.begin(115200);           // set up Serial library at 9600 bps
  while (!Serial);
  // Serial.println("Stepper test!");

  // if (!AFMS.begin()) {         // create with the default frequency 1.6KHz
  // // if (!AFMS.begin(1000)) {  // OR with a different frequency, say 1KHz
  //   Serial.println("Could not find Motor Shield. Check wiring.");
  //   while (1);
  // }
  // Serial.println("Motor Shield found.");

  //myMotor->setSpeed(10);  // 10 rpm
  pinMode(13, OUTPUT);      // set LED pin as output
  digitalWrite(13, HIGH);    // switch off LED pin
}

void loop() {

  if(Serial.available()) {
    char data_rcvd = Serial.read();   // read one byte from serial buffer and save to data_rcvd
    if(data_rcvd == '1') digitalWrite(13, HIGH);  // switch LED On
    else if(data_rcvd == '0') digitalWrite(13, LOW);   // switch LED Off
    delay(1000);
  }
  // Serial.println("Single coil steps");
  // myMotor->step(100, FORWARD, SINGLE);
  // myMotor->step(100, BACKWARD, SINGLE);

  // Serial.println("Double coil steps");
  // myMotor->step(100, FORWARD, DOUBLE);
  // myMotor->step(100, BACKWARD, DOUBLE);

  // Serial.println("Interleave coil steps");
  // myMotor->step(100, FORWARD, INTERLEAVE);
  // myMotor->step(100, BACKWARD, INTERLEAVE);

  // Serial.println("Microstep steps");
  // myMotor->step(50, FORWARD, MICROSTEP);
  // myMotor->step(50, BACKWARD, MICROSTEP);
}