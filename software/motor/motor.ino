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
// to motor port #1 (M1 and M2)
Adafruit_StepperMotor *leftMotor = AFMS.getStepper(200, 1);
Adafruit_StepperMotor *rightMotor = AFMS.getStepper(200, 2);

int steps = 0;
void setup() {
  Serial.begin(115200);           // set up Serial library at 9600 bps
  while (!Serial);
  Serial.println("Stepper test!");

  if (!AFMS.begin()) {         // create with the default frequency 1.6KHz
  // if (!AFMS.begin(1000)) {  // OR with a different frequency, say 1KHz
    Serial.println("Could not find Motor Shield. Check wiring.");
    while (1);
  }
  Serial.println("Motor Shield found.");

  leftMotor->setSpeed(10);  // 10 rpm
  rightMotor->setSpeed(10);

}
void loop() {
  if(Serial.available()) {
    String data_rcvd = Serial.readStringUntil(',');
    if (data_rcvd == "left") {
      
      data_rcvd = Serial.readStringUntil('\n');
      steps = data_rcvd.toInt();
      for(int i = 0; i < abs(steps); i++) {
          
        leftMotor->onestep(steps > 0 ? FORWARD : BACKWARD, INTERLEAVE);
        rightMotor->onestep(steps > 0 ? FORWARD : BACKWARD, INTERLEAVE);
      }
      
    } else if(data_rcvd == "up") {
      data_rcvd = Serial.readStringUntil('\n');
      steps = data_rcvd.toInt();

      for(int i = 0; i < abs(steps); i++) {

        leftMotor->onestep(steps > 0 ? FORWARD : BACKWARD, INTERLEAVE);
        rightMotor->onestep(steps > 0 ? BACKWARD : FORWARD, INTERLEAVE);
      }
    } 
  }

  // if(Serial.available()) {
  //   String data_rcvd = Serial.readStringUntil('\n');   // read one byte from serial buffer and save to data_rcvd
  //   if(Serial.read() == -1)
  //     digitalWrite(13, HIGH);
  //   while(Serial.read() != -1);
    // int val = Serial.parseInt();
    // toggle = toggle == 0 ? 1 : 0;
    // if(data_rcvd == "left")
    //   digitalWrite(13, val > 256 ? HIGH : LOW);
    // // if(data_rcvd == "left") {
    // //   int steps = Serial.read();
    // //   myMotor->step(abs(steps), steps > 0 ? FORWARD : BACKWARD, MICROSTEP);
    // // } else {

    // // }
  
}