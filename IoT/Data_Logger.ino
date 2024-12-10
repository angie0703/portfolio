#include <Ezo_i2c.h>
#include <Wire.h>                //enable I2C.
#include <LiquidCrystal_I2C.h>

Ezo_board ec = Ezo_board(100, "EC");
Ezo_board ph = Ezo_board(99, "PH");
Ezo_board temp = Ezo_board(102, "RTD");
Ezo_board doxy = Ezo_board(97, "DO");
LiquidCrystal_I2C lcd = LiquidCrystal_I2C(0x27, 20, 4);

bool reading_request_phase = true;        

uint32_t next_poll_time = 0;              
const unsigned int response_delay = 1000; 
String message;

void setup()                     //hardware initialization.
{
  Serial.begin(9600);            //enable serial port.
  Wire.begin();                  //enable I2C port
  lcd.init();
  lcd.backlight();
  lcd.setCursor(4, 1);
  lcd.print("DATA  LOGGER");
  delay(5000);
  lcd.clear();

}

void loop() {                                                                     //the main loop.
  lcd.setCursor(0, 0);
  lcd.print("EC: ");
  lcd.setCursor(4, 0);
  lcd.print(ec.get_last_received_reading()/1000);
  lcd.setCursor(9, 0);
  lcd.print("mS/cm");
  lcd.setCursor(0, 1);
  lcd.print("pH: ");
  lcd.setCursor(4, 1);
  lcd.print(ph.get_last_received_reading());
  lcd.setCursor(0, 2);
  lcd.print("Temp.: ");
  lcd.setCursor(7, 2);
  lcd.print(temp.get_last_received_reading());
  lcd.setCursor(13, 2);
  lcd.print("'C");
  lcd.setCursor(0, 3);
  lcd.print("DO: ");
  lcd.setCursor(4, 3);
  lcd.print(doxy.get_last_received_reading());

  if (reading_request_phase)             //if were in the phase where we ask for a reading
  {

    //to let the library know to parse the reading
    ec.send_read_cmd();
    ph.send_read_cmd();
    temp.send_read_cmd();
    doxy.send_read_cmd();

    next_poll_time = millis() + response_delay;         //set when the response will arrive
    reading_request_phase = false;                      //switch to the receiving phase
  }
  else                                                //if were in the receiving phase
  {
    if (millis() >= next_poll_time)                    //and its time to get the response
    {
      ec.receive_read_cmd();
      ph.receive_read_cmd();                              
      temp.receive_read_cmd();
      doxy.receive_read_cmd();
      display_reading();

      reading_request_phase = true;                     //switch back to asking for readings
    }
  }
}

void display_reading() {
  Serial.print(ec.get_last_received_reading()/1000);
  Serial.print(",");
  Serial.print(ph.get_last_received_reading());
  Serial.print(",");
  Serial.print(temp.get_last_received_reading());
  Serial.print(",");
  Serial.println(doxy.get_last_received_reading());
}
