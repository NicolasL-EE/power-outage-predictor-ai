/*
  Reads an analog voltage from A0 every 15 seconds.
  so the Python script can read it.
*/


// Assignments, change for personal use case
const int sensorPin = A0; // pin for voltage read
const int dropDownRatio = 26; // the ratio from arduino measured voltage to actual circuit voltage
const int ledPin = 13; // pin for the LED to light up when outage is predicted

// Measure every 15 seconds (15,000 ms)
unsigned long interval = 1500.0;
unsigned long previousMillis = 0;

void setup() {
  pinMode(sensorPin, INPUT);
  pinMode(ledPin, OUTPUT);

  Serial.begin(9600);
}

void loop() {
  unsigned long currentMillis = millis();

  // This code is to 
  // 1) Read voltage every 15 seconds
  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;

    // Raw ADC (0..1023)
    int sensorValue = analogRead(sensorPin);

    // Convert to 0..5 V
    float measuredVoltage = ((sensorValue / 1023.0) * 5.0);

    // Multiply by voltage drop down ratio
    float circuitVoltage = measuredVoltage * dropDownRatio;

    // Send to Python
    Serial.print("Voltage: ");
    Serial.println(circuitVoltage, 2);
  }

  // Listen for AI inference result
  if (Serial.available() > 0) {
    char incoming = Serial.read();
    if (incoming == '1') {
      digitalWrite(ledPin, HIGH);  // Turn LED on
    } 
    else if (incoming == '0') {
      digitalWrite(ledPin, LOW);   // Turn LED off
    }
  }
}
