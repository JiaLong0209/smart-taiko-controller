/*
 * AI Taiko Firmware (Non-blocking / Stable Sample Rate)
 * - Logic: Reads 4 sensors at a fixed frequency (2000Hz).
 * - Baud Rate: 115200 (Must match the Python script).
 * - Target Sample Rate: 2000Hz (Runs every 500 microseconds).
 */

 // Pin Definitions: Mapping physical pins to Drum parts
 const int PIN_DON_L = A1; // Center Drum (Left)
 const int PIN_DON_R = A2; // Center Drum (Right)
 const int PIN_KA_L  = A0; // Rim (Left)
 const int PIN_KA_R  = A3; // Rim (Right)
 
 // Noise Gate Threshold
 // Signals below this value (0-1023) are ignored to filter out electrical noise.
 const int GATE_THRESHOLD = 35;
 
 // === Timing Control Variables ===
 unsigned long previousMicros = 0;   // Stores the last time we sampled
 const long SAMPLE_INTERVAL = 500;   // Interval in microseconds (500us = 0.5ms)
                                     // 1,000,000 us / 500 us = 2000 Hz (Samples per second)
 
 void setup() {
   // Initialize Serial Communication at 115200 bits per second
   // Fast speed is required to send 2000 lines of data per second without lag.
   Serial.begin(115200);
 }
 
 void loop() {
   // Get current time in microseconds
   unsigned long currentMicros = micros();
 
   // === Non-blocking Timer Logic ===
   // Check if 500 microseconds have passed since the last sample
   if (currentMicros - previousMicros >= SAMPLE_INTERVAL) {
     
     // Update the time marker to now
     previousMicros = currentMicros;
 
     // 1. Read Raw Data from Analog Pins (0-1023)
     int dL = analogRead(PIN_DON_L);
     int dR = analogRead(PIN_DON_R);
     int kL = analogRead(PIN_KA_L);
     int kR = analogRead(PIN_KA_R);
 
     // 2. Data Transmission Logic
     // Only send data if AT LEAST ONE sensor exceeds the threshold.
     // This saves USB bandwidth and Python CPU usage during idle times.
     if (dL > GATE_THRESHOLD || dR > GATE_THRESHOLD || 
         kL > GATE_THRESHOLD || kR > GATE_THRESHOLD) {
       
       // Send data in CSV format: "dL,dR,kL,kR"
       Serial.print(dL); Serial.print(",");
       Serial.print(dR); Serial.print(",");
       Serial.print(kL); Serial.print(",");
       Serial.println(kR); // println adds a newline character at the end
     }
   }
 }