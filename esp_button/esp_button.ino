#include <BleKeyboard.h>
#include <Wire.h>
#include <MPU9250_asukiaaa.h>

MPU9250_asukiaaa mpu;

#define BUTTON_PIN 4
#define GND_PIN 13

BleKeyboard bleKeyboard("ESP32_KEYB", "Gabriel", 100);

bool lastReading = HIGH;
bool stableState = HIGH;
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 30;

const float TAP_THRESHOLD = 5;          
const unsigned long TAP_DEBOUNCE_MS = 180; 
const unsigned long TAP_WINDOW_MS = 1200;  
const float LPF_ALPHA = 0.90;              

float gx_f = 0.0, gy_f = 0.0, gz_f = 0.0;  
unsigned long lastTapTime = 0;
unsigned long firstTapTime = 0;
int tapCount = 0;

void sendCombo() {
  if (bleKeyboard.isConnected()) {
    Serial.println("Sending key_combo");
    bleKeyboard.press(KEY_LEFT_CTRL);
    bleKeyboard.press(KEY_LEFT_ALT);
    bleKeyboard.press('p');
    delay(50);
    bleKeyboard.releaseAll();
  } else {
    Serial.println("BLE not connected");
  }
}

void setupMPU() {
  Wire.begin();
  mpu.setWire(&Wire);
  mpu.beginAccel();

  delay(200);


  mpu.accelUpdate();
  gx_f = mpu.accelX();
  gy_f = mpu.accelY();
  gz_f = mpu.accelZ();
}

bool detectTripleTap() {
  if (mpu.accelUpdate() != 0) {
    return false;
  }

  float ax = mpu.accelX();
  float ay = mpu.accelY();
  float az = mpu.accelZ();

  gx_f = LPF_ALPHA * gx_f + (1.0 - LPF_ALPHA) * ax;
  gy_f = LPF_ALPHA * gy_f + (1.0 - LPF_ALPHA) * ay;
  gz_f = LPF_ALPHA * gz_f + (1.0 - LPF_ALPHA) * az;

  float dx = ax - gx_f;
  float dy = ay - gy_f;
  float dz = az - gz_f;

  float impactMag = sqrt(dx * dx + dy * dy + dz * dz);

  unsigned long now = millis();

  if (impactMag > TAP_THRESHOLD && (now - lastTapTime) > TAP_DEBOUNCE_MS) {
    lastTapTime = now;

    if (tapCount == 0) {
      firstTapTime = now;
      tapCount = 1;
    } else {
      tapCount++;
    }

    Serial.print("Tap detected. Count = ");
    Serial.print(tapCount);
    Serial.print("  impactMag = ");
    Serial.println(impactMag, 3);

    if (tapCount >= 3 && (now - firstTapTime) <= TAP_WINDOW_MS) {
      tapCount = 0;
      return true;
    }
  }

  if (tapCount > 0 && (now - firstTapTime) > TAP_WINDOW_MS) {
    tapCount = 0;
  }

  return false;
}

void setup() {
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(GND_PIN, OUTPUT);
  digitalWrite(GND_PIN, LOW);

  Serial.begin(115200);
  Serial.println("Starting BLE keyboard...");
  bleKeyboard.begin();
  
  setupMPU();
}

void loop() {
  
  bool reading = digitalRead(BUTTON_PIN);

  if (reading != lastReading) {
    lastDebounceTime = millis();
  }

  if ((millis() - lastDebounceTime) > debounceDelay) {
    if (reading != stableState) {
      stableState = reading;

      if (stableState == LOW) {
        sendCombo();
      }
    }
  }

  lastReading = reading;

  if (detectTripleTap()) {
    Serial.println("Triple tap detected");
    sendCombo();
  }

  delay(5);
}