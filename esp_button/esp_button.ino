#include <BleKeyboard.h>

#define BUTTON_PIN 4
#define GND_PIN 13

BleKeyboard bleKeyboard("ESP32_KEYB", "Gabriel", 100);

bool lastReading = HIGH;
bool stableState = HIGH;
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 30;

void setup() {
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(GND_PIN, OUTPUT);
  digitalWrite(GND_PIN, LOW);

  Serial.begin(115200);
  Serial.println("Starting BLE keyboard...");
  bleKeyboard.begin();
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
    }
  }

  lastReading = reading;
  delay(5);
}