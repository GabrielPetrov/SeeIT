#include "BluetoothSerial.h"

#define BUTTON_PIN 15

BluetoothSerial SerialBT;

volatile bool buttonEvent = false;
volatile bool buttonPressed = false;
volatile unsigned long lastInterruptTime = 0;

const unsigned long debounceDelay = 50;

bool btConnected = false;

void btCallback(esp_spp_cb_event_t event, esp_spp_cb_param_t *param) {
  if (event == ESP_SPP_SRV_OPEN_EVT) {
    btConnected = true;
    Serial.println("Bluetooth client CONNECTED");
  }
  else if (event == ESP_SPP_CLOSE_EVT) {
    btConnected = false;
    Serial.println("Bluetooth client DISCONNECTED");
  }
}

void IRAM_ATTR handleButtonInterrupt() {
  unsigned long currentTime = millis();

  if (currentTime - lastInterruptTime > debounceDelay) {
    buttonPressed = (digitalRead(BUTTON_PIN) == LOW);
    buttonEvent = true;
    lastInterruptTime = currentTime;
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  if (!SerialBT.begin("ESP32_Button")) {
    Serial.println("Bluetooth start failed");
    while (true) delay(1000);
  }

  // Register callback
  SerialBT.register_callback(btCallback);

  attachInterrupt(digitalPinToInterrupt(BUTTON_PIN), handleButtonInterrupt, CHANGE);

  Serial.println("ESP32 Bluetooth sender started");
  Serial.println("Waiting for client connection...");
}

void loop() {
  if (buttonEvent) {
    noInterrupts(); 
    bool pressed = buttonPressed;
    buttonEvent = false;
    interrupts();

    if (pressed) {
      Serial.println("Button PRESSED");
      if (btConnected) SerialBT.println("PRESSED");
    } else {
      Serial.println("Button RELEASED");
      if (btConnected) SerialBT.println("RELEASED");
    }
  }
}