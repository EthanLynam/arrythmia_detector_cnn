#include <SPI.h>
#include <WiFiNINA.h>
#include <Arduino.h>
#include <WebSocketsClient.h>

#define SSID "YOUR-WIFI"
#define PASSWORD "YOUR-WIFI-PASSWORD"

#define ECG_PIN A0

const char* serverAddress = "YOUR-LAPTOPS-IPv4-ADDRESS";
const uint16_t serverPort = 9000;

WiFiClient wifi;
WebSocketsClient webSocket;

void webSocketEvent(WStype_t type, uint8_t *payload, size_t length) {

  switch (type) {
    case WStype_DISCONNECTED:
      Serial.println("Disconnected.");
      break;
    case WStype_CONNECTED:
      Serial.println("Connected to server!");
      Serial.println("Sending ECG data to server...");
      break;
    case WStype_ERROR:
      Serial.println("Error occurred.");
      break;
    case WStype_PING:
    case WStype_PONG:
    case WStype_FRAGMENT_TEXT_START:
    case WStype_FRAGMENT_BIN_START:
    case WStype_FRAGMENT:
    case WStype_FRAGMENT_FIN:
      break;
  }
}

void setup() {
    Serial.begin(115200);
      while (!Serial) {
      ;  // wait for serial port to connect.
    }

    WiFi.begin(SSID, PASSWORD);
    Serial.print("Connecting to Wi-Fi...");
    while (WiFi.status() != WL_CONNECTED) {
        Serial.print(".");
        delay(300);
    }
    Serial.println();
    Serial.print("Connected to WiFi: ");
    Serial.println(SSID);

    // initialize WebSocket
    webSocket.begin(serverAddress, serverPort);

    // event handler
    webSocket.onEvent(webSocketEvent);
}

void loop() {
    webSocket.loop();

    int ecgData = analogRead(A0);

    char dataToSend[20];
    sprintf(dataToSend, "%d", ecgData);
    webSocket.sendTXT(dataToSend);

    delay(3); //300hZ to mimick mit-bih 360Hz
}