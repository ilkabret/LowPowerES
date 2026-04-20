#include <PDM.h>

static const char channels = 1;
static const int frequency = 16000;

#define CLAP_THRESHOLD 2000
#define RECORD_SAMPLES 2000      // shorter = more reliable
#define NUM_RECORDINGS 20
#define COOLDOWN_MS 1000         // 1 second between claps

short sampleBuffer[512];
volatile int samplesRead;

// Recording buffer
short recording[RECORD_SAMPLES];
int recordIndex = 0;

bool isRecording = false;
bool aboveThreshold = false;
unsigned long lastClapTime = 0;

int recordingsDone = 0;

unsigned long startTime;

void setup() {
  Serial.begin(115200);
  while (!Serial);

  startTime = millis();  // <-- add this

  PDM.onReceive(onPDMdata);

  if (!PDM.begin(channels, frequency)) {
    Serial.println("Failed to start PDM!");
    while (1);
  }

  // Serial.println("READY");
}

void loop() {
  if (millis() - startTime < 2000) {
    samplesRead = 0;
    return;
  }

  if (!samplesRead) return;

  for (int i = 0; i < samplesRead; i++) {
    int sample = sampleBuffer[i];
    int amplitude = abs(sample);

    // Detect rising edge (clap start)
    if (amplitude > CLAP_THRESHOLD && !aboveThreshold) {
      aboveThreshold = true;

      if (!isRecording && millis() - lastClapTime > COOLDOWN_MS) {
        isRecording = true;
        recordIndex = 0;
        lastClapTime = millis();

        // Serial.print("START ");
        //Serial.println(recordingsDone);
      }
    }

    // Detect falling edge
    if (amplitude < CLAP_THRESHOLD) {
      aboveThreshold = false;
    }

    // Store samples
    if (isRecording) {
      if (recordIndex < RECORD_SAMPLES) {
        recording[recordIndex++] = sample;
      } else {
        // Finish recording
        isRecording = false;

        // Send data
        for (int j = 0; j < RECORD_SAMPLES; j++) {
          Serial.println(recording[j]);
        }

        // Serial.println("END");

        recordingsDone++;

        if (recordingsDone >= NUM_RECORDINGS) {
          // Serial.println("DONE");
          while (1);
        }
      }
    }
  }

  samplesRead = 0;
}

void onPDMdata() {
  int bytesAvailable = PDM.available();
  PDM.read(sampleBuffer, bytesAvailable);
  samplesRead = bytesAvailable / 2;
}