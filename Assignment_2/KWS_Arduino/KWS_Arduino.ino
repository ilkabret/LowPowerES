/* -------- Assignment 1 --------
Topic:  Threshold-triggered keyword spotting
Date:   22/04/2026
Names:  Romain Mathis Noblet (268709)
        Ilka Bretschneider (268664)

  • Continuous RMS monitoring in loop()
  • Inference fires when RMS crosses TRIGGER_THRESHOLD
  • POST_TRIGGER_SAMPLES_MS of audio is captured after the trigger
    so the event is centred in the inference window
  • A REARM_MS window prevents double-firing
*/

#include <PDM.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <math.h>
#include <avr/pgmspace.h>

#include "mfcc_tables.h"
#include "kws_model_data.h"
#include "kws_params.h"

// ─────────────────────────────────────────────────────────────────────────────
//  CONFIGURATION  (audio pipeline — do NOT change)
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int   SAMPLE_RATE      = 16000;
static constexpr int   SAMPLES_PER_OBS  = 2000;
static constexpr int   FRAME_LEN        = 256;
static constexpr int   HOP_LEN          = 128;
static constexpr int   FFT_SIZE         = 256;
static constexpr int   FFT_BINS         = FFT_SIZE / 2 + 1;
static constexpr int   N_MELS           = 26;
static constexpr int   N_MFCC           = 13;
static constexpr int   N_FRAMES         = (SAMPLES_PER_OBS - FRAME_LEN) / HOP_LEN + 1;
static constexpr float LOG_FLOOR        = 1e-10f;
static constexpr int   TENSOR_ARENA_SIZE = 40 * 1024;
static constexpr float CONFIDENCE_THRESHOLD = 0.60f;

// ─────────────────────────────────────────────────────────────────────────────
//  TRIGGER CONFIGURATION  ← tune these three values for your environment
// ─────────────────────────────────────────────────────────────────────────────

// RMS amplitude (0–32767) that fires the trigger.
// Raise if background noise causes false triggers; lower if events are missed.
static constexpr float    TRIGGER_THRESHOLD = 500.0f;

// How many samples to collect AFTER the trigger before running inference.
// 1000 samples = 62.5 ms @ 16 kHz — gives the tail of the sound time to arrive.
// The pre-trigger audio already sitting in the ring buffer provides the head.
static constexpr uint16_t POST_TRIGGER_SAMPLES = 1000;

// Minimum ms between two inferences — prevents the same event firing twice.
static constexpr unsigned long REARM_MS = 500;

// Number of samples used for the RMS check window (must be ≤ SAMPLES_PER_OBS).
static constexpr int RMS_WINDOW = 256;

// ─────────────────────────────────────────────────────────────────────────────
//  GLOBALS
// ─────────────────────────────────────────────────────────────────────────────

// ── Audio ring buffer ────────────────────────────────────────────────────────
static volatile int16_t  audioBuffer[SAMPLES_PER_OBS];
static volatile uint16_t writeIdx   = 0;
static volatile bool     bufferFull = false;

// ── PDM staging buffer ───────────────────────────────────────────────────────
static int16_t pdmBuffer[256];

// ── MFCC working buffers ─────────────────────────────────────────────────────
static float frameBuf [FRAME_LEN];
static float fftRe    [FFT_SIZE];
static float fftIm    [FFT_SIZE];
static float melBuf   [N_MELS];
static float mfccOut  [N_FRAMES][N_MFCC];

// ── TFLite ───────────────────────────────────────────────────────────────────
static uint8_t tensorArena[TENSOR_ARENA_SIZE];
static const tflite::Model*      tfModel     = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor*             inputTensor = nullptr;
static TfLiteTensor*             outputTensor= nullptr;

// ── Trigger state ────────────────────────────────────────────────────────────
enum class TriggerState { LISTENING, COLLECTING };
static TriggerState     triggerState      = TriggerState::LISTENING;
static uint16_t         samplesAfterTrig  = 0;   // counts post-trigger samples
static unsigned long    lastInferenceMs   = 0;    // for REARM_MS cooldown

// ─────────────────────────────────────────────────────────────────────────────
//  PDM CALLBACK
// ─────────────────────────────────────────────────────────────────────────────
static void onPDMData() {
  int available = PDM.available();
  PDM.read(pdmBuffer, available);

  int nSamples = available / sizeof(int16_t);
  for (int i = 0; i < nSamples; i++) {
    audioBuffer[writeIdx] = pdmBuffer[i];
    writeIdx = (writeIdx + 1) % SAMPLES_PER_OBS;
    if (writeIdx == 0) bufferFull = true;

    // While collecting post-trigger audio, count incoming samples
    if (triggerState == TriggerState::COLLECTING) {
      samplesAfterTrig++;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  RMS over the most-recent RMS_WINDOW samples in the ring buffer
// ─────────────────────────────────────────────────────────────────────────────
static float computeRMS() {
  // Read without disabling interrupts — slight tearing is acceptable here
  // because this is only used as a coarse trigger detector.
  uint16_t tail = writeIdx;
  float    sum  = 0.0f;
  for (int i = 0; i < RMS_WINDOW; i++) {
    uint16_t idx = (tail + SAMPLES_PER_OBS - RMS_WINDOW + i) % SAMPLES_PER_OBS;
    float s = (float)audioBuffer[idx];
    sum += s * s;
  }
  return sqrtf(sum / RMS_WINDOW);
}

// ─────────────────────────────────────────────────────────────────────────────
//  FFT — Cooley-Tukey in-place radix-2 DIT  (unchanged)
// ─────────────────────────────────────────────────────────────────────────────
static void computeFFT(float* re, float* im, int n) {
  for (int i = 1, j = 0; i < n; i++) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      float tr = re[i]; re[i] = re[j]; re[j] = tr;
      float ti = im[i]; im[i] = im[j]; im[j] = ti;
    }
  }
  for (int len = 2; len <= n; len <<= 1) {
    float ang = -2.0f * (float)M_PI / (float)len;
    float wRe = cosf(ang), wIm = sinf(ang);
    for (int i = 0; i < n; i += len) {
      float curRe = 1.0f, curIm = 0.0f;
      for (int j = 0; j < len / 2; j++) {
        float uRe = re[i + j], uIm = im[i + j];
        float vRe = re[i+j+len/2]*curRe - im[i+j+len/2]*curIm;
        float vIm = re[i+j+len/2]*curIm + im[i+j+len/2]*curRe;
        re[i+j]       = uRe+vRe; im[i+j]       = uIm+vIm;
        re[i+j+len/2] = uRe-vRe; im[i+j+len/2] = uIm-vIm;
        float newRe = curRe*wRe - curIm*wIm;
        curIm = curRe*wIm + curIm*wRe;
        curRe = newRe;
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  MFCC EXTRACTION  (unchanged)
// ─────────────────────────────────────────────────────────────────────────────
static void computeMFCC(const int16_t* pcm) {
  for (int frame = 0; frame < N_FRAMES; frame++) {
    int start = frame * HOP_LEN;
    for (int i = 0; i < FRAME_LEN; i++) {
      frameBuf[i] = (float)pcm[start + i] / 32768.0f
                    * pgm_read_float(&KWS_HAMMING[i]);
    }
    for (int i = 0; i < FFT_SIZE; i++) { fftRe[i] = frameBuf[i]; fftIm[i] = 0.0f; }
    computeFFT(fftRe, fftIm, FFT_SIZE);

    float powerSpec[FFT_BINS];
    for (int k = 0; k < FFT_BINS; k++)
      powerSpec[k] = (fftRe[k]*fftRe[k] + fftIm[k]*fftIm[k]) / (float)FFT_SIZE;

    for (int m = 0; m < N_MELS; m++) {
      float energy = 0.0f;
      for (int k = 0; k < FFT_BINS; k++)
        energy += pgm_read_float(&KWS_MEL_FBANK[m*FFT_BINS+k]) * powerSpec[k];
      melBuf[m] = logf(energy + LOG_FLOOR);
    }
    for (int c = 0; c < N_MFCC; c++) {
      float acc = 0.0f;
      for (int m = 0; m < N_MELS; m++)
        acc += pgm_read_float(&KWS_DCT[c*N_MELS+m]) * melBuf[m];
      mfccOut[frame][c] = acc;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  NORMALISE  (unchanged)
// ─────────────────────────────────────────────────────────────────────────────
static void normaliseMFCC() {
  for (int f = 0; f < N_FRAMES; f++)
    for (int c = 0; c < N_MFCC; c++) {
      float mu  = pgm_read_float(&KWS_NORM_MEAN[c]);
      float sig = pgm_read_float(&KWS_NORM_STD [c]);
      mfccOut[f][c] = (mfccOut[f][c] - mu) / sig;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  SNAPSHOT  (unchanged)
// ─────────────────────────────────────────────────────────────────────────────
static int16_t pcmSnapshot[SAMPLES_PER_OBS];

static void snapshotAudio() {
  noInterrupts();
  uint16_t startIdx = writeIdx;
  for (int i = 0; i < SAMPLES_PER_OBS; i++)
    pcmSnapshot[i] = audioBuffer[(startIdx + i) % SAMPLES_PER_OBS];
  interrupts();
}

// ─────────────────────────────────────────────────────────────────────────────
//  RUN INFERENCE  — snapshot → MFCC → normalise → TFLite → print
// ─────────────────────────────────────────────────────────────────────────────
static void runInference() {
  snapshotAudio();
  computeMFCC(pcmSnapshot);
  normaliseMFCC();

  float* inPtr = inputTensor->data.f;
  for (int f = 0; f < N_FRAMES; f++)
    for (int c = 0; c < N_MFCC; c++)
      *inPtr++ = mfccOut[f][c];

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println(F("ERROR: Inference failed!"));
    return;
  }

  float* outPtr  = outputTensor->data.f;
  int    bestIdx = 0;
  float  bestConf = outPtr[0];
  for (int i = 1; i < KWS_NUM_CLASSES; i++)
    if (outPtr[i] > bestConf) { bestConf = outPtr[i]; bestIdx = i; }

  Serial.print(F("[KWS] "));
  Serial.print(bestConf >= CONFIDENCE_THRESHOLD
               ? KWS_CLASS_NAMES[bestIdx]
               : "uncertain");
  Serial.print(F("  ("));
  Serial.print((int)(bestConf * 100.0f));
  Serial.print(F("%)  scores: "));
  for (int i = 0; i < KWS_NUM_CLASSES; i++) {
    Serial.print(KWS_CLASS_NAMES[i]);
    Serial.print('=');
    Serial.print((int)(outPtr[i] * 100.0f));
    Serial.print(F("% "));
  }
  Serial.println();
}

// ─────────────────────────────────────────────────────────────────────────────
//  SETUP  (unchanged except banner text)
// ─────────────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  while (!Serial) {}
  Serial.println(F("=== KWS — Threshold-triggered Keyword Spotting ==="));
  Serial.println(F("Classes: clap / tap / snap / whistle"));
  Serial.println();

  PDM.onReceive(onPDMData);
  if (!PDM.begin(1, SAMPLE_RATE)) {
    Serial.println(F("ERROR: PDM init failed."));
    while (true) {}
  }
  Serial.println(F("PDM microphone started at 16 kHz."));

  tfModel = tflite::GetModel(kws_model_data);
  if (tfModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println(F("ERROR: TFLite schema version mismatch!"));
    while (true) {}
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter staticInterpreter(
    tfModel, resolver, tensorArena, TENSOR_ARENA_SIZE);
  interpreter = &staticInterpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println(F("ERROR: AllocateTensors() failed — increase TENSOR_ARENA_SIZE."));
    while (true) {}
  }

  inputTensor  = interpreter->input(0);
  outputTensor = interpreter->output(0);

  if (inputTensor->dims->size != 3 ||
      inputTensor->dims->data[1] != N_FRAMES ||
      inputTensor->dims->data[2] != N_MFCC) {
    Serial.println(F("ERROR: Unexpected input tensor shape!"));
    while (true) {}
  }

  Serial.print(F("TFLite model loaded. Arena used: "));
  Serial.print(interpreter->arena_used_bytes());
  Serial.println(F(" bytes"));

  while (!bufferFull) { delay(10); }
  Serial.println(F("Buffer filled — waiting for trigger."));
  Serial.print(F("Trigger threshold: RMS > "));
  Serial.println(TRIGGER_THRESHOLD);
  Serial.println(F("─────────────────────────────────────────"));
}

// ─────────────────────────────────────────────────────────────────────────────
//  LOOP — threshold-triggered state machine
//
//  LISTENING  →  RMS crosses TRIGGER_THRESHOLD  →  COLLECTING
//  COLLECTING →  POST_TRIGGER_SAMPLES received  →  inference  →  LISTENING
// ─────────────────────────────────────────────────────────────────────────────
void loop() {
  if (!bufferFull) return;

  switch (triggerState) {

    case TriggerState::LISTENING: {
      // Enforce rearm cooldown so one event can't fire twice
      if ((millis() - lastInferenceMs) < REARM_MS) break;

      float rms = computeRMS();
      if (rms > TRIGGER_THRESHOLD) {
        Serial.print(F("[TRIGGER] RMS="));
        Serial.println((int)rms);
        samplesAfterTrig = 0;
        triggerState     = TriggerState::COLLECTING;
      }
      break;
    }

    case TriggerState::COLLECTING:
      // samplesAfterTrig is incremented inside the ISR.
      // Once enough post-trigger audio has arrived, run inference.
      if (samplesAfterTrig >= POST_TRIGGER_SAMPLES) {
        runInference();
        lastInferenceMs = millis();
        triggerState    = TriggerState::LISTENING;
      }
      break;
  }
}