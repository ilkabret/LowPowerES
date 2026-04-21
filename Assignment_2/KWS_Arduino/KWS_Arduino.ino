/**
 * KWS_Arduino.ino
 * ─────────────────────────────────────────────────────────────────────────────
 * On-Device Keyword Spotting — Arduino Nano 33 BLE Sense (Rev1 / Rev2)
 *
 * Hardware  : Arduino Nano 33 BLE Sense (or Rev2)
 * Microphone: Built-in PDM microphone (MP34DT05 / MP34DT06)
 *
 * Pipeline (mirrors the Python training notebook exactly):
 *   PDM audio → ring buffer (2000 int16 @ 16 kHz)
 *   → Frame segmentation (256 samples, 128-sample hop, 50% overlap)
 *   → Hamming window
 *   → Real FFT (256-point) → Power spectrum
 *   → Mel filter bank (26 filters, 300–8000 Hz)
 *   → Log energy  (log(e + 1e-10))
 *   → DCT-II ortho → 13 MFCC coefficients per frame
 *   → Z-score normalisation (training mean / std)
 *   → TFLite Micro 1-D CNN inference
 *   → Serial output of predicted class + confidence
 *
 * Required libraries (install via Library Manager):
 *   • Arduino_TensorFlowLite   (El Gato / tflite-micro)  ≥ 2.4.0
 *   • PDM                      (bundled with Nano 33 BLE board package)
 *
 * Place these generated files in the same sketch folder:
 *   • kws_model_data.h   (from Colab: arduino_weights/kws_model_data.h)
 *   • kws_params.h       (from Colab: arduino_weights/kws_params.h)
 *   • mfcc_tables.h      (this repo — pre-computed Hamming / Mel / DCT tables)
 *
 * Memory budget (Nano 33 BLE Sense: 256 KB SRAM):
 *   Audio ring buffer       4 000 B
 *   MFCC feature buffer       728 B
 *   TFLite tensor arena    ~40 KB  (adjust TENSOR_ARENA_SIZE if needed)
 *   Static tables (flash)  ~20 KB  → stored in PROGMEM, not SRAM
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <PDM.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <math.h>
#include <avr/pgmspace.h>   // PROGMEM / pgm_read_float

// ── Project headers ──────────────────────────────────────────────────────────
#include "mfcc_tables.h"    // Hamming, Mel filter bank, DCT matrix (PROGMEM)
#include "kws_model_data.h" // kws_model_data[], kws_model_len
#include "kws_params.h"     // KWS_NORM_MEAN[], KWS_NORM_STD[], class names, #defines

// ─────────────────────────────────────────────────────────────────────────────
//  CONFIGURATION  (must match training notebook — do NOT change)
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int   SAMPLE_RATE      = 16000;  // Hz
static constexpr int   SAMPLES_PER_OBS  = 2000;   // samples per inference window
static constexpr int   FRAME_LEN        = 256;    // FFT frame length
static constexpr int   HOP_LEN          = 128;    // 50% overlap
static constexpr int   FFT_SIZE         = 256;    // must equal FRAME_LEN
static constexpr int   FFT_BINS         = FFT_SIZE / 2 + 1;  // 129
static constexpr int   N_MELS           = 26;
static constexpr int   N_MFCC           = 13;
static constexpr int   N_FRAMES         = (SAMPLES_PER_OBS - FRAME_LEN) / HOP_LEN + 1; // 14
static constexpr float LOG_FLOOR        = 1e-10f;

// TFLite tensor arena — tune down if you run out of SRAM
static constexpr int   TENSOR_ARENA_SIZE = 40 * 1024;

// Confidence threshold — predictions below this are reported as "uncertain"
static constexpr float CONFIDENCE_THRESHOLD = 0.60f;

// ─────────────────────────────────────────────────────────────────────────────
//  GLOBALS
// ─────────────────────────────────────────────────────────────────────────────

// ── Audio ring buffer (filled by PDM ISR) ────────────────────────────────────
static volatile int16_t  audioBuffer[SAMPLES_PER_OBS];
static volatile uint16_t writeIdx   = 0;   // next position to write
static volatile bool     bufferFull = false;

// ── PDM read buffer (DMA → ISR staging) ──────────────────────────────────────
static int16_t pdmBuffer[256];

// ── MFCC working memory (all on stack / static) ──────────────────────────────
// One frame at a time — re-used across frames to save SRAM
static float   frameBuf [FRAME_LEN];     // windowed frame (float)
static float   fftRe    [FFT_SIZE];      // FFT real part
static float   fftIm    [FFT_SIZE];      // FFT imaginary part
static float   melBuf   [N_MELS];        // Mel energies for one frame
static float   mfccOut  [N_FRAMES][N_MFCC]; // final feature matrix

// ── TFLite ───────────────────────────────────────────────────────────────────
static uint8_t tensorArena[TENSOR_ARENA_SIZE];
static const tflite::Model*        tfModel     = nullptr;
static tflite::MicroInterpreter*   interpreter = nullptr;
static TfLiteTensor*               inputTensor = nullptr;
static TfLiteTensor*               outputTensor= nullptr;

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
    // Mark full once the ring buffer has been written at least once
    if (writeIdx == 0) bufferFull = true;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  FFT — Cooley-Tukey in-place radix-2 DIT
//  Operates on fftRe[] / fftIm[] (length must be a power of two)
// ─────────────────────────────────────────────────────────────────────────────
static void computeFFT(float* re, float* im, int n) {
  // Bit-reversal permutation
  for (int i = 1, j = 0; i < n; i++) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      float tr = re[i]; re[i] = re[j]; re[j] = tr;
      float ti = im[i]; im[i] = im[j]; im[j] = ti;
    }
  }
  // Butterfly stages
  for (int len = 2; len <= n; len <<= 1) {
    float ang = -2.0f * (float)M_PI / (float)len;
    float wRe = cosf(ang), wIm = sinf(ang);
    for (int i = 0; i < n; i += len) {
      float curRe = 1.0f, curIm = 0.0f;
      for (int j = 0; j < len / 2; j++) {
        float uRe = re[i + j];
        float uIm = im[i + j];
        float vRe = re[i + j + len/2] * curRe - im[i + j + len/2] * curIm;
        float vIm = re[i + j + len/2] * curIm + im[i + j + len/2] * curRe;
        re[i + j]         = uRe + vRe;
        im[i + j]         = uIm + vIm;
        re[i + j + len/2] = uRe - vRe;
        im[i + j + len/2] = uIm - vIm;
        float newRe = curRe * wRe - curIm * wIm;
        curIm       = curRe * wIm + curIm * wRe;
        curRe       = newRe;
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  MFCC EXTRACTION
//  Reads a contiguous snapshot of audioBuffer[], computes mfccOut[][].
//  Preprocessing is identical to the Python notebook:
//    1. int16 → float32  /32768
//    2. Frame + Hamming window
//    3. Real FFT → power spectrum  (|X|² / N)
//    4. Mel filter bank (from PROGMEM)
//    5. log(energy + 1e-10)
//    6. DCT-II ortho (from PROGMEM)
//    7. Keep first N_MFCC coefficients
// ─────────────────────────────────────────────────────────────────────────────
static void computeMFCC(const int16_t* pcm) {
  for (int frame = 0; frame < N_FRAMES; frame++) {
    int start = frame * HOP_LEN;

    // ── Step 1 + 2: normalise & apply Hamming window ─────────────────────
    for (int i = 0; i < FRAME_LEN; i++) {
      float s   = (float)pcm[start + i] / 32768.0f;
      float win = pgm_read_float(&KWS_HAMMING[i]);
      frameBuf[i] = s * win;
    }

    // ── Step 3a: load into FFT arrays ────────────────────────────────────
    for (int i = 0; i < FFT_SIZE; i++) {
      fftRe[i] = frameBuf[i];
      fftIm[i] = 0.0f;
    }

    // ── Step 3b: FFT ─────────────────────────────────────────────────────
    computeFFT(fftRe, fftIm, FFT_SIZE);

    // ── Step 3c: power spectrum  |X[k]|² / N  (only bins 0..N/2) ────────
    float powerSpec[FFT_BINS];
    for (int k = 0; k < FFT_BINS; k++) {
      powerSpec[k] = (fftRe[k] * fftRe[k] + fftIm[k] * fftIm[k])
                     / (float)FFT_SIZE;
    }

    // ── Step 4: Mel filter bank  (row-major PROGMEM: mel × fft_bin) ──────
    for (int m = 0; m < N_MELS; m++) {
      float energy = 0.0f;
      for (int k = 0; k < FFT_BINS; k++) {
        energy += pgm_read_float(&KWS_MEL_FBANK[m * FFT_BINS + k])
                  * powerSpec[k];
      }
      // ── Step 5: log energy ────────────────────────────────────────────
      melBuf[m] = logf(energy + LOG_FLOOR);
    }

    // ── Step 6 + 7: DCT-II ortho, keep N_MFCC coefficients ──────────────
    // KWS_DCT is (N_MFCC × N_MELS), row-major.
    for (int c = 0; c < N_MFCC; c++) {
      float acc = 0.0f;
      for (int m = 0; m < N_MELS; m++) {
        acc += pgm_read_float(&KWS_DCT[c * N_MELS + m]) * melBuf[m];
      }
      mfccOut[frame][c] = acc;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  NORMALISE  —  z-score using training mean/std (from kws_params.h)
//  mean/std are stored as [N_MFCC] arrays (one value per coefficient).
// ─────────────────────────────────────────────────────────────────────────────
static void normaliseMFCC() {
  for (int f = 0; f < N_FRAMES; f++) {
    for (int c = 0; c < N_MFCC; c++) {
      float mu  = pgm_read_float(&KWS_NORM_MEAN[c]);
      float sig = pgm_read_float(&KWS_NORM_STD [c]);
      mfccOut[f][c] = (mfccOut[f][c] - mu) / sig;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  COPY SNAPSHOT from volatile ring buffer → local flat array
//  Disables interrupts for the copy to avoid tearing.
// ─────────────────────────────────────────────────────────────────────────────
static void snapshotAudio(int16_t* dst) {
  noInterrupts();
  uint16_t startIdx = writeIdx;   // oldest sample is at writeIdx (ring)
  for (int i = 0; i < SAMPLES_PER_OBS; i++) {
    dst[i] = audioBuffer[(startIdx + i) % SAMPLES_PER_OBS];
  }
  interrupts();
}

// ─────────────────────────────────────────────────────────────────────────────
//  SETUP
// ─────────────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  while (!Serial) { /* wait for USB */ }
  Serial.println(F("=== KWS — On-Device Keyword Spotting ==="));
  Serial.println(F("Classes: clap / tap / snap / silence"));
  Serial.println();

  // ── Initialise PDM microphone ─────────────────────────────────────────────
  PDM.onReceive(onPDMData);
  // Mono, 16 kHz — the Nano 33 BLE Sense PDM library handles decimation
  if (!PDM.begin(1, SAMPLE_RATE)) {
    Serial.println(F("ERROR: PDM init failed — check board variant."));
    while (true) {}
  }
  // Optional: set PDM gain (0–255, default is already reasonable)
  // PDM.setGain(30);
  Serial.println(F("PDM microphone started at 16 kHz."));

  // ── Load TFLite model ─────────────────────────────────────────────────────
  tfModel = tflite::GetModel(kws_model_data);
  if (tfModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println(F("ERROR: TFLite schema version mismatch!"));
    while (true) {}
  }

  // AllOpsResolver is large but safe; swap for MicroMutableOpResolver
  // listing only Conv2D, MaxPool2D, FullyConnected, Softmax, Reshape
  // once everything works, to save a few KB of flash.
  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter staticInterpreter(
    tfModel, resolver, tensorArena, TENSOR_ARENA_SIZE);
  interpreter = &staticInterpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println(F("ERROR: TFLite AllocateTensors() failed — "
                     "increase TENSOR_ARENA_SIZE."));
    while (true) {}
  }

  inputTensor  = interpreter->input(0);
  outputTensor = interpreter->output(0);

  // Sanity-check tensor dimensions
  // Expected input:  [1, N_FRAMES, N_MFCC]  = [1, 14, 13]
  if (inputTensor->dims->size != 3 ||
      inputTensor->dims->data[1] != N_FRAMES ||
      inputTensor->dims->data[2] != N_MFCC) {
    Serial.println(F("ERROR: Unexpected input tensor shape!"));
    Serial.print(F("Got dims: "));
    for (int d = 0; d < inputTensor->dims->size; d++) {
      Serial.print(inputTensor->dims->data[d]);
      Serial.print(' ');
    }
    Serial.println();
    while (true) {}
  }

  Serial.print(F("TFLite model loaded. Arena used: "));
  Serial.print(interpreter->arena_used_bytes());
  Serial.println(F(" bytes"));
  Serial.println(F("Listening... (buffering audio)"));
  Serial.println(F("─────────────────────────────────────────"));

  // Wait until the ring buffer has filled at least once
  while (!bufferFull) { delay(10); }
  Serial.println(F("Buffer filled — inference running."));
}

// ─────────────────────────────────────────────────────────────────────────────
//  LOOP — runs once every ~250 ms (throttled by inferencePeriodMs)
// ─────────────────────────────────────────────────────────────────────────────
static const unsigned long INFERENCE_PERIOD_MS = 250;
static unsigned long lastInferenceMs = 0;
static int16_t       pcmSnapshot[SAMPLES_PER_OBS];

void loop() {
  unsigned long now = millis();
  if (now - lastInferenceMs < INFERENCE_PERIOD_MS) return;
  lastInferenceMs = now;

  if (!bufferFull) return;

  // ── 1. Snapshot audio ─────────────────────────────────────────────────────
  snapshotAudio(pcmSnapshot);

  // ── 2. Compute MFCC ───────────────────────────────────────────────────────
  computeMFCC(pcmSnapshot);

  // ── 3. Normalise ──────────────────────────────────────────────────────────
  normaliseMFCC();

  // ── 4. Copy feature matrix into TFLite input tensor ──────────────────────
  float* inPtr = inputTensor->data.f;
  for (int f = 0; f < N_FRAMES; f++) {
    for (int c = 0; c < N_MFCC; c++) {
      *inPtr++ = mfccOut[f][c];
    }
  }

  // ── 5. Run inference ──────────────────────────────────────────────────────
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println(F("ERROR: Inference failed!"));
    return;
  }

  // ── 6. Read output probabilities ──────────────────────────────────────────
  float* outPtr   = outputTensor->data.f;
  int    bestIdx  = 0;
  float  bestConf = outPtr[0];
  for (int i = 1; i < KWS_NUM_CLASSES; i++) {
    if (outPtr[i] > bestConf) {
      bestConf = outPtr[i];
      bestIdx  = i;
    }
  }

  // ── 7. Print result ───────────────────────────────────────────────────────
  Serial.print(F("[KWS] "));
  if (bestConf >= CONFIDENCE_THRESHOLD) {
    Serial.print(KWS_CLASS_NAMES[bestIdx]);
  } else {
    Serial.print(F("uncertain"));
  }
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
