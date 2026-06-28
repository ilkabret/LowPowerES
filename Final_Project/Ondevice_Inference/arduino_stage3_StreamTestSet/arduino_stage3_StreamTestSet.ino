#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "hazelnut_Distill_b16.h"   // the model C array

// Frozen scoring protocol: per-pixel error -> border crop (4) -> 5x5 blur -> mean.
// Input/output are CHW int8 (3 x 64 x 64), flat: idx = c*4096 + row*64 + col.
#define IMG_H 64
#define IMG_W 64
#define N_CH  3
#define BORDER 4
#define BLUR_K 5   // 5x5 average pool

// Compute mean reconstruction error (the anomaly score) over the image,
// matching the frozen scoring protocol (with border crop / blur)
float reconstruction_error(const signed char* in, const signed char* out,
                           float in_scale, int in_zp,
                           float out_scale, int out_zp, int /*n_unused*/) {
  // 1. Per-pixel squared error, averaged over the 3 channels -> err_map[64*64]
  static float err_map[IMG_H * IMG_W];
  for (int row = 0; row < IMG_H; row++) {
    for (int col = 0; col < IMG_W; col++) {
      float acc = 0.0f;
      for (int c = 0; c < N_CH; c++) {
        int idx = c * (IMG_H * IMG_W) + row * IMG_W + col;
        float a = (in[idx]  - in_zp)  * in_scale;
        float b = (out[idx] - out_zp) * out_scale;
        acc += (a - b) * (a - b);
      }
      err_map[row * IMG_W + col] = acc / N_CH;
    }
  }

  // 2 + 3. For each pixel in the cropped region, compute the 5x5 average
  //        (blur) centered on it, then accumulate for the final mean.
  //        Border crop = we only iterate over rows/cols in [BORDER, H-BORDER).
  int pad = BLUR_K / 2;   // 2
  double sum = 0.0;
  int count = 0;
  for (int row = BORDER; row < IMG_H - BORDER; row++) {
    for (int col = BORDER; col < IMG_W - BORDER; col++) {
      // exact PyTorch avg_pool2d(padding=2) match: divide by 25, OOB = 0
      float blur_acc = 0.0f;
      for (int dr = -pad; dr <= pad; dr++) {
        for (int dc = -pad; dc <= pad; dc++) {
          int r = row + dr, c = col + dc;
          if (r >= 0 && r < IMG_H && c >= 0 && c < IMG_W) {
            blur_acc += err_map[r * IMG_W + c];
          }
          // else: add 0 (zero-padding)
        }
      }
      sum += blur_acc / (BLUR_K * BLUR_K);   // divide by 25 always
      count++;
    }
  }

  // 4. Mean over the cropped, blurred region
  return (float)(sum / count);
}

// Tensor arena — holds model activations. Tune down after first run.
constexpr int kTensorArenaSize = 160 * 1024;  // near the 256 KB ceiling
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void setup() {
  Serial.begin(115200);
  while (!Serial);

  model = tflite::GetModel(hazelnut_Distill_b16);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!"); while (1);
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed — increase arena size"); while (1);
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);

  Serial.print("Arena used (bytes): ");
  Serial.println(interpreter->arena_used_bytes());
  Serial.print("Input bytes: ");  Serial.println(input->bytes);
  Serial.print("Output bytes: "); Serial.println(output->bytes);
  Serial.print("Model len: ");
  Serial.println(hazelnut_Distill_b16_len);
}

void run_one(const signed char* img, int img_len, const char* label) {
  // save a copy of the input separately
  static signed char input_copy[12288];
  for (int i = 0; i < img_len; i++) {
    input->data.int8[i] = img[i];
    input_copy[i] = img[i];        // independent copy
  }

  unsigned long t0 = millis();
  interpreter->Invoke();
  unsigned long t1 = millis();

  // compute error against the saved input copy, not the live tensor
  float score = reconstruction_error(
      input_copy, output->data.int8,
      input->params.scale, input->params.zero_point,
      output->params.scale, output->params.zero_point, img_len);

  Serial.print(label); Serial.print(" score="); Serial.print(score, 6);
  Serial.print(" latency_ms="); Serial.println(t1 - t0);
}

void loop() {
  static signed char input_copy[12288];
  int received = 0;

  // Read 12288 bytes, draining the buffer as they come in
  while (received < 12288) {
    if (Serial.available()) {
      signed char b = (signed char)Serial.read();
      input->data.int8[received] = b;
      input_copy[received] = b;
      received++;
    }
  }

  unsigned long t0 = millis();
  interpreter->Invoke();
  unsigned long t1 = millis();

  float score = reconstruction_error(
      input_copy, output->data.int8,
      input->params.scale, input->params.zero_point,
      output->params.scale, output->params.zero_point, 12288);

  Serial.print("SCORE:"); Serial.print(score, 8);
  Serial.print(",LAT:"); Serial.println(t1 - t0);
}
