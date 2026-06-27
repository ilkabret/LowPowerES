#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "bottle_Prune_50pct_l1.h"   // the model C array
#include "test_img_good.h"          // stored test image (INT8)
#include "test_img_defect.h"

// Tensor arena — holds model activations. Tune down after first run.
constexpr int kTensorArenaSize = 180 * 1024;  // near the 256 KB ceiling
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Compute mean reconstruction error (the anomaly score) over the image,
// matching the frozen scoring protocol (minus border crop / blur, which
// can be added on-device or applied host-side from the raw error).
float reconstruction_error(const signed char* in, const signed char* out,
                           float in_scale, int in_zp,
                           float out_scale, int out_zp, int n) {
  double acc = 0.0;
  for (int i = 0; i < n; i++) {
    float a = (in[i]  - in_zp)  * in_scale;
    float b = (out[i] - out_zp) * out_scale;
    acc += (double)(a - b) * (a - b);
  }
  return (float)(acc / n);
}

void setup() {
  Serial.begin(115200);
  while (!Serial);

  model = tflite::GetModel(bottle_Prune_50pct_l1);
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
}

void run_one(const signed char* img, int img_len, const char* label) {
  // copy stored image into the input tensor
  for (int i = 0; i < img_len; i++) input->data.int8[i] = img[i];

  unsigned long t0 = micros();
  interpreter->Invoke();
  unsigned long t1 = micros();

  float score = reconstruction_error(
      input->data.int8, output->data.int8,
      input->params.scale, input->params.zero_point,
      output->params.scale, output->params.zero_point, img_len);

  Serial.print(label); Serial.print("  score="); Serial.print(score, 6);
  Serial.print("  latency_us="); Serial.println(t1 - t0);
}

void loop() {
  run_one(test_img_good,   test_img_good_len,   "GOOD  ");
  run_one(test_img_defect, test_img_defect_len, "DEFECT");
  delay(1000);
}
