#include "Arduino_BMI270_BMM150.h"

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "model.h"

const float accelerationThreshold = 2.5;
const int numSamples = 119;
const int NUM_FEATURES = 36;

int samplesRead = numSamples;

// buffers for IMU samples
float ax[numSamples], ay[numSamples], az[numSamples];
float gx[numSamples], gy[numSamples], gz[numSamples];

// feature vector
float features[NUM_FEATURES];

// TensorFlow objects
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// memory for TensorFlow
constexpr int tensorArenaSize = 16 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

const char* GESTURES[] = {
  "punch",
  "flex",
  "rightleft",
  "circle"
};

#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))

// -------- Feature Functions --------

float mean(float *x){
  float s=0;
  for(int i=0;i<numSamples;i++) s+=x[i];
  return s/numSamples;
}

float rms(float *x){
  float s=0;
  for(int i=0;i<numSamples;i++) s+=x[i]*x[i];
  return sqrt(s/numSamples);
}

float stddev(float *x){
  float m = mean(x);
  float s=0;
  for(int i=0;i<numSamples;i++) s+=(x[i]-m)*(x[i]-m);
  return sqrt(s/numSamples);
}

float minVal(float *x){
  float m=x[0];
  for(int i=1;i<numSamples;i++) if(x[i]<m) m=x[i];
  return m;
}

float maxVal(float *x){
  float m=x[0];
  for(int i=1;i<numSamples;i++) if(x[i]>m) m=x[i];
  return m;
}

float psdEnergy(float *x){
  float e=0;
  for(int i=0;i<numSamples;i++) {
    e+=x[i]*x[i];
  }
  return e / numSamples;
}

// -------- Feature Extraction --------

void computeFeatures(){

  float* signals[] = {ax, ay, az, gx, gy, gz};
  int idx = 0;

  for(int i = 0; i < 6; i++){
    float m   = mean(signals[i]);
    float s   = stddev(signals[i]);
    float r   = rms(signals[i]);
    float mn  = minVal(signals[i]);
    float mx  = maxVal(signals[i]);
    float psd = psdEnergy(signals[i]);

    features[idx++] = m;
    features[idx++] = s;
    features[idx++] = r;
    features[idx++] = mn;
    features[idx++] = mx;
    features[idx++] = psd;

    Serial.print("Mean: "); Serial.println(m);
    Serial.print("Std: "); Serial.println(s);
    Serial.print("RMS: "); Serial.println(r);
    Serial.print("Min: "); Serial.println(mn);
    Serial.print("Max: "); Serial.println(mx);
    Serial.print("Mean PSD: "); Serial.println(psd);
  }

  // normalize each feature independently
  // features are laid out in blocks of 6 per signal
  // but here we normalize each of the 36 features on its own min/max
  // Since each feature is a single scalar, we normalize across the
  // 6 signals for each feature type (mean, std, rms, min, max, range)

  // reshape: for each feature position (0-5), collect values across 6 signals
  for(int f = 0; f < 6; f++){  // f = feature type (mean, std, rms, min, max, range)

    float minF = features[f];
    float maxF = features[f];

    // find min/max across all 6 signals for this feature type
    for(int s = 1; s < 6; s++){
      float val = features[s * 6 + f];
      if(val < minF) minF = val;
      if(val > maxF) maxF = val;
    }

    float range = maxF - minF;
    if(range < 1e-6) range = 1;

    // normalize
    for(int s = 0; s < 6; s++){
      int idx = s * 6 + f;
      features[idx] = (features[idx] - minF) / range;
    }
  }

  // print normalized features
  Serial.println("Features:");
  for(int i = 0; i < NUM_FEATURES; i++){
    Serial.print(i); Serial.print(": ");
    Serial.println(features[i], 6);
  }
  Serial.println("---------------------");
}

// -------- Setup --------

void setup() {

  Serial.begin(9600);
  while(!Serial);

  if(!IMU.begin()){
    Serial.println("Failed to initialize IMU!");
    while(1);
  }

  Serial.print("Accelerometer sample rate = ");
  Serial.println(IMU.accelerationSampleRate());

  Serial.print("Gyroscope sample rate = ");
  Serial.println(IMU.gyroscopeSampleRate());

  Serial.println("System ready");

  // load TFLite model
  tflModel = tflite::GetModel(model);

  static tflite::MicroInterpreter static_interpreter(
    tflModel,
    tflOpsResolver,
    tensorArena,
    tensorArenaSize
  );

  tflInterpreter = &static_interpreter;

  // tflInterpreter->AllocateTensors();
  if (tflInterpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed!");
    while(1);
}

  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  // sanity check tensor shapes
  Serial.print("Input shape: ");
  Serial.println(tflInputTensor->dims->data[1]);   // should print 36
  Serial.print("Output shape: ");
  Serial.println(tflOutputTensor->dims->data[1]);  // should print 4
}

// -------- Main Loop --------

void loop() {

  float aX, aY, aZ, gX, gY, gZ;

  // wait for movement trigger
  while(samplesRead == numSamples){

    if(IMU.accelerationAvailable()){

      IMU.readAcceleration(aX,aY,aZ);

      float aSum = fabs(aX)+fabs(aY)+fabs(aZ);

      if(aSum >= accelerationThreshold){
        samplesRead = 0;
        break;
      }
    }
  }

  // collect samples
  while(samplesRead < numSamples){

    if(IMU.accelerationAvailable() && IMU.gyroscopeAvailable()){

      IMU.readAcceleration(aX,aY,aZ);
      IMU.readGyroscope(gX,gY,gZ);

      ax[samplesRead]=aX;
      ay[samplesRead]=aY;
      az[samplesRead]=aZ;

      gx[samplesRead]=gX;
      gy[samplesRead]=gY;
      gz[samplesRead]=gZ;

      samplesRead++;

      if(samplesRead == numSamples){

        // compute features
        computeFeatures();

        // copy features to model input
        for(int i=0;i<NUM_FEATURES;i++)
            tflInputTensor->data.f[i] = features[i];

        // now you can safely run inference
        if(tflInterpreter->Invoke()!=kTfLiteOk){
            Serial.println("Invoke failed!");
            while(1);
        }

        // print model output
        for(int i=0;i<NUM_GESTURES;i++){
            Serial.print(GESTURES[i]);
            Serial.print(": ");
            Serial.println(tflOutputTensor->data.f[i],6);
        }
      }
    }
  }
}
