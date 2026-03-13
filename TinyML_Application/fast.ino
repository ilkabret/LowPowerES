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
constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

const char* GESTURES[] = {
  "punch",
  "flex"
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
  for(int i=0;i<numSamples;i++) e+=x[i]*x[i];
  return e;
}

// -------- Feature Extraction --------

void computeFeatures(){

  float* signals[] = {ax, ay, az, gx, gy, gz};

  int idx=0;

  for(int i=0;i<6;i++){
    features[idx++] = mean(signals[i]);
    features[idx++] = stddev(signals[i]);
    features[idx++] = rms(signals[i]);
    features[idx++] = minVal(signals[i]);
    features[idx++] = maxVal(signals[i]);
    features[idx++] = psdEnergy(signals[i]);
  }

  // normalize features (same as Python)
  float minF = features[0];
  float maxF = features[0];

  for(int i=1;i<NUM_FEATURES;i++){
    if(features[i] < minF) minF = features[i];
    if(features[i] > maxF) maxF = features[i];
  }

  for(int i=0;i<NUM_FEATURES;i++){
    features[i] = (features[i] - minF) / (maxF - minF);
  }
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

  tflInterpreter->AllocateTensors();

  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
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

        // run inference
        if(tflInterpreter->Invoke()!=kTfLiteOk){
          Serial.println("Invoke failed!");
          while(1);
        }

        // print probabilities
        for(int i=0;i<NUM_GESTURES;i++){
          Serial.print(GESTURES[i]);
          Serial.print(": ");
          Serial.println(tflOutputTensor->data.f[i],6);
        }

        Serial.println();
      }
    }
  }
}
