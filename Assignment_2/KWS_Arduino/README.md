# KWS_Arduino — On-Device Keyword Spotting

Arduino Nano 33 BLE Sense sketch for real-time keyword spotting using MFCC features
and a TFLite Micro 1-D CNN.

## File structure

```
KWS_Arduino/
├── KWS_Arduino.ino      ← main sketch (open this in the Arduino IDE)
├── mfcc_tables.h        ← pre-computed Hamming / Mel / DCT tables (PROGMEM)
├── kws_params.h         ← normalisation mean/std  ← REPLACE WITH COLAB OUTPUT
└── kws_model_data.h     ← TFLite model byte array ← REPLACE WITH COLAB OUTPUT
```

## Before uploading

1. **Train the model** by running all cells in `KWS_MFCC_Training.ipynb`.
2. **Download** `arduino_weights/kws_params.h` and `arduino_weights/kws_model_data.h`
   from Colab (Section 14 — Download all outputs).
3. **Replace** the two placeholder files in this folder with the downloaded ones.

## Required Arduino libraries

Install via **Sketch → Include Library → Manage Libraries…**:

| Library | Version |
|---|---|
| `Arduino_TensorFlowLite` | ≥ 2.4.0 |
| `PDM` | (bundled with Nano 33 BLE board package) |

Board package: **Arduino Mbed OS Nano Boards** ≥ 3.5.0  
(`Tools → Board → Boards Manager → search "Nano 33 BLE"`)

## Uploading

```
Tools → Board   : Arduino Nano 33 BLE
Tools → Port    : <your port>
Sketch → Upload
```

## Serial Monitor output

Open at **115200 baud**. Each line after inference:

```
[KWS] clap  (87%)  scores: clap=87% tap=5% snap=4% silence=4%
[KWS] silence  (94%)  scores: clap=2% tap=1% snap=3% silence=94%
```

Predictions below 60 % confidence are labelled `uncertain`.

## MFCC pipeline (matches training exactly)

| Step | Detail |
|---|---|
| Audio capture | PDM mic, 16 kHz mono, ring buffer 2000 samples |
| Frame size | 256 samples |
| Hop length | 128 samples (50 % overlap) → 14 frames |
| Window | Hamming (`np.hamming(256)`) |
| FFT | 256-point real FFT → 129 power bins |
| Mel filters | 26 triangular filters, 300–8000 Hz (O'Shaughnessy Mel) |
| Log | `log(energy + 1e-10)` |
| DCT | Type-II, orthonormal (`scipy.fftpack.dct` norm=`'ortho'`) |
| Coefficients | First 13 (MFCC 0–12) |
| Normalisation | Per-coefficient z-score (mean/std from training set) |

## Memory usage (approximate)

| Region | Size |
|---|---|
| Audio ring buffer (SRAM) | 4 000 B |
| MFCC feature buffer (SRAM) | 728 B |
| FFT working arrays (SRAM) | ~4 KB |
| TFLite tensor arena (SRAM) | ~40 KB (adjust `TENSOR_ARENA_SIZE`) |
| Lookup tables (Flash/PROGMEM) | ~53 KB |
| TFLite model (Flash) | <50 KB (INT8 quantised) |
