# Energy-Aware Visual Anomaly Detection on MCUs

> Compression-aware deployment and runtime adaptation of a CNN-based anomaly detector on the **Arduino Nano 33 BLE Sense Rev 2**.

This project trains a compact CNN-based anomaly detector on the **MVTec AD** dataset, systematically compares three CNN compression techniques (post-training quantization, structured filter pruning, knowledge distillation), deploys the resulting variants on a Cortex-M4 microcontroller via **TensorFlow Lite for Microcontrollers**, and implements a runtime **energy-aware inference adaptation** mechanism that selects which compressed model to run based on the remaining energy budget.

The end product is a joint accuracy / memory / latency / energy **Pareto characterization** of compression techniques for visual anomaly detection on real MCU hardware, plus a battery-powered demo of adaptive inference.

---

## Table of contents

1. [Motivation](#motivation)
2. [Project structure](#project-structure)
3. [Pipeline overview](#pipeline-overview)
4. [Hardware](#hardware)
5. [Software requirements](#software-requirements)
6. [Getting started](#getting-started)
7. [Reproducing the benchmark](#reproducing-the-benchmark)
8. [Results](#results)
9. [Runtime adaptation demo](#runtime-adaptation-demo)
10. [Roadmap](#roadmap)
11. [References](#references)
12. [Authors](#authors)
13. [License](#license)

---

## Motivation

Visual anomaly detection is a core component of industrial quality control and edge-deployed authentication systems (counterfeit detection, packaging inspection, etc.). Deployment on microcontrollers brings two challenges:

- The model must fit a tight memory and compute budget (here: 256 KB SRAM, 1 MB Flash, Cortex-M4F @ 64 MHz).
- The energy supply is finite and sometimes variable (battery, harvested energy), which calls for inference strategies that gracefully degrade as energy depletes.

Most published compression studies target supervised classification. Anomaly detection has different loss functions, output structures, and failure modes — so how far compression can be pushed before detection AUROC collapses is an open question. This project provides a systematic, hardware-grounded answer for one representative setting.

## Project structure

```
.
├── README.md
├── data/                          # MVTec AD (not versioned — see Getting started)
├── models/
│   ├── baseline/                  # FP32 trained baseline
│   ├── quantized/                 # INT8 TFLite variants
│   ├── pruned/                    # Structured-pruned variants
│   └── distilled/                 # Knowledge-distilled student variants
├── src/
│   ├── train/                     # Training scripts (baseline + KD student)
│   ├── compress/
│   │   ├── quantize.py            # Post-training INT8 quantization
│   │   ├── prune.py               # Structured filter pruning (L1, Taylor)
│   │   └── distill.py             # Knowledge distillation
│   ├── eval/                      # Offline AUROC + MACs + size evaluation
│   └── convert/                   # Keras → TFLite → C array (xxd)
├── firmware/
│   ├── arduino_ad/                # Arduino sketch (TFLM runtime + dispatcher)
│   ├── models_c/                  # Generated C arrays of all variants
│   └── adaptive_policy/           # Runtime selection policy code
├── benchmarks/
│   ├── on_device/                 # Scripts to drive the board + parse logs
│   └── results/                   # CSVs and plots
├── docs/
│   ├── proposal.pdf
│   └── report/                    # LaTeX final report
└── requirements.txt
```

## Pipeline overview

```
                ┌──────────────────────┐
                │   MVTec AD dataset   │
                └──────────┬───────────┘
                           │  (2–3 categories)
                           ▼
                ┌──────────────────────┐
                │  Train FP32 baseline │
                │  (compact CNN AE /   │
                │   PaDiM-style)       │
                └──────────┬───────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌─────────┐  ┌─────────┐  ┌─────────────┐
        │  PTQ    │  │ Pruning │  │ Distillation │
        │  INT8   │  │ (L1,    │  │ (small       │
        │         │  │  Taylor)│  │  student)    │
        └────┬────┘  └────┬────┘  └──────┬──────┘
             └────────────┼──────────────┘
                          ▼
                 ┌─────────────────┐
                 │  Combinations   │
                 │  (e.g. prune    │
                 │   + INT8, KD    │
                 │   + INT8)       │
                 └────────┬────────┘
                          ▼
              ┌────────────────────────┐
              │ Offline AUROC, params, │
              │ MACs, model size       │
              └────────────┬───────────┘
                           ▼
              ┌────────────────────────┐
              │ TFLite Micro on        │
              │ Arduino Nano 33 BLE    │
              │ → latency, SRAM, Flash,│
              │   energy/inference     │
              └────────────┬───────────┘
                           ▼
              ┌────────────────────────┐
              │ Pareto plots           │
              │ + adaptive runtime     │
              │   policy demo          │
              └────────────────────────┘
```

## Hardware

| Component | Reference |
|-----------|-----------|
| MCU board | Arduino Nano 33 BLE Sense Rev 2 (Nordic nRF52840, Cortex-M4F @ 64 MHz, 256 KB SRAM, 1 MB Flash) |
| Power measurement (optional) | USB current meter or low-side shunt + DMM; Nordic Power Profiler Kit II if available |
| Battery for adaptive demo | Single Li-ion 18650 or 3 × AA, with voltage divider into an analog pin |

> The on-board camera/microphone are **not** used. Inputs are streamed from the host over USB-serial or pre-loaded as C arrays in flash.

## Software requirements

- Python 3.10+
- TensorFlow 2.15+ (with `tensorflow_model_optimization` for pruning, `tflite_micro` for export)
- PyTorch 2.x (optional, only if using Anomalib for baselines)
- Anomalib (optional, for PaDiM / PatchCore / EfficientAD references)
- Arduino IDE 2.x or `arduino-cli`
- `Arduino_TensorFlowLite` library for the board

Install Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Getting started

### 1. Get the dataset

Download MVTec AD from https://www.mvtec.com/company/research/datasets/mvtec-ad and unzip into `data/mvtec/`. Initial experiments focus on 2–3 categories (e.g. `bottle`, `hazelnut`, `metal_nut`).

```
data/mvtec/
├── bottle/
├── hazelnut/
└── metal_nut/
```

### 2. Train the baseline

```bash
python src/train/train_baseline.py --category bottle --epochs 100 --out models/baseline/
```

### 3. Apply the three compression techniques

```bash
# Post-training INT8 quantization
python src/compress/quantize.py --in models/baseline/bottle.h5 --out models/quantized/

# Structured filter pruning (try L1 and Taylor)
python src/compress/prune.py --in models/baseline/bottle.h5 --method l1 --ratio 0.3 --finetune 10
python src/compress/prune.py --in models/baseline/bottle.h5 --method taylor --ratio 0.3 --finetune 10

# Knowledge distillation into a smaller student
python src/compress/distill.py --teacher models/baseline/bottle.h5 --student-config configs/student_tiny.yaml
```

### 4. Convert for the MCU

```bash
python src/convert/to_tflite_micro.py --in models/quantized/bottle_int8.tflite --out firmware/models_c/bottle_int8.h
```

### 5. Flash the Arduino

Open `firmware/arduino_ad/arduino_ad.ino` in the Arduino IDE (or use `arduino-cli`), select the **Arduino Nano 33 BLE Sense Rev 2** board, and upload.

## Reproducing the benchmark

To run the full on-device benchmark (assumes the board is connected over USB):

```bash
python benchmarks/on_device/run_full_benchmark.py \
    --port /dev/ttyACM0 \
    --variants quantized,pruned_l1,pruned_taylor,distilled,kd_int8,prune_int8 \
    --n-images 100 \
    --out benchmarks/results/
```

For each variant the script records: AUROC (from offline eval), parameter count, MAC count, peak SRAM, flash size, mean inference latency, and energy per inference (if a current measurement source is configured).

## Results

Results will appear in `benchmarks/results/` as CSVs and as Pareto plots generated by `benchmarks/results/make_plots.py`. Headline tables:

| Variant | AUROC | Params | Size (KB) | Latency (ms) | Energy (mJ/inf) |
|---------|------:|-------:|----------:|-------------:|----------------:|
| Baseline FP32 (host) | — | — | — | — | — |
| INT8 quantized | — | — | — | — | — |
| Pruned 30% (L1) + INT8 | — | — | — | — | — |
| Pruned 30% (Taylor) + INT8 | — | — | — | — | — |
| Distilled student + INT8 | — | — | — | — | — |

> Numbers will be filled in as on-device measurements are completed. See `docs/report/` for the final analysis.

## Runtime adaptation demo

The Arduino firmware stores multiple compressed variants simultaneously in flash. At each inference the **adaptive policy** in `firmware/adaptive_policy/` selects which variant to execute based on the supply voltage read from an analog pin (battery proxy).

Two policies are implemented:

- **Threshold-based** — voltage range partitioned into bands; one variant per band.
- **Utility-based** — variant chosen to maximize expected AUROC given remaining energy and an estimated number of remaining inferences.

A complementary off-device simulator (`src/eval/simulate_policy.py`) replays the two policies against recorded battery discharge curves and synthetic energy-harvesting traces, which lets us evaluate the policy across many scenarios without rebuilding the hardware setup.

To run the live demo:

1. Disconnect USB; switch the board to battery power.
2. Press the user button to start the inference loop.
3. The on-board LED indicates which variant is currently selected; serial output (when reconnected) shows the voltage trace and the variant timeline.

## Roadmap

- [x] Project proposal accepted
- [ ] Stage 1 — baseline anomaly detector trained on MVTec AD
- [ ] Stage 2 — compression sweep complete (PTQ, pruning, distillation, combinations)
- [ ] Stage 3 — all variants deployed on Arduino with measured latency / memory / energy
- [ ] Stage 4 — adaptive runtime policy implemented and validated on battery
- [ ] Final report and demo video

## References

The four core references for the project:

1. **Defard et al.**, "PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization," ICPR 2020. [arXiv:2011.08785](https://arxiv.org/abs/2011.08785)
2. **Batzner, Heckler, König**, "EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies," WACV 2024. [arXiv:2303.14535](https://arxiv.org/abs/2303.14535)
3. **"Efficient Visual Anomaly Detection at the Edge: Enabling Real-Time Industrial Inspection on Resource-Constrained Devices"** — PaDiM-Lite and PatchCore-Lite for edge deployment. [arXiv:2603.20288](https://arxiv.org/abs/2603.20288)
4. **Abushahla et al.**, "Quantized Neural Networks for Microcontrollers: A Comprehensive Review," 2025. [arXiv:2508.15008](https://arxiv.org/abs/2508.15008)

Complementary references:

- Bergmann et al., "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection," CVPR 2019.
- Li et al., "Pruning Filters for Efficient ConvNets," ICLR 2017.
- Molchanov et al., "Importance Estimation for Neural Network Pruning," CVPR 2019.
- Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," CVPR 2018.
- Hinton, Vinyals, Dean, "Distilling the Knowledge in a Neural Network," NeurIPS workshop 2014.
- David et al., "TensorFlow Lite Micro: Embedded Machine Learning on TinyML Systems," MLSys 2021.
- Lin et al., "MCUNet: Tiny Deep Learning on IoT Devices," NeurIPS 2020.

## Authors

- **Romain Noblet** — ML & compression track (baseline, quantization, pruning, distillation, adaptive policy design, simulation).
- **Ilka Bretschneider** — Embedded & measurement track (TFLite Micro integration, Arduino firmware, on-device profiling, hardware measurement, live demo).

Course: *Low-Power Embedded System* — *Università Degli Studi Di Trento*, supervised by *Professor Kasim Sinan Yildirim*.
*2026*

## License

Code released under the MIT License. The MVTec AD dataset is distributed by MVTec under its own license; please refer to the [dataset page](https://www.mvtec.com/company/research/datasets/mvtec-ad) for terms of use.
