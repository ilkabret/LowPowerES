# 🔍 Energy-Aware Visual Anomaly Detection on MCUs

> ⚡ Compression-aware deployment and runtime adaptation of a CNN-based anomaly detector on the **Arduino Nano 33 BLE Sense Rev 2**.

This project:
- trains a compact convolutional autoencoder on the **MVTec AD** dataset,
- evaluates multiple compression strategies (quantization, pruning, distillation),
- deploys INT8 models on a Cortex-M4 microcontroller via **TensorFlow Lite for Microcontrollers**,
- and studies **energy-aware inference strategies** under limited or variable power budgets.

The final goal is a **hardware-grounded Pareto analysis** of accuracy, latency, memory, and energy, plus an adaptive inference policy that selects models based on energy availability.

---

## 🎯 Motivation

Deploying visual anomaly detection on microcontrollers introduces two core constraints:

- 🧠 **Compute & memory limits** (Cortex-M4, 256 KB SRAM, 1 MB Flash)
- 🔋 **Energy constraints** (battery or harvested power)

This project investigates how far a convolutional autoencoder can be compressed before anomaly detection performance (AUROC) degrades, and whether runtime adaptation can improve system-level utility.

---

## 📂 Project structure
```
.
├── literature/ # Papers and references
├── mvtec_stage1/ # Stage 1 data/experiments (baseline training)
├── mvtec_stage2/ # Compression experiments (pruning, distillation, quantization)
├── mvtec_stage3/ # TFLite conversion + deployment experiments
├── Ondevice_Inference/ # Arduino + test sets + logs (Stage 3/4)
│ ├── arduino_stage3_OneModelInference/
│ ├── arduino_stage3_StreamTestSet/
│ ├── bottle_testset_int8.npz
│ ├── hazelnut_testset_int8.npz
│ ├── streamTestSet.py
│ └── ondevice_results.csv
│
├── Stage1_Baseline.ipynb # Baseline autoencoder training (Colab)
├── Stage2_Compression.ipynb # Pruning / distillation / quantization
├── Stage3_Deployment.ipynb # TFLite + Arduino deployment + measurement
├── Stage4_EnergyAdaptive.ipynb # Energy-aware adaptive inference
├── README.md
└── TensorFlowLiteLibrary.url
```

---

## 🔄 Pipeline overview

```
MVTec AD dataset
↓
Stage 1 — Baseline autoencoder (Colab)
↓
Stage 2 — Compression study
├── L1 / Taylor pruning
├── Knowledge distillation
└── INT8 quantization
↓
Stage 3 — MCU deployment (Arduino Nano 33 BLE Sense)
├── TFLite Micro inference
├── latency / SRAM / Flash measurement
└── on-device AUROC evaluation
↓
Stage 4 — Energy-aware adaptive inference
├── battery / harvesting simulation
├── runtime model switching
└── system-level utility evaluation
```

---

## 🔌 Hardware

| Component | Details |
|-----------|--------|
| MCU | Arduino Nano 33 BLE Sense Rev 2 (nRF52840, Cortex-M4F @ 64 MHz) |
| Memory | 256 KB SRAM / 1 MB Flash |
| Runtime | TensorFlow Lite for Microcontrollers |
| Input | Preloaded test images or USB-serial streaming |
| Energy | Estimated from latency × datasheet power (6.3 mA @ 3.3 V) |

---

## 📓 Notebooks (Colab workflow)

All experiments are implemented as Jupyter notebooks:

- `Stage1_Baseline.ipynb` → train convolutional autoencoder on MVTec AD  
- `Stage2_Compression.ipynb` → pruning (L1 & Taylor), distillation, INT8 quantization + Pareto analysis  
- `Stage3_Deployment.ipynb` → TFLite conversion, Arduino deployment, on-device metrics  
- `Stage4_EnergyAdaptive.ipynb` → energy-aware inference simulation and evaluation  

👉 The recommended workflow is to execute notebooks sequentially in Colab.

---

## 📟 On-device inference artifacts (Stage 3/4)

Located in: `Ondevice_Inference/`


### Arduino inference modes

- `arduino_stage3_OneModelInference/`
  - Single-model inference demo
  - Uses:
    - `bottle_Prune_50pct_l1.h`
    - `test_img_good.h`
    - `test_img_defect.h`
  - Measures:
    - latency
    - SRAM usage
    - flash footprint

- `arduino_stage3_StreamTestSet/`
  - Full dataset streaming evaluation
  - Files:
    - `streamTestSet.py` → host-side evaluation script
    - `bottle_testset_int8.npz`
    - `hazelnut_testset_int8.npz`
    - `ondevice_results.csv`

---

## 📊 Results

Across Stage 2–3, models are evaluated using:

- AUROC (anomaly detection quality)
- MACs (compute cost)
- model size (Flash usage)
- latency (ms / inference)
- estimated energy (mJ / inference)

❌ These metrics are combined into Pareto plots to identify optimal compression–accuracy trade-offs.

---

## ⚡ Stage 4 — Adaptive inference

Stage 4 evaluates runtime policies that select between compressed models depending on energy availability.

Key finding:

> A strong static model (especially `prune50 L1`) often matches or outperforms adaptive strategies in overall utility, since it already provides a robust accuracy–energy trade-off under the defined constraints.

Adaptive inference is therefore not universally superior, but helps illustrate system-level trade-offs under variable energy conditions.

## 👥 Authors

**Romain Noblet** & **Ilka Bretschneider**

Course: *Low-Power Embedded Systems* — University of Trento  
2026  
