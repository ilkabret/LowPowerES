import serial, numpy as np, time, csv
from pathlib import Path
from sklearn.metrics import roc_auc_score

# ============ CONFIG — edit these per model you flash ============
PORT       = 'COM3'
CATEGORY   = 'bottle'                  # 'bottle' or 'hazelnut'
MODEL_NAME = 'bottle_Distill_b16'          # label for the CSV row
NPZ_FILE   = r"C:\Users\ilkab\Studium\LowPowerES\Final_Project\Ondevice_Inference\bottle_testset_int8.npz" # test set for this category
CSV_OUT    = 'ondevice_results.csv'

# ============ ENERGY MODEL (datasheet) ============
SUPPLY_V   = 3.3      # board rail voltage
CURRENT_MA = 6.3      # nRF52840 active current @ 64 MHz (datasheet)
POWER_MW   = SUPPLY_V * CURRENT_MA     # = 20.79 mW

def energy_mJ(latency_ms):
    # E = P * t ;  mW * s = mJ
    return POWER_MW * (latency_ms / 1000.0)

# ============ STREAM TEST SET ============
data = np.load(NPZ_FILE)
imgs, labels = data['imgs'], data['labels']

ser = serial.Serial(PORT, 115200, timeout=30, write_timeout=30)
time.sleep(2)
ser.reset_input_buffer(); ser.reset_output_buffer()

def send_image(img_bytes):
    for i in range(0, len(img_bytes), 64):
        ser.write(img_bytes[i:i+64]); ser.flush()
    return ser.readline().decode(errors='ignore').strip()

scores, lats = [], []
for i, img in enumerate(imgs):
    line = send_image(img.astype(np.int8).tobytes())
    try:
        score = float(line.split('SCORE:')[1].split(',')[0])
        lat   = int(line.split('LAT:')[1])
    except (IndexError, ValueError):
        print(f'  bad response at {i}: {repr(line)}'); continue
    scores.append(score); lats.append(lat)
    if i % 10 == 0:
        print(f'{i}/{len(imgs)}  score={score:.5f}  lat={lat}ms')

ser.close()

# ============ COMPUTE METRICS ============
auroc      = roc_auc_score(labels[:len(scores)], scores)
mean_lat   = float(np.mean(lats))
std_lat    = float(np.std(lats))
mean_energy = energy_mJ(mean_lat)

print('\n' + '='*50)
print(f'Model            : {MODEL_NAME}')
print(f'On-device AUROC  : {auroc:.4f}')
print(f'Latency          : {mean_lat:.0f} ± {std_lat:.0f} ms')
print(f'Power (assumed)  : {POWER_MW:.2f} mW  ({CURRENT_MA} mA @ {SUPPLY_V} V)')
print(f'Energy/inference : {mean_energy:.1f} mJ')
print('='*50)

# ============ APPEND TO CSV ============
csv_path = Path(CSV_OUT)
new_file = not csv_path.exists()
with open(csv_path, 'a', newline='') as f:
    w = csv.writer(f)
    if new_file:
        w.writerow(['model','category','auroc_ondevice','latency_ms','latency_std',
                    'power_mW','energy_mJ','current_mA','supply_V'])
    w.writerow([MODEL_NAME, CATEGORY, round(auroc,4), round(mean_lat,1), round(std_lat,1),
                round(POWER_MW,2), round(mean_energy,1), CURRENT_MA, SUPPLY_V])
print(f'Appended to {CSV_OUT}')