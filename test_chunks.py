import random
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal
import torch
from tsai.all import load_learner
from fastcore.foundation import L

# 1. Load Model & Apply Compatibility Patch
print("Loading footstep_model.pkl...")
learn = load_learner("footstep_model.pkl")

if hasattr(learn.dls, 'train_ds') and isinstance(learn.dls.train_ds.tls, list):
    learn.dls.train_ds.tls = L(learn.dls.train_ds.tls)
if hasattr(learn.dls, 'valid_ds') and isinstance(learn.dls.valid_ds.tls, list):
    learn.dls.valid_ds.tls = L(learn.dls.valid_ds.tls)

learn.model.eval()

WINDOW_SIZE = 120
CUTOFF_40 = 48

# 2. Pick a random test file
excel_files = list(Path("data").glob("*.xlsx")) or list(Path(".").glob("*.xlsx"))
chosen_file = random.choice(excel_files)
df = pd.read_excel(chosen_file).dropna(subset=['ADC_Value'])
adc = df['ADC_Value'].values.astype(np.float32)

actual_person = df['Person_id'].iloc[0]
actual_shoe = df['Footwear'].iloc[0]
secret_label = f"Person_{actual_person}_{actual_shoe}"

# 3. Peak Detection: Anchor the window around a real footstep impact
peaks, _ = scipy.signal.find_peaks(np.abs(adc - np.mean(adc)), height=15, distance=50)

# Filter peaks that have enough room for a full 120-point window
valid_peaks = [p for p in peaks if p >= 15 and (p + (WINDOW_SIZE - 15)) < len(adc)]

if not valid_peaks:
    # Fallback to standard window if no prominent peaks found
    start_idx = random.randint(0, len(adc) - WINDOW_SIZE)
else:
    chosen_peak = random.choice(valid_peaks)
    # Start 15 samples before peak so the initial heel strike is inside the first 40%
    start_idx = chosen_peak - 15

full_step = adc[start_idx : start_idx + WINDOW_SIZE].copy()

# 4. Sequential Stream Simulation (40% base + progressive 60% chunks)
initial_40 = full_step[:CUTOFF_40]
chunk_1 = full_step[CUTOFF_40:72]
chunk_2 = full_step[72:96]
chunk_3 = full_step[96:120]

# Choose evaluation mode: 0 = 40% only, 1 = 60%, 2 = 80%, 3 = full step
stage_choice = random.choice([0, 1, 2, 3])

if stage_choice == 0:
    active_wave = initial_40
    stage_name = "Stage 1: 40% Initial Impact (48 pts)"
elif stage_choice == 1:
    active_wave = full_step[:72]
    stage_name = "Stage 2: 40% + Chunk #1 (72 pts)"
elif stage_choice == 2:
    active_wave = full_step[:96]
    stage_name = "Stage 3: 40% + Chunks #1 & #2 (96 pts)"
else:
    active_wave = full_step
    stage_name = "Stage 4: Complete Footstep (120 pts)"

# 5. Normalize and Pad Remaining Points with Zeros
norm_wave = (active_wave - np.mean(active_wave)) / (np.std(active_wave) + 1e-8)
padded_input = np.pad(norm_wave, (0, WINDOW_SIZE - len(norm_wave)), 'constant')
X_test = padded_input.reshape(1, 1, WINDOW_SIZE).astype(np.float32)

# 6. Predict
with torch.no_grad():
    device = next(learn.model.parameters()).device
    logits = learn.model(torch.tensor(X_test).to(device))
    probabilities = torch.softmax(logits, dim=-1)[0]
    target_idx = torch.argmax(probabilities).item()
    confidence = float(probabilities[target_idx]) * 100
    prediction = learn.dls.vocab[target_idx]

# 7. Results
print("\n" + "=" * 48)
print(f"Evaluation:       {stage_name}")
print(f"Points Evaluated: {len(active_wave)} / {WINDOW_SIZE}")
print(f"Model Guess:      {prediction}")
print(f"Confidence:       {confidence:.2f}%")
print("-" * 48)
input("Press [Enter] to reveal the actual label...")
print(f"Actual Label:     {secret_label} ({chosen_file.name})")
print("=" * 48 + "\n")