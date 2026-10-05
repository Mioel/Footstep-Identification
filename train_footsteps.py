from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal
import torch
from tsai.all import (
    Categorize,
    InceptionTimePlus,
    Learner,
    TSDataLoaders,
    TSDatasets,
    accuracy,
    create_model,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

WINDOW_SIZE = 120
CUTOFF_40 = 48

data_folder = Path("data") if Path("data").exists() else Path(".")
excel_files = list(data_folder.glob("*.xlsx"))

if not excel_files:
    raise FileNotFoundError("No Excel files found.")

all_windows = []
all_labels = []
train_idx = []
valid_idx = []
current_idx = 0

print(f"Found {len(excel_files)} Excel files. Extracting peak-aligned footsteps...")

for file in excel_files:
    df = pd.read_excel(file).dropna(subset=['ADC_Value'])
    adc = df['ADC_Value'].values.astype(np.float32)
    
    person = df['Person_id'].iloc[0]
    footwear = df['Footwear'].iloc[0]
    label = f"Person_{person}_{footwear}"
    
    # Locate heel-strike peaks
    peaks, _ = scipy.signal.find_peaks(np.abs(adc - np.mean(adc)), height=12, distance=50)
    file_step_count = 0
    
    for p in peaks:
        if p < 15 or (p + (WINDOW_SIZE - 15)) >= len(adc):
            continue
            
        full_step = adc[p - 15 : p - 15 + WINDOW_SIZE]
        
        # Variable-Length Masking (Augmentation):
        # Create versions of each step at 40%, 60%, 80%, and 100% completion
        for cutoff in [48, 72, 96, 120]:
            active_part = full_step[:cutoff]
            norm_active = (active_part - np.mean(active_part)) / (np.std(active_part) + 1e-8)
            
            # Zero-pad remaining window
            window = np.pad(norm_active, (0, WINDOW_SIZE - cutoff), 'constant')
            
            all_windows.append(window)
            all_labels.append(label)
            file_step_count += 1
            
    # Chronological Split (first 80% steps to train, last 20% to validation)
    split_point = int(file_step_count * 0.80)
    train_idx.extend(range(current_idx, current_idx + split_point))
    valid_idx.extend(range(current_idx + split_point, current_idx + file_step_count))
    current_idx += file_step_count

print(f"Total training/validation segments: {len(all_windows)}")
print(f"Train size: {len(train_idx)} | Validation size: {len(valid_idx)}")

X = np.array(all_windows).reshape(len(all_windows), 1, WINDOW_SIZE).astype(np.float32)
y = np.array(all_labels)

splits = (train_idx, valid_idx)
dsets = TSDatasets(X, y, tfms=[None, [Categorize()]], splits=splits)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=64, num_workers=0)

model = create_model(InceptionTimePlus, dls=dls)
learn = Learner(dls, model, metrics=accuracy)

print("\nTraining InceptionTime with variable-length footstep patterns...")
learn.fit_one_cycle(20, lr_max=1e-3)

learn.export("footstep_model.pkl")
print("\nRetraining complete! Saved updated 'footstep_model.pkl'.")