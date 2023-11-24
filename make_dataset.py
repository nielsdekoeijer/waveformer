import os
import random
import numpy as np
import torch
import torchaudio

ntrain = 1024# ~40 ms 
npred = 512 # ~20 ms 
nsamples = {
    'train': 100000,
    'test': 1, 
    'validation': 1,
}
ifolder = f"./fma_small_wav/"
ofolder = f"./dataset_{ntrain}_{npred}_{nsamples['train']}_{nsamples['test']}_{nsamples['validation']}/"

wav_files = [f for f in os.listdir(ifolder) if f.endswith('.wav')]

for key, value in nsamples.items():
    ofolder_path = os.path.join(ofolder, key)
    os.makedirs(ofolder_path, exist_ok=True)
    for i in range(value):
        try:
            selected_file = random.choice(wav_files)
            print(f"{key}::{i}::Loading file {selected_file}...")
            waveform, _ = torchaudio.load(os.path.join(ifolder, selected_file))
            waveform = (waveform * 127.5 + 127.5).clamp_(0, 255).to(torch.uint8)
            print(f"{key}::{i}::Loading file {selected_file} OK")
            start_idx = random.randint(0, waveform.shape[1] - (ntrain + npred))
            print(f"{key}::{i}::Selected index {start_idx}")
            item = {
                    "frame": waveform[0, start_idx : start_idx + ntrain + npred].numpy(),
            }

            npy_file_path = f"{i}.npy"
            npy_file_path = os.path.join(ofolder_path, npy_file_path)
            print(f"{key}::{i}::Saving {npy_file_path}...")
            np.save(npy_file_path, item, allow_pickle=True)
            print(f"{key}::{i}::Saving {npy_file_path} OK")
        except KeyboardInterrupt:
            exit(1)
        except:
            print(f"{key}::{i}::Exception")
            continue
