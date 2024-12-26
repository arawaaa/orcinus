import os

import torch
import torchaudio
import pandas as pd

orc_dir = "data/Possible Resident"
orc_trans_dir = "data/GoA Transient"

annotations = pd.DataFrame(columns=["Name", "Animal"])

def iterate_dir(path, animal):
    for fname in os.listdir(path):
        combined = path + "/" + fname
        if (os.path.isdir(combined)):
            iterate_dir(combined, animal)
        if (combined.endswith(".wav")):
            audio_tensor, rate = torchaudio.load(combined)

            audio_tensor = audio_tensor[0, :200000]
            spectro = torchaudio.transforms.Spectrogram(n_fft=512)
            result = spectro(audio_tensor)[65:, :]**1.3 # power to differentiate signal from noise

            fname = fname[:-4]
            torch.save(result, "data/cleaned/" + fname)

            annotations.loc[annotations.shape[1], :] = [fname, animal]

iterate_dir(orc_dir, 0)
iterate_dir(orc_trans_dir, 1)
annotations.to_csv("data/cleaned/annotations.csv")
