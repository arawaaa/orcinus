import torch

import torch.nn as nn
import torch.nn.functional as F

import torchaudio
import matplotlib.pyplot as plt

audio_tensor, rate = torchaudio.load("data/Possible Resident/Abyssal/AB01_Enc3.wav")

audio_tensor = audio_tensor[0, :200000]
spectro = torchaudio.transforms.Spectrogram(n_fft=512)
result = spectro(audio_tensor)[65:, :]**1.7 # power to differentiate signal from noise

to_db = torchaudio.transforms.AmplitudeToDB('amplitude')
print(result.shape)
plt.imshow(to_db(result))
plt.rcParams["figure.figsize"] = (20,10)
plt.show()
