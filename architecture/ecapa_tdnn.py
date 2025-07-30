import torch.nn as nn
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN as ECAPA_MODEL
from speechbrain.lobes.features import Fbank

class ECAPA_TDNN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 192):
        super(ECAPA_TDNN, self).__init__()
        
        self.model = ECAPA_MODEL(input_size = input_dim, lin_neurons=output_dim)
        self.fbank = Fbank(n_mels=input_dim)
    
    def forward(self, x, _):
        mel_spectrogram = self.fbank(x.squeeze(1))
        return self.model(mel_spectrogram)