import os
import copy
import glob
import torch
import random
import torchaudio
import numpy as np
import soundfile as sf
from scipy import signal
import torchaudio.functional as F

class RawBoost_Noise_Augmentation:
    def __init__(self,
                 musan_noises,
                 rir_noises,
                 nBands=5,
                 minF=20,
                 maxF=8000,
                 minBW=100,
                 maxBW=1000,
                 minCoeff=10,
                 maxCoeff=100,
                 minG=0,
                 maxG=0,
                 minBiasLinNonLin=5,
                 maxBiasLinNonLin=20,
                 N_f=5,
                 P=10,
                 g_sd=2,
                 SNRmin=10,
                 SNRmax=40,
                ):
        self.musan_noises = musan_noises
        self.rir_noises = rir_noises
        self.nBands = nBands
        self.minF = minF
        self.maxF = maxF
        self.minBW = minBW
        self.maxBW = maxBW
        self.minCoeff = minCoeff
        self.maxCoeff = maxCoeff
        self.minG = minG
        self.maxG = maxG
        self.minBiasLinNonLin = minBiasLinNonLin
        self.maxBiasLinNonLin = maxBiasLinNonLin
        self.N_f = N_f
        self.P = P
        self.g_sd = g_sd
        self.SNRmin = SNRmin
        self.SNRmax = SNRmax   
    
    def randRange(self, x1, x2, integer):
        y = np.random.uniform(low=x1, high=x2, size=(1,))
        if integer:
            y = int(y)
        return y
    
    def genNotchCoeffs(self,minG,maxG,fs):
        b = 1
        for i in range(0, self.nBands):
            fc = self.randRange(self.minF,self.maxF,0);
            bw = self.randRange(self.minBW,self.maxBW,0);
            c = self.randRange(self.minCoeff,self.maxCoeff,1);
            
            if c/2 == int(c/2):
                c = c + 1
            f1 = fc - bw/2
            f2 = fc + bw/2
            if f1 <= 0:
                f1 = float(1e-3)
            if f2 >= fs/2:
                f2 =  float((fs/2) - (1e-3))
            b = np.convolve(signal.firwin(c, [float(f1), float(f2)], window='hamming', fs=fs),b)

        G = self.randRange(minG,maxG,0); 
        _, h = signal.freqz(b, 1, fs=fs)    
        b = pow(10, G/20)*b/np.amax(abs(h))   
        return b

    def filterFIR(self, x, b):
        N = b.shape[0] + 1
        xpad = np.pad(x, (0, N), 'constant')
        y = signal.lfilter(b, 1, xpad)
        y = y[int(N/2):int(y.shape[0]-N/2)]
        return y
    
    def normWav(self, x, always):
        if always:
            x = x/np.amax(abs(x))
        elif np.amax(abs(x)) > 1:
                x = x/np.amax(abs(x))
        return x
    
    def convolutive_noise(self, x, fs):
        y = [0] * x.shape[0]
        for i in range(0, self.N_f):
            if i == 1:
                minG = self.minG-self.minBiasLinNonLin
                maxG = self.maxG-self.maxBiasLinNonLin
                b = self.genNotchCoeffs(minG,maxG,fs)
            else:
                b = self.genNotchCoeffs(self.minG,self.maxG,fs)
            y = y + self.filterFIR(np.power(x, (i+1)),  b)     
        y = y - np.mean(y)
        y = self.normWav(y,0)
        return y
    
    def impulsive_noise(self, x):
        beta = self.randRange(0, self.P, 0)
        y = copy.deepcopy(x)
        x_len = x.shape[0]
        n = int(x_len*(beta/100))
        p = np.random.permutation(x_len)[:n]
        f_r= np.multiply(((2*np.random.rand(p.shape[0]))-1),((2*np.random.rand(p.shape[0]))-1))
        r = self.g_sd * x[p] * f_r
        y[p] = x[p] + r
        y = self.normWav(y,0)
        return y
    
    def SSI_additive_noise(self,x,fs):
        noise = np.random.normal(0, 1, x.shape[0])
        b = self.genNotchCoeffs(self.minG,self.maxG,fs)
        noise = self.filterFIR(noise, b)
        noise = self.normWav(noise,1)
        SNR = self.randRange(self.SNRmin, self.SNRmax, 0)
        noise = noise / np.linalg.norm(noise,2) * np.linalg.norm(x,2) / 10.0**(0.05 * SNR)
        x = x + noise
        return x
    
    def add_musan_noise_to_waveform(
        self,
        x: np.ndarray,
        category: str = 'noise'
    ) -> np.ndarray:
        
        SNR_RANGES = {
            'noise': (0, 15),
            'speech': (13, 20), 
            'music': (5, 25)
        }
        
        NUM_FILES = {
            'noise': (1, 1),
            'speech': (3, 8),
            'music': (1, 1)
        }
        
        if not self.musan_noises.get(category) or len(self.musan_noises[category]) == 0:
            raise ValueError(f"No {category} files found!")
        
        x_size = x.shape[-1]
        x_dB = self.calculate_decibel(x)
        
        snr_min, snr_max = SNR_RANGES[category]
        file_min, file_max = NUM_FILES[category]
        files = random.sample(
            self.musan_noises[category],
            random.randint(file_min, file_max)
        )
        
        noises = []
        for f in files:
            info = sf.info(f)
            wav_size = int(info.samplerate * info.duration)
            
            if wav_size <= x_size:
                noise, _ = sf.read(f, start=0)
                noise_size = noise.shape[0]
                if noise_size < x_size:
                    shortage = x_size - noise_size
                    noise = np.pad(noise, (0, shortage), 'wrap')
            else:
                index = random.randint(0, wav_size - x_size - 1)
                noise, _ = sf.read(f, start=index, stop=index + x_size)
            
            noises.append(noise)
        
        if len(noises) != 0:
            noise = np.mean(noises, axis=0)
            snr = random.uniform(snr_min, snr_max)
            noise_dB = self.calculate_decibel(noise)
            p = (x_dB - noise_dB - snr)
            x = x + np.sqrt(10 ** (p / 10)) * noise
        
        return x

    def calculate_decibel(self, wav: np.ndarray) -> float:
        return 10 * np.log10(np.mean(wav ** 2) + 1e-4)
    
    def apply_rir(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        path = random.sample(self.rir_noises, 1)[0]
        
        rir, _ = sf.read(path)
        rir = rir.astype(np.float)
        rir = np.expand_dims(rir, 0)
        rir = rir / np.sqrt(np.sum(rir**2))
        
        x = np.expand_dims(x, 0)
        x = signal.convolve(x, rir, mode='full')[:,:len(x[0])]

        x = np.squeeze(x, 0)

        return x
    
    
    def apply(self, raw_waveform: np.ndarray, sr, method="convolutive"):
        methods = ["convolutive", "impulsive", "color_additive", "conv_impulse", "conv_coloradd", "impulse_coloradd", "conv_coloradd_impulse", "conv_impulse_p"]

        def execute_method(selected_method):
            if selected_method=="convolutive":
                return self.convolutive_noise(raw_waveform, sr)
            elif selected_method=="impulsive":
                return self.impulsive_noise(raw_waveform)
            elif selected_method=="color_additive":
                return self.SSI_additive_noise(raw_waveform, sr)
            elif selected_method=="conv_impulse":
                conv_feature = self.convolutive_noise(raw_waveform, sr)                    
                return self.impulsive_noise(conv_feature)
            elif selected_method=="conv_coloradd":
                conv_feature = self.convolutive_noise(raw_waveform, sr)
                return self.SSI_additive_noise(conv_feature, sr)
            elif selected_method=="impulse_coloradd":
                imp_feature = self.impulsive_noise(raw_waveform)
                return self.SSI_additive_noise(imp_feature, sr) 
            elif selected_method=="conv_coloradd_impulse":
                conv_feature = self.convolutive_noise(raw_waveform, sr)
                conv_imp_feature = self.impulsive_noise(conv_feature)
                return self.SSI_additive_noise(conv_imp_feature, sr)
            elif selected_method=="conv_impulse_p":
                conv_feature = self.convolutive_noise(raw_waveform, sr)
                imp_feature = self.impulsive_noise(raw_waveform)
                parallel_feature=conv_feature+imp_feature
                return self.normWav(parallel_feature,0)
            elif selected_method=="musan":
                return self.add_musan_noise_to_waveform(raw_waveform)
            elif selected_method=="rir":
                return self.apply_rir(raw_waveform)
            return raw_waveform

        if method == "random":
            remaining_methods = methods.copy()
            while remaining_methods:
                selected_method = random.choice(remaining_methods)
                remaining_methods.remove(selected_method)
                try:
                    return execute_method(selected_method)
                except:
                    continue
            return raw_waveform
        else:
            return execute_method(method)

# class RawBoost_Noise_Augmentation:
#     def __init__(self,
#                  nBands=5,
#                  minF=20,
#                  maxF=8000,
#                  minBW=100,
#                  maxBW=1000,
#                  minCoeff=10,
#                  maxCoeff=100,
#                  minG=0,
#                  maxG=0,
#                  minBiasLinNonLin=5,
#                  maxBiasLinNonLin=20,
#                  N_f=5,
#                  P=10,
#                  g_sd=2,
#                  SNRmin=10,
#                  SNRmax=40,
#                 ):
#         self.nBands = nBands
#         self.minF = minF
#         self.maxF = maxF
#         self.minBW = minBW
#         self.maxBW = maxBW
#         self.minCoeff = minCoeff
#         self.maxCoeff = maxCoeff
#         self.minG = minG
#         self.maxG = maxG
#         self.minBiasLinNonLin = minBiasLinNonLin
#         self.maxBiasLinNonLin = maxBiasLinNonLin
#         self.N_f = N_f
#         self.P = P
#         self.g_sd = g_sd
#         self.SNRmin = SNRmin
#         self.SNRmax = SNRmax   
    
#     def randRange(self, x1, x2, integer):
#         y = np.random.uniform(low=x1, high=x2, size=(1,))
#         if integer:
#             y = int(y)
#         return y
    
#     def genNotchCoeffs(self,minG,maxG,fs):
#         b = 1
#         for i in range(0, self.nBands):
#             fc = self.randRange(self.minF,self.maxF,0);
#             bw = self.randRange(self.minBW,self.maxBW,0);
#             c = self.randRange(self.minCoeff,self.maxCoeff,1);
            
#             if c/2 == int(c/2):
#                 c = c + 1
#             f1 = fc - bw/2
#             f2 = fc + bw/2
#             if f1 <= 0:
#                 f1 = float(1e-3)
#             if f2 >= fs/2:
#                 f2 =  float((fs/2) - (1e-3))
#             b = np.convolve(signal.firwin(c, [float(f1), float(f2)], window='hamming', fs=fs),b)

#         G = self.randRange(minG,maxG,0); 
#         _, h = signal.freqz(b, 1, fs=fs)    
#         b = pow(10, G/20)*b/np.amax(abs(h))   
#         return b

#     def filterFIR(self, x, b):
#         N = b.shape[0] + 1
#         xpad = np.pad(x, (0, N), 'constant')
#         y = signal.lfilter(b, 1, xpad)
#         y = y[int(N/2):int(y.shape[0]-N/2)]
#         return y
    
#     def normWav(self, x, always):
#         if always:
#             x = x/np.amax(abs(x))
#         elif np.amax(abs(x)) > 1:
#                 x = x/np.amax(abs(x))
#         return x
    
#     def convolutive_noise(self, x, fs):
#         y = [0] * x.shape[0]
#         for i in range(0, self.N_f):
#             if i == 1:
#                 minG = self.minG-self.minBiasLinNonLin
#                 maxG = self.maxG-self.maxBiasLinNonLin
#                 b = self.genNotchCoeffs(minG,maxG,fs)
#             else:
#                 b = self.genNotchCoeffs(self.minG,self.maxG,fs)
#             y = y + self.filterFIR(np.power(x, (i+1)),  b)     
#         y = y - np.mean(y)
#         y = self.normWav(y,0)
#         return y
    
#     def impulsive_noise(self, x):
#         beta = self.randRange(0, self.P, 0)
#         y = copy.deepcopy(x)
#         x_len = x.shape[0]
#         n = int(x_len*(beta/100))
#         p = np.random.permutation(x_len)[:n]
#         f_r= np.multiply(((2*np.random.rand(p.shape[0]))-1),((2*np.random.rand(p.shape[0]))-1))
#         r = self.g_sd * x[p] * f_r
#         y[p] = x[p] + r
#         y = self.normWav(y,0)
#         return y
    
#     def SSI_additive_noise(self,x,fs):
#         noise = np.random.normal(0, 1, x.shape[0])
#         b = self.genNotchCoeffs(self.minG,self.maxG,fs)
#         noise = self.filterFIR(noise, b)
#         noise = self.normWav(noise,1)
#         SNR = self.randRange(self.SNRmin, self.SNRmax, 0)
#         noise = noise / np.linalg.norm(noise,2) * np.linalg.norm(x,2) / 10.0**(0.05 * SNR)
#         x = x + noise
#         return x
    
#     def add_musan_noise_to_waveform(
#         self,
#         waveform: np.ndarray,
#         sample_rate=16000,
#         musan_noise_path="./.././SERVER_DATASETS/DATASET_MUSAN/musan/noise/",
#         snr_db=None
#     ) -> np.ndarray:
        
#         noise_patterns = [
#             os.path.join(musan_noise_path, 'free-sound', '**', '*.wav'),
#             os.path.join(musan_noise_path, 'sound-bible', '**', '*.wav')
#         ]
        
#         all_noise_files = []
#         for pattern in noise_patterns:
#             all_noise_files.extend(glob.glob(pattern, recursive=True))
        
#         if not all_noise_files:
#             raise ValueError(f"No noise files found in {musan_noise_path}")
        
#         noise_file = random.choice(all_noise_files)
#         noise_waveform, noise_sr = torchaudio.load(noise_file)  # [channels, time]
        
#         waveform_tensor = torch.from_numpy(waveform).float()
#         if waveform_tensor.dim() > 1 and waveform_tensor.shape[0] > 1:
#             waveform_tensor = torch.mean(waveform_tensor, dim=0, keepdim=True)
#         else:
#             waveform_tensor = waveform_tensor.unsqueeze(0) if waveform_tensor.dim() == 1 else waveform_tensor

#         if noise_waveform.shape[0] > 1:
#             noise_waveform = torch.mean(noise_waveform, dim=0, keepdim=True)

#         if noise_sr != sample_rate:
#             resampler = torchaudio.transforms.Resample(orig_freq=noise_sr, new_freq=sample_rate)
#             noise_waveform = resampler(noise_waveform)

#         audio_length = waveform_tensor.shape[-1]
#         noise_length = noise_waveform.shape[-1]
        
#         if noise_length < audio_length:
#             repeat_factor = (audio_length // noise_length) + 1
#             noise_waveform = noise_waveform.repeat(1, repeat_factor)
        
#         noise_waveform = noise_waveform[..., :audio_length]
        
#         if snr_db is None:
#             snr_db = random.uniform(0, 15)
        
#         signal_power = torch.mean(waveform_tensor ** 2)
#         noise_power = torch.mean(noise_waveform ** 2)
#         snr_linear = 10 ** (snr_db / 10)
#         noise_scale = torch.sqrt(signal_power / (noise_power * snr_linear))
#         scaled_noise = noise_scale * noise_waveform

#         noisy_waveform = waveform_tensor + scaled_noise

#         return noisy_waveform.squeeze(0).cpu().numpy()
    
#     def apply_rir(
#         self,
#         raw_waveform: np.ndarray,
#         rir_base_path: str = "./.././SERVER_DATASETS/DATASET_RIRS_NOISES/RIRS_NOISES/",
#         sample_rate: int = 16000
#     ) -> np.ndarray:
#         rir_folders = ['pointsource_noises', 'real_rirs_isotropic_noises', 'simulated_rirs']
#         selected_folder = random.choice(rir_folders)
#         folder_path = os.path.join(rir_base_path, selected_folder)
        
#         rir_files = []
#         for root, _, files in os.walk(folder_path):
#             for file in files:
#                 if file.lower().endswith('.wav'):
#                     rir_files.append(os.path.join(root, file))
        
#         if not rir_files:
#             raise ValueError(f"No RIR files found in {folder_path}")
        
#         rir_file = random.choice(rir_files)
        
#         rir_waveform, rir_sr = torchaudio.load(rir_file)
        
#         if rir_waveform.shape[0] > 1:
#             rir_waveform = torch.mean(rir_waveform, dim=0, keepdim=True)
        
#         if rir_sr != sample_rate:
#             resampler = torchaudio.transforms.Resample(orig_freq=rir_sr, new_freq=sample_rate)
#             rir_waveform = resampler(rir_waveform)
        
#         rir_waveform = rir_waveform / torch.linalg.vector_norm(rir_waveform, ord=2)
        
#         waveform_tensor = torch.from_numpy(raw_waveform).float()
#         if waveform_tensor.dim() == 1:
#             waveform_tensor = waveform_tensor.unsqueeze(0)  # [1, time]
        
#         convolved = F.fftconvolve(waveform_tensor, rir_waveform)
        
#         original_length = waveform_tensor.shape[-1]
#         if convolved.shape[-1] > original_length:
#             convolved = convolved[..., :original_length]
        
#         return convolved.squeeze(0).cpu().numpy()
    
    
#     def apply(self, raw_waveform: np.ndarray, sr, method="convolutive"):
#         methods = ["convolutive", "impulsive", "color_additive", "conv_impulse", "conv_coloradd", "impulse_coloradd", "conv_coloradd_impulse", "conv_impulse_p", "musan", "rir"]

#         def execute_method(selected_method):
#             if selected_method=="convolutive":
#                 return self.convolutive_noise(raw_waveform, sr)
#             elif selected_method=="impulsive":
#                 return self.impulsive_noise(raw_waveform)
#             elif selected_method=="color_additive":
#                 return self.SSI_additive_noise(raw_waveform, sr)
#             elif selected_method=="conv_impulse":
#                 conv_feature = self.convolutive_noise(raw_waveform, sr)                    
#                 return self.impulsive_noise(conv_feature)
#             elif selected_method=="conv_coloradd":
#                 conv_feature = self.convolutive_noise(raw_waveform, sr)
#                 return self.SSI_additive_noise(conv_feature, sr)
#             elif selected_method=="impulse_coloradd":
#                 imp_feature = self.impulsive_noise(raw_waveform)
#                 return self.SSI_additive_noise(imp_feature, sr) 
#             elif selected_method=="conv_coloradd_impulse":
#                 conv_feature = self.convolutive_noise(raw_waveform, sr)
#                 conv_imp_feature = self.impulsive_noise(conv_feature)
#                 return self.SSI_additive_noise(conv_imp_feature, sr)
#             elif selected_method=="conv_impulse_p":
#                 conv_feature = self.convolutive_noise(raw_waveform, sr)
#                 imp_feature = self.impulsive_noise(raw_waveform)
#                 parallel_feature=conv_feature+imp_feature
#                 return self.normWav(parallel_feature,0)
#             elif selected_method=="musan":
#                 return self.add_musan_noise_to_waveform(raw_waveform)
#             elif selected_method=="rir":
#                 return self.apply_rir(raw_waveform)
#             return raw_waveform

#         if method == "random":
#             remaining_methods = methods.copy()
#             while remaining_methods:
#                 selected_method = random.choice(remaining_methods)
#                 remaining_methods.remove(selected_method)
#                 try:
#                     return execute_method(selected_method)
#                 except:
#                     continue
#             return raw_waveform
#         else:
#             return execute_method(method)
