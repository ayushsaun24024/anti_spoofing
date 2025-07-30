import torch.nn as nn
import torch, torchaudio
from transformers import WhisperModel, WhisperFeatureExtractor

class WAVLM(torch.nn.Module):
    def __init__(self, num_trainable_layers=2):
        super().__init__()
        bundle = torchaudio.pipelines.WAVLM_BASE_PLUS
        self.model = bundle.get_model()
        self.num_trainable_layers = num_trainable_layers
        
        self._freeze_all_parameters()
        
        self.encoder_layers = self.model.encoder.transformer.layers
        self.total_layers = len(self.encoder_layers)
        
        self.attention = nn.Sequential(
            nn.Linear(768, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256)
        )
        
    def _freeze_all_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def _unfreeze_wavlm_layers(self, num_layers):
        num_layers = min(num_layers, self.total_layers)
        
        start_idx = self.total_layers - 1
        for i in range(num_layers):
            layer_idx = start_idx - i
            for param in self.encoder_layers[layer_idx].parameters():
                param.requires_grad = True
            
    def _freeze_attention_classifier(self):
        for param in self.attention.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = False
            
    def _unfreeze_attention_classifier(self):
        for param in self.attention.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
    
    def train(self, mode=True):
        super().train(mode)
        
        self._freeze_all_parameters()
        
        self._unfreeze_attention_classifier()
        
        if self.num_trainable_layers > 0:
            self._unfreeze_wavlm_layers(self.num_trainable_layers)
        
    def eval(self):
        super().eval()
        self._freeze_all_parameters()
        self._freeze_attention_classifier()
        return self
        
    def forward(self, waveform):
        features, _ = self.model(waveform.squeeze(1))
        
        attn_scores = self.attention(features)
        attn_weights = torch.softmax(attn_scores, dim=1)

        x_pooled = torch.sum(features * attn_weights, dim=1)
        
        return self.classifier(x_pooled)

class WHISPER(nn.Module):
    def __init__(self, model_name, train_encoder=False, num_encoder_layers=0, device="cpu"):
        super().__init__()
        
        self.device = device
        self.model = WhisperModel.from_pretrained(model_name).to(self.device)
        self.feature_extractor = WhisperFeatureExtractor()
        
        self.encoder = self.model.encoder        
        self.train_encoder = train_encoder
        self.num_encoder_layers = num_encoder_layers
        self.encoder_layers = self.encoder.layers
        self.total_encoder_layers = len(self.encoder_layers)
        
        self.attention = nn.Sequential(
            nn.Linear(768, 256),
            nn.Tanh(),
            nn.LayerNorm(256),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )

        self._freeze_all_parameters()
        self.train()

    def _freeze_all_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

    def _train_encoder_layers(self, num_layers):
        num_layers = min(num_layers, self.total_encoder_layers)
        for i in range(1, num_layers + 1):
            for param in self.encoder_layers[-i].parameters():
                param.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self._freeze_all_parameters()
            
            for param in self.attention.parameters():
                param.requires_grad = True
            for param in self.classifier.parameters():
                param.requires_grad = True

            if self.train_encoder and self.num_encoder_layers > 0:
                self._train_encoder_layers(self.num_encoder_layers)
        return self

    def eval(self):
        super().eval()
        for param in self.attention.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = False
        self._freeze_all_parameters()
        return self

    def forward(self, waveform):
        inputs = self.feature_extractor(
            waveform.squeeze(1).cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features
        
        encoder_outputs = self.model.encoder(inputs.to(self.device))
        extracted_features = encoder_outputs.last_hidden_state
        
        
        attention_scores = self.attention(extracted_features)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = torch.sum(attention_weights * extracted_features, dim=1)

        return self.classifier(context_vector)

class WHISPER_TINY(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()

        self.device = device
        print(device)
        self.feature_extractor = WhisperFeatureExtractor()
        self.model = WhisperModel.from_pretrained("openai/whisper-small").to(device)
        for param in self.model.parameters():
            param.requires_grad = False

        self.feature_down = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )


        self.seq_reduce = nn.Sequential(
            nn.Linear(1500, 1125),
            nn.ReLU(),
            nn.Linear(1125, 840),
            nn.ReLU(),
            nn.Linear(840, 640),
            nn.ReLU(),
            nn.Linear(640, 470),
            nn.ReLU(),
            nn.Linear(470, 355),
            nn.ReLU(),
            nn.Linear(355, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.final = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.squeeze(1)

        inputs = self.feature_extractor(
            x.squeeze(1).cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features
        
        encoder_outputs = self.model.encoder(inputs.to(self.device)).last_hidden_state

        x = self.feature_down(encoder_outputs)
        x = x.transpose(1, 2)
        x = self.seq_reduce(x)
        x = x.squeeze(-1)
        out = self.final(x)

        return out

class HuBERTFeature(nn.Module):
    def __init__(self, DEVICE="cpu"):
        super(HuBERTFeature, self).__init__()
        bundle = torchaudio.pipelines.HUBERT_BASE
        self.hubert_model = bundle.get_model()

        checkpoint = torch.load("./hubert_models/iter3/hubert_iter3_final.pt", map_location=DEVICE)
        self.hubert_model.load_state_dict(checkpoint['model_state_dict'])

        for param in self.hubert_model.parameters():
            param.requires_grad = False

    def forward(self, audio):
        audio = audio.squeeze(1)
        with torch.no_grad():
            layer_results, _ = self.hubert_model.extract_features(audio)
            x = layer_results[-1] 

        return x