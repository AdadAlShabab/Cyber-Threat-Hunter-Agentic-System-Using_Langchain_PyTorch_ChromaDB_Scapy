import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class ThreatDetector(nn.Module):
    def __init__(self, input_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, bidirectional=True)
        self.attention = nn.MultiheadAttention(128, 4)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        return self.classifier(attn_out[-1])

class AnomalyEvaluator:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        self.scaler = RobustScaler()
        
    def analyze_traffic(self, packet_stream):
        processed = self.preprocess(packet_stream)
        with torch.no_grad():
            outputs = self.model(processed)
        return torch.nn.functional.softmax(outputs, dim=1)
