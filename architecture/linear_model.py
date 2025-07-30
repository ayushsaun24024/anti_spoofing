import torch

class LinearLayer(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(LinearLayer, self).__init__()

        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.relu1 = torch.nn.ReLU()
        
        self.fc2 = torch.nn.Linear(128, 64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.relu2 = torch.nn.ReLU()
        
        self.fc3 = torch.nn.Linear(64, output_dim)
    
    def forward(self, x):
        
        x = x.squeeze(1)
        
        x = self.fc1(x)  
        x = self.bn1(x)  
        x = self.relu1(x)
        
        x = self.fc2(x)  
        x = self.bn2(x)  
        x = self.relu2(x)
        
        x = self.fc3(x)
        return x