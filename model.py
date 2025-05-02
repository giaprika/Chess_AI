import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import board_to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)

class AlphaZeroNet(nn.Module):
    def __init__(self, input_channels=20, n_res_blocks=19):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, 256, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(256)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(n_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(256, 128, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(128)
        self.policy_fc = nn.Conv2d(128, 73, kernel_size=1)

        # Value head
        self.value_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.res_blocks(x)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))  
        policy = self.policy_fc(policy)                       
        policy = policy.permute(0, 2, 3, 1).contiguous()       
        policy = policy.view(policy.size(0), -1)               

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))     
        value = value.view(value.size(0), -1)                  
        value = F.relu(self.value_fc1(value))                 
        value = torch.tanh(self.value_fc2(value))              

        return policy, value
    
    def predict(self, board):
        self.eval()
        with torch.no_grad():
            x = board_to_tensor(board).unsqueeze(0).to(device)  # shape: (1, 20, 8, 8)
            policy_logits, value = self.forward(x)
            policy = F.softmax(policy_logits, dim=1)
            policy = policy.squeeze(0).cpu().numpy()
            return policy, value.item()
