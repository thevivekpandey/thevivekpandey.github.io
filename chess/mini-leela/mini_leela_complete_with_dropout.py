"""
ChessNet with Dropout for regularization
Only showing the modified network - rest stays the same
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Basic residual block with optional dropout"""

    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class ChessNetWithDropout(nn.Module):
    """
    ResNet-based neural network with dropout regularization
    """

    def __init__(self, input_channels: int = 19, num_res_blocks: int = 4,
                 num_channels: int = 128, dropout: float = 0.3):
        super().__init__()

        # Initial convolutional block
        self.conv_input = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)

        # Residual tower with dropout
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels, dropout=dropout) for _ in range(num_res_blocks)
        ])

        # Policy head with dropout (reduced channels to reduce parameters)
        self.policy_conv = nn.Conv2d(num_channels, 16, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(16)
        self.policy_dropout = nn.Dropout(dropout)
        self.policy_fc = nn.Linear(16 * 8 * 8, 4096)

        # Value head with dropout (reduced channels to reduce parameters)
        self.value_conv = nn.Conv2d(num_channels, 16, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_dropout = nn.Dropout(dropout)
        self.value_fc1 = nn.Linear(16 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Shared representation
        x = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            x = block(x)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_dropout(policy)
        policy_logits = self.policy_fc(policy)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = self.value_dropout(value)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy_logits, value
