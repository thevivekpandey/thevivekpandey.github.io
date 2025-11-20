"""
ChessNet for Position Value Prediction
Predicts stockfish evaluation score instead of moves
This is pure value prediction - no policy head needed!
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


class ChessNetValue(nn.Module):
    """
    ResNet-based chess network for VALUE PREDICTION
    Predicts stockfish evaluation score for a position
    No policy head - just value estimation
    """

    def __init__(self, input_channels: int = 19, num_res_blocks: int = 8,
                 num_channels: int = 256, dropout: float = 0.1):
        super().__init__()

        # Initial convolutional block
        self.conv_input = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)

        # Residual tower with dropout
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels, dropout=dropout) for _ in range(num_res_blocks)
        ])

        # Value head - predicts centipawn evaluation
        # We'll use a deeper value head since this is our only task
        self.value_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_dropout = nn.Dropout(dropout)

        # Deeper FC layers for better value estimation
        self.value_fc1 = nn.Linear(32 * 8 * 8, 512)
        self.value_fc2 = nn.Linear(512, 256)
        self.value_fc3 = nn.Linear(256, 1)

    def forward(self, x):
        """
        Args:
            x: (batch, 19, 8, 8) board representation

        Returns:
            value: (batch, 1) position evaluation in centipawns
                   Positive = advantage for white, negative = advantage for black
        """
        # Shared representation
        x = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            x = block(x)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = self.value_dropout(value)
        value = F.relu(self.value_fc1(value))
        value = self.value_dropout(value)
        value = F.relu(self.value_fc2(value))
        value = self.value_fc3(value)  # No activation - raw centipawn output

        return value
