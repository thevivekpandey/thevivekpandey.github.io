"""
ChessNet with Source/Destination Move Encoding
Predicts best move by separating source and destination squares
Much more efficient than 4096-output encoding!
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


class ChessNetSourceDest(nn.Module):
    """
    ResNet-based chess network with SOURCE/DEST move prediction
    Predicts which square to move FROM and which square to move TO
    """

    def __init__(self, input_channels: int = 19, num_res_blocks: int = 8,
                 num_channels: int = 256, dropout: float = 0.0):
        super().__init__()

        # Initial convolutional block
        self.conv_input = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)

        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels, dropout=dropout) for _ in range(num_res_blocks)
        ])

        # Policy head - SOURCE/DEST encoding
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Separate predictions for source and destination squares
        policy_input_size = 32 * 8 * 8  # 2048
        self.policy_source = nn.Linear(policy_input_size, 64)  # Which square to move FROM
        self.policy_dest = nn.Linear(policy_input_size, 64)    # Which square to move TO

        # Value head (predicts position evaluation)
        self.value_conv = nn.Conv2d(num_channels, 16, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.value_fc1 = nn.Linear(16 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        """
        Args:
            x: (batch, 19, 8, 8) board representation

        Returns:
            source_logits: (batch, 64) - logits for source square
            dest_logits: (batch, 64) - logits for destination square
            value: (batch, 1) - position evaluation (-1 to +1)
        """
        # Shared representation
        x = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            x = block(x)

        # Policy head - source/dest
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        if self.policy_dropout is not None:
            policy = self.policy_dropout(policy)

        source_logits = self.policy_source(policy)
        dest_logits = self.policy_dest(policy)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        if self.value_dropout is not None:
            value = self.value_dropout(value)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # Output in [-1, 1]

        return source_logits, dest_logits, value
