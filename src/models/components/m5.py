import torch.nn as nn

class ConvBlock(nn.Module):
  def __init__(self, kernel_size, in_channels, out_channels):
    super().__init__()
    self.seq_of_layers = nn.Sequential(
        nn.Conv2d(kernel_size = kernel_size,
                  in_channels = in_channels,
                  out_channels = out_channels),
        nn.BatchNorm2d(num_features = out_channels),
        nn.ReLU()
    )

  def forward(self, x):
    return self.seq_of_layers(x)

class M5(nn.Module):
  def __init__(self):
    super().__init__()
    self.seq_of_layers = nn.Sequential(
        ConvBlock(5, 1, 32),
        ConvBlock(5, 32, 64),
        ConvBlock(5, 64, 96),
        ConvBlock(5, 96, 128),
        ConvBlock(5, 128, 160),
        nn.Flatten(),
        nn.Linear(in_features = 10240,
                  out_features = 10),
        nn.BatchNorm1d(num_features = 10)
    )

  def forward(self, x):
    return self.seq_of_layers(x)