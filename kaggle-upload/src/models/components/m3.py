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
  
class M3(nn.Module):
  def __init__(self):
    super().__init__()
    self.seq_of_layers = nn.Sequential(
        ConvBlock(3, 1, 32),
        ConvBlock(3, 32, 48),
        ConvBlock(3, 48, 64),
        ConvBlock(3, 64, 80),
        ConvBlock(3, 80, 96),
        ConvBlock(3, 96, 112),
        ConvBlock(3, 112, 128),
        ConvBlock(3, 128, 144),
        ConvBlock(3, 144, 160),
        ConvBlock(3, 160, 176),
        nn.Flatten(),
        nn.Linear(in_features = 11264,
                  out_features = 10),
        nn.BatchNorm1d(num_features = 10)
    )

  def forward(self, x):
    return self.seq_of_layers(x)