import torch.nn as nn


class ResidualBlock(nn.Module):
  expansion = 1

  def __init__(self, input_channels, output_channels, stride=1):
    super().__init__()

    self.residual_function = nn.Sequential(
      nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False),
      nn.BatchNorm2d(output_channels),
      nn.ReLU(),
      nn.Conv2d(output_channels, output_channels * ResidualBlock.expansion, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(output_channels * ResidualBlock.expansion),)
    
    self.shortcut = nn.Sequential()
    self.relu = nn.ReLU()

    # Maybe shortcut size should fit to the result of residual function
    if stride != 1 or input_channels != ResidualBlock.expansion * output_channels:
      self.shortcut = nn.Sequential(
        nn.Conv2d(input_channels, output_channels * ResidualBlock.expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(output_channels * ResidualBlock.expansion))

  def forward(self, x):
    x = self.residual_function(x) + self.shortcut(x)
    x = self.relu(x)
    return x

class BottleneckBlock(nn.Module):
  expansion = 4

  def __init__(self, input_channels, output_channels, stride=1):
    super().__init__()

    self.residual_function = nn.Sequential(
      nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, bias=False),
      nn.BatchNorm2d(output_channels),
      nn.ReLU(),
      nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False),
      nn.BatchNorm2d(output_channels),
      nn.ReLU(),
      nn.Conv2d(output_channels, output_channels * BottleneckBlock.expansion, kernel_size=1, stride=1, bias=False),
      nn.BatchNorm2d(output_channels * BottleneckBlock.expansion),)
    
    self.shortcut = nn.Sequential()
    self.relu = nn.ReLU()

    if stride != 1 or input_channels != output_channels * BottleneckBlock.expansion:
      self.shortcut = nn.Sequential(
        nn.Conv2d(input_channels, output_channels * BottleneckBlock.expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(output_channels * BottleneckBlock.expansion))
    
  def forward(self, x):
    x = self.residual_function(x) + self.shortcut(x)
    x = self.relu(x)
    return x

class ResNet(nn.Module):
  def __init__(self, block, num_block, num_classes=10, init_weights=True):
      super().__init__()

      self.input_channels = 64
      
      self.conv1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

      self.conv2_x = self.make_layer(block, output_channels=64, num_blocks=num_block[0], stride=1)
      self.conv3_x = self.make_layer(block, output_channels=128, num_blocks=num_block[1], stride=2)
      self.conv4_x = self.make_layer(block, output_channels=256, num_blocks=num_block[2], stride=2)
      self.conv5_x = self.make_layer(block, output_channels=512, num_blocks=num_block[3], stride=2)

      self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
      self.fc = nn.Linear(512 * block.expansion, num_classes)

      # Weights initialization
      if init_weights:
        self.initialize_weights()

  def forward(self, x):
    output = self.conv1(x)
    output = self.conv2_x(output)
    x = self.conv3_x(output)
    x = self.conv4_x(x)
    x = self.conv5_x(x)
    x = self.avg_pool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x

  def make_layer(self, block, output_channels, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []

    for stride in strides:
      layers.append(block(self.input_channels, output_channels, stride))
      self.input_channels = output_channels * block.expansion

    return nn.Sequential(*layers)
  
  def initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)