import torch.nn as nn


class Net(nn.Module):
  def __init__(self):
      super(Net, self).__init__()

      self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(num_features=16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2))
      self.layer2 = nn.Sequential(
        nn.Conv2d(16, 32, 3, 1, 1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, 2))
      self.layer3 = nn.Sequential(
        nn.Conv2d(32, 64, 3, 1, 1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2))
      self.layer4 = nn.Sequential(
        nn.Conv2d(64, 128, 3, 1, 1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2, 2))
      self.layer5 = nn.Sequential(
        nn.Conv2d(128, 256, 3, 1, 1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(2, 2))
      self.global_avg_pool = nn.Sequential(
        nn.Conv2d(256, 17, 3, 1, 1),
        nn.BatchNorm2d(17),
        nn.LeakyReLU(),
        nn.AdaptiveAvgPool2d((1, 1)))
  
  def forward(self, x):
    result = self.layer1(x)
    result = self.layer2(result)
    result = self.layer3(result)
    result = self.layer4(result)
    result = self.layer5(result)
    result = self.global_avg_pool(result)
    result = result.view(-1, 17)

    return result
