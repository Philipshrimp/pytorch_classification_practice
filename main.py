import glob
import numpy as np
import os
from PIL import Image
from pandas.core.common import flatten
import random
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model import Net


class LoadDataset(Dataset):
  def __init__(self, image_paths, transform=False):
    self.image_paths = image_paths
    self.transform = transform
  
  def __len__(self):
    return len(self.image_paths)
  
  def __getitem__(self, index):
    image_file_path = self.image_paths[index]
    image = Image.open(image_file_path)
    image = image.convert("RGB")

    label = image_file_path.split('/')[-2]
    label = class_to_idx[label]

    if self.transform is not None:
      image = self.transform(image)
    
    return image, label

if __name__ == "__main__":
  with open('config.yaml') as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)

  _epoch = config['epoch']
  _batch_size = config['batch_size']
  _learning_rate = config['learning_rate']

  dataset_path = config['file_path']
  if not os.path.exists(dataset_path):
    print("Error: invalid dataset path")
    exit()

  train_image_paths = []
  classes = []
  
  for data_path in glob.glob(dataset_path + 'train/*'):
    classes.append(data_path.split('/')[-1]) 
    train_image_paths.append(glob.glob(data_path + '/*'))

  train_image_paths = list(flatten(train_image_paths))
  random.shuffle(train_image_paths)

  train_image_ratio = 0.8
  train_image_paths = train_image_paths[:int(train_image_ratio*len(train_image_paths))]
  valid_image_paths = train_image_paths[int(train_image_ratio*len(train_image_paths)):]

  idx_to_class = {i:j for i, j in enumerate(classes)}
  class_to_idx = {value:key for key,value in idx_to_class.items()}

  train_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(10.),
    transforms.ToTensor()])
  test_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()])

  train_dataset = LoadDataset(train_image_paths, train_transforms)
  validation_dataset = LoadDataset(valid_image_paths, test_transforms)

  train_dataloader = DataLoader(train_dataset, batch_size=_batch_size, shuffle=True)
  validation_dataloader = DataLoader(validation_dataset, batch_size=_batch_size, shuffle=False)

  # Test
  test_image_paths = []
  for data_path in glob.glob(dataset_path + 'test/*'):
    test_image_paths.append(glob.glob(data_path + '/*'))

  test_image_paths = list(flatten(test_image_paths))
  test_dataset = LoadDataset(test_image_paths, test_transforms)
  test_dataloader = DataLoader(test_dataset, batch_size=_batch_size, shuffle=False)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  model = Net().to(device)

  num_classes = len(classes)
  loss_function = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), _learning_rate)

  for e in range(_epoch):
    for idx_batch, item in enumerate(train_dataloader):
      images = item[0].to(device)
      labels = item[1].to(device)

      outputs = model(images)
      loss = loss_function(outputs, labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if (idx_batch + 1) % _batch_size == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(e + 1, _epoch, loss.item()))

  model.eval()

  with torch.no_grad():
    correct = 0
    total = 0

    for item in validation_dataloader:
      images = item[0].to(device)
      labels = item[1].to(device)

      outputs = model(images)

      _, predicted = torch.max(outputs.data, 1)
      total += len(labels)
      correct += (predicted == labels).sum().item()
    
    print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))
