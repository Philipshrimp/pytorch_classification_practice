import glob
from pandas.core.common import flatten
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import yaml


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

    classes = ['kimchi', 'seasoned_chicken', 'ramen', 'rice', 'side_dish',
      'pizza', 'dessert', 'dumpling', 'noodle', 'seafood', 'soup', 
      'fried_chicken', 'snack_food', 'meat', 'hamburger', 'drink', 'porridge']
    idx_to_class = {i:j for i, j in enumerate(classes)}
    class_to_idx = {value:key for key,value in idx_to_class.items()}

    label = image_file_path.split('/')[-2]
    label = class_to_idx[label]

    if self.transform is not None:
      image = self.transform(image)
    
    return image, label

with open('config.yaml') as config_file:
  config = yaml.load(config_file, Loader=yaml.FullLoader)

_batch_size = config['batch_size']
dataset_path = config['file_path']

test_image_paths = []
for data_path in glob.glob(dataset_path + 'test/*'):
  test_image_paths.append(data_path)

test_transforms = transforms.Compose([
  transforms.Resize((128, 128)),
  transforms.ToTensor()])

test_image_paths = list(flatten(test_image_paths))
test_dataset = LoadDataset(test_image_paths, test_transforms)
test_dataloader = DataLoader(test_dataset, batch_size=_batch_size, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = torch.load("./models/weights.pt")

with torch.no_grad():
  correct = 0
  total = 0

  for item in test_dataloader:
    print(len(item))
    images = item[0].to(device)
    labels = item[1].to(device)

    outputs = model(images)

    _, predicted = torch.max(outputs.data, 1)
    total += len(labels)
    correct += (predicted == labels).sum().item()

    print(labels)
print("DONE")