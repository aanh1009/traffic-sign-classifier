import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Subset
from PIL import Image, UnidentifiedImageError
import os
import numpy as np
import torchvision

#transform data
data_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#import dataset + preprocessing
train_dataset = torchvision.datasets.GTSRB(root = 'data', split = 'train', download = True, transform = data_transforms)

portion = 0.1   # Use 10% of the dataset for training and validation
total_size = len(train_dataset)
subset_size = int(portion * total_size)
indices = np.random.choice(total_size, subset_size, replace=False)

train_size = int(0.8 * subset_size)
val_size = subset_size - train_size
train_indices = indices[:train_size]
val_indices = indices[train_size:]

trainset = DataLoader(torch.utils.data.Subset(train_dataset,train_indices), batch_size =64)
testset = DataLoader(torch.utils.data.Subset(train_dataset,val_indices), batch_size = 64)

#define model architecture
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(3, 16, padding =1, kernel_size = 3, stride = 1) #64x64
    self.conv2 = nn.Conv2d(16, 32, padding =1, kernel_size = 3, stride = 1) #32x32
    self.conv3 = nn.Conv2d(32, 64, padding =1, kernel_size = 3, stride = 1) #16x16
    self.dropout = nn.Dropout(0.25)
    self.fc1 = nn.Linear(64*16*16, 256)
    self.fc2 = nn.Linear(256,43)
  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x,2)
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x,2)
    x = F.relu(self.conv3(x))
    x = F.max_pool2d(x,2)
    x = self.dropout(x)
    x = torch.flatten(x,1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    return x
  
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0005)

#train and validate the model
epochs = 24
for epoch in range(epochs):
  running_loss = 0.0
  for images, labels in trainset:
    outputs = model(images)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    running_loss +=loss.item()
  print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(trainset)}')

correct = 0
total = 0
preds = []
label = []
with torch.no_grad():
    for images, labels in testset:
        label += labels.tolist()
        preds += list(model(images).argmax(dim=1).numpy())
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the model on the test images: {:.2f}%'.format(100 * correct / total))


#save the trained model to pickle file
data = {'data_transforms' : data_transforms}
import pickle
with open('saved_steps.pkl','wb') as file:
    pickle.dump(data,file)

with open('saved_steps.pkl','rb') as file:
    data = pickle.load(file)

with open('model.pt', 'wb') as file:
    torch.save(model, file)

with open('model.pt', 'rb') as file:
    the_model = torch.load(file)

# Export to TorchScript
model_scripted = torch.jit.script(model) 
model_scripted.save('model_scripted.pt')

model = torch.jit.load('model_scripted.pt')
model.eval()